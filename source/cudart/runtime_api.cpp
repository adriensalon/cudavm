#include <cudart/cudavm_registration.h>
#include "../cuda/driver_api.hpp"
#include <cstring>
#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <map>
#include <memory>
#include <vector>
#if defined(__clang__) && defined(_WIN32)
#pragma clang diagnostic ignored "-Wdll-attribute-on-redeclaration"
#endif
namespace cudavm {
namespace {
struct failure : std::runtime_error {
    CUresult code;
    failure(CUresult value, const char* text) : std::runtime_error(text), code(value) {}
};
struct symbol_failure : std::runtime_error {
    cudaError_t code;
    symbol_failure(cudaError_t value, const char* text) : std::runtime_error(text), code(value) {}
};
void symbol_require(bool condition, cudaError_t code, const char* text) { if (!condition) throw symbol_failure(code, text); }
void require(bool condition, CUresult code, const char* text) { if (!condition) throw failure(code, text); }
void check(CUresult code) { if (code != CUDA_SUCCESS) throw failure(code, cudavmGetDiagnostic()); }
struct nvidia_managed_binding { void** host; std::string symbol; size_t bytes; const void* alias = nullptr; };
struct program_image {
    cudavmBackend kind;
    std::vector<std::byte> bytes;
    std::unordered_map<std::string, std::vector<cudavmKernelParameter>> kernels;
    bool retiring = false;
    std::vector<nvidia_managed_binding> managed;
    std::shared_ptr<void> managed_image;
    bool managed_conflict = false;
};
struct nvidia_registration { std::string identity; };
struct nvidia_launch_frame { dim3 grid, block; size_t shared; cudaStream_t stream; };
struct binding { std::string identity, symbol; };
struct runtime_module_cache { uint64_t generation = 0; std::unordered_map<std::string, CUmodule> modules; };
struct runtime_function_entry { const void* token; CUcontext owner; uint64_t generation; std::string identity; CUkernel kernel; };
struct state {
    std::mutex mutex;
    uint64_t epoch = 0;
    std::unordered_map<int, CUcontext> primaries;
    std::map<CUcontext, runtime_module_cache> modules;
    std::unordered_map<std::string, program_image> programs;
    std::unordered_map<std::string, CUlibrary> libraries;
    std::unordered_map<const void*, binding> bindings;
    std::unordered_map<const void*, binding> symbols;
    std::unordered_map<CUfunction, runtime_function_entry> function_entries;
    std::map<void**, nvidia_registration> nvidia_registrations;
    uintptr_t next_nvidia_registration = 0x10000000;
    // Inactive weak definitions in registration order; the active one lives
    // in bindings/symbols. An empty vector still marks NVIDIA ownership.
    std::unordered_map<const void*, std::vector<binding>> nvidia_kernel_owners, nvidia_symbol_owners;
    std::unordered_map<const void*, cudaKernel_t> compiler_entries;
    std::map<cudaKernel_t, const void*> compiler_kernels;
    uintptr_t next_compiler_kernel = 0x20000000;
};
state& global() { static state* value = new state; return *value; }
thread_local int runtime_device = 0;
thread_local uint64_t thread_epoch = 0;
thread_local cudaError_t last_error = cudaSuccess;
thread_local std::vector<nvidia_launch_frame> nvidia_launch_frames;
thread_local std::unique_lock<std::mutex>* runtime_mutex_lock = nullptr;
template<class F> CUresult runtime_driver_wait(F function) {
    auto* lock = runtime_mutex_lock;
    require(lock && lock->owns_lock(), CUDA_ERROR_UNKNOWN, "blocking Driver call requires a Runtime boundary");
    lock->unlock();
    CUresult result;
    try { result = function(); } catch (...) { lock->lock(); throw; }
    lock->lock(); return result;
}
cudaError_t runtime_error(CUresult code) {
    switch (code) {
#define DRIVER_STATUS(driver, runtime) case driver: return runtime;
#include "driver_status.inl"
#undef DRIVER_STATUS
    default: return cudaErrorUnknown;
    }
}

template<class F> cudaError_t runtime_call(F function) noexcept {
    cudaError_t result = cudaSuccess;
    try {
        require(!cudavmDriverInHostFunction(), CUDA_ERROR_NOT_PERMITTED, "CUDA calls are forbidden inside host functions");
        std::unique_lock lock(global().mutex);
        struct boundary {
            std::unique_lock<std::mutex>* previous = runtime_mutex_lock;
            explicit boundary(std::unique_lock<std::mutex>& value) { runtime_mutex_lock = &value; }
            ~boundary() { runtime_mutex_lock = previous; }
        } scope(lock);
        auto epoch = cudavmDriverEpoch();
        if (epoch != global().epoch) {
            global().primaries.clear(); global().modules.clear(); global().libraries.clear(); global().function_entries.clear(); global().epoch = epoch;
        }
        if (epoch != thread_epoch) { runtime_device = 0; thread_epoch = epoch; }
        function();
    } catch (const symbol_failure& e) { cudavmDriverSetDiagnostic(e.what()); result = e.code; }
      catch (const failure& e) { cudavmDriverSetDiagnostic(e.what()); result = runtime_error(e.code); }
      catch (const std::bad_alloc&) { cudavmDriverSetDiagnostic("host allocation failed"); result = cudaErrorMemoryAllocation; }
      catch (const std::exception& e) { cudavmDriverSetDiagnostic(e.what()); result = cudaErrorUnknown; }
      catch (...) { cudavmDriverSetDiagnostic("unknown Runtime failure"); result = cudaErrorUnknown; }
    if (result != cudaSuccess) last_error = result;
    return result;
}
CUcontext retain_primary(int device) {
    auto found = global().primaries.find(device);
    if (found != global().primaries.end()) {
        if (cudavmDriverContextExists(found->second)) return found->second;
        // Driver reset preserves our existing retain. Reactivate the primary
        // with a temporary retain, then balance only that temporary ownership.
        if (cudavmDriverPrimaryDevice(found->second) == device) {
            CUcontext handle = nullptr; check(cuDevicePrimaryCtxRetain(&handle, device));
            check(cuDevicePrimaryCtxRelease(device));
            // The legacy reset ABI may have consumed our sole usage. In that
            // case the temporary release deactivated the context again, and
            // we must restore the Runtime's ownership rather than return it.
            if (!cudavmDriverContextExists(handle)) check(cuDevicePrimaryCtxRetain(&handle, device));
            return handle;
        }
    }
    CUcontext handle = nullptr;
    check(cuDevicePrimaryCtxRetain(&handle, device));
    try { global().primaries.insert_or_assign(device, handle); }
    catch (...) { cuDevicePrimaryCtxRelease(device); throw; }
    return handle;
}
CUcontext runtime_current() {
    check(cuInit(0));
    CUcontext handle = nullptr;
    check(cuCtxGetCurrent(&handle));
    if (handle) {
        if (cudavmDriverContextIsPrimary(handle)) {
            handle = retain_primary(cudavmDriverPrimaryDevice(handle));
        }
        return handle;
    }
    int count = 0; check(cuDeviceGetCount(&count));
    require(count != 0, CUDA_ERROR_NO_DEVICE, "selected backend has no GPU devices");
    handle = retain_primary(runtime_device); check(cuCtxSetCurrent(handle)); return handle;
}
void register_program(const cudavmProgram* program, const char* kernel = nullptr, const binding* owner = nullptr, const char* bound_symbol = nullptr) {
    require(program->abi_version == 1 && program->backend == cudavmBackendCuda, CUDA_ERROR_NOT_SUPPORTED, "unsupported compiler module ABI or backend");
    require(program->identity && *program->identity && program->image && program->image_size && (!program->kernel_count || program->kernels),
        CUDA_ERROR_INVALID_IMAGE, "incomplete compiler module descriptor");
    require(static_cast<const std::byte*>(program->image)[program->image_size - 1] == std::byte(0), CUDA_ERROR_INVALID_IMAGE, "PTX image is not terminated");
    program_image candidate{program->backend, {}, {}};
    for (size_t i = 0; i < program->kernel_count; ++i) {
        const auto& item = program->kernels[i];
        require(item.symbol && *item.symbol && (!item.parameter_count || item.parameters), CUDA_ERROR_INVALID_IMAGE, "invalid kernel descriptor");
        std::vector<cudavmKernelParameter> parameters;
        size_t end = 0;
        for (size_t p = 0; p < item.parameter_count; ++p) {
            auto value = item.parameters[p];
            require(value.size && value.alignment && !(value.alignment & (value.alignment - 1))
                && value.offset % value.alignment == 0 && value.offset >= end
                && value.size <= std::numeric_limits<size_t>::max() - value.offset,
                CUDA_ERROR_INVALID_IMAGE, "invalid kernel parameter layout");
            end = value.offset + value.size;
            parameters.push_back(value);
        }
        require(candidate.kernels.emplace(item.symbol, std::move(parameters)).second, CUDA_ERROR_INVALID_IMAGE, "duplicate compiler kernel symbol");
    }
    require(!kernel || candidate.kernels.contains(kernel), CUDA_ERROR_NOT_FOUND, "kernel is not present in its compiler module");
    require(!owner || (owner->identity == program->identity && owner->symbol == bound_symbol),
        CUDA_ERROR_INVALID_IMAGE, "host token already belongs to a different program or symbol");
    auto& programs = global().programs;
    auto found = programs.find(program->identity);
    if (found == programs.end()) {
        candidate.bytes.assign(static_cast<const std::byte*>(program->image), static_cast<const std::byte*>(program->image) + program->image_size);
        programs.emplace(program->identity, std::move(candidate));
    } else {
        const auto& existing = found->second;
        bool matches = existing.kind == candidate.kind && existing.bytes.size() == program->image_size
            && std::memcmp(existing.bytes.data(), program->image, program->image_size) == 0
            && existing.kernels.size() == candidate.kernels.size();
        for (const auto& [name, parameters] : candidate.kernels) {
            auto other = existing.kernels.find(name);
            if (other == existing.kernels.end() || other->second.size() != parameters.size()) { matches = false; break; }
            for (size_t i = 0; i < parameters.size(); ++i)
                matches &= parameters[i].size == other->second[i].size && parameters[i].alignment == other->second[i].alignment && parameters[i].offset == other->second[i].offset;
        }
        require(matches, CUDA_ERROR_INVALID_IMAGE, "compiler module identity reused for different contents");
    }
}
void bind_kernel(const void* token, const cudavmProgram* program, const char* symbol) {
    require(token && program && symbol && *symbol, CUDA_ERROR_INVALID_VALUE, "invalid compiler kernel binding");
    auto found = global().bindings.find(token);
    register_program(program, symbol, found == global().bindings.end() ? nullptr : &found->second, symbol);
    global().bindings.insert_or_assign(token, binding{program->identity, symbol});
}
void bind_symbol(const void* token, const cudavmProgram* program, const char* symbol) {
    require(token && program && symbol && *symbol, CUDA_ERROR_INVALID_VALUE, "invalid compiler symbol binding");
    auto found = global().symbols.find(token);
    register_program(program, nullptr, found == global().symbols.end() ? nullptr : &found->second, symbol);
    global().symbols.insert_or_assign(token, binding{program->identity, symbol});
}

std::shared_ptr<void> runtime_managed_image(program_image& program) {
    if (program.managed.empty()) return {};
    if (!program.managed_image) {
        std::vector<const char*> names; std::vector<const void*> slots; std::vector<size_t> sizes;
        for (auto& entry : program.managed) { names.push_back(entry.symbol.c_str()); slots.push_back(entry.host); sizes.push_back(entry.bytes); }
        void* image = nullptr; check(cudavmDriverCreateManagedRegistration(&image, program.bytes.data(), program.bytes.size(), names.data(), slots.data(), sizes.data(), names.size()));
        program.managed_image = std::shared_ptr<void>(image, [](void* pointer) { cudavmDriverReleaseManagedRegistration(pointer); });
    }
    return program.managed_image;
}
CUlibrary runtime_library(const std::string& identity) {
    auto program = global().programs.find(identity);
    require(program != global().programs.end() && !program->second.retiring, CUDA_ERROR_NOT_PERMITTED, "registered program is unloading");
    require(!program->second.managed_conflict, CUDA_ERROR_NOT_SUPPORTED, "managed registration contains conflicting host slots");
    if (auto it = global().libraries.find(identity); it != global().libraries.end()) return it->second;
    // Unregistration may erase the program while the Driver call releases our
    // mutex. Own the bytes until loading finishes, then reject stale publication.
    auto managed = runtime_managed_image(program->second); auto image = program->second.bytes; auto epoch = global().epoch;
    CUlibrary candidate = nullptr;
    check(runtime_driver_wait([&] { return managed ? cudavmDriverLoadManagedRegistration(&candidate, managed.get())
        : cudavmDriverLoadRegisteredLibrary(&candidate, image.data(), image.size()); }));
    CUlibrary library = nullptr; bool inserted = false;
    try {
        require(epoch == cudavmDriverEpoch(), CUDA_ERROR_NOT_INITIALIZED, "Driver shutdown while loading a registered library");
        auto current = global().programs.find(identity);
        require(current != global().programs.end() && !current->second.retiring, CUDA_ERROR_NOT_PERMITTED, "registered program was unloaded during loading");
        auto selected = global().libraries.emplace(identity, candidate); library = selected.first->second; inserted = selected.second;
    } catch (...) { runtime_driver_wait([&] { return cuLibraryUnload(candidate); }); throw; }
    if (!inserted && candidate != library) check(runtime_driver_wait([&] { return cuLibraryUnload(candidate); }));
    return library;
}
CUkernel runtime_registered_kernel(const void* token) {
    auto found = global().bindings.find(token);
    symbol_require(token && found != global().bindings.end(), cudaErrorInvalidDeviceFunction, "kernel entry is not registered with this Runtime");
    auto value = found->second; check(cuInit(0)); auto library = runtime_library(value.identity);
    CUkernel kernel = nullptr; check(cuLibraryGetKernel(&kernel, library, value.symbol.c_str())); return kernel;
}
CUmodule runtime_module(binding value, CUcontext owner) {
    auto identity = value.identity;
    // Libraries retain registration identities across reset. Their independently
    // owned native modules retain the existing per-context reset contract.
    auto library = runtime_library(identity);
    std::erase_if(global().modules, [](const auto& entry) { return entry.second.generation != cudavmDriverContextGeneration(entry.first); });
    std::erase_if(global().function_entries, [](const auto& entry) { return entry.second.generation != cudavmDriverContextGeneration(entry.second.owner); });
    auto generation = cudavmDriverContextGeneration(owner);
    auto& cache = global().modules[owner]; cache.generation = generation;
    auto& modules = cache.modules;
    auto found = modules.find(identity);
    CUmodule module = nullptr;
    if (found == modules.end()) {
        CUmodule candidate = nullptr;
        check(runtime_driver_wait([&] { return cuLibraryGetModule(&candidate, library); }));
        auto current = global().programs.find(identity); auto loaded = global().libraries.find(identity);
        require(current != global().programs.end() && !current->second.retiring && loaded != global().libraries.end() && loaded->second == library,
            CUDA_ERROR_NOT_PERMITTED, "registered program was unloaded during module binding");
        require(generation && generation == cudavmDriverContextGeneration(owner), CUDA_ERROR_INVALID_CONTEXT, "context was reset or destroyed while loading a module");
        auto& target = global().modules[owner]; target.generation = generation;
        module = target.modules.emplace(identity, candidate).first->second;
    } else module = found->second;
    return module;
}
CUfunction runtime_function(binding value, CUcontext owner) {
    auto symbol = value.symbol;
    CUfunction function = nullptr; check(cuModuleGetFunction(&function, runtime_module(value, owner), symbol.c_str()));
    if (!global().function_entries.contains(function)) for (auto& [token, entry] : global().bindings)
        if (entry.identity == value.identity && entry.symbol == value.symbol) {
            CUkernel kernel = nullptr; check(cuLibraryGetKernel(&kernel, global().libraries.at(value.identity), symbol.c_str()));
            global().function_entries.emplace(function, runtime_function_entry{token, owner, cudavmDriverContextGeneration(owner), value.identity, kernel}); break;
        }
    return function;
}
CUfunction runtime_kernel(const void* token) {
    auto owner = runtime_current();
    if (cudavmDriverIsLibraryKernel(token)) {
        CUfunction function = nullptr; check(cuKernelGetFunction(&function, reinterpret_cast<CUkernel>(const_cast<void*>(token)))); return function;
    }
    auto found = global().bindings.find(token);
    require(token && found != global().bindings.end(), CUDA_ERROR_NOT_FOUND, "kernel is not registered with this cudavm runtime");
    return runtime_function(found->second, owner);
}
struct symbol_storage { CUdeviceptr address; size_t size; };
symbol_storage runtime_symbol(const void* token) {
    auto it = global().symbols.find(token);
    symbol_require(token && it != global().symbols.end(), cudaErrorInvalidSymbol, "symbol is not registered with this cudavm runtime");
    auto value = it->second;
    auto module = runtime_module(value, runtime_current());
    symbol_storage result{};
    auto status = runtime_driver_wait([&] { return cuModuleGetGlobal(&result.address, &result.size, module, value.symbol.c_str()); });
    symbol_require(status != CUDA_ERROR_NOT_FOUND, cudaErrorInvalidSymbol, "symbol is not present in its compiler module");
    check(status);
    return result;
}
CUdeviceptr symbol_region(symbol_storage storage, size_t bytes, size_t offset) {
    require(offset <= storage.size && bytes <= storage.size - offset, CUDA_ERROR_INVALID_VALUE, "copy exceeds symbol storage");
    return storage.address + offset;
}
#include "nvidia_registration_state.inl"
#include "arrays.inl"
#include "image_objects.inl"
#include "memory.inl"
#include "launch_state.inl"
#include "device_state.inl"
#include "stream_state.inl"
#include "graphs.inl"
#include "graph_memory.inl"
}
} // namespace cudavm
using namespace cudavm;
#define RUNTIME_API(name, parameters, ...) extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI name parameters { return runtime_call([&] { __VA_ARGS__; }); }
#include "nvidia_registration_exports.inl"
#include "interop.inl"
#include "memory_exports.inl"
#include "array_exports.inl"
#include "image_object_exports.inl"
#include "host_memory_exports.inl"
#include "stream_exports.inl"
#include "async_memory_exports.inl"
#include "memory_pool_exports.inl"
#include "graph_exports.inl"
#include "graph_memory_exports.inl"
#include "graph_advanced_exports.inl"
#include "graph_interop_exports.inl"
#include "graph_generic_exports.inl"
#include "graph_lifetime_exports.inl"
#include "graph_priority_exports.inl"
#include "graph_parameter_exports.inl"
#include "graph_diagnostic_exports.inl"
#include "capture_exports.inl"
#include "procedure_address_exports.inl"
#include "context_exports.inl"
#include "function_exports.inl"
extern "C" CUDAVM_CUDART_API cudaError_t cudavmBindKernel(const void* token, const cudavmProgram* program, const char* symbol) {
    return runtime_call([&] { bind_kernel(token, program, symbol); });
}
extern "C" CUDAVM_CUDART_API cudaError_t cudavmBindSymbol(const void* token, const cudavmProgram* program, const char* symbol) {
    return runtime_call([&] { bind_symbol(token, program, symbol); });
}
RUNTIME_API(cudaGetSymbolAddress, (void** output, const void* symbol),
    require(output, CUDA_ERROR_INVALID_VALUE, "null symbol address output"); auto storage = runtime_symbol(symbol);
    *output = reinterpret_cast<void*>(static_cast<std::uintptr_t>(storage.address)))
RUNTIME_API(cudaGetSymbolSize, (size_t* output, const void* symbol),
    require(output, CUDA_ERROR_INVALID_VALUE, "null symbol size output"); auto storage = runtime_symbol(symbol); *output = storage.size)
RUNTIME_API(cudaMemcpyToSymbol, (const void* symbol, const void* source, size_t bytes, size_t offset, cudaMemcpyKind kind),
    auto destination = symbol_region(runtime_symbol(symbol), bytes, offset);
    symbol_require(kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice || kind == cudaMemcpyDefault,
        cudaErrorInvalidMemcpyDirection, "invalid copy-to-symbol direction");
    if (!bytes) return;
    auto pointer = reinterpret_cast<std::uintptr_t>(source);
    bool device = cudavmDriverIsDevicePointer(pointer);
    if (kind == cudaMemcpyDefault) kind = device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    symbol_require(device == (kind == cudaMemcpyDeviceToDevice), cudaErrorInvalidMemcpyDirection, "symbol copy direction disagrees with source ownership");
    if (device) check(runtime_driver_wait([&] { return cuMemcpyDtoD(destination, pointer, bytes); })); else check(runtime_driver_wait([&] { return cuMemcpyHtoD(destination, source, bytes); })))
RUNTIME_API(cudaMemcpyFromSymbol, (void* destination, const void* symbol, size_t bytes, size_t offset, cudaMemcpyKind kind),
    auto source = symbol_region(runtime_symbol(symbol), bytes, offset);
    symbol_require(kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice || kind == cudaMemcpyDefault,
        cudaErrorInvalidMemcpyDirection, "invalid copy-from-symbol direction");
    if (!bytes) return;
    auto pointer = reinterpret_cast<std::uintptr_t>(destination);
    bool device = cudavmDriverIsDevicePointer(pointer);
    if (kind == cudaMemcpyDefault) kind = device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
    symbol_require(device == (kind == cudaMemcpyDeviceToDevice), cudaErrorInvalidMemcpyDirection, "symbol copy direction disagrees with destination ownership");
    if (device) check(runtime_driver_wait([&] { return cuMemcpyDtoD(pointer, source, bytes); })); else check(runtime_driver_wait([&] { return cuMemcpyDtoH(destination, source, bytes); })))
namespace cudavm {
namespace {
struct runtime_library_launch_scope {
    bool pushed = false;
    runtime_library_launch_scope(const void* token, cudaStream_t stream) {
        if (!cudavmDriverIsLibraryKernel(token) || !stream || stream == cudaStreamLegacy || stream == cudaStreamPerThread) return;
        CUcontext owner = nullptr; check(cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &owner)); check(cuCtxPushCurrent(owner)); pushed = true;
    }
    ~runtime_library_launch_scope() { if (pushed) { CUcontext owner = nullptr; cuCtxPopCurrent(&owner); } }
};
cudaError_t runtime_launch(const void* token, dim3 grid, dim3 block, void** parameters, size_t shared, cudaStream_t stream,
                           bool cooperative, const cudaLaunchConfig_t* extended, bool compiler = false) {
    bool invalid_configuration = !grid.x || !grid.y || !grid.z || !block.x || !block.y || !block.z || shared > std::numeric_limits<unsigned>::max();
    auto result = runtime_call([&] {
        if (compiler) token = compiler_kernel_binding(reinterpret_cast<cudaKernel_t>(const_cast<void*>(token)));
        runtime_library_launch_scope binding(token, stream);
        runtime_current();
        require(!invalid_configuration, CUDA_ERROR_INVALID_VALUE, "invalid Runtime launch configuration");
        auto function = runtime_kernel(token);
        auto launch_function = cudavmDriverIsLibraryKernel(token) ? reinterpret_cast<CUfunction>(const_cast<void*>(token)) : function;
        size_t parameter_count = 0; check(cuFuncGetParamCount(function, &parameter_count));
        require(!parameter_count || parameters, CUDA_ERROR_INVALID_VALUE, "missing Runtime kernel parameters");
        for (size_t i = 0; i < parameter_count; ++i) require(parameters[i], CUDA_ERROR_INVALID_VALUE, "null Runtime kernel parameter storage");
        int threads = 0, dynamic_shared = 0;
        check(cuFuncGetAttribute(&threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function));
        check(cuFuncGetAttribute(&dynamic_shared, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, function));
        uint64_t product = block.x;
        invalid_configuration = shared > size_t(dynamic_shared) || product > unsigned(threads) ||
            block.y > unsigned(threads) / product;
        if (!invalid_configuration) { product *= block.y; invalid_configuration = block.z > unsigned(threads) / product; }
        CUdevice device = 0; check(cuCtxGetDevice(&device));
        const unsigned grids[]{grid.x, grid.y, grid.z}, blocks[]{block.x, block.y, block.z};
        for (unsigned axis = 0; axis < 3 && !invalid_configuration; ++axis) {
            int maximum = 0;
            auto status = cuDeviceGetAttribute(&maximum, static_cast<CUdevice_attribute>(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X + axis), device);
            if (status == CUDA_SUCCESS) invalid_configuration = blocks[axis] > unsigned(maximum);
            else if (status != CUDA_ERROR_NOT_SUPPORTED) check(status);
            status = cuDeviceGetAttribute(&maximum, static_cast<CUdevice_attribute>(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X + axis), device);
            if (status == CUDA_SUCCESS) invalid_configuration |= grids[axis] > unsigned(maximum);
            else if (status != CUDA_ERROR_NOT_SUPPORTED) check(status);
        }
        require(!invalid_configuration, CUDA_ERROR_INVALID_VALUE, "Runtime launch exceeds function or device resources");
        if (extended) {
            std::vector<CUlaunchAttribute> attributes; auto config = driver_launch_config(extended, attributes);
            check(runtime_driver_wait([&] { return cuLaunchKernelEx(&config, launch_function, parameters, nullptr); }));
        } else if (cooperative) {
            check(runtime_driver_wait([&] { return cuLaunchCooperativeKernel(launch_function, grid.x, grid.y, grid.z, block.x, block.y, block.z, static_cast<unsigned>(shared), reinterpret_cast<CUstream>(stream), parameters); }));
        } else {
            check(runtime_driver_wait([&] { return cuLaunchKernel(launch_function, grid.x, grid.y, grid.z, block.x, block.y, block.z, static_cast<unsigned>(shared), reinterpret_cast<CUstream>(stream), parameters, nullptr); }));
        }
    });
    if (invalid_configuration && result == cudaErrorInvalidValue) last_error = result = cudaErrorInvalidConfiguration;
    return result;
}
#include "reset_state.inl"
}
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaLaunchKernel(const void* token, dim3 grid, dim3 block, void** parameters, size_t shared, cudaStream_t stream) {
    return runtime_launch(token, grid, block, parameters, shared, stream, false, nullptr);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI __cudaLaunchKernel(cudaKernel_t kernel, dim3 grid, dim3 block, void** parameters, size_t shared, cudaStream_t stream) {
    return runtime_launch(kernel, grid, block, parameters, shared, stream, false, nullptr, true);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI __cudaLaunchKernel_ptsz(cudaKernel_t kernel, dim3 grid, dim3 block, void** parameters, size_t shared, cudaStream_t stream) {
    return runtime_launch(kernel, grid, block, parameters, shared, stream ? stream : cudaStreamPerThread, false, nullptr, true);
}
#include "launch_exports.inl"
#include "library_exports.inl"
#include "device_exports.inl"
RUNTIME_API(cudaGetDeviceCount, (int* count), require(count, CUDA_ERROR_INVALID_VALUE, "null device count"); check(cuInit(0)); int value = 0; check(cuDeviceGetCount(&value)); require(value != 0, CUDA_ERROR_NO_DEVICE, "selected backend has no GPU devices"); *count = value)
RUNTIME_API(cudaDeviceGetAttribute, (int* output, cudaDeviceAttr attr, int device), require(output && attr > 0 && attr < cudaDevAttrMax, CUDA_ERROR_INVALID_VALUE, "invalid Runtime device attribute query"); check(cuInit(0)); check(cuDeviceGetAttribute(output, static_cast<CUdevice_attribute>(attr), device)))
RUNTIME_API(cudaSetDevice, (int device), check(cuInit(0)); auto owner = retain_primary(device); check(cuCtxSetCurrent(owner)); runtime_device = device)
RUNTIME_API(cudaGetDevice, (int* device), require(device, CUDA_ERROR_INVALID_VALUE, "null device output"); check(cuInit(0));
    CUcontext owner = nullptr; check(cuCtxGetCurrent(&owner)); int selected = owner ? cudavmDriverPrimaryDevice(owner) : runtime_device;
    if (owner && selected < 0) check(cuCtxGetDevice(&selected));
    else { CUdevice native = 0; check(cuDeviceGet(&native, selected)); selected = native; }
    *device = selected)
RUNTIME_API(cudaMalloc, (void** output, size_t bytes), require(output, CUDA_ERROR_INVALID_VALUE, "null allocation output"); runtime_current(); CUdeviceptr pointer = 0; if (bytes) check(runtime_driver_wait([&] { return cuMemAlloc(&pointer, bytes); })); *output = reinterpret_cast<void*>(static_cast<std::uintptr_t>(pointer)))
RUNTIME_API(cudaFree, (void* pointer), runtime_current(); if (pointer) check(runtime_driver_wait([&] { return cuMemFree(reinterpret_cast<std::uintptr_t>(pointer)); })))
RUNTIME_API(cudaMemcpy, (void* destination, const void* source, size_t bytes, cudaMemcpyKind kind),
    runtime_current(); require(kind >= cudaMemcpyHostToHost && kind <= cudaMemcpyDefault, CUDA_ERROR_INVALID_VALUE, "invalid copy direction");
    if (!bytes) return;
    CUdeviceptr to = reinterpret_cast<std::uintptr_t>(destination), from = reinterpret_cast<std::uintptr_t>(source);
    auto types = runtime_copy_types(destination, source, kind);
    kind = types.to == CU_MEMORYTYPE_DEVICE ? (types.from == CU_MEMORYTYPE_DEVICE ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice)
                                          : (types.from == CU_MEMORYTYPE_DEVICE ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
    switch (kind) {
    case cudaMemcpyHostToHost: require(destination && source, CUDA_ERROR_INVALID_VALUE, "null host copy pointer"); std::memcpy(destination, source, bytes); break;
    case cudaMemcpyHostToDevice: check(runtime_driver_wait([&] { return cuMemcpyHtoD(to, source, bytes); })); break;
    case cudaMemcpyDeviceToHost: check(runtime_driver_wait([&] { return cuMemcpyDtoH(destination, from, bytes); })); break;
    case cudaMemcpyDeviceToDevice: check(runtime_driver_wait([&] { return cuMemcpyDtoD(to, from, bytes); })); break;
    default: break;
    })
RUNTIME_API(cudaMemset, (void* destination, int pattern, size_t bytes), runtime_current(); check(runtime_driver_wait([&] { return cuMemsetD8(reinterpret_cast<std::uintptr_t>(destination), static_cast<unsigned char>(pattern), bytes); })))
RUNTIME_API(cudaDeviceSynchronize, (), runtime_current(); check(runtime_driver_wait([&] { return cuCtxSynchronize(); })))
RUNTIME_API(cudaDeviceReset, (), runtime_reset())
RUNTIME_API(cudaRuntimeGetVersion, (int* version), require(version, CUDA_ERROR_INVALID_VALUE, "null version output"); *version = CUDART_VERSION)
RUNTIME_API(cudaDriverGetVersion, (int* version), check(cuDriverGetVersion(version)))
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaGetLastError() { auto result = last_error; last_error = cudaSuccess; return result; }
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaPeekAtLastError() { return last_error; }
extern "C" CUDAVM_CUDART_API const char* CUDARTAPI cudaGetErrorName(cudaError_t error) {
    switch (error) {
#define RUNTIME_ERROR(value, description) case value: return #value;
#include "error_catalog.inl"
#undef RUNTIME_ERROR
    default: return "unrecognized error code";
    }
}
extern "C" CUDAVM_CUDART_API const char* CUDARTAPI cudaGetErrorString(cudaError_t error) {
    switch (error) {
#define RUNTIME_ERROR(value, description) case value: return description;
#include "error_catalog.inl"
#undef RUNTIME_ERROR
    default: return "unrecognized error code";
    }
}
