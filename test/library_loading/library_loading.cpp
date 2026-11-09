#include "test_framework.hpp"
#include "module_fixtures.hpp"
#include "../module_loading/module_oracle.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

namespace cudavm {
using library_loading_bytes = std::vector<unsigned char>;
std::string_view library_loading_case = "contextless";
void library_loading_runtime(cudaError_t status) {
    require(status == cudaSuccess, std::string("library Runtime API: ") + cudaGetErrorString(status));
}
template<class T> struct library_loading_output {
    static_assert(alignof(T) <= 16);
    std::array<unsigned char, 16> before; T value{}; std::array<unsigned char, 16> after;
    library_loading_output() { before.fill(0x35); after.fill(0x79); }
    T get() const {
        require(std::all_of(before.begin(), before.end(), [](auto byte) { return byte == 0x35; })
            && std::all_of(after.begin(), after.end(), [](auto byte) { return byte == 0x79; }), "library query exact output guards"); return value;
    }
};
struct library_loading_restore {
    CUcontext value{};
    library_loading_restore() { CONFORMANCE_CUDA(cuCtxGetCurrent(&value)); }
    ~library_loading_restore() { cuCtxSetCurrent(value); }
};
void library_loading_current(CUcontext expected) {
    library_loading_output<CUcontext> output; CONFORMANCE_CUDA(cuCtxGetCurrent(&output.value));
    require(output.get() == expected, "library operation preserves public current context");
}
struct library_loading_context {
    CUcontext value{}, previous{};
    explicit library_loading_context(CUdevice device) {
        CONFORMANCE_CUDA(cuCtxGetCurrent(&previous)); CONFORMANCE_CUDA(cuCtxCreate(&value, nullptr, 0, device));
    }
    ~library_loading_context() { if (value) { cuCtxSetCurrent(value); cuCtxDestroy(value); cuCtxSetCurrent(previous); } }
    library_loading_context(const library_loading_context&) = delete;
};
library_loading_bytes library_loading_image(bool source) {
    if (!source) { library_loading_bytes bytes(module_loading_ptx.begin(), module_loading_ptx.end()); bytes.push_back(0); return bytes; }
    compiler_fixture_program program(std::string(module_loading_source), "library_loading.cu");
    const auto status = program.compile({"-arch=sm_75", "--std=c++17"});
    require(status == NVRTC_SUCCESS, "library source compilation: " + program.text(true));
    return compiler_fixture_bytes([&](size_t* size) { compiler_fixture_program::check(nvrtcGetCUBINSize(program.value, size)); },
        [&](unsigned char* bytes) { compiler_fixture_program::check(nvrtcGetCUBIN(program.value, reinterpret_cast<char*>(bytes))); });
}
struct library_loading_file {
    std::filesystem::path root, path;
    explicit library_loading_file(const library_loading_bytes& bytes) {
        root = std::filesystem::current_path() / ("library_loading_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        require(std::filesystem::create_directory(root), "create owned library file directory"); path = root / "input.image";
        try {
            std::ofstream output(path, std::ios::binary); output.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
            output.close(); require(bool(output), "write exact library file bytes");
        } catch (...) { cleanup(); throw; }
    }
    ~library_loading_file() { cleanup(); }
    void cleanup() const { std::error_code error; std::filesystem::remove(path, error); std::filesystem::remove(root, error); }
    library_loading_file(const library_loading_file&) = delete;
};
struct library_loading_program {
    CUlibrary value{};
    bool runtime;
    library_loading_program(const library_loading_bytes& image, bool use_runtime, bool file = false) : runtime(use_runtime) {
        auto bytes = image; library_loading_output<CUlibrary> output;
        library_loading_output<cudaLibrary_t> runtime_output;
        if (file) {
            library_loading_file input(bytes); auto path = input.path.string();
            if (runtime) library_loading_runtime(cudaLibraryLoadFromFile(&runtime_output.value, path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
            else CONFORMANCE_CUDA(cuLibraryLoadFromFile(&output.value, path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
            std::fill(path.begin(), path.end(), '!'); require(std::filesystem::remove(input.path), "retire library file before lookup");
        } else if (runtime) library_loading_runtime(cudaLibraryLoadData(&runtime_output.value, bytes.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
        else CONFORMANCE_CUDA(cuLibraryLoadData(&output.value, bytes.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
        value = runtime ? reinterpret_cast<CUlibrary>(runtime_output.get()) : output.get();
        require(value != nullptr, "library load publishes guarded handle"); std::fill(bytes.begin(), bytes.end(), 0x57);
    }
    ~library_loading_program() { if (value) cuLibraryUnload(value); }
    library_loading_program(const library_loading_program&) = delete;
    cudaLibrary_t runtime_value() const { return reinterpret_cast<cudaLibrary_t>(value); }
    CUkernel kernel(const char* name) const {
        library_loading_output<CUkernel> first; library_loading_output<cudaKernel_t> second;
        CONFORMANCE_CUDA(cuLibraryGetKernel(&first.value, value, name));
        library_loading_runtime(cudaLibraryGetKernel(&second.value, runtime_value(), name));
        require(first.get() && first.value == reinterpret_cast<CUkernel>(second.get()), "Driver/Runtime library kernel identity agrees"); return first.value;
    }
    CUdeviceptr global(const char* name) const {
        library_loading_output<CUdeviceptr> pointer; library_loading_output<size_t> size, runtime_size;
        library_loading_output<void*> runtime_pointer;
        CONFORMANCE_CUDA(cuLibraryGetGlobal(&pointer.value, &size.value, value, name));
        library_loading_runtime(cudaLibraryGetGlobal(&runtime_pointer.value, &runtime_size.value, runtime_value(), name));
        require(pointer.get() && size.get() == 4 && runtime_size.get() == 4
            && pointer.value == uintptr_t(runtime_pointer.get()), "Driver/Runtime library global address and extent agree"); return pointer.value;
    }
    void close() {
        if (runtime) library_loading_runtime(cudaLibraryUnload(runtime_value())); else CONFORMANCE_CUDA(cuLibraryUnload(value));
        value = nullptr;
    }
};
void library_loading_metadata(const library_loading_program& program) {
    library_loading_output<unsigned> count, runtime_count;
    CONFORMANCE_CUDA(cuLibraryGetKernelCount(&count.value, program.value));
    library_loading_runtime(cudaLibraryGetKernelCount(&runtime_count.value, program.runtime_value()));
    require(count.get() == 2 && runtime_count.get() == 2, "both library inventories match the two declared kernels");
    const auto alpha = program.kernel("module_alpha"), beta = program.kernel("module_beta");
    require(alpha != beta, "distinct contextless kernel identities");
    for (bool runtime : {false, true}) for (unsigned capacity : {0u, 1u, 2u, 3u}) {
        library_loading_output<std::array<CUkernel, 3>> output;
        library_loading_output<std::array<cudaKernel_t, 3>> runtime_output;
        if (runtime) library_loading_runtime(cudaLibraryEnumerateKernels(runtime_output.value.data(), capacity, program.runtime_value()));
        else CONFORMANCE_CUDA(cuLibraryEnumerateKernels(output.value.data(), capacity, program.value));
        auto values = output.get(); const auto runtime_values = runtime_output.get();
        if (runtime) std::transform(runtime_values.begin(), runtime_values.end(), values.begin(), [](auto kernel) { return reinterpret_cast<CUkernel>(kernel); });
        for (unsigned i = 0; i < values.size(); ++i)
            require(i < std::min(capacity, 2u) ? values[i] == alpha || values[i] == beta : values[i] == nullptr,
                "library enumeration honors capacity and complete kernel inventory");
        if (capacity >= 2) require(values[0] != values[1], "library enumeration has no duplicates");
    }
    for (const auto& [kernel, name] : std::array<std::pair<CUkernel, const char*>, 2>{{{alpha, "module_alpha"}, {beta, "module_beta"}}}) {
        library_loading_output<CUlibrary> owner; library_loading_output<const char*> text;
        CONFORMANCE_CUDA(cuKernelGetLibrary(&owner.value, kernel)); CONFORMANCE_CUDA(cuKernelGetName(&text.value, kernel));
        require(owner.get() == program.value && text.get() && std::string_view(text.value) == name, "contextless kernel name and parent library");
        library_loading_output<size_t> parameters; CONFORMANCE_CUDA(cuKernelGetParamCount(kernel, &parameters.value));
        require(parameters.get() == 3, "library declared parameter count");
        for (size_t index = 0; index < 3; ++index) {
            library_loading_output<size_t> offset, bytes;
            CONFORMANCE_CUDA(cuKernelGetParamInfo(kernel, index, &offset.value, &bytes.value));
            require(offset.get() == index * 8 && bytes.get() == (index < 2 ? 8u : 4u), "library parameter metadata agrees with independent ABI");
        }
    }
}
void library_loading_execute(const library_loading_program& program, uint32_t expected_seed = 13, bool no_current = false) {
    constexpr uint32_t guard = 0xa95b371d; constexpr unsigned capacity = 41, guards = 9;
    const auto seed_address = program.global("module_seed"), bias_address = program.global("module_bias");
    std::array<uint32_t, 3> seed{guard, guard, guard}, bias{guard, guard, guard};
    CONFORMANCE_CUDA(cuMemcpyDtoH(seed.data() + 1, seed_address, 4)); CONFORMANCE_CUDA(cuMemcpyDtoH(bias.data() + 1, bias_address, 4));
    require(seed == std::array<uint32_t, 3>{guard, expected_seed, guard} && bias == std::array<uint32_t, 3>{guard, 29, guard},
        "library context storage and host guards");
    std::vector<uint32_t> input(capacity + guards * 2, guard), expected(input.size()), actual(input.size()), preserved(input.size());
    constexpr std::array<uint32_t, 4> edges{0, 0xffffffff, 0x80000000, 0x7fffffff};
    for (unsigned i = 0; i < capacity; ++i) input[guards + i] = i < edges.size() ? edges[i] : i * 0x193759bdu;
    device_buffer source(input.size() * 4), output(input.size() * 4);
    CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
    auto in = source.value + guards * 4, out = output.value + guards * 4;
    struct stream_owner { CUstream value{}; ~stream_owner() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } } } stream;
    CONFORMANCE_CUDA(cuStreamCreate(&stream.value, CU_STREAM_NON_BLOCKING));
    library_loading_restore restore;
    for (bool beta : {false, true}) for (bool runtime : {false, true}) for (unsigned count : {0u, 1u, 33u, capacity}) {
        std::fill(expected.begin(), expected.end(), guard);
        CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
        const auto kernel = program.kernel(beta ? "module_beta" : "module_alpha");
        void* parameters[]{&out, &in, &count};
        if (no_current) CONFORMANCE_CUDA(cuCtxSetCurrent(nullptr));
        if (runtime) library_loading_runtime(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(2), dim3(31), parameters, 0, reinterpret_cast<cudaStream_t>(stream.value)));
        else CONFORMANCE_CUDA(cuLaunchKernel(reinterpret_cast<CUfunction>(kernel), 2, 1, 1, 31, 1, 1, 0, stream.value, parameters, nullptr));
        library_loading_current(no_current ? nullptr : restore.value);
        if (no_current) CONFORMANCE_CUDA(cuCtxSetCurrent(restore.value));
        CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
        for (unsigned i = 0; i < count; ++i) expected[guards + i] = module_boundary_expected(beta, input[guards + i], expected_seed, i);
        expect_words(actual, expected, "library public launches and entire guarded output");
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
        expect_words(preserved, input, "library launches preserve all input bytes");
    }
    CONFORMANCE_CUDA(cuMemcpyDtoH(seed.data() + 1, seed_address, 4)); CONFORMANCE_CUDA(cuMemcpyDtoH(bias.data() + 1, bias_address, 4));
    require(seed == std::array<uint32_t, 3>{guard, expected_seed, guard} && bias == std::array<uint32_t, 3>{guard, 29, guard},
        "library launches preserve context storage and host guards");
}
void library_loading_contexts(const library_loading_program& program, CUdevice device) {
    library_loading_restore restore;
    const auto kernel = program.kernel("module_alpha"); const auto first_seed = program.global("module_seed");
    library_loading_output<CUmodule> first_module; library_loading_output<CUfunction> first_function;
    CONFORMANCE_CUDA(cuLibraryGetModule(&first_module.value, program.value)); CONFORMANCE_CUDA(cuKernelGetFunction(&first_function.value, kernel));
    first_module.get(); first_function.get(); uint32_t value = 71; CONFORMANCE_CUDA(cuMemcpyHtoD(first_seed, &value, 4));
    {
        library_loading_context second(device); library_loading_output<CUmodule> module; library_loading_output<CUfunction> function;
        CONFORMANCE_CUDA(cuLibraryGetModule(&module.value, program.value)); CONFORMANCE_CUDA(cuKernelGetFunction(&function.value, kernel));
        require(module.get() != first_module.value && function.get() != first_function.value, "simultaneous contexts own distinct module/function bindings");
        CUfunction alias{}; CONFORMANCE_CUDA(cuModuleGetFunction(&alias, module.value, "module_alpha"));
        require(alias == function.value && kernel == program.kernel("module_alpha"), "contextless identity selects its context's bound function");
        require(cuModuleUnload(module.value) == CUDA_ERROR_NOT_PERMITTED, "borrowed library module cannot be unloaded directly");
        library_loading_execute(program);
    }
    library_loading_current(restore.value); library_loading_execute(program, 71);
    {
        library_loading_context replacement(device); require(kernel == program.kernel("module_alpha"), "library kernel survives retired context generation");
        library_loading_execute(program);
    }
    library_loading_current(restore.value); library_loading_execute(program, 71);
}
int library_loading_dynamic(CUfunction function) {
    library_loading_output<int> output; CONFORMANCE_CUDA(cuFuncGetAttribute(&output.value, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, function)); return output.get();
}
void library_loading_resources(const library_loading_program& program, CUdevice device) {
    const auto kernel = program.kernel("module_alpha");
    CONFORMANCE_CUDA(cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 64, kernel, device));
    CUfunction first{}; CONFORMANCE_CUDA(cuKernelGetFunction(&first, kernel)); require(library_loading_dynamic(first) == 64, "device-wide setting reaches current binding");
    CONFORMANCE_CUDA(cuFuncSetAttribute(first, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 128));
    library_loading_runtime(cudaKernelSetAttributeForDevice(reinterpret_cast<cudaKernel_t>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, 96, device));
    require(library_loading_dynamic(first) == 128, "local function setting survives a later Runtime device-wide update");
    {
        library_loading_context second(device); CUfunction function{}; CONFORMANCE_CUDA(cuKernelGetFunction(&function, kernel));
        require(library_loading_dynamic(function) == 96, "device-wide setting reaches a future context binding");
        CONFORMANCE_CUDA(cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 80, kernel, device));
        require(library_loading_dynamic(function) == 80, "device-wide setting updates existing binding without override");
        library_loading_execute(program);
    }
    require(library_loading_dynamic(first) == 128, "another context's update preserves local function override"); library_loading_execute(program);
}
struct library_loading_gate {
    std::mutex mutex; std::condition_variable changed; bool entered = false, released = false, expired = false;
    static void CUDA_CB hold(void* argument) {
        auto& gate = *static_cast<library_loading_gate*>(argument); std::unique_lock lock(gate.mutex);
        gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(15), [&] { return gate.released; })) gate.expired = true;
    }
    void await() { std::unique_lock lock(mutex); require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "library retirement gate entry deadline"); }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
    ~library_loading_gate() { release(); }
};
void library_loading_retirement(library_loading_program& program) {
    library_loading_execute(program); const auto kernel = program.kernel("module_alpha");
    constexpr uint32_t guard = 0xa95b371d; std::array<uint32_t, 17> input, expected, actual;
    input.fill(guard); expected.fill(guard); input[8] = 0xfffffff1u;
    device_buffer source(sizeof(input)), output(sizeof(expected));
    CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), sizeof(input))); CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), sizeof(expected)));
    auto in = source.value + 32, out = output.value + 32; unsigned count = 1; void* parameters[]{&out, &in, &count};
    CUstream stream{}; CONFORMANCE_CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    // Materialize this exact launch shape before holding accepted work.
    CONFORMANCE_CUDA(cuLaunchKernel(reinterpret_cast<CUfunction>(kernel), 1, 1, 1, 1, 1, 1, 0, stream, parameters, nullptr));
    CONFORMANCE_CUDA(cuStreamSynchronize(stream)); CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), sizeof(expected)));
    library_loading_gate gate;
    struct cleanup { library_loading_gate& gate; CUstream& stream; ~cleanup() { gate.release(); if (stream) { cuStreamSynchronize(stream); cuStreamDestroy(stream); } } } cleanup{gate, stream};
    CONFORMANCE_CUDA(cuLaunchHostFunc(stream, library_loading_gate::hold, &gate)); gate.await();
    CONFORMANCE_CUDA(cuLaunchKernel(reinterpret_cast<CUfunction>(kernel), 1, 1, 1, 1, 1, 1, 0, stream, parameters, nullptr));
    std::mutex mutex; std::condition_variable changed; bool entered = false, done = false; CUresult status = CUDA_ERROR_UNKNOWN;
    cudaError_t runtime_status = cudaErrorUnknown;
    std::thread worker([&] {
        { std::lock_guard lock(mutex); entered = true; changed.notify_all(); }
        if (program.runtime) runtime_status = cudaLibraryUnload(program.runtime_value()); else status = cuLibraryUnload(program.value);
        { std::lock_guard lock(mutex); done = true; changed.notify_all(); }
    });
    struct join_worker {
        library_loading_gate& gate; std::thread& worker;
        ~join_worker() { gate.release(); if (worker.joinable()) worker.join(); }
    } join{gate, worker};
    bool started = false, premature = false;
    { std::unique_lock lock(mutex); started = changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; });
      if (started) premature = changed.wait_for(lock, std::chrono::milliseconds(40), [&] { return done; }); }
    gate.release(); worker.join();
    if (program.runtime) library_loading_runtime(runtime_status); else CONFORMANCE_CUDA(status); program.value = nullptr;
    bool expired; { std::lock_guard lock(gate.mutex); expired = gate.expired; }
    require(started && !premature && !expired, "library unload waits for accepted gated users");
    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, sizeof(actual))); expected[8] = module_boundary_expected(false, input[8], 13, 0);
    expect_words(actual, expected, "library retirement preserves queued result and complete allocation guards");
    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), source.value, sizeof(actual))); expect_words(actual, input, "library retirement preserves queued input");
}
void library_loading_validation(const library_loading_program& program, const library_loading_bytes& image) {
    library_loading_output<CUlibrary> output;
    library_loading_output<cudaLibrary_t> runtime_output;
    require(cuLibraryLoadData(nullptr, image.data(), nullptr, nullptr, 0, nullptr, nullptr, 0) == CUDA_ERROR_INVALID_VALUE
        && cuLibraryLoadData(&output.value, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0) == CUDA_ERROR_INVALID_VALUE
        && cuLibraryLoadData(&output.value, image.data(), nullptr, nullptr, 1, nullptr, nullptr, 0) == CUDA_ERROR_INVALID_VALUE,
        "Driver invalid library load arguments"); output.get();
    require(cudaLibraryLoadData(nullptr, image.data(), nullptr, nullptr, 0, nullptr, nullptr, 0) == cudaErrorInvalidValue
        && cudaLibraryLoadData(&runtime_output.value, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0) == cudaErrorInvalidValue,
        "Runtime invalid library load arguments"); runtime_output.get();
    library_loading_output<CUkernel> kernel;
    library_loading_output<cudaKernel_t> runtime_kernel;
    require(cuLibraryGetKernel(&kernel.value, program.value, "owned_missing_kernel") == CUDA_ERROR_NOT_FOUND
        && cudaLibraryGetKernel(&runtime_kernel.value, program.runtime_value(), "owned_missing_kernel") == cudaErrorSymbolNotFound,
        "both surfaces report a missing library kernel"); kernel.get(); runtime_kernel.get();
    require(cuLibraryGetKernel(nullptr, program.value, "module_alpha") == CUDA_ERROR_INVALID_VALUE
        && cuLibraryGetKernelCount(nullptr, program.value) == CUDA_ERROR_INVALID_VALUE
        && cuLibraryEnumerateKernels(nullptr, 1, program.value) == CUDA_ERROR_INVALID_VALUE, "null library query outputs");
    library_loading_output<CUdeviceptr> pointer; library_loading_output<size_t> bytes, offset;
    require(cuLibraryGetGlobal(&pointer.value, &bytes.value, program.value, "owned_missing_global") == CUDA_ERROR_NOT_FOUND,
        "missing library global status"); pointer.get(); bytes.get();
    library_loading_output<void*> runtime_pointer;
    require(cudaLibraryGetGlobal(&runtime_pointer.value, &bytes.value, program.runtime_value(), "owned_missing_global") == cudaErrorSymbolNotFound,
        "Runtime missing library global has symbol-not-found status"); runtime_pointer.get(); bytes.get();
    require(cuKernelGetParamInfo(program.kernel("module_alpha"), 3, &offset.value, &bytes.value) == CUDA_ERROR_INVALID_VALUE,
        "library parameter index boundary"); offset.get(); bytes.get(); library_loading_execute(program);
}
void execute_library_loading() {
    CUdevice device{}; CONFORMANCE_CUDA(cuCtxGetDevice(&device)); library_loading_restore restore;
    for (bool source : {false, true}) {
        const auto image = library_loading_image(source);
        for (bool runtime : {false, true}) {
            if (library_loading_case == "contextless") {
                CONFORMANCE_CUDA(cuCtxSetCurrent(nullptr)); library_loading_program program(image, runtime);
                library_loading_current(nullptr); library_loading_metadata(program); library_loading_current(nullptr);
                CONFORMANCE_CUDA(cuCtxSetCurrent(restore.value)); library_loading_execute(program, 13, true); program.close();
            } else {
                library_loading_program program(image, runtime, library_loading_case == "files");
                if (library_loading_case == "files") library_loading_execute(program);
                else if (library_loading_case == "contexts") library_loading_contexts(program, device);
                else if (library_loading_case == "resources") library_loading_resources(program, device);
                else if (library_loading_case == "retirement") { library_loading_retirement(program); continue; }
                else { require(library_loading_case == "validation", "known library case"); library_loading_validation(program, image); }
                program.close();
            }
        }
    }
}
void library_loading_sources() { library_loading_image(true); }
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "sources") return cudavm::run_host_conformance(argc, argv, cudavm::library_loading_sources);
    if (argc == 3) { cudavm::library_loading_case = argv[2]; argc = 2; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_library_loading, 0, cudavm::probe_execution);
}
