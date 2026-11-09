#include "test_framework.hpp"
#include <cstdio>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace cudavm {
struct guarded_tensor_map {
    std::array<unsigned char, 128> before;
    CUtensorMap value;
    std::array<unsigned char, 128> after;
    guarded_tensor_map() { before.fill(0x93); after.fill(0x57); std::memset(&value, 0xa5, sizeof(value)); }
    void guards() const {
        require(std::all_of(before.begin(), before.end(), [](auto x) { return x == 0x93; })
            && std::all_of(after.begin(), after.end(), [](auto x) { return x == 0x57; }), "tensor map exact output extent");
    }
};
static_assert(offsetof(guarded_tensor_map, value) == 128 && offsetof(guarded_tensor_map, after) == 256);
struct tensor_map_shape {
    CUtensorMapDataType type = CU_TENSOR_MAP_DATA_TYPE_UINT32;
    unsigned rank = 2;
    void* address;
    std::array<cuuint64_t, 5> dimensions{16, 8, 4, 3, 2};
    std::array<cuuint64_t, 4> strides{64, 512, 2048, 6144};
    std::array<cuuint32_t, 5> box{8, 4, 4, 3, 2}, elements{1, 1, 1, 1, 1};
    CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
    CUtensorMapL2promotion promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    CUtensorMapFloatOOBfill fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    explicit tensor_map_shape(CUdeviceptr pointer) : address(reinterpret_cast<void*>(static_cast<uintptr_t>(pointer))) {}
    CUresult encode(CUtensorMap* output, decltype(&cuTensorMapEncodeTiled) function = &cuTensorMapEncodeTiled) const {
        const auto dim_copy = dimensions;
        const auto stride_copy = strides;
        const auto box_copy = box, element_copy = elements;
        const auto result = function(output, type, rank, address, dimensions.data(), rank == 1 ? nullptr : strides.data(),
            box.data(), elements.data(), interleave, swizzle, promotion, fill);
        require(dim_copy == dimensions && stride_copy == strides && box_copy == box && element_copy == elements,
            "tensor map shape inputs are unchanged");
        return result;
    }
};
struct tensor_map_storage {
    static constexpr size_t size = 16384;
    device_buffer first{size}, second{size};
    std::vector<unsigned char> input;
    tensor_map_storage() : input(size) {
        for (size_t i = 0; i < size; ++i) input[i] = static_cast<unsigned char>(0x71 + i * 19 + (i >> 6));
        CONFORMANCE_CUDA(cuMemcpyHtoD(first.value, input.data(), size));
        CONFORMANCE_CUDA(cuMemcpyHtoD(second.value, input.data(), size));
    }
    void preserved() const {
        std::vector<unsigned char> output(size);
        for (auto pointer : {first.value, second.value}) {
            CONFORMANCE_CUDA(cuMemcpyDtoH(output.data(), pointer, size));
            require(output == input, "tensor map metadata operations preserve device allocations");
        }
    }
};
std::string_view tensor_map_backend, tensor_map_case;
bool tensor_map_native_hardware() {
    CUdevice device{}; int major{};
    CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    return tensor_map_backend == "cuda" && major >= 9;
}
void tensor_map_discovery() {
    for (const auto* name : {"cuTensorMapEncodeTiled", "cuTensorMapReplaceAddress"}) {
        void* pointer{}; CUdriverProcAddressQueryResult status{};
        CONFORMANCE_CUDA(cuGetProcAddress(name, &pointer, 12000, 0, &status));
        require(pointer && status == CU_GET_PROC_ADDRESS_SUCCESS, "tensor map introduction-version procedure discovery");
        if (std::string_view(name) == "cuTensorMapEncodeTiled") {
            const auto encode = reinterpret_cast<decltype(&cuTensorMapEncodeTiled)>(pointer);
            require(encode(nullptr, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, nullptr, nullptr, nullptr, nullptr, nullptr,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_ERROR_INVALID_VALUE, "tensor encode resolved ABI is callable");
        } else require(reinterpret_cast<decltype(&cuTensorMapReplaceAddress)>(pointer)(nullptr, nullptr) == CUDA_ERROR_INVALID_VALUE,
            "tensor replace resolved ABI is callable");
        CONFORMANCE_CUDA(cuGetProcAddress(name, &pointer, 11080, 0, &status));
        require(!pointer && status == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "tensor map entry points require CUDA 12.0");
    }
}
struct tensor_map_callback_gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false;
    static void CUDA_CB callback(void* data) {
        auto& gate = *static_cast<tensor_map_callback_gate*>(data);
        std::unique_lock lock(gate.mutex); gate.entered = true; gate.changed.notify_all();
        gate.changed.wait(lock, [&] { return gate.released; });
    }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
};
void tensor_map_pending(const tensor_map_shape& shape, guarded_tensor_map& output, CUresult expected) {
    tensor_map_callback_gate gate;
    struct pending_stream {
        CUstream value{};
        pending_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
        ~pending_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    } stream;
    std::promise<CUresult> completion;
    auto result = completion.get_future();
    std::jthread worker;
    struct release_gate { tensor_map_callback_gate& gate; ~release_gate() { gate.release(); } } release{gate};
    CUcontext context{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&context));
    CONFORMANCE_CUDA(cuLaunchHostFunc(stream.value, tensor_map_callback_gate::callback, &gate));
    {
        std::unique_lock lock(gate.mutex);
        require(gate.changed.wait_for(lock, std::chrono::seconds(5), [&] { return gate.entered; }), "tensor map pending-work callback entered");
    }
    worker = std::jthread([&] {
        try {
            CONFORMANCE_CUDA(cuCtxSetCurrent(context));
            const auto status = shape.encode(&output.value);
            completion.set_value(status); cuCtxSetCurrent(nullptr);
        } catch (...) { cuCtxSetCurrent(nullptr); completion.set_exception(std::current_exception()); }
    });
    const bool completed_while_held = result.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    gate.release(); worker.join();
    CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
    require(completed_while_held, "host tensor descriptor encoding does not drain an earlier held stream callback");
    require(result.get() == expected, "pending stream work does not change the tensor encoder result");
    output.guards();
}
void tensor_map_validation() {
    tensor_map_discovery();
    tensor_map_storage storage;
    const tensor_map_shape legal(storage.first.value);
    guarded_tensor_map output;
    auto reject = [&](CUresult result, const char* label) {
        require(result == CUDA_ERROR_INVALID_VALUE, std::string("invalid tensor map request: ") + label);
        output.guards();
        const auto* bytes = reinterpret_cast<const unsigned char*>(&output.value);
        require(std::all_of(bytes, bytes + sizeof(output.value), [](auto x) { return x == 0xa5; }),
            std::string("rejected tensor map preserves descriptor: ") + label);
    };
    reject(legal.encode(nullptr), "null output");
    auto* misaligned = reinterpret_cast<CUtensorMap*>(reinterpret_cast<unsigned char*>(&output.value) + 1);
    reject(legal.encode(misaligned), "misaligned output");
    for (unsigned rank : {0u, 6u}) { auto value = legal; value.rank = rank; reject(value.encode(&output.value), "rank"); }
    for (int invalid : {-1, 99}) { auto value = legal; value.type = static_cast<CUtensorMapDataType>(invalid); reject(value.encode(&output.value), "data type"); }
    for (unsigned missing = 0; missing < 4; ++missing)
        reject(cuTensorMapEncodeTiled(&output.value, legal.type, 2, legal.address,
            missing == 0 ? nullptr : legal.dimensions.data(), missing == 1 ? nullptr : legal.strides.data(),
            missing == 2 ? nullptr : legal.box.data(), missing == 3 ? nullptr : legal.elements.data(),
            legal.interleave, legal.swizzle, legal.promotion, legal.fill), "required shape array");
    for (unsigned axis : {0u, 1u}) {
        for (cuuint64_t bad : {cuuint64_t{0}, (cuuint64_t{1} << 32) + 1}) {
            auto value = legal; value.dimensions[axis] = bad; reject(value.encode(&output.value), "dimension");
        }
        for (unsigned bad : {0u, 257u}) { auto value = legal; value.box[axis] = bad; reject(value.encode(&output.value), "box"); }
    }
    for (cuuint64_t bad : {cuuint64_t{15}, cuuint64_t{1} << 40}) {
        auto value = legal; value.strides[0] = bad; reject(value.encode(&output.value), "byte stride");
    }
    for (unsigned bad : {0u, 9u}) { auto value = legal; value.elements[1] = bad; reject(value.encode(&output.value), "element stride"); }
    { auto value = legal; value.box[0] = 3; reject(value.encode(&output.value), "inner box byte alignment"); }
    { auto value = legal; value.address = reinterpret_cast<void*>(static_cast<uintptr_t>(storage.first.value + 1)); reject(value.encode(&output.value), "global address"); }
    { auto value = legal; value.interleave = static_cast<CUtensorMapInterleave>(99); reject(value.encode(&output.value), "interleave enum"); }
    { auto value = legal; value.interleave = CU_TENSOR_MAP_INTERLEAVE_16B; reject(value.encode(&output.value), "interleave rank"); }
    { auto value = legal; value.swizzle = static_cast<CUtensorMapSwizzle>(99); reject(value.encode(&output.value), "swizzle enum"); }
    { auto value = legal; value.promotion = static_cast<CUtensorMapL2promotion>(99); reject(value.encode(&output.value), "promotion enum"); }
    { auto value = legal; value.fill = static_cast<CUtensorMapFloatOOBfill>(99); reject(value.encode(&output.value), "fill enum"); }
    reject(cuTensorMapReplaceAddress(nullptr, legal.address), "replace null output");
    reject(cuTensorMapReplaceAddress(misaligned, legal.address), "replace misaligned output");
    reject(cuTensorMapReplaceAddress(&output.value, reinterpret_cast<void*>(static_cast<uintptr_t>(storage.second.value + 1))), "replace misaligned address");
    const auto result = legal.encode(&output.value);
    if (tensor_map_native_hardware() && native_cuda_version() >= 12000) CONFORMANCE_CUDA(result);
    else {
        require(result == CUDA_ERROR_NOT_SUPPORTED, "legal tensor map control reports unavailable encoder");
        std::puts("Encoding control unavailable; validation checks do not establish successful descriptor encoding.");
    }
    tensor_map_pending(legal, output, result);
    output.guards(); storage.preserved();
}

// The system driver is an independent ABI reference; never decode an opaque map.
struct native_tensor_map_reference {
#if defined(_WIN32)
    HMODULE library = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    void* find(const char* name) { return reinterpret_cast<void*>(GetProcAddress(library, name)); }
#else
    void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    void* find(const char* name) { return dlsym(library, name); }
#endif
    decltype(&cuTensorMapEncodeTiled) encode{};
    decltype(&cuTensorMapReplaceAddress) replace{};
    decltype(&cuCtxPushCurrent) push{};
    decltype(&cuCtxPopCurrent) pop{};
    bool pushed = false;
    template<class T> T symbol(const char* name) {
        auto pointer = find(name);
        if (!pointer) {
            using resolver = CUresult (CUDAAPI*)(const char*, void**, int, cuuint64_t);
            auto resolve = reinterpret_cast<resolver>(find("cuGetProcAddress"));
            if (resolve) CONFORMANCE_CUDA(resolve(name, &pointer, 12000, 0));
        }
        require(pointer != nullptr, std::string("native tensor map reference entry point: ") + name);
        return reinterpret_cast<T>(pointer);
    }
    native_tensor_map_reference() {
        require(library != nullptr, "system CUDA driver is available for independent tensor map reference");
        encode = symbol<decltype(encode)>("cuTensorMapEncodeTiled");
        replace = symbol<decltype(replace)>("cuTensorMapReplaceAddress");
        push = symbol<decltype(push)>("cuCtxPushCurrent_v2"); pop = symbol<decltype(pop)>("cuCtxPopCurrent_v2");
    }
    void enter(CUdeviceptr pointer) {
        auto attribute = symbol<decltype(&cuPointerGetAttribute)>("cuPointerGetAttribute"); CUcontext context{};
        CONFORMANCE_CUDA(attribute(&context, CU_POINTER_ATTRIBUTE_CONTEXT, pointer));
        CONFORMANCE_CUDA(push(context)); pushed = true;
    }
    ~native_tensor_map_reference() {
        if (pushed) { CUcontext prior{}; pop(&prior); }
#if defined(_WIN32)
        if (library) FreeLibrary(library);
#else
        if (library) dlclose(library);
#endif
    }
};
void tensor_map_positive() {
    require_capability(tensor_map_native_hardware(), "native tensor map encoding requires compute capability 9.0 or higher");
    require_capability(native_cuda_version() >= 12000, "native tensor map encoding requires a CUDA 12.0 Driver service");
    tensor_map_storage storage;
    native_tensor_map_reference reference; reference.enter(storage.first.value);
    for (auto type : {CU_TENSOR_MAP_DATA_TYPE_UINT32, CU_TENSOR_MAP_DATA_TYPE_FLOAT32}) for (unsigned rank = 1; rank <= 5; ++rank)
        for (unsigned first_stride : {0u, 1u}) {
            tensor_map_shape shape(storage.first.value); shape.rank = rank; shape.type = type; shape.elements[0] = first_stride;
            guarded_tensor_map actual, expected;
            CONFORMANCE_CUDA(shape.encode(&expected.value, reference.encode));
            CONFORMANCE_CUDA(shape.encode(&actual.value));
            auto compare = [&] {
                actual.guards(); expected.guards();
                require(!std::memcmp(&actual.value, &expected.value, sizeof(CUtensorMap)),
                    "opaque tensor map forwarding agrees with independent native API rank=" + std::to_string(rank));
            };
            compare();
            if (tensor_map_case == "replacement") for (auto pointer : {storage.second.value, storage.first.value}) {
                auto* address = reinterpret_cast<void*>(static_cast<uintptr_t>(pointer));
                CONFORMANCE_CUDA(reference.replace(&expected.value, address));
                CONFORMANCE_CUDA(cuTensorMapReplaceAddress(&actual.value, address)); compare();
            }
        }
    storage.preserved();
}
void tensor_map_execute() {
    if (tensor_map_case == "validation") tensor_map_validation();
    else if (tensor_map_case == "encoding" || tensor_map_case == "replacement") tensor_map_positive();
    else require(false, "unknown tensor map case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::tensor_map_backend = argv[1]; cudavm::tensor_map_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::tensor_map_execute);
}
