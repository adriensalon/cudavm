#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>

namespace cudavm {
std::string_view surface_case;
void surface_runtime(cudaError_t status) {
    require(status == cudaSuccess, std::string("surface Runtime: ") + cudaGetErrorString(status));
}
template<class T> struct surface_output {
    std::array<unsigned char, 16> before;
    T value{};
    std::array<unsigned char, 16> after;
    surface_output() { before.fill(0xb5); after.fill(0x39); }
    T get() const {
        require(std::all_of(before.begin(), before.end(), [](auto byte) { return byte == 0xb5; }) &&
                std::all_of(after.begin(), after.end(), [](auto byte) { return byte == 0x39; }), "surface output exact width");
        return value;
    }
};
struct surface_array {
    CUarray value{};
    size_t width = 11, height = 3;
    unsigned channels;
    std::vector<unsigned char> bytes;
    explicit surface_array(unsigned count) : channels(count), bytes(width * height * channels * 4) {
        CUDA_ARRAY3D_DESCRIPTOR descriptor{};
        descriptor.Width = width; descriptor.Height = height; descriptor.Format = CU_AD_FORMAT_UNSIGNED_INT32;
        descriptor.NumChannels = channels; descriptor.Flags = CUDA_ARRAY3D_SURFACE_LDST;
        CONFORMANCE_CUDA(cuArray3DCreate(&value, &descriptor));
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<unsigned char>((i * 53 + channels * 31) ^ (i >> 4));
        transfer(bytes.data(), false);
    }
    ~surface_array() { if (value) cuArrayDestroy(value); }
    void transfer(void* host, bool read) const {
        CUDA_MEMCPY2D copy{}; copy.WidthInBytes = width * channels * 4; copy.Height = height;
        copy.srcMemoryType = read ? CU_MEMORYTYPE_ARRAY : CU_MEMORYTYPE_HOST;
        copy.dstMemoryType = read ? CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_ARRAY;
        copy.srcArray = copy.dstArray = value; copy.srcHost = copy.dstHost = host;
        copy.srcPitch = copy.dstPitch = copy.WidthInBytes;
        CONFORMANCE_CUDA(cuMemcpy2D(&copy));
    }
    void verify() const {
        std::vector<unsigned char> actual(bytes.size()); transfer(actual.data(), true);
        require(actual == bytes, "surface lifecycle preserves full backing bytes");
    }
};
struct surface_handle {
    CUsurfObject value{};
    bool runtime;
    surface_handle(const surface_array& array, bool use_runtime) : runtime(use_runtime) {
        surface_output<CUsurfObject> output;
        if (runtime) {
            cudaResourceDesc descriptor{}; descriptor.resType = cudaResourceTypeArray;
            descriptor.res.array.array = reinterpret_cast<cudaArray_t>(array.value);
            surface_runtime(cudaCreateSurfaceObject(&output.value, &descriptor)); descriptor = {};
        } else {
            CUDA_RESOURCE_DESC descriptor{}; descriptor.resType = CU_RESOURCE_TYPE_ARRAY; descriptor.res.array.hArray = array.value;
            CONFORMANCE_CUDA(cuSurfObjectCreate(&output.value, &descriptor)); descriptor = {};
        }
        value = output.get(); require(value != 0, "nonnull opaque surface handle");
    }
    ~surface_handle() { if (value) cuSurfObjectDestroy(value); }
    void close() {
        if (runtime) CONFORMANCE_CUDA(cuSurfObjectDestroy(value));
        else surface_runtime(cudaDestroySurfaceObject(value));
        value = 0;
    }
};
void surface_query(const surface_handle& object, const surface_array& array) {
    surface_output<CUDA_RESOURCE_DESC> driver;
    surface_output<cudaResourceDesc> runtime;
    CONFORMANCE_CUDA(cuSurfObjectGetResourceDesc(&driver.value, object.value));
    surface_runtime(cudaGetSurfaceObjectResourceDesc(&runtime.value, object.value));
    const auto d = driver.get(); const auto r = runtime.get();
    require(d.resType == CU_RESOURCE_TYPE_ARRAY && d.flags == 0 && d.res.array.hArray == array.value &&
            r.resType == cudaResourceTypeArray && r.res.array.array == reinterpret_cast<cudaArray_t>(array.value), "surface resource identity across facades");
    CUDA_ARRAY3D_DESCRIPTOR descriptor{}; CONFORMANCE_CUDA(cuArray3DGetDescriptor(&descriptor, array.value));
    require(descriptor.Width == array.width && descriptor.Height == array.height && descriptor.Depth == 0 &&
            descriptor.Format == CU_AD_FORMAT_UNSIGNED_INT32 && descriptor.NumChannels == array.channels &&
            descriptor.Flags == CUDA_ARRAY3D_SURFACE_LDST, "surface creation preserves array descriptor");
}
void surface_descriptors() {
    for (bool runtime : {false, true}) for (unsigned channels : {1u, 2u, 4u}) {
        surface_array backing(channels); surface_handle object(backing, runtime);
        surface_query(object, backing); object.close(); backing.verify();
    }
}
void surface_siblings() {
    for (bool runtime : {false, true}) {
        surface_array backing(4);
        surface_handle first(backing, runtime), second(backing, !runtime);
        surface_query(first, backing); surface_query(second, backing);
        first.close(); surface_query(second, backing); backing.verify();
        surface_handle replacement(backing, runtime);
        second.close(); surface_query(replacement, backing); replacement.close(); backing.verify();
    }
}
void surface_contexts() {
    CUcontext original{}; CUdevice device{};
    CONFORMANCE_CUDA(cuCtxGetCurrent(&original)); CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    surface_array first(1); surface_handle a(first, false);
    CUcontext other{}; CONFORMANCE_CUDA(cuCtxCreate(&other, nullptr, 0, device));
    struct context_cleanup { CUcontext other, original; ~context_cleanup() { if (other) cuCtxDestroy(other); cuCtxSetCurrent(original); } } cleanup{other, original};
    {
        surface_array second(4); surface_handle b(second, true);
        surface_query(b, second);
        CONFORMANCE_CUDA(cuCtxSetCurrent(original)); surface_query(a, first);
        CONFORMANCE_CUDA(cuCtxSetCurrent(other)); surface_query(b, second); b.close(); second.verify();
    }
    CONFORMANCE_CUDA(cuCtxSetCurrent(original));
    CONFORMANCE_CUDA(cuCtxDestroy(other)); cleanup.other = nullptr;
    surface_query(a, first); a.close(); first.verify();
}
void surface_validation() {
    surface_array backing(1);
    CUDA_RESOURCE_DESC driver{}; driver.resType = CU_RESOURCE_TYPE_ARRAY; driver.res.array.hArray = backing.value;
    cudaResourceDesc runtime{}; runtime.resType = cudaResourceTypeArray; runtime.res.array.array = reinterpret_cast<cudaArray_t>(backing.value);
    CUsurfObject output{};
    require(cuSurfObjectCreate(nullptr, &driver) == CUDA_ERROR_INVALID_VALUE, "missing Driver surface output");
    require(cuSurfObjectCreate(&output, nullptr) == CUDA_ERROR_INVALID_VALUE, "missing Driver surface resource");
    require(cudaCreateSurfaceObject(nullptr, &runtime) == cudaErrorInvalidValue, "missing Runtime surface output");
    require(cudaCreateSurfaceObject(&output, nullptr) == cudaErrorInvalidValue, "missing Runtime surface resource");
    driver.flags = 1;
    require(cuSurfObjectCreate(&output, &driver) == CUDA_ERROR_INVALID_VALUE, "reserved surface resource flags");
    driver.flags = 0;
    for (auto kind : {CU_RESOURCE_TYPE_LINEAR, CU_RESOURCE_TYPE_PITCH2D, static_cast<CUresourcetype>(7)}) {
        driver.resType = kind; runtime.resType = static_cast<cudaResourceType>(kind);
        require(cuSurfObjectCreate(&output, &driver) == CUDA_ERROR_INVALID_VALUE, "Driver surface requires valid array resource type");
        require(cudaCreateSurfaceObject(&output, &runtime) == cudaErrorInvalidValue, "Runtime surface requires valid array resource type");
    }
    require(cudaGetLastError() == cudaErrorInvalidValue && cudaGetLastError() == cudaSuccess, "surface Runtime last error clears");
    for (bool use_runtime : {false, true}) {
        surface_handle repaired(backing, use_runtime);
        require(cuSurfObjectGetResourceDesc(nullptr, repaired.value) == CUDA_ERROR_INVALID_VALUE, "missing Driver surface query output");
        require(cudaGetSurfaceObjectResourceDesc(nullptr, repaired.value) == cudaErrorInvalidValue, "missing Runtime surface query output");
        surface_query(repaired, backing); repaired.close(); backing.verify(); cudaGetLastError();
    }
}
void execute_surface_objects() {
    if (surface_case == "descriptors") surface_descriptors();
    else if (surface_case == "siblings") surface_siblings();
    else if (surface_case == "contexts") surface_contexts();
    else if (surface_case == "validation") surface_validation();
    else throw std::runtime_error("unknown surface_objects case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::surface_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_surface_objects);
}
