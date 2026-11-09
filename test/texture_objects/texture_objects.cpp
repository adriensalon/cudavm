#include "test_framework.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <bit>

namespace cudavm {
std::string_view texture_case;
void texture_runtime(cudaError_t status) {
    require(status == cudaSuccess, std::string("texture Runtime: ") + cudaGetErrorString(status));
}
template<class T> struct texture_output {
    std::array<unsigned char, 16> before;
    T value{};
    std::array<unsigned char, 16> after;
    texture_output() { before.fill(0x63); after.fill(0xa7); }
    T get() const {
        require(std::all_of(before.begin(), before.end(), [](auto byte) { return byte == 0x63; }) &&
                std::all_of(after.begin(), after.end(), [](auto byte) { return byte == 0xa7; }), "texture query exact output guards");
        return value;
    }
};
struct texture_backing {
    CUarray array{}; CUdeviceptr pointer{};
    size_t width = 8, height = 4, pitch = 0;
    unsigned channels;
    CUresourcetype type;
    std::vector<unsigned char> bytes;
    explicit texture_backing(CUresourcetype kind, unsigned count = 1) : channels(count), type(kind) {
        if (type == CU_RESOURCE_TYPE_ARRAY) {
            CUDA_ARRAY3D_DESCRIPTOR descriptor{}; descriptor.Width = width; descriptor.Height = height;
            descriptor.Format = CU_AD_FORMAT_UNSIGNED_INT32; descriptor.NumChannels = channels;
            CONFORMANCE_CUDA(cuArray3DCreate(&array, &descriptor)); pitch = width * channels * 4;
        } else if (type == CU_RESOURCE_TYPE_PITCH2D)
            CONFORMANCE_CUDA(cuMemAllocPitch(&pointer, &pitch, width * channels * 4, height, 16));
        else { pitch = width * channels * 4; CONFORMANCE_CUDA(cuMemAlloc(&pointer, pitch * height)); }
        bytes.resize(pitch * height);
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<unsigned char>((i * 37 + 53) ^ (i >> 5));
        transfer(bytes.data(), false);
    }
    ~texture_backing() { if (array) cuArrayDestroy(array); if (pointer) cuMemFree(pointer); }
    void transfer(void* host, bool read) const {
        if (!array) {
            if (read) CONFORMANCE_CUDA(cuMemcpyDtoH(host, pointer, bytes.size()));
            else CONFORMANCE_CUDA(cuMemcpyHtoD(pointer, host, bytes.size()));
            return;
        }
        CUDA_MEMCPY2D copy{}; copy.WidthInBytes = pitch; copy.Height = height;
        copy.srcMemoryType = read ? CU_MEMORYTYPE_ARRAY : CU_MEMORYTYPE_HOST;
        copy.dstMemoryType = read ? CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_ARRAY;
        copy.srcArray = copy.dstArray = array; copy.srcHost = copy.dstHost = host;
        copy.srcPitch = copy.dstPitch = pitch; CONFORMANCE_CUDA(cuMemcpy2D(&copy));
    }
    void verify() const {
        std::vector<unsigned char> actual(bytes.size()); transfer(actual.data(), true);
        require(actual == bytes, "texture lifecycle preserves all backing bytes and pitched padding");
    }
    CUDA_RESOURCE_DESC driver() const {
        CUDA_RESOURCE_DESC value{}; value.resType = type;
        if (array) value.res.array.hArray = array;
        else if (type == CU_RESOURCE_TYPE_LINEAR) value.res.linear = {pointer, CU_AD_FORMAT_UNSIGNED_INT32, channels, bytes.size()};
        else value.res.pitch2D = {pointer, CU_AD_FORMAT_UNSIGNED_INT32, channels, width, height, pitch};
        return value;
    }
    cudaResourceDesc runtime() const {
        cudaResourceDesc value{}; value.resType = static_cast<cudaResourceType>(type);
        cudaChannelFormatDesc channel{32, channels >= 2 ? 32 : 0, channels == 4 ? 32 : 0, channels == 4 ? 32 : 0, cudaChannelFormatKindUnsigned};
        if (array) value.res.array.array = reinterpret_cast<cudaArray_t>(array);
        else if (type == CU_RESOURCE_TYPE_LINEAR) value.res.linear = {reinterpret_cast<void*>(uintptr_t(pointer)), channel, bytes.size()};
        else value.res.pitch2D = {reinterpret_cast<void*>(uintptr_t(pointer)), channel, width, height, pitch};
        return value;
    }
};
struct texture_handle {
    CUtexObject value{};
    bool runtime;
    unsigned normalized;
    texture_handle(const texture_backing& backing, bool use_runtime, unsigned coordinates = 0)
        : runtime(use_runtime), normalized(coordinates) {
        texture_output<CUtexObject> output;
        if (runtime) {
            auto resource = backing.runtime(); cudaTextureDesc sampler{};
            sampler.readMode = cudaReadModeElementType; sampler.normalizedCoords = normalized;
            sampler.maxAnisotropy = 1;
            for (auto& mode : sampler.addressMode) mode = cudaAddressModeClamp;
            sampler.borderColor[0] = 1.25f; sampler.borderColor[1] = -2.5f; sampler.borderColor[2] = 3.75f; sampler.borderColor[3] = -0.0f;
            texture_runtime(cudaCreateTextureObject(&output.value, &resource, &sampler, nullptr));
            resource = {}; sampler = {};
        } else {
            auto resource = backing.driver(); CUDA_TEXTURE_DESC sampler{};
            sampler.flags = CU_TRSF_READ_AS_INTEGER | (normalized ? CU_TRSF_NORMALIZED_COORDINATES : 0);
            sampler.maxAnisotropy = 1;
            for (auto& mode : sampler.addressMode) mode = CU_TR_ADDRESS_MODE_CLAMP;
            sampler.borderColor[0] = 1.25f; sampler.borderColor[1] = -2.5f; sampler.borderColor[2] = 3.75f; sampler.borderColor[3] = -0.0f;
            CONFORMANCE_CUDA(cuTexObjectCreate(&output.value, &resource, &sampler, nullptr));
            resource = {}; sampler = {};
        }
        value = output.get(); require(value != 0, "created texture is a nonnull opaque handle");
    }
    ~texture_handle() { if (value) cuTexObjectDestroy(value); }
    void close() {
        if (runtime) CONFORMANCE_CUDA(cuTexObjectDestroy(value));
        else texture_runtime(cudaDestroyTextureObject(value));
        value = 0;
    }
};
void texture_query(const texture_handle& object, const texture_backing& backing) {
    texture_output<CUDA_RESOURCE_DESC> driver_resource;
    texture_output<cudaResourceDesc> runtime_resource;
    CONFORMANCE_CUDA(cuTexObjectGetResourceDesc(&driver_resource.value, object.value));
    texture_runtime(cudaGetTextureObjectResourceDesc(&runtime_resource.value, object.value));
    const auto d = driver_resource.get(); const auto r = runtime_resource.get();
    require(d.resType == backing.type && d.flags == 0 && r.resType == static_cast<cudaResourceType>(backing.type), "texture resource kind and flags");
    cudaChannelFormatDesc channel{};
    if (backing.array) require(d.res.array.hArray == backing.array && r.res.array.array == reinterpret_cast<cudaArray_t>(backing.array), "texture array identity");
    else if (backing.type == CU_RESOURCE_TYPE_LINEAR) {
        require(d.res.linear.devPtr == backing.pointer && d.res.linear.format == CU_AD_FORMAT_UNSIGNED_INT32 && d.res.linear.numChannels == backing.channels &&
                d.res.linear.sizeInBytes == backing.bytes.size() && r.res.linear.devPtr == reinterpret_cast<void*>(uintptr_t(backing.pointer)) && r.res.linear.sizeInBytes == backing.bytes.size(), "linear texture extent and identity");
        channel = r.res.linear.desc;
    } else {
        require(d.res.pitch2D.devPtr == backing.pointer && d.res.pitch2D.format == CU_AD_FORMAT_UNSIGNED_INT32 && d.res.pitch2D.numChannels == backing.channels &&
                d.res.pitch2D.width == backing.width && d.res.pitch2D.height == backing.height && d.res.pitch2D.pitchInBytes == backing.pitch &&
                r.res.pitch2D.devPtr == reinterpret_cast<void*>(uintptr_t(backing.pointer)) && r.res.pitch2D.width == backing.width && r.res.pitch2D.height == backing.height && r.res.pitch2D.pitchInBytes == backing.pitch, "pitched texture geometry and identity");
        channel = r.res.pitch2D.desc;
    }
    if (!backing.array) require(channel.x == 32 && channel.y == (backing.channels >= 2 ? 32 : 0) && channel.z == (backing.channels == 4 ? 32 : 0) && channel.w == (backing.channels == 4 ? 32 : 0) && channel.f == cudaChannelFormatKindUnsigned, "Runtime texture channels");
    texture_output<CUDA_TEXTURE_DESC> driver_sampler;
    texture_output<cudaTextureDesc> runtime_sampler;
    CONFORMANCE_CUDA(cuTexObjectGetTextureDesc(&driver_sampler.value, object.value));
    texture_runtime(cudaGetTextureObjectTextureDesc(&runtime_sampler.value, object.value));
    const auto ds = driver_sampler.get(); const auto rs = runtime_sampler.get();
    require(ds.flags == (CU_TRSF_READ_AS_INTEGER | (object.normalized ? CU_TRSF_NORMALIZED_COORDINATES : 0)) &&
            rs.readMode == cudaReadModeElementType && rs.normalizedCoords == int(object.normalized), "owned texture flags and read mode");
    for (unsigned i = 0; i < 3; ++i) require(ds.addressMode[i] == CU_TR_ADDRESS_MODE_CLAMP && rs.addressMode[i] == cudaAddressModeClamp, "owned sampler address mode");
    require(ds.filterMode == CU_TR_FILTER_MODE_POINT && rs.filterMode == cudaFilterModePoint && ds.maxAnisotropy == 1 && rs.maxAnisotropy == 1 &&
            ds.mipmapFilterMode == CU_TR_FILTER_MODE_POINT && rs.mipmapFilterMode == cudaFilterModePoint && ds.mipmapLevelBias == 0 && rs.mipmapLevelBias == 0 &&
            ds.minMipmapLevelClamp == 0 && ds.maxMipmapLevelClamp == 0 && rs.minMipmapLevelClamp == 0 && rs.maxMipmapLevelClamp == 0, "owned sampler filter and mip fields");
    const std::array<float, 4> border{1.25f, -2.5f, 3.75f, -0.0f};
    for (unsigned i = 0; i < 4; ++i) require(std::bit_cast<uint32_t>(ds.borderColor[i]) == std::bit_cast<uint32_t>(border[i]) &&
        std::bit_cast<uint32_t>(rs.borderColor[i]) == std::bit_cast<uint32_t>(border[i]), "owned border-color bits");
}
void texture_resources(bool arrays) {
    for (bool runtime : {false, true}) for (unsigned channels : {1u, 2u, 4u}) {
        for (auto kind : {CU_RESOURCE_TYPE_ARRAY, CU_RESOURCE_TYPE_LINEAR, CU_RESOURCE_TYPE_PITCH2D}) {
            if ((kind == CU_RESOURCE_TYPE_ARRAY) != arrays) continue;
            texture_backing backing(kind, channels);
            texture_handle object(backing, runtime);
            texture_query(object, backing); object.close(); backing.verify();
        }
    }
}
void texture_siblings() {
    for (bool runtime : {false, true}) {
        texture_backing backing(CU_RESOURCE_TYPE_ARRAY, 4);
        texture_handle first(backing, runtime), second(backing, !runtime, 1);
        texture_query(first, backing); texture_query(second, backing);
        first.close(); texture_query(second, backing); backing.verify();
        texture_handle replacement(backing, runtime);
        texture_query(replacement, backing); second.close(); texture_query(replacement, backing);
        replacement.close(); backing.verify();
    }
}
void texture_contexts() {
    CUcontext original{}; CUdevice device{};
    CONFORMANCE_CUDA(cuCtxGetCurrent(&original)); CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    texture_backing first(CU_RESOURCE_TYPE_ARRAY); texture_handle a(first, false);
    CUcontext other{}; CONFORMANCE_CUDA(cuCtxCreate(&other, nullptr, 0, device));
    struct context_cleanup { CUcontext other, original; ~context_cleanup() { if (other) cuCtxDestroy(other); cuCtxSetCurrent(original); } } cleanup{other, original};
    {
        texture_backing second(CU_RESOURCE_TYPE_ARRAY, 4); texture_handle b(second, true, 1);
        texture_query(b, second);
        CONFORMANCE_CUDA(cuCtxSetCurrent(original)); texture_query(a, first);
        CONFORMANCE_CUDA(cuCtxSetCurrent(other)); texture_query(b, second);
        b.close(); second.verify();
    }
    CONFORMANCE_CUDA(cuCtxSetCurrent(original));
    CONFORMANCE_CUDA(cuCtxDestroy(other)); cleanup.other = nullptr;
    texture_query(a, first); a.close(); first.verify();
}
void texture_validation() {
    texture_backing backing(CU_RESOURCE_TYPE_ARRAY);
    auto driver_resource = backing.driver(); CUDA_TEXTURE_DESC driver_sampler{}; driver_sampler.flags = CU_TRSF_READ_AS_INTEGER;
    auto runtime_resource = backing.runtime(); cudaTextureDesc runtime_sampler{}; runtime_sampler.readMode = cudaReadModeElementType;
    CUtexObject output = 0;
    require(cuTexObjectCreate(nullptr, &driver_resource, &driver_sampler, nullptr) == CUDA_ERROR_INVALID_VALUE, "missing Driver texture output");
    require(cuTexObjectCreate(&output, nullptr, &driver_sampler, nullptr) == CUDA_ERROR_INVALID_VALUE, "missing Driver resource");
    require(cuTexObjectCreate(&output, &driver_resource, nullptr, nullptr) == CUDA_ERROR_INVALID_VALUE, "missing Driver sampler");
    require(cudaCreateTextureObject(nullptr, &runtime_resource, &runtime_sampler, nullptr) == cudaErrorInvalidValue, "missing Runtime texture output");
    require(cudaCreateTextureObject(&output, nullptr, &runtime_sampler, nullptr) == cudaErrorInvalidValue, "missing Runtime resource");
    require(cudaCreateTextureObject(&output, &runtime_resource, nullptr, nullptr) == cudaErrorInvalidValue, "missing Runtime sampler");
    driver_resource.flags = 1;
    require(cuTexObjectCreate(&output, &driver_resource, &driver_sampler, nullptr) == CUDA_ERROR_INVALID_VALUE, "reserved resource flags");
    driver_resource = backing.driver(); driver_sampler.flags = 0x80000000u;
    require(cuTexObjectCreate(&output, &driver_resource, &driver_sampler, nullptr) == CUDA_ERROR_INVALID_VALUE, "reserved sampler flags");
    driver_sampler.flags = CU_TRSF_READ_AS_INTEGER; driver_resource.resType = static_cast<CUresourcetype>(7);
    require(cuTexObjectCreate(&output, &driver_resource, &driver_sampler, nullptr) == CUDA_ERROR_INVALID_VALUE, "invalid Driver resource type");
    runtime_resource.resType = static_cast<cudaResourceType>(7);
    require(cudaCreateTextureObject(&output, &runtime_resource, &runtime_sampler, nullptr) == cudaErrorInvalidValue, "invalid Runtime resource type");
    runtime_resource = backing.runtime();
    runtime_sampler.readMode = static_cast<cudaTextureReadMode>(7);
    require(cudaCreateTextureObject(&output, &runtime_resource, &runtime_sampler, nullptr) == cudaErrorInvalidValue, "invalid Runtime read mode");
    runtime_sampler.readMode = cudaReadModeElementType; runtime_sampler.filterMode = static_cast<cudaTextureFilterMode>(7);
    require(cudaCreateTextureObject(&output, &runtime_resource, &runtime_sampler, nullptr) == cudaErrorInvalidValue, "invalid Runtime filter");
    require(cudaGetLastError() == cudaErrorInvalidValue && cudaGetLastError() == cudaSuccess, "texture last error clears");
    for (bool runtime : {false, true}) {
        texture_handle repaired(backing, runtime); texture_query(repaired, backing);
        require(cuTexObjectGetResourceDesc(nullptr, repaired.value) == CUDA_ERROR_INVALID_VALUE, "missing Driver resource query output");
        require(cuTexObjectGetTextureDesc(nullptr, repaired.value) == CUDA_ERROR_INVALID_VALUE, "missing Driver sampler query output");
        require(cudaGetTextureObjectResourceDesc(nullptr, repaired.value) == cudaErrorInvalidValue, "missing Runtime resource query output");
        require(cudaGetTextureObjectTextureDesc(nullptr, repaired.value) == cudaErrorInvalidValue, "missing Runtime sampler query output");
        texture_query(repaired, backing); repaired.close(); backing.verify(); cudaGetLastError();
    }
}
void execute_texture_objects() {
    if (texture_case == "arrays") texture_resources(true);
    else if (texture_case == "pointers") texture_resources(false);
    else if (texture_case == "siblings") texture_siblings();
    else if (texture_case == "contexts") texture_contexts();
    else if (texture_case == "validation") texture_validation();
    else throw std::runtime_error("unknown texture_objects case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::texture_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_texture_objects);
}
