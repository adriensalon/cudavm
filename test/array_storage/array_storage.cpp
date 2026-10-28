#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <iostream>

namespace cudavm {
void check_array_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define ARRAY_RUNTIME(call) ::cudavm::check_array_runtime((call), #call)

struct array_format_case {
    CUarray_format format;
    unsigned bytes;
    cudaChannelFormatKind kind;
};
constexpr std::array<array_format_case, 8> array_formats{{
    {CU_AD_FORMAT_UNSIGNED_INT8, 1, cudaChannelFormatKindUnsigned},
    {CU_AD_FORMAT_SIGNED_INT8, 1, cudaChannelFormatKindSigned},
    {CU_AD_FORMAT_UNSIGNED_INT16, 2, cudaChannelFormatKindUnsigned},
    {CU_AD_FORMAT_SIGNED_INT16, 2, cudaChannelFormatKindSigned},
    {CU_AD_FORMAT_UNSIGNED_INT32, 4, cudaChannelFormatKindUnsigned},
    {CU_AD_FORMAT_SIGNED_INT32, 4, cudaChannelFormatKindSigned},
    {CU_AD_FORMAT_HALF, 2, cudaChannelFormatKindFloat},
    {CU_AD_FORMAT_FLOAT, 4, cudaChannelFormatKindFloat}}};

struct array_shape_case {
    const char* name;
    size_t width, height, depth;
    unsigned flags;
    bool legacy;
};
constexpr std::array<array_shape_case, 13> array_shapes{{
    {"legacy_1d", 5, 0, 0, 0, true},
    {"legacy_2d", 7, 3, 0, 0, true},
    {"1d_unit", 1, 0, 0, 0, false},
    {"1d", 5, 0, 0, 0, false},
    {"2d_unit", 1, 1, 0, 0, false},
    {"2d", 7, 3, 0, 0, false},
    {"3d_unit", 1, 1, 1, 0, false},
    {"3d", 5, 3, 2, 0, false},
    {"layered_1d", 5, 0, 3, CUDA_ARRAY3D_LAYERED, false},
    {"layered_2d", 5, 3, 3, CUDA_ARRAY3D_LAYERED, false},
    {"cubemap", 3, 3, 6, CUDA_ARRAY3D_CUBEMAP, false},
    {"layered_cubemap", 3, 3, 12, CUDA_ARRAY3D_CUBEMAP | CUDA_ARRAY3D_LAYERED, false},
    {"gather", 7, 3, 0, CUDA_ARRAY3D_TEXTURE_GATHER, false}}};

unsigned runtime_array_flags(unsigned flags) {
    return (flags & CUDA_ARRAY3D_LAYERED ? cudaArrayLayered : 0)
        | (flags & CUDA_ARRAY3D_SURFACE_LDST ? cudaArraySurfaceLoadStore : 0)
        | (flags & CUDA_ARRAY3D_CUBEMAP ? cudaArrayCubemap : 0)
        | (flags & CUDA_ARRAY3D_TEXTURE_GATHER ? cudaArrayTextureGather : 0)
        | (flags & CUDA_ARRAY3D_DEFERRED_MAPPING ? cudaArrayDeferredMapping : 0)
        | (flags & CUDA_ARRAY3D_SPARSE ? cudaArraySparse : 0)
        | (flags & CUDA_ARRAY3D_COLOR_ATTACHMENT ? cudaArrayColorAttachment : 0)
        | (flags & (CUDA_ARRAY3D_DEPTH_TEXTURE | CUDA_ARRAY3D_VIDEO_ENCODE_DECODE))
        | (flags & 0x80000000u);
}
cudaChannelFormatDesc array_channel(const array_format_case& format, unsigned channels) {
    const auto bits = int(format.bytes * 8);
    return {bits, channels >= 2 ? bits : 0, channels == 4 ? bits : 0,
        channels == 4 ? bits : 0, format.kind};
}
bool same_channel(const cudaChannelFormatDesc& a, const cudaChannelFormatDesc& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w && a.f == b.f;
}

struct stored_array {
    CUarray value{};
    bool runtime = false;
    stored_array() = default;
    stored_array(const stored_array&) = delete;
    stored_array& operator=(const stored_array&) = delete;
    ~stored_array() {
        if (value) {
            if (runtime) cudaFreeArray(reinterpret_cast<cudaArray_t>(value));
            else cuArrayDestroy(value);
        }
    }
    void destroy() {
        if (runtime) ARRAY_RUNTIME(cudaFreeArray(reinterpret_cast<cudaArray_t>(value)));
        else CONFORMANCE_CUDA(cuArrayDestroy(value));
        value = nullptr;
    }
};

void create_stored_array(stored_array& array, const CUDA_ARRAY3D_DESCRIPTOR& descriptor,
                         const cudaChannelFormatDesc& channel, bool legacy, bool runtime) {
    array.runtime = runtime;
    if (runtime) {
        auto request = channel;
        cudaArray_t value{};
        if (legacy) ARRAY_RUNTIME(cudaMallocArray(&value, &request, descriptor.Width,
            descriptor.Height, runtime_array_flags(descriptor.Flags)));
        else ARRAY_RUNTIME(cudaMalloc3DArray(&value, &request,
            {descriptor.Width, descriptor.Height, descriptor.Depth}, runtime_array_flags(descriptor.Flags)));
        array.value = reinterpret_cast<CUarray>(value);
        request = {}; // The caller's descriptor does not remain the array's state.
    } else if (legacy) {
        CUDA_ARRAY_DESCRIPTOR request{descriptor.Width, descriptor.Height, descriptor.Format, descriptor.NumChannels};
        CUarray value{};
        CONFORMANCE_CUDA(cuArrayCreate(&value, &request));
        array.value = value;
        request = {};
    } else {
        auto request = descriptor;
        CUarray value{};
        CONFORMANCE_CUDA(cuArray3DCreate(&value, &request));
        array.value = value;
        request = {};
    }
    require(array.value != nullptr, "array creation publishes a handle");
}

void verify_driver_array_descriptors(CUarray array, const CUDA_ARRAY3D_DESCRIPTOR& expected) {
    CUDA_ARRAY3D_DESCRIPTOR actual{};
    CONFORMANCE_CUDA(cuArray3DGetDescriptor(&actual, array));
    require(actual.Width == expected.Width && actual.Height == expected.Height && actual.Depth == expected.Depth,
        "3D descriptor preserves logical extents, including zeros and unit dimensions");
    require(actual.Format == expected.Format && actual.NumChannels == expected.NumChannels
        && actual.Flags == expected.Flags, "Driver format, channel count and flags");
    if (!expected.Depth) {
        CUDA_ARRAY_DESCRIPTOR legacy{};
        CONFORMANCE_CUDA(cuArrayGetDescriptor(&legacy, array));
        require(legacy.Width == expected.Width && legacy.Height == expected.Height
            && legacy.Format == expected.Format && legacy.NumChannels == expected.NumChannels,
            "legacy descriptor agreement");
    }
}
void verify_array_descriptors(CUarray array, const CUDA_ARRAY3D_DESCRIPTOR& expected,
                              const cudaChannelFormatDesc& channel) {
    verify_driver_array_descriptors(array, expected);
    const auto runtime_array = reinterpret_cast<cudaArray_t>(array);
    cudaChannelFormatDesc actual_channel{};
    ARRAY_RUNTIME(cudaGetChannelDesc(&actual_channel, runtime_array));
    require(same_channel(actual_channel, channel), "Runtime channel descriptor agreement");
    // Exercise each documented combination of optional outputs independently.
    for (unsigned outputs = 0; outputs < 8; ++outputs) {
        cudaExtent extent{};
        unsigned flags = ~0u;
        actual_channel = {};
        ARRAY_RUNTIME(cudaArrayGetInfo(outputs & 1 ? &actual_channel : nullptr,
            outputs & 2 ? &extent : nullptr, outputs & 4 ? &flags : nullptr, runtime_array));
        if (outputs & 1) require(same_channel(actual_channel, channel), "Runtime array channel information");
        if (outputs & 2) require(extent.width == expected.Width && extent.height == expected.Height
            && extent.depth == expected.Depth, "Runtime array extent information");
        if (outputs & 4) require(flags == runtime_array_flags(expected.Flags), "Runtime array flags");
    }
}

constexpr unsigned char array_guard = 0x9d;
unsigned char array_storage_pattern(size_t x, size_t y, size_t z, unsigned seed) {
    return static_cast<unsigned char>((x * 73 + y * 29 + z * 113 + seed * 47) ^ (x >> 3));
}
void transfer_array_storage(CUarray array, const CUDA_ARRAY3D_DESCRIPTOR& descriptor, unsigned component_bytes,
                             unsigned seed, bool upload, bool runtime) {
    const size_t row_bytes = descriptor.Width * descriptor.NumChannels * component_bytes;
    const size_t height = std::max(size_t(1), descriptor.Height), depth = std::max(size_t(1), descriptor.Depth);
    const size_t pitch = row_bytes + (upload ? 11 : 19), slice_height = height + (upload ? 2 : 1);
    std::vector<unsigned char> expected(32 + pitch * slice_height * depth, array_guard);
    for (size_t z = 0; z < depth; ++z) for (size_t y = 0; y < height; ++y)
        for (size_t x = 0; x < row_bytes; ++x)
            expected[16 + (z * slice_height + y) * pitch + x] = array_storage_pattern(x, y, z, seed);
    auto host = upload ? expected : std::vector<unsigned char>(expected.size(), array_guard);
    if (runtime) {
        cudaMemcpy3DParms copy{};
        cudaPitchedPtr pointer{host.data() + 16, pitch, row_bytes, slice_height};
        if (upload) { copy.srcPtr = pointer; copy.dstArray = reinterpret_cast<cudaArray_t>(array); }
        else { copy.srcArray = reinterpret_cast<cudaArray_t>(array); copy.dstPtr = pointer; }
        copy.extent = {descriptor.Width, height, depth}; // Array widths are elements in the Runtime API.
        copy.kind = upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
        ARRAY_RUNTIME(cudaMemcpy3D(&copy));
    } else {
        CUDA_MEMCPY3D copy{};
        if (upload) {
            copy.srcMemoryType = CU_MEMORYTYPE_HOST; copy.srcHost = host.data() + 16;
            copy.srcPitch = pitch; copy.srcHeight = slice_height;
            copy.dstMemoryType = CU_MEMORYTYPE_ARRAY; copy.dstArray = array;
        } else {
            copy.srcMemoryType = CU_MEMORYTYPE_ARRAY; copy.srcArray = array;
            copy.dstMemoryType = CU_MEMORYTYPE_HOST; copy.dstHost = host.data() + 16;
            copy.dstPitch = pitch; copy.dstHeight = slice_height;
        }
        copy.WidthInBytes = row_bytes; copy.Height = height; copy.Depth = depth;
        CONFORMANCE_CUDA(cuMemcpy3D(&copy));
    }
    for (size_t i = 0; i < expected.size(); ++i)
        if (host[i] != expected[i])
            throw std::runtime_error(std::string(upload ? "upload source preservation" : "array storage readback")
                + ": byte " + std::to_string(i));
}

void numeric_array_storage() {
    for (bool runtime : {false, true}) for (const auto& format : array_formats)
    for (unsigned channels : {1u, 2u, 4u}) for (const auto& shape : array_shapes)
    for (unsigned surface : {0u, unsigned(CUDA_ARRAY3D_SURFACE_LDST)}) {
        if (!runtime && shape.legacy && surface) continue; // cuArrayCreate has no flags parameter.
        const auto context = std::string(runtime ? "Runtime" : "Driver") + " array shape=" + shape.name
            + " format=" + std::to_string(unsigned(format.format)) + " channels=" + std::to_string(channels)
            + " flags=" + std::to_string(shape.flags | surface);
        try {
            const CUDA_ARRAY3D_DESCRIPTOR descriptor{shape.width, shape.height, shape.depth,
                format.format, channels, shape.flags | surface};
            const auto channel = array_channel(format, channels);
            require(same_channel(cudaCreateChannelDesc(channel.x, channel.y, channel.z, channel.w, channel.f), channel),
                "Runtime channel descriptor construction");
            stored_array first, survivor, replacement;
            create_stored_array(first, descriptor, channel, shape.legacy, runtime);
            create_stored_array(survivor, descriptor, channel, shape.legacy && !surface, !runtime);
            verify_array_descriptors(first.value, descriptor, channel);
            verify_array_descriptors(survivor.value, descriptor, channel);
            transfer_array_storage(first.value, descriptor, format.bytes, 1, true, runtime);
            transfer_array_storage(survivor.value, descriptor, format.bytes, 2, true, !runtime);
            // Both APIs read storage allocated and uploaded by the other facade.
            transfer_array_storage(first.value, descriptor, format.bytes, 1, false, !runtime);
            transfer_array_storage(survivor.value, descriptor, format.bytes, 2, false, runtime);
            transfer_array_storage(first.value, descriptor, format.bytes, 3, true, !runtime);
            transfer_array_storage(first.value, descriptor, format.bytes, 3, false, runtime);
            transfer_array_storage(survivor.value, descriptor, format.bytes, 2, false, !runtime);
            verify_array_descriptors(first.value, descriptor, channel);
            const auto retired = first.value;
            first.destroy();
            CUDA_ARRAY3D_DESCRIPTOR invalid{};
            require(cuArray3DGetDescriptor(&invalid, retired) == CUDA_ERROR_INVALID_HANDLE, "retired array lookup");
            require(cuArrayDestroy(retired) == CUDA_ERROR_INVALID_HANDLE, "retired array destruction");
            create_stored_array(replacement, descriptor, channel, shape.legacy, runtime);
            transfer_array_storage(replacement.value, descriptor, format.bytes, 4, true, runtime);
            transfer_array_storage(survivor.value, descriptor, format.bytes, 2, false, runtime);
            transfer_array_storage(replacement.value, descriptor, format.bytes, 4, false, !runtime);
            verify_array_descriptors(survivor.value, descriptor, channel);
            verify_array_descriptors(replacement.value, descriptor, channel);
            replacement.destroy();
            survivor.destroy();
        } catch (const std::exception& error) {
            throw std::runtime_error(context + ": " + error.what());
        }
    }
}

void reject_array_descriptor(const CUDA_ARRAY3D_DESCRIPTOR& descriptor, bool legacy) {
    CUarray rejected{};
    CUDA_ARRAY_DESCRIPTOR request{descriptor.Width, descriptor.Height, descriptor.Format, descriptor.NumChannels};
    const auto status = legacy ? cuArrayCreate(&rejected, &request) : cuArray3DCreate(&rejected, &descriptor);
    if (status == CUDA_SUCCESS) CONFORMANCE_CUDA(cuArrayDestroy(rejected));
    require(status == CUDA_ERROR_INVALID_VALUE, "invalid Driver array descriptor format="
        + std::to_string(unsigned(descriptor.Format)) + " channels=" + std::to_string(descriptor.NumChannels)
        + " flags=" + std::to_string(descriptor.Flags) + ": status=" + std::to_string(status));
}
struct encoded_array_case {
    CUarray_format format;
    unsigned channels, bits, block_width, block_height, block_bytes;
    cudaChannelFormatKind kind;
    bool runtime_allocation = true;
};
constexpr encoded_array_case encoded_arrays[]{
    {CU_AD_FORMAT_UNORM_INT8X1, 1, 8, 1, 1, 1, cudaChannelFormatKindUnsignedNormalized8X1},
    {CU_AD_FORMAT_UNORM_INT8X2, 2, 8, 1, 1, 2, cudaChannelFormatKindUnsignedNormalized8X2},
    {CU_AD_FORMAT_UNORM_INT8X4, 4, 8, 1, 1, 4, cudaChannelFormatKindUnsignedNormalized8X4},
    {CU_AD_FORMAT_UNORM_INT16X1, 1, 16, 1, 1, 2, cudaChannelFormatKindUnsignedNormalized16X1},
    {CU_AD_FORMAT_UNORM_INT16X2, 2, 16, 1, 1, 4, cudaChannelFormatKindUnsignedNormalized16X2},
    {CU_AD_FORMAT_UNORM_INT16X4, 4, 16, 1, 1, 8, cudaChannelFormatKindUnsignedNormalized16X4},
    {CU_AD_FORMAT_SNORM_INT8X1, 1, 8, 1, 1, 1, cudaChannelFormatKindSignedNormalized8X1},
    {CU_AD_FORMAT_SNORM_INT8X2, 2, 8, 1, 1, 2, cudaChannelFormatKindSignedNormalized8X2},
    {CU_AD_FORMAT_SNORM_INT8X4, 4, 8, 1, 1, 4, cudaChannelFormatKindSignedNormalized8X4},
    {CU_AD_FORMAT_SNORM_INT16X1, 1, 16, 1, 1, 2, cudaChannelFormatKindSignedNormalized16X1},
    {CU_AD_FORMAT_SNORM_INT16X2, 2, 16, 1, 1, 4, cudaChannelFormatKindSignedNormalized16X2},
    {CU_AD_FORMAT_SNORM_INT16X4, 4, 16, 1, 1, 8, cudaChannelFormatKindSignedNormalized16X4},
    {CU_AD_FORMAT_BC1_UNORM, 4, 8, 4, 4, 8, cudaChannelFormatKindUnsignedBlockCompressed1},
    {CU_AD_FORMAT_BC1_UNORM_SRGB, 4, 8, 4, 4, 8, cudaChannelFormatKindUnsignedBlockCompressed1SRGB},
    {CU_AD_FORMAT_BC2_UNORM, 4, 8, 4, 4, 16, cudaChannelFormatKindUnsignedBlockCompressed2},
    {CU_AD_FORMAT_BC2_UNORM_SRGB, 4, 8, 4, 4, 16, cudaChannelFormatKindUnsignedBlockCompressed2SRGB},
    {CU_AD_FORMAT_BC3_UNORM, 4, 8, 4, 4, 16, cudaChannelFormatKindUnsignedBlockCompressed3},
    {CU_AD_FORMAT_BC3_UNORM_SRGB, 4, 8, 4, 4, 16, cudaChannelFormatKindUnsignedBlockCompressed3SRGB},
    {CU_AD_FORMAT_BC4_UNORM, 1, 8, 4, 4, 8, cudaChannelFormatKindUnsignedBlockCompressed4},
    {CU_AD_FORMAT_BC4_SNORM, 1, 8, 4, 4, 8, cudaChannelFormatKindSignedBlockCompressed4},
    {CU_AD_FORMAT_BC5_UNORM, 2, 8, 4, 4, 16, cudaChannelFormatKindUnsignedBlockCompressed5},
    {CU_AD_FORMAT_BC5_SNORM, 2, 8, 4, 4, 16, cudaChannelFormatKindSignedBlockCompressed5},
    {CU_AD_FORMAT_BC6H_UF16, 3, 16, 4, 4, 16, cudaChannelFormatKindUnsignedBlockCompressed6H},
    {CU_AD_FORMAT_BC6H_SF16, 3, 16, 4, 4, 16, cudaChannelFormatKindSignedBlockCompressed6H},
    {CU_AD_FORMAT_BC7_UNORM, 4, 8, 4, 4, 16, cudaChannelFormatKindUnsignedBlockCompressed7},
    {CU_AD_FORMAT_BC7_UNORM_SRGB, 4, 8, 4, 4, 16, cudaChannelFormatKindUnsignedBlockCompressed7SRGB},
    {CU_AD_FORMAT_UNORM_INT_101010_2, 4, 10, 1, 1, 4, cudaChannelFormatKindUnsignedNormalized1010102},
    {CU_AD_FORMAT_Y410, 4, 10, 1, 1, 4, cudaChannelFormatKindUnsignedNormalized1010102, false},
    {CU_AD_FORMAT_YUY2, 4, 8, 1, 1, 4, cudaChannelFormatKindUnsigned8Packed422, false},
    {CU_AD_FORMAT_Y210, 4, 16, 1, 1, 8, cudaChannelFormatKindUnsigned, false},
    {CU_AD_FORMAT_Y216, 4, 16, 1, 1, 8, cudaChannelFormatKindUnsigned, false},
    {CU_AD_FORMAT_AYUV, 4, 8, 1, 1, 4, cudaChannelFormatKindUnsigned8Packed444, false},
    {CU_AD_FORMAT_Y416, 4, 16, 1, 1, 8, cudaChannelFormatKindUnsigned, false},
    {CU_AD_FORMAT_UINT8_PACKED_422, 4, 8, 1, 1, 4, cudaChannelFormatKindUnsigned8Packed422},
    {CU_AD_FORMAT_UINT8_PACKED_444, 4, 8, 1, 1, 4, cudaChannelFormatKindUnsigned8Packed444},
};
void encoded_array_storage() {
    for (const auto& format : encoded_arrays) for (bool runtime : {false, true})
    for (const auto& shape : std::array<std::array<size_t, 3>, 4>{{{1, 1, 0}, {5, 7, 0}, {16, 16, 2}, {7, 3, 3}}})
    for (bool legacy : {false, true}) {
        if (legacy && shape[2]) continue;
        if (runtime && !format.runtime_allocation) continue;
        const auto label = std::string(runtime ? "Runtime" : "Driver") + " encoded format="
            + std::to_string(unsigned(format.format));
        try {
            const auto flags = shape[0] == 7 ? CUDA_ARRAY3D_LAYERED : 0u;
            const auto width = shape[0];
            const CUDA_ARRAY3D_DESCRIPTOR descriptor{width, shape[1], shape[2], format.format, format.channels, flags};
            const cudaChannelFormatDesc channel{int(format.bits), format.channels > 1 ? int(format.bits) : 0,
                format.channels > 2 ? int(format.bits) : 0,
                format.channels > 3 ? (format.bits == 10 ? 2 : int(format.bits)) : 0, format.kind};
            stored_array array;
            create_stored_array(array, descriptor, channel, legacy, runtime);
            verify_array_descriptors(array.value, descriptor, channel);
            const auto columns = (width + format.block_width - 1) / format.block_width;
            const auto rows = (shape[1] + format.block_height - 1) / format.block_height;
            const auto depth = std::max(size_t(1), shape[2]);
            const size_t row_bytes = columns * format.block_bytes, pitch = row_bytes + 32, slice_height = rows + 1;
            std::vector<unsigned char> source(32 + pitch * slice_height * depth, array_guard);
            for (size_t z = 0; z < depth; ++z) for (size_t y = 0; y < rows; ++y)
                for (size_t x = 0; x < row_bytes; ++x)
                    source[16 + (z * slice_height + y) * pitch + x] = array_storage_pattern(x, y, z, 11);
            const auto preserved = source;
            std::vector<unsigned char> output(source.size(), array_guard);
            for (bool upload : {true, false}) {
                auto* host = upload ? source.data() : output.data();
                if (runtime == upload) {
                    CUDA_MEMCPY3D copy{};
                    if (upload) {
                        copy.srcMemoryType = CU_MEMORYTYPE_HOST; copy.srcHost = host + 16;
                        copy.srcPitch = pitch; copy.srcHeight = slice_height;
                        copy.dstMemoryType = CU_MEMORYTYPE_ARRAY; copy.dstArray = array.value;
                    } else {
                        copy.srcMemoryType = CU_MEMORYTYPE_ARRAY; copy.srcArray = array.value;
                        copy.dstMemoryType = CU_MEMORYTYPE_HOST; copy.dstHost = host + 16;
                        copy.dstPitch = pitch; copy.dstHeight = slice_height;
                    }
                    copy.WidthInBytes = row_bytes; copy.Height = rows; copy.Depth = depth;
                    CONFORMANCE_CUDA(cuMemcpy3D(&copy));
                } else {
                    cudaMemcpy3DParms copy{};
                    const cudaPitchedPtr pointer{host + 16, pitch, row_bytes, slice_height};
                    if (upload) { copy.srcPtr = pointer; copy.dstArray = reinterpret_cast<cudaArray_t>(array.value); }
                    else { copy.srcArray = reinterpret_cast<cudaArray_t>(array.value); copy.dstPtr = pointer; }
                    copy.extent = {columns, rows, depth};
                    copy.kind = upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
                    ARRAY_RUNTIME(cudaMemcpy3D(&copy));
                }
            }
            require(output == preserved, "encoded storage blocks and host guards");
            require(source == preserved, "encoded upload source preservation");
            verify_array_descriptors(array.value, descriptor, channel);
            if (format.block_width > 1) {
                CUDA_MEMCPY2D incomplete{};
                incomplete.srcMemoryType = CU_MEMORYTYPE_HOST; incomplete.srcHost = source.data() + 16;
                incomplete.srcPitch = pitch; incomplete.dstMemoryType = CU_MEMORYTYPE_ARRAY; incomplete.dstArray = array.value;
                incomplete.WidthInBytes = format.block_bytes - 1; incomplete.Height = 1;
                require(cuMemcpy2D(&incomplete) == CUDA_ERROR_INVALID_VALUE, "compressed byte copies require complete blocks");
                require(cudaMemcpy2DToArray(reinterpret_cast<cudaArray_t>(array.value), 0, 0, source.data() + 16,
                    pitch, format.block_bytes - 1, 1, cudaMemcpyHostToDevice) == cudaErrorInvalidValue,
                    "Runtime compressed byte copies require complete blocks");
            }
            array.destroy();
            // Fixed-format channel counts are part of the allocation contract.
            if (format.runtime_allocation) {
                auto invalid_channel = channel; ++invalid_channel.x;
                cudaArray_t rejected{};
                const auto status = legacy ? cudaMallocArray(&rejected, &invalid_channel, descriptor.Width, descriptor.Height, runtime_array_flags(flags))
                    : cudaMalloc3DArray(&rejected, &invalid_channel, {descriptor.Width, descriptor.Height, descriptor.Depth}, runtime_array_flags(flags));
                if (status == cudaSuccess) ARRAY_RUNTIME(cudaFreeArray(rejected));
                require(status == cudaErrorInvalidChannelDescriptor, "fixed-format Runtime channel bits");
            }
            auto invalid = descriptor;
            invalid.NumChannels = format.channels == 1 ? 2 : 1;
            reject_array_descriptor(invalid, false);
            if (format.block_width == 4) {
                invalid = descriptor; invalid.Flags |= CUDA_ARRAY3D_SURFACE_LDST;
                reject_array_descriptor(invalid, false);
            }
        } catch (const std::exception& error) {
            throw std::runtime_error(label + ": " + error.what());
        }
    }
}
void array_validation() {
    const CUDA_ARRAY3D_DESCRIPTOR valid{4, 3, 0, CU_AD_FORMAT_UNSIGNED_INT8, 1, 0};
    const CUDA_ARRAY_DESCRIPTOR legacy{4, 3, CU_AD_FORMAT_UNSIGNED_INT8, 1};
    const auto channel = array_channel(array_formats[0], 1);
    CUarray output{};
    require(cuArrayCreate(nullptr, &legacy) == CUDA_ERROR_INVALID_VALUE, "null legacy output");
    require(cuArrayCreate(&output, nullptr) == CUDA_ERROR_INVALID_VALUE, "null legacy descriptor");
    require(cuArray3DCreate(nullptr, &valid) == CUDA_ERROR_INVALID_VALUE, "null 3D output");
    require(cuArray3DCreate(&output, nullptr) == CUDA_ERROR_INVALID_VALUE, "null 3D descriptor");
    for (bool use_legacy : {false, true}) {
        auto descriptor = valid; descriptor.Width = 0;
        reject_array_descriptor(descriptor, use_legacy);
        for (unsigned channels : {0u, 3u, 5u}) {
            descriptor = valid; descriptor.NumChannels = channels;
            reject_array_descriptor(descriptor, use_legacy);
        }
        for (auto format : {CUarray_format(0), CU_AD_FORMAT_MAX}) {
            descriptor = valid; descriptor.Format = format;
            reject_array_descriptor(descriptor, use_legacy);
        }
    }
    constexpr std::array<array_shape_case, 9> invalid_shapes{{
        {"depth_without_height", 4, 0, 2, 0, false},
        {"layered_without_layers", 4, 3, 0, CUDA_ARRAY3D_LAYERED, false},
        {"nonsquare_cubemap", 4, 3, 6, CUDA_ARRAY3D_CUBEMAP, false},
        {"cubemap_faces", 4, 4, 5, CUDA_ARRAY3D_CUBEMAP, false},
        {"layered_cubemap_faces", 4, 4, 7, CUDA_ARRAY3D_CUBEMAP | CUDA_ARRAY3D_LAYERED, false},
        {"gather_1d", 4, 0, 0, CUDA_ARRAY3D_TEXTURE_GATHER, false},
        {"gather_3d", 4, 3, 2, CUDA_ARRAY3D_TEXTURE_GATHER, false},
        {"gather_layered", 4, 3, 2, CUDA_ARRAY3D_TEXTURE_GATHER | CUDA_ARRAY3D_LAYERED, false},
        {"unknown_flags", 4, 3, 0, 0x80000000u, false}}};
    for (const auto& shape : invalid_shapes) {
        const CUDA_ARRAY3D_DESCRIPTOR descriptor{shape.width, shape.height, shape.depth,
            CU_AD_FORMAT_UNSIGNED_INT8, 1, shape.flags};
        reject_array_descriptor(descriptor, false);
        cudaArray_t rejected{};
        const auto status = cudaMalloc3DArray(&rejected, &channel,
            {shape.width, shape.height, shape.depth}, runtime_array_flags(shape.flags));
        if (status == cudaSuccess) ARRAY_RUNTIME(cudaFreeArray(rejected));
        require(status == cudaErrorInvalidValue, std::string("invalid Runtime array shape: ") + shape.name);
    }
    cudaArray_t runtime_output{};
    for (bool legacy_api : {false, true}) {
        auto allocate = [&](cudaArray_t* destination, const cudaChannelFormatDesc* description, size_t width) {
            return legacy_api ? cudaMallocArray(destination, description, width, 3, 0)
                : cudaMalloc3DArray(destination, description, {width, 3, 0}, 0);
        };
        require(allocate(nullptr, &channel, 4) == cudaErrorInvalidValue, "null Runtime allocation output");
        require(allocate(&runtime_output, nullptr, 4) == cudaErrorInvalidValue, "null Runtime channel descriptor");
        const auto status = allocate(&runtime_output, &channel, 0);
        if (status == cudaSuccess) ARRAY_RUNTIME(cudaFreeArray(runtime_output));
        require(status == cudaErrorInvalidValue, "zero Runtime array width");
        for (const cudaChannelFormatDesc invalid : std::array<cudaChannelFormatDesc, 7>{{
            {7, 0, 0, 0, cudaChannelFormatKindUnsigned}, {8, 0, 0, 0, cudaChannelFormatKindFloat},
            {8, 16, 0, 0, cudaChannelFormatKindUnsigned}, {8, 8, 8, 0, cudaChannelFormatKindUnsigned},
            {8, 0, 8, 0, cudaChannelFormatKindUnsigned}, {0, 0, 0, 0, cudaChannelFormatKindUnsigned},
            {8, 0, 0, 0, cudaChannelFormatKindNone}}}) {
            const auto result = allocate(&runtime_output, &invalid, 4);
            if (result == cudaSuccess) ARRAY_RUNTIME(cudaFreeArray(runtime_output));
            require(result == cudaErrorInvalidChannelDescriptor, "invalid Runtime channel layout");
        }
    }
    stored_array array;
    create_stored_array(array, valid, channel, false, false);
    require(cuArrayGetDescriptor(nullptr, array.value) == CUDA_ERROR_INVALID_VALUE, "null legacy query output");
    require(cuArray3DGetDescriptor(nullptr, array.value) == CUDA_ERROR_INVALID_VALUE, "null 3D query output");
    require(cudaGetChannelDesc(nullptr, reinterpret_cast<cudaArray_t>(array.value)) == cudaErrorInvalidValue,
        "null Runtime channel output");
    CUDA_ARRAY_DESCRIPTOR invalid_legacy{};
    CUDA_ARRAY3D_DESCRIPTOR invalid_3d{};
    require(cuArrayGetDescriptor(&invalid_legacy, nullptr) == CUDA_ERROR_INVALID_HANDLE, "null legacy array handle");
    require(cuArray3DGetDescriptor(&invalid_3d, nullptr) == CUDA_ERROR_INVALID_HANDLE, "null 3D array handle");
    require(cuArrayDestroy(nullptr) == CUDA_ERROR_INVALID_HANDLE, "null destruction handle");
    CUdevice device{};
    CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    CUDA_ARRAY_MEMORY_REQUIREMENTS requirements{};
    require(cuArrayGetMemoryRequirements(&requirements, array.value, device) == CUDA_ERROR_INVALID_VALUE,
        "ordinary arrays have no deferred-mapping memory requirements");
    require(cuArrayGetMemoryRequirements(nullptr, array.value, device) == CUDA_ERROR_INVALID_VALUE,
        "null Driver memory requirements output");
    int runtime_device = -1;
    ARRAY_RUNTIME(cudaGetDevice(&runtime_device));
    cudaArrayMemoryRequirements runtime_requirements{};
    require(cudaArrayGetMemoryRequirements(&runtime_requirements, reinterpret_cast<cudaArray_t>(array.value), runtime_device)
        == cudaErrorInvalidValue, "ordinary Runtime arrays have no deferred-mapping memory requirements");
    require(cudaArrayGetMemoryRequirements(nullptr, reinterpret_cast<cudaArray_t>(array.value), runtime_device)
        == cudaErrorInvalidValue, "null Runtime memory requirements output");
    array.destroy();
}

struct array_host_storage {
    unsigned char* value{};
    explicit array_host_storage(size_t bytes) {
        void* pointer{};
        CONFORMANCE_CUDA(cuMemAllocHost(&pointer, bytes));
        value = static_cast<unsigned char*>(pointer);
    }
    ~array_host_storage() { if (value) cuMemFreeHost(value); }
    array_host_storage(const array_host_storage&) = delete;
    array_host_storage& operator=(const array_host_storage&) = delete;
};
struct array_transfer_stream {
    CUstream value{};
    array_transfer_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~array_transfer_stream() { if (value) cuStreamDestroy(value); }
    array_transfer_stream(const array_transfer_stream&) = delete;
    array_transfer_stream& operator=(const array_transfer_stream&) = delete;
};
void queued_array_retirement() {
    const CUDA_ARRAY3D_DESCRIPTOR descriptor{64, 3, 2, CU_AD_FORMAT_UNSIGNED_INT16, 2, 0};
    constexpr size_t row_bytes = 64 * 2 * 2, bytes = row_bytes * 3 * 2;
    const auto channel = array_channel(array_formats[2], 2);
    for (bool runtime : {false, true}) {
        std::vector<unsigned char> expected(bytes + 32, array_guard);
        for (size_t i = 0; i < bytes; ++i)
            expected[i + 16] = array_storage_pattern(i % row_bytes, (i / row_bytes) % 3, i / (row_bytes * 3), 5);
        array_host_storage source(expected.size()), output(expected.size());
        std::copy(expected.begin(), expected.end(), source.value);
        std::fill_n(output.value, expected.size(), array_guard);
        array_transfer_stream stream;
        stored_array array;
        create_stored_array(array, descriptor, channel, false, runtime);
        if (runtime) {
            cudaMemcpy3DParms upload{};
            upload.srcPtr = {source.value + 16, row_bytes, row_bytes, 3};
            upload.dstArray = reinterpret_cast<cudaArray_t>(array.value);
            upload.extent = {64, 3, 2}; upload.kind = cudaMemcpyHostToDevice;
            ARRAY_RUNTIME(cudaMemcpy3DAsync(&upload, reinterpret_cast<cudaStream_t>(stream.value)));
            CUDA_MEMCPY3D download{};
            download.srcMemoryType = CU_MEMORYTYPE_ARRAY; download.srcArray = array.value;
            download.dstMemoryType = CU_MEMORYTYPE_HOST; download.dstHost = output.value + 16;
            download.dstPitch = row_bytes; download.dstHeight = 3;
            download.WidthInBytes = row_bytes; download.Height = 3; download.Depth = 2;
            CONFORMANCE_CUDA(cuMemcpy3DAsync(&download, stream.value));
        } else {
            CUDA_MEMCPY3D upload{};
            upload.srcMemoryType = CU_MEMORYTYPE_HOST; upload.srcHost = source.value + 16;
            upload.srcPitch = row_bytes; upload.srcHeight = 3;
            upload.dstMemoryType = CU_MEMORYTYPE_ARRAY; upload.dstArray = array.value;
            upload.WidthInBytes = row_bytes; upload.Height = 3; upload.Depth = 2;
            CONFORMANCE_CUDA(cuMemcpy3DAsync(&upload, stream.value));
            cudaMemcpy3DParms download{};
            download.srcArray = reinterpret_cast<cudaArray_t>(array.value);
            download.dstPtr = {output.value + 16, row_bytes, row_bytes, 3};
            download.extent = {64, 3, 2}; download.kind = cudaMemcpyDeviceToHost;
            ARRAY_RUNTIME(cudaMemcpy3DAsync(&download, reinterpret_cast<cudaStream_t>(stream.value)));
        }
        // No stream/context synchronization precedes retirement. Observe the
        // accepted transfers only after the stream's defined completion point.
        array.destroy();
        CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
        require(std::equal(expected.begin(), expected.end(), output.value), "queued array retirement preserves readback and guards");
        require(std::equal(expected.begin(), expected.end(), source.value), "queued array upload preserves source and guards");
        CONFORMANCE_CUDA(cuStreamDestroy(stream.value)); stream.value = nullptr;
        CONFORMANCE_CUDA(cuMemFreeHost(source.value)); source.value = nullptr;
        CONFORMANCE_CUDA(cuMemFreeHost(output.value)); output.value = nullptr;
    }
}

struct array_test_context {
    CUcontext previous{}, value{};
    array_test_context() {
        CONFORMANCE_CUDA(cuCtxGetCurrent(&previous));
        CUdevice device{};
        CONFORMANCE_CUDA(cuCtxGetDevice(&device));
        CONFORMANCE_CUDA(cuCtxCreate(&value, nullptr, 0, device));
    }
    ~array_test_context() {
        if (value) cuCtxDestroy(value);
        cuCtxSetCurrent(previous);
    }
    array_test_context(const array_test_context&) = delete;
    array_test_context& operator=(const array_test_context&) = delete;
    void destroy() {
        CONFORMANCE_CUDA(cuCtxDestroy(value)); value = nullptr;
        CONFORMANCE_CUDA(cuCtxSetCurrent(previous));
    }
};
void context_array_retirement() {
    const CUDA_ARRAY3D_DESCRIPTOR descriptor{7, 3, 0, CU_AD_FORMAT_UNSIGNED_INT8, 1, 0};
    const auto channel = array_channel(array_formats[0], 1);
    stored_array survivor;
    create_stored_array(survivor, descriptor, channel, false, false);
    transfer_array_storage(survivor.value, descriptor, 1, 6, true, false);
    for (bool runtime : {false, true}) {
        array_test_context context;
        CUDA_ARRAY3D_DESCRIPTOR invalid{};
        require(cuArray3DGetDescriptor(&invalid, survivor.value) == CUDA_ERROR_INVALID_CONTEXT,
            "array descriptors require the owning context");
        require(cuArrayDestroy(survivor.value) == CUDA_ERROR_INVALID_CONTEXT,
            "foreign context cannot retire a live array");
        stored_array abandoned;
        create_stored_array(abandoned, descriptor, channel, false, runtime);
        transfer_array_storage(abandoned.value, descriptor, 1, 7, true, runtime);
        const auto retired = abandoned.value;
        abandoned.value = nullptr; // The context owns retirement in this case.
        context.destroy();
        require(cuArray3DGetDescriptor(&invalid, retired) == CUDA_ERROR_INVALID_HANDLE, "context destruction retires array lookup");
        require(cuArrayDestroy(retired) == CUDA_ERROR_INVALID_HANDLE, "context destruction retires array ownership");
        transfer_array_storage(survivor.value, descriptor, 1, 6, false, runtime);
        verify_array_descriptors(survivor.value, descriptor, channel);
    }
    survivor.destroy();
}

struct planar_array_case {
    CUarray_format format;
    unsigned bytes, planes, subsample_x, subsample_y;
    cudaChannelFormatKind kind;
    bool runtime_allocation = true;
};
constexpr planar_array_case planar_arrays[]{
    {CU_AD_FORMAT_NV12, 1, 2, 2, 2, cudaChannelFormatKindNV12},
    {CU_AD_FORMAT_P010, 2, 2, 2, 2, cudaChannelFormatKindUnsigned16SemiPlanar420, false},
    {CU_AD_FORMAT_P016, 2, 2, 2, 2, cudaChannelFormatKindUnsigned16SemiPlanar420, false},
    {CU_AD_FORMAT_NV16, 1, 2, 2, 1, cudaChannelFormatKindUnsigned8SemiPlanar422, false},
    {CU_AD_FORMAT_P210, 2, 2, 2, 1, cudaChannelFormatKindUnsigned16SemiPlanar422, false},
    {CU_AD_FORMAT_P216, 2, 2, 2, 1, cudaChannelFormatKindUnsigned16SemiPlanar422, false},
    {CU_AD_FORMAT_Y444_PLANAR8, 1, 3, 1, 1, cudaChannelFormatKindUnsigned8Planar444, false},
    {CU_AD_FORMAT_Y444_PLANAR10, 2, 3, 1, 1, cudaChannelFormatKindUnsigned16Planar444, false},
    {CU_AD_FORMAT_YUV444_8bit_SemiPlanar, 1, 2, 1, 1, cudaChannelFormatKindUnsigned8SemiPlanar444, false},
    {CU_AD_FORMAT_YUV444_16bit_SemiPlanar, 2, 2, 1, 1, cudaChannelFormatKindUnsigned16SemiPlanar444, false},
    {CU_AD_FORMAT_UINT8_SEMIPLANAR_420, 1, 2, 2, 2, cudaChannelFormatKindUnsigned8SemiPlanar420},
    {CU_AD_FORMAT_UINT16_SEMIPLANAR_420, 2, 2, 2, 2, cudaChannelFormatKindUnsigned16SemiPlanar420},
    {CU_AD_FORMAT_UINT8_SEMIPLANAR_422, 1, 2, 2, 1, cudaChannelFormatKindUnsigned8SemiPlanar422},
    {CU_AD_FORMAT_UINT16_SEMIPLANAR_422, 2, 2, 2, 1, cudaChannelFormatKindUnsigned16SemiPlanar422},
    {CU_AD_FORMAT_UINT8_SEMIPLANAR_444, 1, 2, 1, 1, cudaChannelFormatKindUnsigned8SemiPlanar444},
    {CU_AD_FORMAT_UINT16_SEMIPLANAR_444, 2, 2, 1, 1, cudaChannelFormatKindUnsigned16SemiPlanar444},
    {CU_AD_FORMAT_UINT8_PLANAR_420, 1, 3, 2, 2, cudaChannelFormatKindUnsigned8Planar420},
    {CU_AD_FORMAT_UINT16_PLANAR_420, 2, 3, 2, 2, cudaChannelFormatKindUnsigned16Planar420},
    {CU_AD_FORMAT_UINT8_PLANAR_422, 1, 3, 2, 1, cudaChannelFormatKindUnsigned8Planar422},
    {CU_AD_FORMAT_UINT16_PLANAR_422, 2, 3, 2, 1, cudaChannelFormatKindUnsigned16Planar422},
    {CU_AD_FORMAT_UINT8_PLANAR_444, 1, 3, 1, 1, cudaChannelFormatKindUnsigned8Planar444},
    {CU_AD_FORMAT_UINT16_PLANAR_444, 2, 3, 1, 1, cudaChannelFormatKindUnsigned16Planar444}
};
void planar_array_storage() {
    for (const auto& format : planar_arrays) for (unsigned layers : {0u, 3u}) for (bool runtime : {false, true})
    for (bool legacy : {false, true}) {
        if (legacy && layers) continue;
        if (runtime && !format.runtime_allocation) continue; // Driver-only legacy allocation spellings.
        const auto label = std::string(runtime ? "Runtime" : "Driver") + " planar format="
            + std::to_string(unsigned(format.format)) + " layers=" + std::to_string(layers);
        try {
            const auto flags = layers ? CUDA_ARRAY3D_LAYERED : 0u;
            const CUDA_ARRAY3D_DESCRIPTOR descriptor{16, 12, layers, format.format, 3, flags};
            const int bits = int(format.bytes * 8);
            const cudaChannelFormatDesc channel{bits, bits, bits, 0, format.kind};
            stored_array parent;
            create_stored_array(parent, descriptor, channel, legacy, runtime);
            CUDA_ARRAY3D_DESCRIPTOR actual{};
            CONFORMANCE_CUDA(cuArray3DGetDescriptor(&actual, parent.value));
            require(actual.Width == 16 && actual.Height == 12 && actual.Depth == layers
                && actual.Format == format.format && actual.NumChannels == 3 && actual.Flags == flags,
                "parent planar descriptor");
            verify_array_descriptors(parent.value, descriptor, channel);
            std::vector<CUarray> planes;
            for (unsigned plane = 0; plane < format.planes; ++plane) {
                CUarray handle{};
                CONFORMANCE_CUDA(cuArrayGetPlane(&handle, parent.value, plane));
                planes.push_back(handle);
                cudaArray_t runtime_handle{};
                ARRAY_RUNTIME(cudaArrayGetPlane(&runtime_handle, reinterpret_cast<cudaArray_t>(parent.value), plane));
                const unsigned channels = plane && format.planes == 2 ? 2 : 1;
                const CUDA_ARRAY3D_DESCRIPTOR expected{plane ? 16 / format.subsample_x : 16,
                    plane ? 12 / format.subsample_y : 12, layers,
                    format.bytes == 1 ? CU_AD_FORMAT_UNSIGNED_INT8 : CU_AD_FORMAT_UNSIGNED_INT16, channels, flags};
                const auto plane_channel = array_channel(format.bytes == 1 ? array_formats[0] : array_formats[2], channels);
                verify_array_descriptors(handle, expected, plane_channel);
                transfer_array_storage(handle, expected, format.bytes, 20 + plane, true, false);
                transfer_array_storage(reinterpret_cast<CUarray>(runtime_handle), expected, format.bytes, 20 + plane, false, true);
            }
            // Read all planes again after every neighbor has been updated.
            for (unsigned plane = 0; plane < planes.size(); ++plane) {
                CUDA_ARRAY3D_DESCRIPTOR expected{};
                CONFORMANCE_CUDA(cuArray3DGetDescriptor(&expected, planes[plane]));
                transfer_array_storage(planes[plane], expected, format.bytes, 20 + plane, false, false);
            }
            CUarray rejected{};
            require(cuArrayGetPlane(&rejected, parent.value, format.planes) == CUDA_ERROR_INVALID_VALUE,
                "out-of-range plane index");
            require(cuArrayGetPlane(nullptr, parent.value, 0) == CUDA_ERROR_INVALID_VALUE, "null plane output");
            require(cudaArrayGetPlane(nullptr, reinterpret_cast<cudaArray_t>(parent.value), 0) == cudaErrorInvalidValue,
                "null Runtime plane output");
            parent.destroy();
            for (auto plane : planes)
                require(cuArray3DGetDescriptor(&actual, plane) == CUDA_ERROR_INVALID_HANDLE, "parent retirement invalidates plane handles");
            if (format.runtime_allocation) {
                auto invalid_channel = channel; ++invalid_channel.y;
                cudaArray_t rejected{};
                const auto status = legacy ? cudaMallocArray(&rejected, &invalid_channel, descriptor.Width, descriptor.Height, runtime_array_flags(flags))
                    : cudaMalloc3DArray(&rejected, &invalid_channel, {descriptor.Width, descriptor.Height, descriptor.Depth}, runtime_array_flags(flags));
                if (status == cudaSuccess) ARRAY_RUNTIME(cudaFreeArray(rejected));
                require(status == cudaErrorInvalidChannelDescriptor, "planar Runtime channel bits");
            }
        } catch (const std::exception& error) {
            throw std::runtime_error(label + ": " + error.what());
        }
    }
}
void deferred_array_storage_contract() {
    auto check = [](const CUDA_ARRAY3D_DESCRIPTOR& descriptor, const cudaChannelFormatDesc& channel,
                    size_t payload, bool runtime) {
        stored_array array;
        create_stored_array(array, descriptor, channel, false, runtime);
        verify_array_descriptors(array.value, descriptor, channel);
        CUdevice device{};
        CONFORMANCE_CUDA(cuCtxGetDevice(&device));
        CUDA_ARRAY_MEMORY_REQUIREMENTS first{}, again{};
        CONFORMANCE_CUDA(cuArrayGetMemoryRequirements(&first, array.value, device));
        CONFORMANCE_CUDA(cuArrayGetMemoryRequirements(&again, array.value, device));
        require(first.size >= payload && first.alignment && (first.alignment & (first.alignment - 1)) == 0
            && first.size % first.alignment == 0, "deferred allocation size and mapping alignment");
        require(first.size == again.size && first.alignment == again.alignment, "stable deferred memory requirements");
        cudaArrayMemoryRequirements actual{};
        int ordinal = -1;
        ARRAY_RUNTIME(cudaGetDevice(&ordinal));
        ARRAY_RUNTIME(cudaArrayGetMemoryRequirements(&actual, reinterpret_cast<cudaArray_t>(array.value), ordinal));
        require(actual.size == first.size && actual.alignment == first.alignment, "Driver/Runtime deferred requirements agreement");
        require(cuArrayGetMemoryRequirements(&again, array.value, -1) == CUDA_ERROR_INVALID_VALUE,
            "invalid memory-requirement device");
        array.destroy();
    };
    for (bool runtime : {false, true}) {
        for (const auto& format : array_formats) for (unsigned channels : {1u, 2u, 4u}) {
            const CUDA_ARRAY3D_DESCRIPTOR d{11,7,0,format.format,channels,CUDA_ARRAY3D_DEFERRED_MAPPING};
            check(d, array_channel(format, channels), 11 * 7 * format.bytes * channels, runtime);
        }
        for (const auto& shape : array_shapes) {
            const CUDA_ARRAY3D_DESCRIPTOR d{shape.width,shape.height,shape.depth,CU_AD_FORMAT_UNSIGNED_INT16,2,
                shape.flags | CUDA_ARRAY3D_DEFERRED_MAPPING};
            check(d, array_channel(array_formats[2], 2),
                d.Width * std::max(size_t(1),d.Height) * std::max(size_t(1),d.Depth) * 4, runtime);
        }
        for (const auto& format : encoded_arrays) {
            if (runtime && !format.runtime_allocation) continue;
            const CUDA_ARRAY3D_DESCRIPTOR d{11,7,0,format.format,format.channels,CUDA_ARRAY3D_DEFERRED_MAPPING};
            const cudaChannelFormatDesc channel{int(format.bits),format.channels > 1 ? int(format.bits) : 0,
                format.channels > 2 ? int(format.bits) : 0,
                format.channels > 3 ? (format.bits == 10 ? 2 : int(format.bits)) : 0,format.kind};
            const size_t bytes = (11 + format.block_width - 1) / format.block_width * format.block_bytes
                * ((7 + format.block_height - 1) / format.block_height);
            check(d, channel, bytes, runtime);
        }
        for (const auto& format : planar_arrays) {
            if (runtime && !format.runtime_allocation) continue;
            const CUDA_ARRAY3D_DESCRIPTOR d{16,12,0,format.format,3,CUDA_ARRAY3D_DEFERRED_MAPPING};
            const int bits = int(format.bytes * 8);
            check(d, {bits,bits,bits,0,format.kind},
                (16 * 12 + 2 * (16 / format.subsample_x) * (12 / format.subsample_y)) * format.bytes, runtime);
        }
    }
}
void mipmap_memory_requirements() {
    for (bool deferred : {false, true}) {
        const CUDA_ARRAY3D_DESCRIPTOR descriptor{16,8,0,CU_AD_FORMAT_UNSIGNED_INT16,2,
            deferred ? CUDA_ARRAY3D_DEFERRED_MAPPING : 0u};
        CUmipmappedArray parent{};
        CONFORMANCE_CUDA(cuMipmappedArrayCreate(&parent, &descriptor, 4));
        try {
            CUdevice device{};
            CONFORMANCE_CUDA(cuCtxGetDevice(&device));
            CUDA_ARRAY_MEMORY_REQUIREMENTS requirements{};
            const auto status = cuMipmappedArrayGetMemoryRequirements(&requirements, parent, device);
            if (!deferred) require(status == CUDA_ERROR_INVALID_VALUE, "ordinary mipmap has no deferred requirements");
            else {
                CONFORMANCE_CUDA(status);
                const size_t payload = (16 * 8 + 8 * 4 + 4 * 2 + 2 * 1) * 4;
                require(requirements.size >= payload && requirements.alignment
                    && !(requirements.alignment & (requirements.alignment - 1))
                    && requirements.size % requirements.alignment == 0, "whole-mipmap deferred memory requirements");
                cudaArrayMemoryRequirements actual{};
                int ordinal = -1;
                ARRAY_RUNTIME(cudaGetDevice(&ordinal));
                ARRAY_RUNTIME(cudaMipmappedArrayGetMemoryRequirements(&actual,
                    reinterpret_cast<cudaMipmappedArray_t>(parent), ordinal));
                require(actual.size == requirements.size && actual.alignment == requirements.alignment,
                    "Runtime whole-mipmap requirements agreement");
            }
            CONFORMANCE_CUDA(cuMipmappedArrayDestroy(parent)); parent = nullptr;
        } catch (...) {
            if (parent) cuMipmappedArrayDestroy(parent);
            throw;
        }
    }
}
void array_usage_flags() {
    for (const auto descriptor : std::array<CUDA_ARRAY3D_DESCRIPTOR, 4>{{
        {7, 3, 0, CU_AD_FORMAT_UNSIGNED_INT16, 1, CUDA_ARRAY3D_DEPTH_TEXTURE},
        {7, 3, 0, CU_AD_FORMAT_FLOAT, 1, CUDA_ARRAY3D_DEPTH_TEXTURE},
        {7, 3, 0, CU_AD_FORMAT_UNSIGNED_INT8, 4, CUDA_ARRAY3D_COLOR_ATTACHMENT},
        {7, 3, 0, CU_AD_FORMAT_UNSIGNED_INT8, 4, CUDA_ARRAY3D_VIDEO_ENCODE_DECODE}}}) {
        const auto& format = descriptor.Format == CU_AD_FORMAT_FLOAT ? array_formats[7]
            : descriptor.Format == CU_AD_FORMAT_UNSIGNED_INT16 ? array_formats[2] : array_formats[0];
        const auto channel = array_channel(format, descriptor.NumChannels);
        stored_array array;
        create_stored_array(array, descriptor, channel, false, false);
        verify_array_descriptors(array.value, descriptor, channel);
        transfer_array_storage(array.value, descriptor, format.bytes, 31, true, false);
        transfer_array_storage(array.value, descriptor, format.bytes, 31, false, true);
        array.destroy();
    }
    CUdevice device{};
    CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    int sparse = 0;
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&sparse, CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED, device));
    const CUDA_ARRAY3D_DESCRIPTOR descriptor{64,64,0,CU_AD_FORMAT_UNSIGNED_INT8,1,CUDA_ARRAY3D_SPARSE};
    for (unsigned api : {0u,1u,2u}) {
        stored_array array;
        array.runtime = api != 0;
        if (!api) {
            CUarray candidate{};
            const auto status = cuArray3DCreate(&candidate, &descriptor);
            if (status == CUDA_SUCCESS) array.value = candidate;
            if (!sparse) {
                require(status == CUDA_ERROR_NOT_SUPPORTED, "Driver sparse admission must match endpoint capability");
                continue;
            }
            CONFORMANCE_CUDA(status);
        } else {
            cudaArray_t candidate{};
            const auto channel = array_channel(array_formats[0], 1);
            const auto status = api == 1 ? cudaMallocArray(&candidate, &channel, 64, 64, cudaArraySparse)
                : cudaMalloc3DArray(&candidate, &channel, {64,64,0}, cudaArraySparse);
            if (status == cudaSuccess) array.value = reinterpret_cast<CUarray>(candidate);
            if (!sparse) {
                require(status == cudaErrorNotSupported, "Runtime sparse admission must match endpoint capability");
                continue;
            }
            ARRAY_RUNTIME(status);
        }
        verify_array_descriptors(array.value, descriptor, array_channel(array_formats[0], 1));
        CUDA_ARRAY_SPARSE_PROPERTIES properties{};
        CONFORMANCE_CUDA(cuArrayGetSparseProperties(&properties, array.value));
        require(properties.tileExtent.width && properties.tileExtent.height && properties.tileExtent.depth
            && properties.miptailFirstLevel == 0
            && !(properties.flags & ~CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL), "sparse allocation layout properties");
        require(properties.flags & CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL ? properties.miptailSize > 0 : properties.miptailSize == 0,
            "single-array sparse mip tail size");
        cudaArraySparseProperties runtime{};
        ARRAY_RUNTIME(cudaArrayGetSparseProperties(&runtime, reinterpret_cast<cudaArray_t>(array.value)));
        require(runtime.tileExtent.width == properties.tileExtent.width && runtime.tileExtent.height == properties.tileExtent.height
            && runtime.tileExtent.depth == properties.tileExtent.depth && runtime.miptailFirstLevel == properties.miptailFirstLevel
            && runtime.miptailSize == properties.miptailSize && runtime.flags == properties.flags, "Runtime sparse properties agreement");
        array.destroy();
    }
    if (!sparse) std::cout << "sparse storage facility absent; admission rejection checked" << std::endl;
    auto invalid = descriptor; invalid.Flags |= CUDA_ARRAY3D_DEFERRED_MAPPING;
    reject_array_descriptor(invalid, false);
}
void array_limit_boundaries() {
    struct limit_case { unsigned flags, dimensions; CUdevice_attribute width, height, depth; };
    const limit_case cases[]{
        {0,1,CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH,{},{}},
        {0,2,CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH,CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT,{}},
        {0,3,CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH,CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT,CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH},
        {CUDA_ARRAY3D_LAYERED,1,CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH,{},CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS},
        {CUDA_ARRAY3D_LAYERED,2,CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH,CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT,CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS},
        {CUDA_ARRAY3D_CUBEMAP,2,CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH,{},{}},
        {CUDA_ARRAY3D_CUBEMAP|CUDA_ARRAY3D_LAYERED,2,CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH,{},CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS},
        {CUDA_ARRAY3D_TEXTURE_GATHER,2,CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH,CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT,{}},
        {CUDA_ARRAY3D_SURFACE_LDST,1,CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH,{},{}},
        {CUDA_ARRAY3D_SURFACE_LDST,2,CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH,CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT,{}},
        {CUDA_ARRAY3D_SURFACE_LDST,3,CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH,CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT,CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH},
        {CUDA_ARRAY3D_SURFACE_LDST|CUDA_ARRAY3D_LAYERED,1,CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH,{},CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS},
        {CUDA_ARRAY3D_SURFACE_LDST|CUDA_ARRAY3D_LAYERED,2,CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH,CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT,CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS},
        {CUDA_ARRAY3D_SURFACE_LDST|CUDA_ARRAY3D_CUBEMAP,2,CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH,{},{}},
        {CUDA_ARRAY3D_SURFACE_LDST|CUDA_ARRAY3D_CUBEMAP|CUDA_ARRAY3D_LAYERED,2,CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH,{},CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS}
    };
    CUdevice device{};
    CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    auto maximum = [&](CUdevice_attribute attribute) {
        int result = 0;
        CONFORMANCE_CUDA(cuDeviceGetAttribute(&result, attribute, device));
        require(result > 0, "array allocation limit must admit its array shape");
        return size_t(result);
    };
    const auto channel = array_channel(array_formats[0], 1);
    for (const auto& shape : cases) {
        const bool cube = (shape.flags & CUDA_ARRAY3D_CUBEMAP) != 0;
        const bool layered = (shape.flags & CUDA_ARRAY3D_LAYERED) != 0;
        const CUDA_ARRAY3D_DESCRIPTOR unit{1, shape.dimensions > 1 ? 1u : 0u,
            cube ? 6u : (layered || shape.dimensions == 3 ? 1u : 0u), CU_AD_FORMAT_UNSIGNED_INT8, 1, shape.flags};
        stored_array small;
        create_stored_array(small, unit, channel, false, false);
        transfer_array_storage(small.value, unit, 1, 41, true, false);
        transfer_array_storage(small.value, unit, 1, 41, false, true);
        small.destroy();
        const std::array<CUdevice_attribute, 3> attributes{shape.width, shape.height, shape.depth};
        for (unsigned axis = 0; axis < attributes.size(); ++axis) {
            if (!attributes[axis]) continue;
            auto bound = maximum(attributes[axis]);
            if (shape.dimensions == 3 && !(shape.flags & CUDA_ARRAY3D_SURFACE_LDST))
                bound = std::max(bound, maximum(static_cast<CUdevice_attribute>(
                    unsigned(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE) + axis)));
            auto invalid = unit;
            const auto beyond = cube && axis == 2 ? (bound / 6 + 1) * 6 : bound + 1;
            if (!axis) { invalid.Width = beyond; if (cube) invalid.Height = beyond; }
            else if (axis == 1) invalid.Height = beyond;
            else invalid.Depth = beyond;
            reject_array_descriptor(invalid, false);
            cudaArray_t rejected{};
            const auto status = cudaMalloc3DArray(&rejected, &channel,
                {invalid.Width, invalid.Height, invalid.Depth}, runtime_array_flags(invalid.Flags));
            if (status == cudaSuccess) ARRAY_RUNTIME(cudaFreeArray(rejected));
            require(status == cudaErrorInvalidValue, "Runtime array limit rejection");
        }
    }
    reject_array_descriptor({SIZE_MAX, 2, 0, CU_AD_FORMAT_UNSIGNED_INT32, 4, 0}, false);
}
void planar_array_retirement() {
    for (bool runtime : {false, true}) {
        const CUDA_ARRAY3D_DESCRIPTOR descriptor{16,12,0,CU_AD_FORMAT_NV12,3,0};
        stored_array parent;
        create_stored_array(parent, descriptor, {8,8,8,0,cudaChannelFormatKindNV12}, false, runtime);
        CUarray luma{};
        CONFORMANCE_CUDA(cuArrayGetPlane(&luma, parent.value, 0));
        transfer_array_storage(luma, {16,12,0,CU_AD_FORMAT_UNSIGNED_INT8,1,0}, 1, 51, true, runtime);
        std::vector<unsigned char> expected(16 * 12 + 32, array_guard);
        for (size_t y = 0; y < 12; ++y) for (size_t x = 0; x < 16; ++x)
            expected[16 + y * 16 + x] = array_storage_pattern(x, y, 0, 51);
        array_host_storage output(expected.size());
        std::fill_n(output.value, expected.size(), array_guard);
        array_transfer_stream stream;
        CUDA_MEMCPY2D copy{};
        copy.srcMemoryType = CU_MEMORYTYPE_ARRAY; copy.srcArray = luma;
        copy.dstMemoryType = CU_MEMORYTYPE_HOST; copy.dstHost = output.value + 16; copy.dstPitch = 16;
        copy.WidthInBytes = 16; copy.Height = 12;
        CONFORMANCE_CUDA(cuMemcpy2DAsync(&copy, stream.value));
        parent.destroy();
        CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
        require(std::equal(expected.begin(), expected.end(), output.value), "parent retirement retains queued plane storage");
        CUDA_ARRAY3D_DESCRIPTOR retired{};
        require(cuArray3DGetDescriptor(&retired, luma) == CUDA_ERROR_INVALID_HANDLE, "queued plane retirement invalidates handle");
        CONFORMANCE_CUDA(cuStreamDestroy(stream.value)); stream.value = nullptr;
        CONFORMANCE_CUDA(cuMemFreeHost(output.value)); output.value = nullptr;
    }
}
std::string_view array_storage_group = "all";
void execute_array_storage() {
    const auto selected = array_storage_group;
    require(selected == "all" || selected == "numeric" || selected == "formats" || selected == "planes"
        || selected == "deferred" || selected == "flags" || selected == "limits" || selected == "validation"
        || selected == "lifetime", "unknown array storage case");
    auto run = [&](std::string_view name, void (*execute)()) {
        if (selected == "all" || selected == name) {
            std::cout << "array_storage=" << name << std::endl;
            execute();
        }
    };
    run("numeric", numeric_array_storage);
    run("formats", encoded_array_storage);
    run("planes", planar_array_storage);
    run("deferred", deferred_array_storage_contract);
    run("deferred", mipmap_memory_requirements);
    run("flags", array_usage_flags);
    run("limits", array_limit_boundaries);
    run("validation", array_validation);
    if (selected == "all" || selected == "lifetime") {
        queued_array_retirement();
        planar_array_retirement();
        context_array_retirement();
    }
}
}
int main(int argc, char** argv) {
    if (argc == 3) {
        cudavm::array_storage_group = argv[2];
        --argc;
    }
    return cudavm::run_conformance(argc, argv, cudavm::execute_array_storage);
}
