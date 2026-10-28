#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>

namespace cudavm {
void check_mipmap_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define MIPMAP_RUNTIME(call) ::cudavm::check_mipmap_runtime((call), #call)

struct mipmap_shape {
    const char* name;
    size_t width, height, depth;
    unsigned flags;
};
constexpr mipmap_shape mipmap_shapes[]{
    {"unit", 1, 0, 0, 0}, {"1d", 9, 0, 0, 0}, {"2d", 9, 5, 0, 0},
    {"height_major", 1, 9, 0, 0}, {"3d", 9, 5, 3, 0}, {"depth_major", 2, 3, 17, 0},
    {"layered_1d", 2, 0, 17, CUDA_ARRAY3D_LAYERED},
    {"layered_2d", 9, 5, 17, CUDA_ARRAY3D_LAYERED},
    {"cubemap", 9, 9, 6, CUDA_ARRAY3D_CUBEMAP},
    {"layered_cubemap", 9, 9, 18, CUDA_ARRAY3D_CUBEMAP | CUDA_ARRAY3D_LAYERED}};
struct mipmap_format {
    CUarray_format driver;
    cudaChannelFormatDesc channel;
    unsigned channels, element_bytes;
};
constexpr mipmap_format mipmap_formats[]{
    {CU_AD_FORMAT_UNSIGNED_INT8, {8, 0, 0, 0, cudaChannelFormatKindUnsigned}, 1, 1},
    {CU_AD_FORMAT_SIGNED_INT16, {16, 16, 0, 0, cudaChannelFormatKindSigned}, 2, 4},
    {CU_AD_FORMAT_UNSIGNED_INT32, {32, 32, 32, 32, cudaChannelFormatKindUnsigned}, 4, 16},
    {CU_AD_FORMAT_FLOAT, {32, 32, 32, 32, cudaChannelFormatKindFloat}, 4, 16}};

unsigned mipmap_runtime_flags(unsigned flags) {
    return (flags & CUDA_ARRAY3D_LAYERED ? cudaArrayLayered : 0)
        | (flags & CUDA_ARRAY3D_CUBEMAP ? cudaArrayCubemap : 0)
        | (flags & CUDA_ARRAY3D_SURFACE_LDST ? cudaArraySurfaceLoadStore : 0);
}
unsigned mipmap_level_count(const mipmap_shape& shape, unsigned requested) {
    size_t largest = std::max(shape.width, shape.height);
    if (!(shape.flags & (CUDA_ARRAY3D_LAYERED | CUDA_ARRAY3D_CUBEMAP)))
        largest = std::max(largest, shape.depth);
    unsigned count = 1;
    while (largest > 1) { largest /= 2; ++count; }
    return std::min(std::max(1u, requested), count);
}
CUDA_ARRAY3D_DESCRIPTOR mipmap_level_descriptor(const mipmap_shape& shape,
                                               const mipmap_format& format, unsigned level) {
    CUDA_ARRAY3D_DESCRIPTOR result{shape.width, shape.height, shape.depth,
        format.driver, format.channels, shape.flags};
    for (unsigned i = 0; i < level; ++i) {
        result.Width = std::max(size_t(1), result.Width / 2);
        if (result.Height) result.Height = std::max(size_t(1), result.Height / 2);
        if (result.Depth && !(shape.flags & (CUDA_ARRAY3D_LAYERED | CUDA_ARRAY3D_CUBEMAP)))
            result.Depth = std::max(size_t(1), result.Depth / 2);
    }
    return result;
}

struct mipmap_allocation {
    CUmipmappedArray value{};
    bool runtime;
    mipmap_allocation(const mipmap_shape& shape, const mipmap_format& format,
                      unsigned levels, bool use_runtime) : runtime(use_runtime) {
        if (runtime) {
            auto channel = format.channel;
            cudaMipmappedArray_t result{};
            MIPMAP_RUNTIME(cudaMallocMipmappedArray(&result, &channel,
                {shape.width, shape.height, shape.depth}, levels, mipmap_runtime_flags(shape.flags)));
            value = reinterpret_cast<CUmipmappedArray>(result);
        } else {
            auto descriptor = mipmap_level_descriptor(shape, format, 0);
            CONFORMANCE_CUDA(cuMipmappedArrayCreate(&value, &descriptor, levels));
        }
        require(value != nullptr, "mipmap allocation publishes a handle");
    }
    ~mipmap_allocation() {
        if (value) {
            if (runtime) cudaFreeMipmappedArray(reinterpret_cast<cudaMipmappedArray_t>(value));
            else cuMipmappedArrayDestroy(value);
        }
    }
    mipmap_allocation(const mipmap_allocation&) = delete;
    mipmap_allocation& operator=(const mipmap_allocation&) = delete;
    void destroy(bool use_runtime) {
        if (use_runtime) MIPMAP_RUNTIME(cudaFreeMipmappedArray(reinterpret_cast<cudaMipmappedArray_t>(value)));
        else CONFORMANCE_CUDA(cuMipmappedArrayDestroy(value));
        value = nullptr;
    }
    CUarray level(unsigned index) const {
        CUarray first{}, again{};
        cudaArray_t other{};
        CONFORMANCE_CUDA(cuMipmappedArrayGetLevel(&first, value, index));
        CONFORMANCE_CUDA(cuMipmappedArrayGetLevel(&again, value, index));
        MIPMAP_RUNTIME(cudaGetMipmappedArrayLevel(&other, reinterpret_cast<cudaMipmappedArray_t>(value), index));
        require(first && first == again && first == reinterpret_cast<CUarray>(other),
            "repeated Driver/Runtime queries return the same level handle");
        return first;
    }
};

void verify_mipmap_level(CUarray level, const CUDA_ARRAY3D_DESCRIPTOR& expected,
                          const mipmap_format& format) {
    CUDA_ARRAY3D_DESCRIPTOR actual{};
    CONFORMANCE_CUDA(cuArray3DGetDescriptor(&actual, level));
    require(actual.Width == expected.Width && actual.Height == expected.Height && actual.Depth == expected.Depth
        && actual.Format == expected.Format && actual.NumChannels == expected.NumChannels && actual.Flags == expected.Flags,
        "Driver level dimensions, format, channels and flags");
    cudaChannelFormatDesc channel{};
    cudaExtent extent{};
    unsigned flags = ~0u;
    MIPMAP_RUNTIME(cudaArrayGetInfo(&channel, &extent, &flags, reinterpret_cast<cudaArray_t>(level)));
    require(extent.width == expected.Width && extent.height == expected.Height && extent.depth == expected.Depth,
        "Runtime preserves absent dimensions and level/layer/face extents");
    require(channel.x == format.channel.x && channel.y == format.channel.y && channel.z == format.channel.z
        && channel.w == format.channel.w && channel.f == format.channel.f
        && flags == mipmap_runtime_flags(expected.Flags), "Runtime level format and flags");
}
void reject_mipmap_level(const mipmap_allocation& parent, unsigned index) {
    auto untouched = parent.level(0);
    const auto original = untouched;
    require(cuMipmappedArrayGetLevel(&untouched, parent.value, index) == CUDA_ERROR_INVALID_VALUE,
        "Driver rejects a level beyond the allocated chain");
    require(untouched == original, "failed Driver level query preserves output");
    auto runtime_output = reinterpret_cast<cudaArray_t>(original);
    require(cudaGetMipmappedArrayLevel(&runtime_output, reinterpret_cast<cudaMipmappedArray_t>(parent.value), index)
        == cudaErrorInvalidValue, "Runtime rejects a level beyond the allocated chain");
    require(runtime_output == reinterpret_cast<cudaArray_t>(original), "failed Runtime level query preserves output");
}
void mipmap_levels() {
    for (const auto& base : mipmap_shapes) for (const auto& format : mipmap_formats)
    for (bool runtime : {false, true}) for (unsigned surface : {0u, unsigned(CUDA_ARRAY3D_SURFACE_LDST)})
    for (unsigned requested : {0u, 1u, 2u, 99u, std::numeric_limits<unsigned>::max()}) {
        auto shape = base;
        shape.flags |= surface;
        try {
            mipmap_allocation parent(shape, format, requested, runtime);
            const auto count = mipmap_level_count(shape, requested);
            std::vector<CUarray> handles;
            for (unsigned i = 0; i < count; ++i) {
                auto level = parent.level(i);
                require(std::find(handles.begin(), handles.end(), level) == handles.end(), "distinct level handles");
                handles.push_back(level);
                verify_mipmap_level(level, mipmap_level_descriptor(shape, format, i), format);
            }
            reject_mipmap_level(parent, count);
            reject_mipmap_level(parent, std::numeric_limits<unsigned>::max());
            parent.destroy(!runtime);
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string(shape.name) + " format=" + std::to_string(unsigned(format.driver))
                + " flags=" + std::to_string(shape.flags) + " requested=" + std::to_string(requested)
                + (runtime ? " Runtime: " : " Driver: ") + error.what());
        }
    }
}

constexpr unsigned char mipmap_guard = 0xb7;
struct mipmap_host_storage {
    unsigned char* value{};
    size_t width, height, depth, pitch, slice_height;
    std::vector<unsigned char> expected;
    mipmap_host_storage(const CUDA_ARRAY3D_DESCRIPTOR& descriptor, unsigned element_bytes, unsigned seed, bool upload)
        : width(descriptor.Width), height(std::max(size_t(1), descriptor.Height)),
          depth(std::max(size_t(1), descriptor.Depth)), pitch(width * element_bytes + (upload ? 13 : 21)),
          slice_height(height + (upload ? 2 : 3)), expected(32 + pitch * slice_height * depth, mipmap_guard) {
        void* pointer{};
        CONFORMANCE_CUDA(cuMemAllocHost(&pointer, expected.size()));
        value = static_cast<unsigned char*>(pointer);
        for (size_t z = 0; z < depth; ++z) for (size_t y = 0; y < height; ++y)
        for (size_t x = 0; x < width * element_bytes; ++x)
            expected[16 + (z * slice_height + y) * pitch + x]
                = static_cast<unsigned char>(x * 37 + y * 71 + z * 113 + seed * 43);
        if (upload) std::copy(expected.begin(), expected.end(), value);
        else std::fill_n(value, expected.size(), mipmap_guard);
    }
    ~mipmap_host_storage() { if (value) cuMemFreeHost(value); }
    mipmap_host_storage(const mipmap_host_storage&) = delete;
    mipmap_host_storage& operator=(const mipmap_host_storage&) = delete;
    void verify(const std::string& operation) const {
        for (size_t i = 0; i < expected.size(); ++i)
            if (value[i] != expected[i]) throw std::runtime_error(operation + ": byte " + std::to_string(i));
    }
};
struct mipmap_stream {
    CUstream value{};
    mipmap_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~mipmap_stream() {
        if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); }
    }
    mipmap_stream(const mipmap_stream&) = delete;
    mipmap_stream& operator=(const mipmap_stream&) = delete;
};
void copy_mipmap_host(CUarray level, mipmap_host_storage& host, unsigned element_bytes,
                      bool upload, bool runtime, CUstream stream = nullptr) {
    if (runtime) {
        cudaMemcpy3DParms copy{};
        cudaPitchedPtr pointer{host.value + 16, host.pitch, host.width * element_bytes, host.slice_height};
        if (upload) { copy.srcPtr = pointer; copy.dstArray = reinterpret_cast<cudaArray_t>(level); }
        else { copy.srcArray = reinterpret_cast<cudaArray_t>(level); copy.dstPtr = pointer; }
        copy.extent = {host.width, host.height, host.depth};
        copy.kind = upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
        if (stream) MIPMAP_RUNTIME(cudaMemcpy3DAsync(&copy, reinterpret_cast<cudaStream_t>(stream)));
        else MIPMAP_RUNTIME(cudaMemcpy3D(&copy));
    } else {
        CUDA_MEMCPY3D copy{};
        if (upload) {
            copy.srcMemoryType = CU_MEMORYTYPE_HOST; copy.srcHost = host.value + 16;
            copy.srcPitch = host.pitch; copy.srcHeight = host.slice_height;
            copy.dstMemoryType = CU_MEMORYTYPE_ARRAY; copy.dstArray = level;
        } else {
            copy.srcMemoryType = CU_MEMORYTYPE_ARRAY; copy.srcArray = level;
            copy.dstMemoryType = CU_MEMORYTYPE_HOST; copy.dstHost = host.value + 16;
            copy.dstPitch = host.pitch; copy.dstHeight = host.slice_height;
        }
        copy.WidthInBytes = host.width * element_bytes; copy.Height = host.height; copy.Depth = host.depth;
        if (stream) CONFORMANCE_CUDA(cuMemcpy3DAsync(&copy, stream));
        else CONFORMANCE_CUDA(cuMemcpy3D(&copy));
    }
}
unsigned mipmap_seed(unsigned level, bool second) { return 3 + level * 7 + (second ? 41 : 0); }
void copy_mipmap_corner(CUarray source, CUarray destination, const CUDA_ARRAY3D_DESCRIPTOR& from,
                         const CUDA_ARRAY3D_DESCRIPTOR& to, unsigned element_bytes, bool runtime) {
    const size_t sy = std::max(size_t(1), from.Height) - 1, sz = std::max(size_t(1), from.Depth) - 1;
    const size_t dy = std::max(size_t(1), to.Height) - 1, dz = std::max(size_t(1), to.Depth) - 1;
    if (runtime) {
        cudaMemcpy3DParms copy{};
        copy.srcArray = reinterpret_cast<cudaArray_t>(source); copy.srcPos = {from.Width - 1, sy, sz};
        copy.dstArray = reinterpret_cast<cudaArray_t>(destination); copy.dstPos = {to.Width - 1, dy, dz};
        copy.extent = {1, 1, 1}; copy.kind = cudaMemcpyDeviceToDevice;
        MIPMAP_RUNTIME(cudaMemcpy3D(&copy));
    } else {
        CUDA_MEMCPY3D copy{};
        copy.srcMemoryType = CU_MEMORYTYPE_ARRAY; copy.srcArray = source;
        copy.srcXInBytes = (from.Width - 1) * element_bytes; copy.srcY = sy; copy.srcZ = sz;
        copy.dstMemoryType = CU_MEMORYTYPE_ARRAY; copy.dstArray = destination;
        copy.dstXInBytes = (to.Width - 1) * element_bytes; copy.dstY = dy; copy.dstZ = dz;
        copy.WidthInBytes = element_bytes; copy.Height = copy.Depth = 1;
        CONFORMANCE_CUDA(cuMemcpy3D(&copy));
    }
}
void mipmap_transfers() {
    for (const auto& base : mipmap_shapes) for (const auto& format : mipmap_formats)
    for (bool runtime : {false, true}) for (unsigned surface : {0u, unsigned(CUDA_ARRAY3D_SURFACE_LDST)}) {
        auto shape = base;
        shape.flags |= surface;
        try {
            mipmap_allocation first(shape, format, 99, runtime), second(shape, format, 99, !runtime);
            const unsigned count = mipmap_level_count(shape, 99), last = count - 1;
            // Initialize every level in two live parents before observing any level.
            for (unsigned i = 0; i < count; ++i) for (bool other : {false, true}) {
                const auto descriptor = mipmap_level_descriptor(shape, format, i);
                mipmap_host_storage source(descriptor, format.element_bytes, mipmap_seed(i, other), true);
                copy_mipmap_host((other ? second : first).level(i), source, format.element_bytes, true, runtime != other);
                source.verify("mipmap upload source and guards");
            }
            // Copy one corner texel from the smallest level into the largest level
            // of the other parent. This crosses levels, layers/faces and API units.
            const auto from = mipmap_level_descriptor(shape, format, last);
            const auto to = mipmap_level_descriptor(shape, format, 0);
            copy_mipmap_corner(first.level(last), second.level(0), from, to, format.element_bytes, runtime);
            // Reverse traversal observes every untouched level and every neighboring byte.
            for (unsigned remaining = count; remaining; --remaining) for (bool other : {false, true}) {
                const auto i = remaining - 1;
                const auto descriptor = mipmap_level_descriptor(shape, format, i);
                mipmap_host_storage output(descriptor, format.element_bytes, mipmap_seed(i, other), false);
                if (other && i == 0) {
                    mipmap_host_storage source(from, format.element_bytes, mipmap_seed(last, false), true);
                    const auto source_offset = 16 + ((source.depth - 1) * source.slice_height + source.height - 1)
                        * source.pitch + (source.width - 1) * format.element_bytes;
                    const auto target_offset = 16 + ((output.depth - 1) * output.slice_height + output.height - 1)
                        * output.pitch + (output.width - 1) * format.element_bytes;
                    std::copy_n(source.expected.begin() + source_offset, format.element_bytes,
                        output.expected.begin() + target_offset);
                }
                copy_mipmap_host((other ? second : first).level(i), output, format.element_bytes, false, runtime == other);
                output.verify("mipmap independent levels, layers/faces and readback guards");
            }
            first.destroy(!runtime);
            second.destroy(runtime);
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string(shape.name) + " format=" + std::to_string(unsigned(format.driver))
                + " flags=" + std::to_string(shape.flags) + (runtime ? " Runtime: " : " Driver: ") + error.what());
        }
    }
}

void mipmap_validation() {
    const auto& format = mipmap_formats[0];
    const auto& shape = mipmap_shapes[2];
    auto valid = mipmap_level_descriptor(shape, format, 0);
    CUmipmappedArray output{};
    require(cuMipmappedArrayCreate(nullptr, &valid, 3) == CUDA_ERROR_INVALID_VALUE, "null Driver allocation output");
    require(cuMipmappedArrayCreate(&output, nullptr, 3) == CUDA_ERROR_INVALID_VALUE, "null Driver descriptor");
    cudaMipmappedArray_t runtime_output{};
    require(cudaMallocMipmappedArray(nullptr, &format.channel, {9, 5, 0}, 3, 0) == cudaErrorInvalidValue,
        "null Runtime allocation output");
    require(cudaMallocMipmappedArray(&runtime_output, nullptr, {9, 5, 0}, 3, 0) == cudaErrorInvalidValue,
        "null Runtime channel descriptor");
    constexpr mipmap_shape invalid_shapes[]{
        {"zero_width", 0, 5, 0, 0}, {"missing_height", 9, 0, 3, 0},
        {"missing_layers", 9, 5, 0, CUDA_ARRAY3D_LAYERED},
        {"nonsquare_cube", 9, 5, 6, CUDA_ARRAY3D_CUBEMAP},
        {"cube_faces", 9, 9, 5, CUDA_ARRAY3D_CUBEMAP},
        {"layered_cube_faces", 9, 9, 7, CUDA_ARRAY3D_CUBEMAP | CUDA_ARRAY3D_LAYERED}};
    for (const auto& invalid : invalid_shapes) {
        auto descriptor = mipmap_level_descriptor(invalid, format, 0);
        output = nullptr;
        auto status = cuMipmappedArrayCreate(&output, &descriptor, 3);
        if (status == CUDA_SUCCESS) CONFORMANCE_CUDA(cuMipmappedArrayDestroy(output));
        require(status == CUDA_ERROR_INVALID_VALUE, std::string("invalid Driver mipmap shape: ") + invalid.name);
        runtime_output = nullptr;
        auto result = cudaMallocMipmappedArray(&runtime_output, &format.channel,
            {invalid.width, invalid.height, invalid.depth}, 3, mipmap_runtime_flags(invalid.flags));
        if (result == cudaSuccess) MIPMAP_RUNTIME(cudaFreeMipmappedArray(runtime_output));
        require(result == cudaErrorInvalidValue, std::string("invalid Runtime mipmap shape: ") + invalid.name);
    }
    mipmap_allocation parent(shape, format, 2, false);
    require(cuMipmappedArrayGetLevel(nullptr, parent.value, 0) == CUDA_ERROR_INVALID_VALUE, "null Driver level output");
    require(cudaGetMipmappedArrayLevel(nullptr, reinterpret_cast<cudaMipmappedArray_t>(parent.value), 0)
        == cudaErrorInvalidValue, "null Runtime level output");
    CUarray level{};
    cudaArray_t runtime_level{};
    require(cuMipmappedArrayGetLevel(&level, nullptr, 0) == CUDA_ERROR_INVALID_HANDLE, "null Driver parent");
    require(cudaGetMipmappedArrayLevel(&runtime_level, nullptr, 0) == cudaErrorInvalidResourceHandle, "null Runtime parent");
    require(cuMipmappedArrayDestroy(nullptr) == CUDA_ERROR_INVALID_HANDLE, "null Driver destruction handle");
    reject_mipmap_level(parent, 2);
    // Failed operations must leave the allocated levels usable.
    verify_mipmap_level(parent.level(1), mipmap_level_descriptor(shape, format, 1), format);
    parent.destroy(true);
}

void mipmap_lifetime() {
    const auto& format = mipmap_formats[1];
    for (bool runtime : {false, true}) for (const auto& shape : mipmap_shapes) {
        mipmap_allocation parent(shape, format, 99, runtime), survivor(shape, format, 99, !runtime);
        const auto count = mipmap_level_count(shape, 99);
        std::vector<CUarray> retired;
        std::vector<std::unique_ptr<mipmap_host_storage>> sources, outputs;
        for (unsigned i = 0; i < count; ++i) {
            const auto descriptor = mipmap_level_descriptor(shape, format, i);
            mipmap_host_storage source(descriptor, format.element_bytes, i + 101, true);
            copy_mipmap_host(survivor.level(i), source, format.element_bytes, true, !runtime);
            source.verify("surviving parent upload source and guards");
        }
        // The stream is retired before the pinned buffers on exceptional exits.
        mipmap_stream stream;
        for (unsigned i = 0; i < count; ++i) {
            const auto descriptor = mipmap_level_descriptor(shape, format, i);
            auto level = parent.level(i);
            retired.push_back(level);
            sources.push_back(std::make_unique<mipmap_host_storage>(descriptor, format.element_bytes, i + 73, true));
            outputs.push_back(std::make_unique<mipmap_host_storage>(descriptor, format.element_bytes, i + 73, false));
            copy_mipmap_host(level, *sources.back(), format.element_bytes, true, runtime, stream.value);
            copy_mipmap_host(level, *outputs.back(), format.element_bytes, false, !runtime, stream.value);
        }
        // No explicit synchronization before destroying the owner of all levels.
        const auto old_parent = parent.value;
        parent.destroy(!runtime);
        CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
        for (unsigned i = 0; i < count; ++i) {
            sources[i]->verify("retired parent upload source and guards");
            outputs[i]->verify("retired parent queued level readback and guards");
            CUDA_ARRAY3D_DESCRIPTOR descriptor{};
            require(cuArray3DGetDescriptor(&descriptor, retired[i]) == CUDA_ERROR_INVALID_HANDLE, "parent retires level handles");
        }
        CUarray invalid{};
        require(cuMipmappedArrayGetLevel(&invalid, old_parent, 0) == CUDA_ERROR_INVALID_HANDLE, "retired parent lookup");
        require(cuMipmappedArrayDestroy(old_parent) == CUDA_ERROR_INVALID_HANDLE, "repeated parent destruction");
        mipmap_allocation replacement(shape, format, 99, runtime);
        for (unsigned i = 0; i < count; ++i) {
            const auto descriptor = mipmap_level_descriptor(shape, format, i);
            mipmap_host_storage output(descriptor, format.element_bytes, i + 101, false);
            copy_mipmap_host(replacement.level(i), *sources[i], format.element_bytes, true, !runtime);
            copy_mipmap_host(survivor.level(i), output, format.element_bytes, false, !runtime);
            output.verify("surviving parent storage after replacement allocation");
            std::fill_n(outputs[i]->value, outputs[i]->expected.size(), mipmap_guard);
            copy_mipmap_host(replacement.level(i), *outputs[i], format.element_bytes, false, runtime);
            outputs[i]->verify("replacement parent level storage");
            sources[i]->verify("replacement parent upload source and guards");
        }
        replacement.destroy(!runtime);
        survivor.destroy(runtime);
    }
}

std::string_view mipmap_case = "all";
void execute_mipmapped_arrays() {
    require(mipmap_case == "all" || mipmap_case == "levels" || mipmap_case == "transfers"
        || mipmap_case == "validation" || mipmap_case == "lifetime", "unknown mipmapped_arrays case");
    const std::array<std::pair<std::string_view, void (*)()>, 4> cases{{
        {"levels", mipmap_levels}, {"transfers", mipmap_transfers},
        {"validation", mipmap_validation}, {"lifetime", mipmap_lifetime}}};
    for (const auto& [name, execute] : cases) if (mipmap_case == "all" || mipmap_case == name) {
        std::cout << "mipmapped_arrays=" << name << std::endl;
        execute();
    }
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::mipmap_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_mipmapped_arrays);
}
