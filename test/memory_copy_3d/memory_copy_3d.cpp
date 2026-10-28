#include "conformance.hpp"
#include <algorithm>
#include <limits>

namespace cudavm {
constexpr size_t copy_3d_width = 44, copy_3d_rows = 5, copy_3d_slices = 4;
constexpr unsigned char copy_3d_guard = 0xc3;
unsigned char copy_3d_pattern(size_t x, size_t y, size_t z, bool source) {
    return static_cast<unsigned char>(source ? x * 37 + y * 71 + z * 113 + 17 : x * 19 + y * 43 + z * 59 + 0x89);
}
struct copy_3d_endpoint {
    unsigned kind;
    size_t pitch, slice_rows, prefix;
    void* host{};
    CUdeviceptr device{};
    CUarray array{};
    std::vector<unsigned char> expected;
    explicit copy_3d_endpoint(unsigned type)
        : kind(type), pitch(type == 0 ? 60 : copy_3d_width), slice_rows(type == 0 ? 7 : type == 1 ? 9 : copy_3d_rows), prefix(type == 0 ? 16 : 0) {
        if (kind == 0) CONFORMANCE_CUDA(cuMemAllocHost(&host, pitch * slice_rows * copy_3d_slices + 32));
        else if (kind == 1) CONFORMANCE_CUDA(cuMemAllocPitch(&device, &pitch, copy_3d_width, slice_rows * copy_3d_slices, 4));
        else {
            CUDA_ARRAY3D_DESCRIPTOR descriptor{};
            descriptor.Width = copy_3d_width / 4; descriptor.Height = copy_3d_rows; descriptor.Depth = copy_3d_slices;
            descriptor.Format = CU_AD_FORMAT_UNSIGNED_INT16; descriptor.NumChannels = 2;
            CONFORMANCE_CUDA(cuArray3DCreate(&array, &descriptor));
        }
        require(pitch >= copy_3d_width && pitch <= (std::numeric_limits<size_t>::max() - 32) / slice_rows / copy_3d_slices, "3D endpoint extent");
        expected.resize(pitch * slice_rows * copy_3d_slices + prefix * 2);
    }
    copy_3d_endpoint(const copy_3d_endpoint&) = delete;
    copy_3d_endpoint& operator=(const copy_3d_endpoint&) = delete;
    ~copy_3d_endpoint() {
        if (host) cuMemFreeHost(host);
        if (device) cuMemFree(device);
        if (array) cuArrayDestroy(array);
    }
    void source(CUDA_MEMCPY3D& operation) const {
        if (kind == 0) {
            operation.srcMemoryType = CU_MEMORYTYPE_HOST; operation.srcHost = static_cast<unsigned char*>(host) + prefix;
            operation.srcPitch = pitch; operation.srcHeight = slice_rows;
        } else if (kind == 1) {
            operation.srcMemoryType = CU_MEMORYTYPE_DEVICE; operation.srcDevice = device;
            operation.srcPitch = pitch; operation.srcHeight = slice_rows;
        } else { operation.srcMemoryType = CU_MEMORYTYPE_ARRAY; operation.srcArray = array; }
    }
    void destination(CUDA_MEMCPY3D& operation) const {
        if (kind == 0) {
            operation.dstMemoryType = CU_MEMORYTYPE_HOST; operation.dstHost = static_cast<unsigned char*>(host) + prefix;
            operation.dstPitch = pitch; operation.dstHeight = slice_rows;
        } else if (kind == 1) {
            operation.dstMemoryType = CU_MEMORYTYPE_DEVICE; operation.dstDevice = device;
            operation.dstPitch = pitch; operation.dstHeight = slice_rows;
        } else { operation.dstMemoryType = CU_MEMORYTYPE_ARRAY; operation.dstArray = array; }
    }
    void reset(bool is_source) {
        std::fill(expected.begin(), expected.end(), copy_3d_guard);
        for (size_t z = 0; z < copy_3d_slices; ++z) for (size_t y = 0; y < copy_3d_rows; ++y)
        for (size_t x = 0; x < copy_3d_width; ++x)
            expected[prefix + (z * slice_rows + y) * pitch + x] = copy_3d_pattern(x, y, z, is_source);
        if (kind == 0) std::copy(expected.begin(), expected.end(), static_cast<unsigned char*>(host));
        else if (kind == 1) CONFORMANCE_CUDA(cuMemcpyHtoD(device, expected.data(), expected.size()));
        else {
            CUDA_MEMCPY3D operation{};
            operation.srcMemoryType = CU_MEMORYTYPE_HOST; operation.srcHost = expected.data();
            operation.srcPitch = pitch; operation.srcHeight = slice_rows;
            destination(operation);
            operation.WidthInBytes = copy_3d_width; operation.Height = copy_3d_rows; operation.Depth = copy_3d_slices;
            CONFORMANCE_CUDA(cuMemcpy3D(&operation));
        }
    }
    void verify(const std::string& context) const {
        std::vector<unsigned char> actual(expected.size(), copy_3d_guard);
        if (kind == 0) std::copy_n(static_cast<unsigned char*>(host), actual.size(), actual.begin());
        else if (kind == 1) CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), device, actual.size()));
        else {
            CUDA_MEMCPY3D operation{};
            source(operation); operation.dstMemoryType = CU_MEMORYTYPE_HOST; operation.dstHost = actual.data();
            operation.dstPitch = pitch; operation.dstHeight = slice_rows;
            operation.WidthInBytes = copy_3d_width; operation.Height = copy_3d_rows; operation.Depth = copy_3d_slices;
            CONFORMANCE_CUDA(cuMemcpy3D(&operation));
        }
        for (size_t i = 0; i < actual.size(); ++i)
            if (actual[i] != expected[i]) throw std::runtime_error(context + ": byte " + std::to_string(i)
                + " got " + std::to_string(actual[i]) + ", expected " + std::to_string(expected[i]));
    }
    void close() {
        if (host) { CONFORMANCE_CUDA(cuMemFreeHost(host)); host = nullptr; }
        if (device) { CONFORMANCE_CUDA(cuMemFree(device)); device = 0; }
        if (array) { CONFORMANCE_CUDA(cuArrayDestroy(array)); array = nullptr; }
    }
};
struct copy_3d_stream {
    CUstream value{};
    copy_3d_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~copy_3d_stream() { if (value) cuStreamDestroy(value); }
};
void execute_memory_copy_3d() {
    copy_3d_stream stream;
    // Source XYZ, destination XYZ, width bytes/height rows/depth slices.
    const std::array<std::array<size_t, 9>, 7> regions{{
        {0, 0, 0, 0, 0, 0, copy_3d_width, copy_3d_rows, copy_3d_slices},
        {copy_3d_width - 4, copy_3d_rows - 1, copy_3d_slices - 1,
         copy_3d_width - 4, copy_3d_rows - 1, copy_3d_slices - 1, 4, 1, 1},
        {4, 1, 1, 12, 2, 2, 8, 2, 2}, {8, 2, 0, 4, 1, 2, 20, 3, 1},
        {4, 1, 1, 8, 2, 2, 0, 2, 1}, {0, 0, 0, 0, 0, 0, copy_3d_width, 0, copy_3d_slices},
        {0, 0, 0, 0, 0, 0, copy_3d_width, copy_3d_rows, 0}}};
    for (unsigned source_kind = 0; source_kind < 3; ++source_kind)
    for (unsigned destination_kind = 0; destination_kind < 3; ++destination_kind) {
        copy_3d_endpoint source(source_kind), destination(destination_kind);
        for (bool asynchronous : {false, true}) for (const auto& region : regions) {
            const auto [source_x, source_y, source_z, destination_x, destination_y, destination_z, width, height, depth] = region;
            source.reset(true); destination.reset(false);
            for (size_t z = 0; z < depth; ++z) for (size_t y = 0; y < height; ++y) for (size_t x = 0; x < width; ++x)
                destination.expected[destination.prefix + ((destination_z + z) * destination.slice_rows + destination_y + y) * destination.pitch + destination_x + x]
                    = copy_3d_pattern(source_x + x, source_y + y, source_z + z, true);
            CUDA_MEMCPY3D operation{};
            source.source(operation); destination.destination(operation);
            operation.srcXInBytes = source_x; operation.srcY = source_y; operation.srcZ = source_z;
            operation.dstXInBytes = destination_x; operation.dstY = destination_y; operation.dstZ = destination_z;
            operation.WidthInBytes = width; operation.Height = height; operation.Depth = depth;
            if (asynchronous) {
                CONFORMANCE_CUDA(cuMemcpy3DAsync(&operation, stream.value));
                CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
            } else CONFORMANCE_CUDA(cuMemcpy3D(&operation));
            const auto context = "3D copy source=" + std::to_string(source_kind) + " destination=" + std::to_string(destination_kind)
                + " sx=" + std::to_string(source_x) + " sy=" + std::to_string(source_y) + " sz=" + std::to_string(source_z)
                + " dx=" + std::to_string(destination_x) + " dy=" + std::to_string(destination_y) + " dz=" + std::to_string(destination_z)
                + " width=" + std::to_string(width) + " height=" + std::to_string(height) + " depth=" + std::to_string(depth)
                + " async=" + std::to_string(asynchronous);
            destination.verify(context + " destination and padding"); source.verify(context + " source preservation");
        }
        destination.close(); source.close();
    }
    CONFORMANCE_CUDA(cuStreamDestroy(stream.value)); stream.value = nullptr;
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::execute_memory_copy_3d);
}
