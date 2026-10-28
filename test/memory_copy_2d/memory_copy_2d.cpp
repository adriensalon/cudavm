#include "conformance.hpp"
#include <algorithm>
#include <limits>

namespace cudavm {
constexpr size_t copy_2d_width = 116, copy_2d_rows = 7, copy_2d_host_pitch = 140;
constexpr unsigned char copy_2d_guard = 0xad;
unsigned char copy_2d_pattern(size_t x, size_t y, bool source) {
    return static_cast<unsigned char>(source ? x * 37 + y * 71 + 13 : x * 19 + y * 43 + 0x81);
}
struct copy_2d_endpoint {
    unsigned kind;
    size_t pitch, prefix;
    void* host{};
    CUdeviceptr device{};
    CUarray array{};
    std::vector<unsigned char> expected;
    explicit copy_2d_endpoint(unsigned type) : kind(type), pitch(type == 0 ? copy_2d_host_pitch : copy_2d_width), prefix(type == 0 ? 16 : 0) {
        if (kind == 0) CONFORMANCE_CUDA(cuMemAllocHost(&host, pitch * copy_2d_rows + 32));
        else if (kind == 1) CONFORMANCE_CUDA(cuMemAllocPitch(&device, &pitch, copy_2d_width, copy_2d_rows, 4));
        else {
            CUDA_ARRAY_DESCRIPTOR descriptor{};
            descriptor.Width = copy_2d_width / 4; descriptor.Height = copy_2d_rows;
            descriptor.Format = CU_AD_FORMAT_UNSIGNED_INT16; descriptor.NumChannels = 2;
            CONFORMANCE_CUDA(cuArrayCreate(&array, &descriptor));
        }
        require(pitch >= copy_2d_width && pitch <= (std::numeric_limits<size_t>::max() - 32) / copy_2d_rows, "2D endpoint pitch extent");
        expected.resize(pitch * copy_2d_rows + prefix * 2);
    }
    copy_2d_endpoint(const copy_2d_endpoint&) = delete;
    copy_2d_endpoint& operator=(const copy_2d_endpoint&) = delete;
    ~copy_2d_endpoint() {
        if (host) cuMemFreeHost(host);
        if (device) cuMemFree(device);
        if (array) cuArrayDestroy(array);
    }
    void source(CUDA_MEMCPY2D& operation) const {
        if (kind == 0) {
            operation.srcMemoryType = CU_MEMORYTYPE_HOST;
            operation.srcHost = static_cast<unsigned char*>(host) + prefix;
            operation.srcPitch = pitch;
        } else if (kind == 1) {
            operation.srcMemoryType = CU_MEMORYTYPE_DEVICE; operation.srcDevice = device; operation.srcPitch = pitch;
        } else { operation.srcMemoryType = CU_MEMORYTYPE_ARRAY; operation.srcArray = array; }
    }
    void destination(CUDA_MEMCPY2D& operation) const {
        if (kind == 0) {
            operation.dstMemoryType = CU_MEMORYTYPE_HOST;
            operation.dstHost = static_cast<unsigned char*>(host) + prefix;
            operation.dstPitch = pitch;
        } else if (kind == 1) {
            operation.dstMemoryType = CU_MEMORYTYPE_DEVICE; operation.dstDevice = device; operation.dstPitch = pitch;
        } else { operation.dstMemoryType = CU_MEMORYTYPE_ARRAY; operation.dstArray = array; }
    }
    void reset(bool is_source) {
        std::fill(expected.begin(), expected.end(), copy_2d_guard);
        for (size_t y = 0; y < copy_2d_rows; ++y) for (size_t x = 0; x < copy_2d_width; ++x)
            expected[prefix + y * pitch + x] = copy_2d_pattern(x, y, is_source);
        if (kind == 0) std::copy(expected.begin(), expected.end(), static_cast<unsigned char*>(host));
        else if (kind == 1) CONFORMANCE_CUDA(cuMemcpyHtoD(device, expected.data(), expected.size()));
        else {
            CUDA_MEMCPY2D operation{};
            operation.srcMemoryType = CU_MEMORYTYPE_HOST; operation.srcHost = expected.data(); operation.srcPitch = pitch;
            destination(operation); operation.WidthInBytes = copy_2d_width; operation.Height = copy_2d_rows;
            CONFORMANCE_CUDA(cuMemcpy2D(&operation));
        }
    }
    void verify(const std::string& context) const {
        std::vector<unsigned char> actual(expected.size(), copy_2d_guard);
        if (kind == 0) std::copy_n(static_cast<unsigned char*>(host), actual.size(), actual.begin());
        else if (kind == 1) CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), device, actual.size()));
        else {
            CUDA_MEMCPY2D operation{};
            source(operation); operation.dstMemoryType = CU_MEMORYTYPE_HOST; operation.dstHost = actual.data(); operation.dstPitch = pitch;
            operation.WidthInBytes = copy_2d_width; operation.Height = copy_2d_rows;
            CONFORMANCE_CUDA(cuMemcpy2D(&operation));
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
struct copy_2d_stream {
    CUstream value{};
    copy_2d_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~copy_2d_stream() { if (value) cuStreamDestroy(value); }
};
void execute_memory_copy_2d() {
    copy_2d_stream stream;
    // Source X bytes/Y rows, destination X bytes/Y rows, width bytes/height rows.
    const std::array<std::array<size_t, 6>, 6> regions{{
        {0, 0, 0, 0, copy_2d_width, copy_2d_rows}, {4, 1, 12, 2, 4, 1},
        {12, 2, 20, 1, 28, 3}, {copy_2d_width - 16, copy_2d_rows - 2, copy_2d_width - 16, copy_2d_rows - 2, 16, 2},
        {4, 1, 8, 2, 0, 3}, {0, 0, 0, 0, copy_2d_width, 0}}};
    for (unsigned source_kind = 0; source_kind < 3; ++source_kind)
    for (unsigned destination_kind = 0; destination_kind < 3; ++destination_kind) {
        copy_2d_endpoint source(source_kind), destination(destination_kind);
        for (bool asynchronous : {false, true}) for (const auto& region : regions) {
            const auto [source_x, source_y, destination_x, destination_y, width, height] = region;
            source.reset(true); destination.reset(false);
            for (size_t y = 0; y < height; ++y) for (size_t x = 0; x < width; ++x)
                destination.expected[destination.prefix + (destination_y + y) * destination.pitch + destination_x + x]
                    = copy_2d_pattern(source_x + x, source_y + y, true);
            CUDA_MEMCPY2D operation{};
            source.source(operation); destination.destination(operation);
            operation.srcXInBytes = source_x; operation.srcY = source_y;
            operation.dstXInBytes = destination_x; operation.dstY = destination_y;
            operation.WidthInBytes = width; operation.Height = height;
            if (asynchronous) {
                CONFORMANCE_CUDA(cuMemcpy2DAsync(&operation, stream.value));
                CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
            } else CONFORMANCE_CUDA(cuMemcpy2D(&operation));
            const auto context = "2D copy source=" + std::to_string(source_kind) + " destination=" + std::to_string(destination_kind)
                + " sx=" + std::to_string(source_x) + " sy=" + std::to_string(source_y)
                + " dx=" + std::to_string(destination_x) + " dy=" + std::to_string(destination_y)
                + " width=" + std::to_string(width) + " height=" + std::to_string(height)
                + " async=" + std::to_string(asynchronous);
            destination.verify(context + " destination and padding"); source.verify(context + " source preservation");
        }
        destination.close(); source.close();
    }
    CONFORMANCE_CUDA(cuStreamDestroy(stream.value)); stream.value = nullptr;
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::execute_memory_copy_2d);
}
