#include "conformance.hpp"
#include <algorithm>

namespace cudavm {
struct fill_stream {
    CUstream value{};
    fill_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~fill_stream() { if (value) cuStreamDestroy(value); }
};
void linear_fill(CUdeviceptr pointer, unsigned bytes, uint32_t pattern, size_t count, bool asynchronous, CUstream stream) {
    if (bytes == 1) {
        if (asynchronous) CONFORMANCE_CUDA(cuMemsetD8Async(pointer, static_cast<unsigned char>(pattern), count, stream));
        else CONFORMANCE_CUDA(cuMemsetD8(pointer, static_cast<unsigned char>(pattern), count));
    } else if (bytes == 2) {
        if (asynchronous) CONFORMANCE_CUDA(cuMemsetD16Async(pointer, static_cast<unsigned short>(pattern), count, stream));
        else CONFORMANCE_CUDA(cuMemsetD16(pointer, static_cast<unsigned short>(pattern), count));
    } else {
        if (asynchronous) CONFORMANCE_CUDA(cuMemsetD32Async(pointer, pattern, count, stream));
        else CONFORMANCE_CUDA(cuMemsetD32(pointer, pattern, count));
    }
}
void pitched_fill(CUdeviceptr pointer, size_t pitch, unsigned bytes, uint32_t pattern,
                  size_t width, size_t height, bool asynchronous, CUstream stream) {
    if (bytes == 1) {
        if (asynchronous) CONFORMANCE_CUDA(cuMemsetD2D8Async(pointer, pitch, static_cast<unsigned char>(pattern), width, height, stream));
        else CONFORMANCE_CUDA(cuMemsetD2D8(pointer, pitch, static_cast<unsigned char>(pattern), width, height));
    } else if (bytes == 2) {
        if (asynchronous) CONFORMANCE_CUDA(cuMemsetD2D16Async(pointer, pitch, static_cast<unsigned short>(pattern), width, height, stream));
        else CONFORMANCE_CUDA(cuMemsetD2D16(pointer, pitch, static_cast<unsigned short>(pattern), width, height));
    } else {
        if (asynchronous) CONFORMANCE_CUDA(cuMemsetD2D32Async(pointer, pitch, pattern, width, height, stream));
        else CONFORMANCE_CUDA(cuMemsetD2D32(pointer, pitch, pattern, width, height));
    }
}
void expect_fill_bytes(const std::vector<unsigned char>& actual, const std::vector<unsigned char>& expected,
                       const std::string& context) {
    require(actual.size() == expected.size(), context + ": extent");
    for (size_t i = 0; i < actual.size(); ++i)
        if (actual[i] != expected[i]) throw std::runtime_error(context + ": byte " + std::to_string(i)
            + " got " + std::to_string(actual[i]) + ", expected " + std::to_string(expected[i]));
}
void execute_memory_fill() {
    constexpr unsigned char guard = 0xcb;
    fill_stream stream;
    for (unsigned bytes : {1u, 2u, 4u}) {
        const std::array<uint32_t, 4> patterns{0, UINT32_MAX, uint32_t(1) << (bytes * 8 - 1), 0x97ac53e1};
        constexpr unsigned capacity = 41;
        std::vector<unsigned char> initial(capacity * bytes + 32, guard), expected(initial.size()), actual(initial.size());
        device_buffer storage(initial.size());
        for (bool asynchronous : {false, true}) for (uint32_t pattern : patterns)
        for (size_t count : {size_t(0), size_t(1), size_t(17), size_t(40), size_t(41)}) {
            expected = initial;
            for (size_t element = 0; element < count; ++element)
                for (unsigned byte = 0; byte < bytes; ++byte)
                    expected[16 + element * bytes + byte] = static_cast<unsigned char>(pattern >> (byte * 8));
            CONFORMANCE_CUDA(cuMemcpyHtoD(storage.value, initial.data(), initial.size()));
            linear_fill(storage.value + 16, bytes, pattern, count, asynchronous, stream.value);
            if (asynchronous) CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), storage.value, actual.size()));
            expect_fill_bytes(actual, expected, "linear fill bytes=" + std::to_string(bytes)
                + " count=" + std::to_string(count) + " pattern=" + std::to_string(pattern)
                + " async=" + std::to_string(asynchronous));
        }
        constexpr size_t pitch = 128, rows = 5;
        initial.assign(pitch * rows + 32, guard); expected.resize(initial.size()); actual.resize(initial.size());
        device_buffer pitched(initial.size());
        const size_t row_capacity = pitch / bytes;
        // column, row, element width, row count; every selected rectangle is live.
        const std::array<std::array<size_t, 4>, 7> rectangles{{
            {0, 0, 0, 0}, {0, 0, row_capacity, rows}, {1, 1, row_capacity - 2, 3},
            {3, 2, 7, 1}, {row_capacity - 1, rows - 1, 1, 1},
            {0, 0, row_capacity, 0}, {2, 1, 0, 3}}};
        for (bool asynchronous : {false, true}) for (uint32_t pattern : patterns)
        for (const auto& rectangle : rectangles) {
            const auto [column, row, width, height] = rectangle;
            expected = initial;
            for (size_t y = 0; y < height; ++y) for (size_t x = 0; x < width; ++x)
                for (unsigned byte = 0; byte < bytes; ++byte)
                    expected[16 + (row + y) * pitch + (column + x) * bytes + byte]
                        = static_cast<unsigned char>(pattern >> (byte * 8));
            CONFORMANCE_CUDA(cuMemcpyHtoD(pitched.value, initial.data(), initial.size()));
            pitched_fill(pitched.value + 16 + row * pitch + column * bytes, pitch, bytes,
                pattern, width, height, asynchronous, stream.value);
            if (asynchronous) CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), pitched.value, actual.size()));
            expect_fill_bytes(actual, expected, "pitched fill bytes=" + std::to_string(bytes)
                + " column=" + std::to_string(column) + " row=" + std::to_string(row)
                + " width=" + std::to_string(width) + " height=" + std::to_string(height)
                + " pattern=" + std::to_string(pattern) + " async=" + std::to_string(asynchronous));
        }
    }
    CONFORMANCE_CUDA(cuStreamDestroy(stream.value));
    stream.value = nullptr;
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::execute_memory_fill);
}
