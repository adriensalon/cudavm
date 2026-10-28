#include "conformance.hpp"
#include <algorithm>

namespace cudavm {
struct copy_host_buffer {
    void* value{};
    explicit copy_host_buffer(size_t bytes) { CONFORMANCE_CUDA(cuMemAllocHost(&value, bytes)); }
    ~copy_host_buffer() { if (value) cuMemFreeHost(value); }
    unsigned char* data() const { return static_cast<unsigned char*>(value); }
};
struct copy_stream {
    CUstream value{};
    copy_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~copy_stream() { if (value) cuStreamDestroy(value); }
};
void expect_copy_bytes(std::span<const unsigned char> actual, std::span<const unsigned char> expected,
                       const std::string& context) {
    require(actual.size() == expected.size(), context + ": extent");
    for (size_t i = 0; i < actual.size(); ++i)
        if (actual[i] != expected[i]) throw std::runtime_error(context + ": byte " + std::to_string(i)
            + " got " + std::to_string(actual[i]) + ", expected " + std::to_string(expected[i]));
}
void execute_memory_copy() {
    constexpr size_t capacity = 137, guard_bytes = 16, extent = capacity + guard_bytes * 2;
    constexpr unsigned char guard = 0xe9;
    std::vector<unsigned char> source(extent, guard), initial(extent, guard), expected(extent), actual(extent);
    for (size_t i = 0; i < capacity; ++i) {
        source[i + guard_bytes] = static_cast<unsigned char>(i * 73 + 29);
        initial[i + guard_bytes] = static_cast<unsigned char>(i * 19 + 0x87);
    }
    copy_host_buffer host_source(extent), host_destination(extent);
    device_buffer device_source(extent), device_destination(extent);
    copy_stream stream;
    // Independent source offset, destination offset and byte count.
    const std::array<std::array<size_t, 3>, 10> regions{{
        {0, 0, 0}, {0, 0, capacity}, {1, 3, 1}, {3, 1, 7}, {7, 13, 33},
        {capacity - 17, capacity - 17, 17}, {capacity - 1, capacity - 1, 1},
        {16, 32, 64}, {0, capacity - 64, 64}, {capacity - 64, 0, 64}}};
    for (bool asynchronous : {false, true}) for (unsigned direction = 0; direction < 3; ++direction)
    for (const auto& region : regions) {
        const auto [source_offset, destination_offset, bytes] = region;
        std::copy(source.begin(), source.end(), host_source.data());
        std::copy(initial.begin(), initial.end(), host_destination.data());
        CONFORMANCE_CUDA(cuMemcpyHtoD(device_source.value, source.data(), extent));
        CONFORMANCE_CUDA(cuMemcpyHtoD(device_destination.value, initial.data(), extent));
        expected = initial;
        std::copy_n(source.begin() + guard_bytes + source_offset, bytes,
            expected.begin() + guard_bytes + destination_offset);
        const auto from = device_source.value + guard_bytes + source_offset;
        const auto to = device_destination.value + guard_bytes + destination_offset;
        const auto* host_from = host_source.data() + guard_bytes + source_offset;
        auto* host_to = host_destination.data() + guard_bytes + destination_offset;
        if (direction == 0) {
            if (asynchronous) CONFORMANCE_CUDA(cuMemcpyHtoDAsync(to, host_from, bytes, stream.value));
            else CONFORMANCE_CUDA(cuMemcpyHtoD(to, host_from, bytes));
        } else if (direction == 1) {
            if (asynchronous) CONFORMANCE_CUDA(cuMemcpyDtoDAsync(to, from, bytes, stream.value));
            else CONFORMANCE_CUDA(cuMemcpyDtoD(to, from, bytes));
        } else {
            if (asynchronous) CONFORMANCE_CUDA(cuMemcpyDtoHAsync(host_to, from, bytes, stream.value));
            else CONFORMANCE_CUDA(cuMemcpyDtoH(host_to, from, bytes));
        }
        if (asynchronous) CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
        const auto context = "memory copy direction=" + std::to_string(direction)
            + " source=" + std::to_string(source_offset) + " destination=" + std::to_string(destination_offset)
            + " bytes=" + std::to_string(bytes) + " async=" + std::to_string(asynchronous);
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), device_destination.value, extent));
        expect_copy_bytes(actual, direction == 2 ? initial : expected, context + " device destination");
        expect_copy_bytes({host_destination.data(), extent}, direction == 2 ? expected : initial, context + " host destination");
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), device_source.value, extent));
        expect_copy_bytes(actual, source, context + " device source preservation");
        expect_copy_bytes({host_source.data(), extent}, source, context + " host source preservation");
    }
    CONFORMANCE_CUDA(cuStreamDestroy(stream.value)); stream.value = nullptr;
    CONFORMANCE_CUDA(cuMemFreeHost(host_source.value)); host_source.value = nullptr;
    CONFORMANCE_CUDA(cuMemFreeHost(host_destination.value)); host_destination.value = nullptr;
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::execute_memory_copy);
}
