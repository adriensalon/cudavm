#include "fixtures.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>

namespace cudavm {
void check_mapped_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define MAPPED_RUNTIME(call) ::cudavm::check_mapped_runtime((call), #call)

struct mapped_host_buffer {
    static constexpr size_t words = 320, bytes = words * sizeof(uint32_t), alignment = 65536;
    unsigned kind;
    void* value{};
    CUdeviceptr alias{};
    std::vector<unsigned char> backing;
    mapped_host_buffer(unsigned form, bool portable) : kind(form) {
        const unsigned flags = CU_MEMHOSTALLOC_DEVICEMAP | (portable ? CU_MEMHOSTALLOC_PORTABLE : 0);
        if (kind < 2) {
            if (kind == 0) CONFORMANCE_CUDA(cuMemHostAlloc(&value, bytes, flags));
            else MAPPED_RUNTIME(cudaHostAlloc(&value, bytes, flags));
        } else {
            backing.resize(alignment * 3, 0xa7);
            auto address = reinterpret_cast<uintptr_t>(backing.data());
            value = reinterpret_cast<void*>(((address + alignment - 1) & ~(uintptr_t(alignment) - 1)) + 4);
            if (kind == 2) CONFORMANCE_CUDA(cuMemHostRegister(value, bytes, flags));
            else MAPPED_RUNTIME(cudaHostRegister(value, bytes, flags));
        }
    }
    ~mapped_host_buffer() { if (value) { if (kind < 2) cuMemFreeHost(value); else cuMemHostUnregister(value); } }
    mapped_host_buffer(const mapped_host_buffer&) = delete;
    mapped_host_buffer& operator=(const mapped_host_buffer&) = delete;
    void release() {
        if (kind == 0) MAPPED_RUNTIME(cudaFreeHost(value));
        else if (kind == 1) CONFORMANCE_CUDA(cuMemFreeHost(value));
        else if (kind == 2) MAPPED_RUNTIME(cudaHostUnregister(value));
        else CONFORMANCE_CUDA(cuMemHostUnregister(value));
        value = nullptr;
    }
    void initialize(std::span<const uint32_t> values) { std::memcpy(value, values.data(), bytes); }
    void expect(std::span<const uint32_t> expected, const std::string& operation) const {
        std::vector<uint32_t> actual(words);
        std::memcpy(actual.data(), value, bytes);
        expect_words(actual, expected, operation);
        if (!backing.empty()) {
            const auto offset = static_cast<const unsigned char*>(value) - backing.data();
            for (size_t i = 0; i < backing.size(); ++i)
                if ((i < size_t(offset) || i >= size_t(offset) + bytes) && backing[i] != 0xa7)
                    throw std::runtime_error(operation + ": registration backing guard " + std::to_string(i));
        }
    }
    void query_alias() {
        CONFORMANCE_CUDA(cuMemHostGetDevicePointer(&alias, value, 0));
        require(alias != 0, "mapped device address is nonzero");
        for (size_t offset : {size_t(0), size_t(1), bytes / 2, bytes - 1}) {
            CUdeviceptr driver{};
            void* runtime{};
            auto* pointer = static_cast<unsigned char*>(value) + offset;
            CONFORMANCE_CUDA(cuMemHostGetDevicePointer(&driver, pointer, 0));
            MAPPED_RUNTIME(cudaHostGetDevicePointer(&runtime, pointer, 0));
            require(driver == alias + offset && reinterpret_cast<uintptr_t>(runtime) == driver,
                "Driver/Runtime mapped interior aliases preserve byte offsets");
        }
        CUdeviceptr invalid{};
        void* runtime{};
        require(cuMemHostGetDevicePointer(nullptr, value, 0) == CUDA_ERROR_INVALID_VALUE, "null Driver alias output");
        require(cudaHostGetDevicePointer(nullptr, value, 0) == cudaErrorInvalidValue, "null Runtime alias output");
        require(cuMemHostGetDevicePointer(&invalid, value, 1) == CUDA_ERROR_INVALID_VALUE, "invalid Driver alias flags");
        require(cudaHostGetDevicePointer(&runtime, value, 1) == cudaErrorInvalidValue, "invalid Runtime alias flags");
    }
};
struct mapped_stream {
    CUstream value{};
    mapped_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~mapped_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
};
struct mapped_event {
    CUevent value{};
    mapped_event() { CONFORMANCE_CUDA(cuEventCreate(&value, CU_EVENT_DISABLE_TIMING)); }
    ~mapped_event() { if (value) cuEventDestroy(value); }
};
void mapped_launch(CUfunction kernel, CUdeviceptr out, CUdeviceptr in, unsigned count,
                    uint32_t seed, unsigned threads, CUstream stream) {
    void* args[]{&out, &in, &count, &seed};
    CONFORMANCE_CUDA(cuLaunchKernel(kernel, 3, 1, 1, threads, 1, 1, 0, stream, args, nullptr));
}
void mapped_expected(std::vector<uint32_t>& output, size_t destination, const std::vector<uint32_t>& input,
                        size_t source, unsigned count, unsigned threads, uint32_t seed) {
    for (unsigned i = 0; i < std::min(count, threads * 3); ++i)
        output[destination + i] = (input[source + i] ^ (seed + i * uint32_t(17))) + i;
}
void execute_mapped_host_memory(const loaded_program& program) {
    CUdevice device{};
    int can_map = 0;
    CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&can_map, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, device));
    require(can_map != 0, "mapped host memory requires advertised support");
    const auto kernel = program.function("mapped_host_transform");
    constexpr size_t source_offset = 17, destination_offset = 23;
    constexpr uint32_t guard = 0xd3e7a591;
    for (unsigned kind = 0; kind < 4; ++kind) for (bool portable : {false, true}) {
        mapped_host_buffer source(kind, portable), middle(kind, portable), destination(kind, portable);
        // Stream destruction drains both dependency chains before buffers retire.
        mapped_stream first, second;
        mapped_event event;
        source.query_alias(); middle.query_alias(); destination.query_alias();
        for (unsigned threads : {1u, 13u, 33u, 64u})
        for (unsigned count : {0u, 1u, threads * 3 / 2 + 1, threads * 3, threads * 3 + 7}) {
            std::vector<uint32_t> input(mapped_host_buffer::words, guard), expected(input.size(), guard), final = expected;
            const uint32_t seed = 0xffffffe1u + count;
            for (unsigned i = 0; i < threads * 3; ++i)
                input[source_offset + i] = (0x9e3779b9u * (i + count + 1)) ^ (threads << (i % 7));
            source.initialize(input); middle.initialize(expected); destination.initialize(final);
            mapped_expected(expected, destination_offset, input, source_offset, count, threads, seed);
            mapped_launch(kernel, middle.alias + destination_offset * 4, source.alias + source_offset * 4,
                count, seed, threads, first.value);
            CONFORMANCE_CUDA(cuEventRecord(event.value, first.value));
            CONFORMANCE_CUDA(cuStreamWaitEvent(second.value, event.value, 0));
            mapped_expected(final, source_offset, expected, destination_offset, count, threads, seed ^ 0x5a5a5a5a);
            mapped_launch(kernel, destination.alias + source_offset * 4, middle.alias + destination_offset * 4,
                count, seed ^ 0x5a5a5a5a, threads, second.value);
            MAPPED_RUNTIME(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(second.value)));
            // No readback API: the CPU observes completed stores at its host address.
            source.expect(input, "mapped input preservation");
            middle.expect(expected, "mapped publication after producer completion");
            destination.expect(final, "mapped event-dependent consumer and guards");
            const auto before_inplace = final;
            mapped_expected(final, source_offset, before_inplace, source_offset, count, threads, seed + 3);
            mapped_launch(kernel, destination.alias + source_offset * 4, destination.alias + source_offset * 4,
                count, seed + 3, threads, second.value);
            CONFORMANCE_CUDA(cuStreamSynchronize(second.value));
            destination.expect(final, "mapped in-place word access and guards");
            source.expect(input, "mapped in-place pass preserves independent input");
            middle.expect(expected, "mapped in-place pass preserves independent producer output");
        }
        source.release(); middle.release(); destination.release();
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_mapped_host_memory, 1, CU_CTX_MAP_HOST);
}
