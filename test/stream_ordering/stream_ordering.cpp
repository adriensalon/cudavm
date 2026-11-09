#include "test_fixtures.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <iostream>

namespace cudavm {
void check_stream_ordering_runtime(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
#define STREAM_ORDERING_RUNTIME(call) ::cudavm::check_stream_ordering_runtime((call), #call)
constexpr size_t stream_ordering_guard = 8, stream_ordering_words = 224, stream_ordering_slot_words = 240, stream_ordering_slots = 6;
void* stream_ordering_pointer(CUdeviceptr pointer) { return reinterpret_cast<void*>(static_cast<uintptr_t>(pointer)); }
struct stream_ordering_stream {
    CUstream value{};
    explicit stream_ordering_stream(bool runtime) {
        if (runtime) { cudaStream_t result{}; STREAM_ORDERING_RUNTIME(cudaStreamCreateWithFlags(&result, cudaStreamNonBlocking)); value = reinterpret_cast<CUstream>(result); }
        else CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING));
    }
    ~stream_ordering_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    void complete(bool runtime) const {
        if (runtime) STREAM_ORDERING_RUNTIME(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(value))); else CONFORMANCE_CUDA(cuStreamSynchronize(value));
        CONFORMANCE_CUDA(cuStreamQuery(value)); STREAM_ORDERING_RUNTIME(cudaStreamQuery(reinterpret_cast<cudaStream_t>(value)));
    }
};
struct stream_ordering_storage {
    device_buffer original{stream_ordering_words * sizeof(uint32_t)}, copy{stream_ordering_words * sizeof(uint32_t)};
    void* input{}; void* output{}; std::vector<uint32_t> source;
    stream_ordering_storage() : source(stream_ordering_slot_words) {
        CONFORMANCE_CUDA(cuMemAllocHost(&input, source.size() * sizeof(uint32_t)));
        try { CONFORMANCE_CUDA(cuMemAllocHost(&output, stream_ordering_slots * stream_ordering_slot_words * sizeof(uint32_t))); }
        catch (...) { cuMemFreeHost(input); input = nullptr; throw; }
    }
    ~stream_ordering_storage() { cuCtxSynchronize(); if (output) cuMemFreeHost(output); if (input) cuMemFreeHost(input); }
    std::vector<uint32_t> initialize(CUstream stream, unsigned seed, bool runtime) {
        for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<uint32_t>(0xfffffff0ull + uint64_t(i) * 0x1020305 + seed);
        std::memcpy(input, source.data(), source.size() * sizeof(uint32_t)); std::memset(output, 0xd3, stream_ordering_slots * stream_ordering_slot_words * sizeof(uint32_t));
        const auto* payload = static_cast<const uint32_t*>(input) + stream_ordering_guard;
        if (runtime) STREAM_ORDERING_RUNTIME(cudaMemcpyAsync(stream_ordering_pointer(original.value), payload, stream_ordering_words * sizeof(uint32_t), cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)));
        else CONFORMANCE_CUDA(cuMemcpyHtoDAsync(original.value, payload, stream_ordering_words * sizeof(uint32_t), stream));
        return {source.begin() + stream_ordering_guard, source.begin() + stream_ordering_guard + stream_ordering_words};
    }
    void read(CUstream stream, unsigned slot, CUdeviceptr pointer, bool runtime) const {
        require(slot < stream_ordering_slots, "bounded FIFO snapshot slot"); auto* host = static_cast<uint32_t*>(output) + slot * stream_ordering_slot_words + stream_ordering_guard;
        if (runtime) STREAM_ORDERING_RUNTIME(cudaMemcpyAsync(host, stream_ordering_pointer(pointer), stream_ordering_words * sizeof(uint32_t), cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream)));
        else CONFORMANCE_CUDA(cuMemcpyDtoHAsync(host, pointer, stream_ordering_words * sizeof(uint32_t), stream));
    }
    void expect(unsigned slot, std::span<const uint32_t> state) const {
        std::vector<uint32_t> expected(stream_ordering_slot_words, 0xd3d3d3d3); std::copy(state.begin(), state.end(), expected.begin() + stream_ordering_guard);
        expect_words({static_cast<uint32_t*>(output) + slot * stream_ordering_slot_words, stream_ordering_slot_words}, expected, "stream FIFO snapshot, inactive lanes and complete guards");
        require(std::memcmp(input, source.data(), source.size() * sizeof(uint32_t)) == 0, "FIFO upload preserves all source bytes and guards");
    }
};
void stream_ordering_apply(std::vector<uint32_t>& expected, unsigned count, uint32_t add) {
    for (uint64_t i = 0; i < std::min(count, 192u); ++i)
        expected[stream_ordering_guard + i] = static_cast<uint32_t>(uint64_t(expected[stream_ordering_guard + i]) * 3 + add + i * 7);
}
void stream_ordering_launch(const loaded_program& program, CUstream stream, CUdeviceptr pointer, unsigned count, unsigned add, bool runtime) {
    pointer += stream_ordering_guard * sizeof(uint32_t); void* arguments[]{&pointer, &count, &add};
    if (runtime) STREAM_ORDERING_RUNTIME(cudaLaunchKernel(reinterpret_cast<const void*>(program.library_kernel("replay_step")), dim3(3), dim3(64), arguments, 0, reinterpret_cast<cudaStream_t>(stream)));
    else CONFORMANCE_CUDA(cuLaunchKernel(program.function("replay_step"), 3, 1, 1, 64, 1, 1, 0, stream, arguments, nullptr));
    pointer = 0; count = add = 0; std::fill(std::begin(arguments), std::end(arguments), nullptr);
}
void stream_ordering_fill(CUstream stream, CUdeviceptr address, unsigned byte, size_t bytes, bool runtime) {
    if (runtime) STREAM_ORDERING_RUNTIME(cudaMemsetAsync(stream_ordering_pointer(address), byte, bytes, reinterpret_cast<cudaStream_t>(stream)));
    else CONFORMANCE_CUDA(cuMemsetD8Async(address, static_cast<unsigned char>(byte), bytes, stream));
}
void execute_stream_ordering(const loaded_program& program) {
    stream_ordering_storage storage;
    for (bool runtime_kernel : {false, true}) {
        if (runtime_kernel && !program.library_kernel("replay_step")) continue;
        for (bool runtime_transfers : {false, true}) {
            std::cout << "stream_ordering runtime_kernel=" << runtime_kernel << " runtime_transfers=" << runtime_transfers << std::endl;
            stream_ordering_stream stream(runtime_transfers);
            for (unsigned count : {0u, 1u, 97u, 192u, 211u}) {
                auto expected = storage.initialize(stream.value, count, runtime_transfers); std::array<std::vector<uint32_t>, stream_ordering_slots> snapshots;
                stream_ordering_launch(program, stream.value, storage.original.value, count, 17, runtime_kernel);
                storage.read(stream.value, 0, storage.original.value, runtime_transfers); stream_ordering_apply(expected, count, 17); snapshots[0] = expected;
                stream_ordering_fill(stream.value, storage.original.value + (stream_ordering_guard + 29) * sizeof(uint32_t), 0x47, 37 * sizeof(uint32_t), runtime_transfers);
                stream_ordering_launch(program, stream.value, storage.original.value, 97, 0xfffffff1, runtime_kernel);
                storage.read(stream.value, 1, storage.original.value, runtime_transfers);
                std::fill_n(expected.begin() + stream_ordering_guard + 29, 37, 0x47474747); stream_ordering_apply(expected, 97, 0xfffffff1); snapshots[1] = expected;
                if (runtime_transfers) STREAM_ORDERING_RUNTIME(cudaMemcpyAsync(stream_ordering_pointer(storage.copy.value), stream_ordering_pointer(storage.original.value), stream_ordering_words * sizeof(uint32_t), cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream.value)));
                else CONFORMANCE_CUDA(cuMemcpyDtoDAsync(storage.copy.value, storage.original.value, stream_ordering_words * sizeof(uint32_t), stream.value));
                stream_ordering_launch(program, stream.value, storage.original.value, 192, 0x12345678, runtime_kernel);
                storage.read(stream.value, 2, storage.copy.value, runtime_transfers); snapshots[2] = expected;
                storage.read(stream.value, 3, storage.original.value, runtime_transfers); stream_ordering_apply(expected, 192, 0x12345678); snapshots[3] = expected;
                stream_ordering_fill(stream.value, storage.original.value, 0x71, stream_ordering_words * sizeof(uint32_t), runtime_transfers);
                stream_ordering_launch(program, stream.value, storage.original.value, 0, 0xffffffff, runtime_kernel);
                storage.read(stream.value, 4, storage.original.value, runtime_transfers); snapshots[4].assign(stream_ordering_words, 0x71717171);
                storage.read(stream.value, 5, storage.copy.value, runtime_transfers); snapshots[5] = snapshots[2]; stream.complete(!runtime_transfers);
                for (unsigned slot = 0; slot < stream_ordering_slots; ++slot) storage.expect(slot, snapshots[slot]);
            }
        }
    }
}
}
int main(int argc, char** argv) { return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_stream_ordering); }
