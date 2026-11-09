#include "test_framework.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

namespace cudavm {
constexpr size_t memory_wait_bytes = 96, memory_wait_host_bytes = 137, memory_wait_host_offset = 17;
constexpr uint64_t memory_wait_marker = 0x8123456789abcdefull;
struct memory_wait_stream {
    CUstream value{}; memory_wait_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~memory_wait_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    void complete() const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10); CUresult status;
        do { status = cuStreamQuery(value); if (status != CUDA_ERROR_NOT_READY) break; std::this_thread::yield(); }
        while (std::chrono::steady_clock::now() < deadline);
        CONFORMANCE_CUDA(status);
    }
};
struct memory_wait_storage {
    device_buffer device{memory_wait_bytes}; void* host{}; std::vector<unsigned char> initial;
    memory_wait_storage() : initial(memory_wait_bytes) {
        for (size_t i = 0; i < initial.size(); ++i) initial[i] = static_cast<unsigned char>(i * 37 + 13);
        CONFORMANCE_CUDA(cuMemAllocHost(&host, memory_wait_host_bytes));
    }
    ~memory_wait_storage() { if (host) { cuCtxSynchronize(); cuMemFreeHost(host); } }
    void reset(size_t offset, uint64_t value, bool wide) {
        for (size_t i = 0; i < initial.size(); ++i) initial[i] = static_cast<unsigned char>(i * 37 + 13);
        const auto narrow = static_cast<uint32_t>(value);
        std::memcpy(initial.data() + offset, wide ? static_cast<const void*>(&value) : static_cast<const void*>(&narrow), wide ? 8 : 4);
        CONFORMANCE_CUDA(cuMemcpyHtoD(device.value, initial.data(), initial.size())); std::memset(host, 0xd3, memory_wait_host_bytes);
    }
    CUresult read(CUstream stream) const { return cuMemcpyDtoHAsync(static_cast<unsigned char*>(host) + memory_wait_host_offset, device.value, memory_wait_bytes, stream); }
    void expect(size_t offset, uint64_t value, bool wide, bool marker = true) const {
        std::vector<unsigned char> expected(memory_wait_host_bytes, 0xd3); std::copy(initial.begin(), initial.end(), expected.begin() + memory_wait_host_offset);
        const auto narrow = static_cast<uint32_t>(value);
        std::memcpy(expected.data() + memory_wait_host_offset + offset, wide ? static_cast<const void*>(&value) : static_cast<const void*>(&narrow), wide ? 8 : 4);
        if (marker) std::memcpy(expected.data() + memory_wait_host_offset + 40, &memory_wait_marker, sizeof(memory_wait_marker));
        require(std::memcmp(host, expected.data(), expected.size()) == 0, "wait predicate visibility, unmodified neighbors and complete device/host guards");
    }
};
CUresult memory_wait_submit(CUstream stream, CUdeviceptr address, uint64_t value, bool wide, unsigned predicate) {
    return wide ? cuStreamWaitValue64(stream, address, value, predicate) : cuStreamWaitValue32(stream, address, static_cast<uint32_t>(value), predicate);
}
CUresult memory_wait_publish(CUstream stream, CUdeviceptr address, uint64_t value, bool wide) {
    return wide ? cuStreamWriteValue64(stream, address, value, 0) : cuStreamWriteValue32(stream, address, static_cast<uint32_t>(value), 0);
}
struct memory_wait_case { uint64_t initial, target, release; unsigned predicate; bool blocked; };
void memory_wait_check(memory_wait_storage& storage, const memory_wait_stream& waiter, const memory_wait_stream& producer,
                       const memory_wait_case& input, bool wide, size_t offset = 8) {
    storage.reset(offset, input.initial, wide);
    // Submit the release before asserting any waiter/marker/readback result so
    // a failed assertion cannot leave a valid wait permanently blocked.
    const auto wait_status = memory_wait_submit(waiter.value, storage.device.value + offset, input.target, wide, input.predicate);
    const auto marker_status = cuStreamWriteValue64(waiter.value, storage.device.value + 40, memory_wait_marker, 0);
    const auto read_status = storage.read(waiter.value);
    auto pending = CUDA_SUCCESS;
    if (input.blocked) {
        const auto observation_end = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
        do { pending = cuStreamQuery(waiter.value); if (pending != CUDA_ERROR_NOT_READY) break; std::this_thread::yield(); }
        while (std::chrono::steady_clock::now() < observation_end);
    }
    const auto release_status = input.blocked ? memory_wait_publish(producer.value, storage.device.value + offset, input.release, wide) : CUDA_SUCCESS;
    CONFORMANCE_CUDA(release_status); producer.complete(); waiter.complete();
    CONFORMANCE_CUDA(wait_status); CONFORMANCE_CUDA(marker_status); CONFORMANCE_CUDA(read_status);
    if (input.blocked) require(pending == CUDA_ERROR_NOT_READY, "unsatisfied wait holds downstream work until the producer publishes");
    storage.expect(offset, input.blocked ? input.release : input.initial, wide);
}
void memory_wait_predicates(bool wide) {
    memory_wait_storage storage; memory_wait_stream waiter, producer;
    const uint64_t all = wide ? UINT64_MAX : uint64_t(UINT32_MAX), top = uint64_t(1) << (wide ? 63 : 31);
    const std::array<memory_wait_case, 18> cases{{
        {0, 0, 0, CU_STREAM_WAIT_VALUE_EQ, false}, {all, all, all, CU_STREAM_WAIT_VALUE_EQ, false},
        {1, 2, 2, CU_STREAM_WAIT_VALUE_EQ, true}, {top, top + 1, top + 1, CU_STREAM_WAIT_VALUE_EQ, true},
        {0, all, 0, CU_STREAM_WAIT_VALUE_GEQ, false}, {all, 0, 0, CU_STREAM_WAIT_VALUE_GEQ, true},
        {top - 1, 0, 0, CU_STREAM_WAIT_VALUE_GEQ, false}, {top, 0, top - 1, CU_STREAM_WAIT_VALUE_GEQ, true},
        {0, top, top, CU_STREAM_WAIT_VALUE_GEQ, true}, {all, all - 1, 0, CU_STREAM_WAIT_VALUE_GEQ, false},
        {0, top, top, CU_STREAM_WAIT_VALUE_AND, true}, {top, top, 0, CU_STREAM_WAIT_VALUE_AND, false},
        {4, 2, 6, CU_STREAM_WAIT_VALUE_AND, true}, {all, 1, 0, CU_STREAM_WAIT_VALUE_AND, false},
        {all, 0, all ^ top, CU_STREAM_WAIT_VALUE_NOR, true}, {top, all ^ top, 0, CU_STREAM_WAIT_VALUE_NOR, true},
        {0, all ^ top, 0, CU_STREAM_WAIT_VALUE_NOR, false}, {all ^ uint64_t(1), 0, 0, CU_STREAM_WAIT_VALUE_NOR, false}}};
    for (size_t i = 0; i < cases.size(); ++i) { std::cout << "wait width=" << (wide ? 64 : 32) << " row=" << i << std::endl; memory_wait_check(storage, waiter, producer, cases[i], wide); }
    if (wide) {
        memory_wait_check(storage, waiter, producer, {1, 0x100000001ull, 0x100000001ull, CU_STREAM_WAIT_VALUE_EQ, true}, true);
        memory_wait_check(storage, waiter, producer, {0x100000000ull, 1, 0x100000001ull, CU_STREAM_WAIT_VALUE_AND, true}, true);
        memory_wait_check(storage, waiter, producer, {0xffffffffull, 0xffffffff00000000ull, 0, CU_STREAM_WAIT_VALUE_NOR, true}, true);
    }
}
void memory_wait_addresses() {
    memory_wait_storage storage; memory_wait_stream waiter, producer;
    for (bool wide : {false, true}) for (size_t offset : {size_t(0), size_t(16), memory_wait_bytes - (wide ? 8 : 4)}) {
        const uint64_t value = wide ? 0x123456789abcdef0ull : 0x9abcdef0u;
        memory_wait_check(storage, waiter, producer, {value, value, value, CU_STREAM_WAIT_VALUE_EQ, false}, wide, offset);
        memory_wait_check(storage, waiter, producer, {0, value, value, CU_STREAM_WAIT_VALUE_EQ, true}, wide, offset);
    }
}
void memory_wait_validation() {
    memory_wait_storage storage; memory_wait_stream waiter, producer;
    for (bool wide : {false, true}) {
        storage.reset(8, 0, wide);
        for (CUdeviceptr address : {CUdeviceptr(0), storage.device.value + 1, storage.device.value + memory_wait_bytes,
             storage.device.value + memory_wait_bytes - (wide ? 4 : 2)})
            require(memory_wait_submit(waiter.value, address, 0, wide, CU_STREAM_WAIT_VALUE_EQ) == CUDA_ERROR_INVALID_VALUE, "invalid wait addresses reject before work publication");
        require(memory_wait_submit(waiter.value, storage.device.value + 8, 0, wide, 4) == CUDA_ERROR_INVALID_VALUE, "invalid wait predicate flags reject");
        CONFORMANCE_CUDA(storage.read(waiter.value)); waiter.complete(); storage.expect(8, 0, wide, false);
        memory_wait_check(storage, waiter, producer, {0, 17, 17, CU_STREAM_WAIT_VALUE_EQ, true}, wide);
    }
}
std::string memory_wait_selected_case;
void execute_memory_wait() {
    if (memory_wait_selected_case == "predicates32") memory_wait_predicates(false);
    else if (memory_wait_selected_case == "predicates64") memory_wait_predicates(true);
    else if (memory_wait_selected_case == "addresses") memory_wait_addresses();
    else if (memory_wait_selected_case == "validation") memory_wait_validation();
    else require(false, "unknown stream memory wait case");
}
}
int main(int argc, char** argv) {
    if (argc != 3) return 1; cudavm::memory_wait_selected_case = argv[2]; return cudavm::run_conformance(2, argv, cudavm::execute_memory_wait);
}
