#include "test_framework.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace cudavm {
constexpr size_t memory_write_bytes = 128, memory_write_slot_bytes = 160, memory_write_slots = 3, memory_write_offset = 13;
struct memory_write_storage {
    device_buffer device{memory_write_bytes}; void* input{}; void* output{}; std::vector<unsigned char> initial;
    memory_write_storage() : initial(memory_write_slot_bytes) {
        CONFORMANCE_CUDA(cuMemAllocHost(&input, memory_write_slot_bytes));
        try { CONFORMANCE_CUDA(cuMemAllocHost(&output, memory_write_slots * memory_write_slot_bytes)); }
        catch (...) { cuMemFreeHost(input); input = nullptr; throw; }
    }
    ~memory_write_storage() { cuCtxSynchronize(); if (output) cuMemFreeHost(output); if (input) cuMemFreeHost(input); }
    void reset(CUstream stream, unsigned seed) {
        for (size_t i = 0; i < initial.size(); ++i) initial[i] = static_cast<unsigned char>(i * 37 + seed);
        std::memcpy(input, initial.data(), initial.size()); std::memset(output, 0xd3, memory_write_slots * memory_write_slot_bytes);
        CONFORMANCE_CUDA(cuMemcpyHtoDAsync(device.value, static_cast<unsigned char*>(input) + memory_write_offset, memory_write_bytes, stream));
    }
    void read(CUstream stream, unsigned slot, CUdeviceptr address = 0) const {
        require(slot < memory_write_slots, "bounded write-value readback slot");
        CONFORMANCE_CUDA(cuMemcpyDtoHAsync(static_cast<unsigned char*>(output) + slot * memory_write_slot_bytes + memory_write_offset, address ? address : device.value, memory_write_bytes, stream));
    }
    std::vector<unsigned char> expected(size_t offset, uint64_t value, bool wide) const {
        std::vector<unsigned char> bytes(initial.begin() + memory_write_offset, initial.begin() + memory_write_offset + memory_write_bytes);
        const uint32_t narrow = static_cast<uint32_t>(value); std::memcpy(bytes.data() + offset, wide ? static_cast<const void*>(&value) : static_cast<const void*>(&narrow), wide ? 8 : 4); return bytes;
    }
    void expect(unsigned slot, std::span<const unsigned char> bytes) const {
        std::vector<unsigned char> expected(memory_write_slot_bytes, 0xd3); std::copy(bytes.begin(), bytes.end(), expected.begin() + memory_write_offset);
        require(std::memcmp(static_cast<unsigned char*>(output) + slot * memory_write_slot_bytes, expected.data(), expected.size()) == 0, "write-value complete payload and pinned-host guards");
        require(std::memcmp(input, initial.data(), initial.size()) == 0, "write-value upload preserves every source byte and guard");
    }
    void untouched(unsigned first) const {
        const auto* begin = static_cast<unsigned char*>(output) + first * memory_write_slot_bytes;
        require(std::all_of(begin, static_cast<const unsigned char*>(output) + memory_write_slots * memory_write_slot_bytes, [](unsigned char byte) { return byte == 0xd3; }), "unused write-value readback slots remain untouched");
    }
};
struct memory_write_stream {
    CUstream value{}; memory_write_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~memory_write_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    void synchronize() const { CONFORMANCE_CUDA(cuStreamSynchronize(value)); }
};
CUresult memory_write_submit(CUstream stream, CUdeviceptr address, uint64_t value, bool wide, unsigned flags = 0) {
    return wide ? cuStreamWriteValue64(stream, address, value, flags) : cuStreamWriteValue32(stream, address, static_cast<uint32_t>(value), flags);
}
void memory_write_values(bool wide) {
    memory_write_storage storage; memory_write_stream stream;
    const uint64_t mask = wide ? UINT64_MAX : uint64_t(UINT32_MAX);
    const std::array<uint64_t, 6> values{0, 1, mask, uint64_t(1) << (wide ? 63 : 31), 0x8123456789abcdefull & mask, 0xfedcba9876543210ull & mask};
    for (size_t offset : {size_t(0), size_t(16), memory_write_bytes - (wide ? 8 : 4)}) for (auto value : values) {
        storage.reset(stream.value, 17); CONFORMANCE_CUDA(memory_write_submit(stream.value, storage.device.value + offset, value, wide)); storage.read(stream.value, 0);
        CONFORMANCE_CUDA(memory_write_submit(stream.value, storage.device.value + offset, value ^ mask, wide)); storage.read(stream.value, 1); stream.synchronize();
        storage.expect(0, storage.expected(offset, value, wide)); storage.expect(1, storage.expected(offset, value ^ mask, wide)); storage.untouched(2);
    }
}
void memory_write_overlap() {
    memory_write_storage storage; memory_write_stream stream; storage.reset(stream.value, 31);
    CONFORMANCE_CUDA(cuStreamWriteValue64(stream.value, storage.device.value + 8, 0x0123456789abcdefull, 0)); storage.read(stream.value, 0);
    CONFORMANCE_CUDA(cuStreamWriteValue32(stream.value, storage.device.value + 12, 0xfedcba98u, 0)); storage.read(stream.value, 1);
    CONFORMANCE_CUDA(cuStreamWriteValue32(stream.value, storage.device.value + 8, 0x76543210u, 0)); storage.read(stream.value, 2); stream.synchronize();
    storage.expect(0, storage.expected(8, 0x0123456789abcdefull, true)); storage.expect(1, storage.expected(8, 0xfedcba9889abcdefull, true)); storage.expect(2, storage.expected(8, 0xfedcba9876543210ull, true));
}
void memory_write_ordering() {
    memory_write_storage storage; device_buffer copy(memory_write_bytes); memory_write_stream stream;
    for (bool wide : {false, true}) {
        storage.reset(stream.value, 53); const uint64_t mask = wide ? UINT64_MAX : uint64_t(UINT32_MAX), first = 0x8123456789abcdefull & mask, second = 0xfedcba9876543210ull & mask;
        CONFORMANCE_CUDA(memory_write_submit(stream.value, storage.device.value + 8, first, wide));
        CONFORMANCE_CUDA(cuMemcpyDtoDAsync(copy.value, storage.device.value, memory_write_bytes, stream.value));
        CONFORMANCE_CUDA(memory_write_submit(stream.value, storage.device.value + 8, second, wide));
        storage.read(stream.value, 0, copy.value); storage.read(stream.value, 1); stream.synchronize();
        storage.expect(0, storage.expected(8, first, wide)); storage.expect(1, storage.expected(8, second, wide)); storage.untouched(2);
    }
}
void memory_write_validation() {
    memory_write_storage storage; memory_write_stream stream;
    for (bool wide : {false, true}) {
        storage.reset(stream.value, 71);
        for (CUdeviceptr address : {CUdeviceptr(0), storage.device.value + 1, storage.device.value + memory_write_bytes,
             storage.device.value + memory_write_bytes - (wide ? 4 : 2)})
            require(memory_write_submit(stream.value, address, 17, wide) == CUDA_ERROR_INVALID_VALUE, "invalid write-value addresses reject");
        for (unsigned flags : {unsigned(CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER), 2u, UINT32_MAX})
            require(memory_write_submit(stream.value, storage.device.value + 8, 17, wide, flags) == CUDA_ERROR_INVALID_VALUE, "current write-value API rejects nonzero flags before publication");
        storage.read(stream.value, 0); CONFORMANCE_CUDA(memory_write_submit(stream.value, storage.device.value + 8, 0x8123456789abcdefull, wide)); storage.read(stream.value, 1); stream.synchronize();
        const std::vector<unsigned char> initial(storage.initial.begin() + memory_write_offset, storage.initial.begin() + memory_write_offset + memory_write_bytes);
        storage.expect(0, initial); storage.expect(1, storage.expected(8, 0x8123456789abcdefull, wide)); storage.untouched(2);
    }
}
std::string memory_write_case;
void execute_memory_write() {
    if (memory_write_case == "values32") memory_write_values(false);
    else if (memory_write_case == "values64") memory_write_values(true);
    else if (memory_write_case == "overlap") memory_write_overlap();
    else if (memory_write_case == "ordering") memory_write_ordering();
    else if (memory_write_case == "validation") memory_write_validation();
    else require(false, "unknown stream memory write case");
}
}
int main(int argc, char** argv) {
    if (argc != 3) return 1; cudavm::memory_write_case = argv[2]; return cudavm::run_conformance(2, argv, cudavm::execute_memory_write);
}
