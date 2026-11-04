#include "conformance.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace cudavm {
constexpr size_t memory_batch_bytes = 128, memory_batch_slot_bytes = 160, memory_batch_offset = 13, memory_batch_slots = 3;
struct memory_batch_stream {
    CUstream value{}; memory_batch_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~memory_batch_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    void synchronize() const { CONFORMANCE_CUDA(cuStreamSynchronize(value)); }
};
struct memory_batch_storage {
    device_buffer device{memory_batch_bytes}; void* host{}; std::vector<unsigned char> initial;
    memory_batch_storage() : initial(memory_batch_bytes) {
        for (size_t i = 0; i < initial.size(); ++i) initial[i] = static_cast<unsigned char>(i * 37 + 13);
        CONFORMANCE_CUDA(cuMemAllocHost(&host, memory_batch_slots * memory_batch_slot_bytes));
        try { reset(); } catch (...) { cuMemFreeHost(host); host = nullptr; throw; }
    }
    ~memory_batch_storage() { if (host) { cuCtxSynchronize(); cuMemFreeHost(host); } }
    void reset() { CONFORMANCE_CUDA(cuMemcpyHtoD(device.value, initial.data(), initial.size())); std::memset(host, 0xd3, memory_batch_slots * memory_batch_slot_bytes); }
    void read(CUstream stream, unsigned slot) const {
        require(slot < memory_batch_slots, "bounded batch readback slot");
        CONFORMANCE_CUDA(cuMemcpyDtoHAsync(static_cast<unsigned char*>(host) + slot * memory_batch_slot_bytes + memory_batch_offset, device.value, memory_batch_bytes, stream));
    }
    void expect(unsigned slot, std::span<const unsigned char> bytes) const {
        std::vector<unsigned char> expected(memory_batch_slot_bytes, 0xd3); std::copy(bytes.begin(), bytes.end(), expected.begin() + memory_batch_offset);
        require(std::memcmp(static_cast<unsigned char*>(host) + slot * memory_batch_slot_bytes, expected.data(), expected.size()) == 0, "batch operation payload, neighboring bytes and complete host guards");
    }
    void untouched(unsigned first) const {
        const auto* begin = static_cast<const unsigned char*>(host) + first * memory_batch_slot_bytes;
        const auto* end = static_cast<const unsigned char*>(host) + memory_batch_slots * memory_batch_slot_bytes;
        require(std::all_of(begin, end, [](unsigned char byte) { return byte == 0xd3; }), "unused batch readback slots remain untouched");
    }
};
CUstreamBatchMemOpParams memory_batch_write(CUdeviceptr address, uint64_t value, bool wide = false) {
    CUstreamBatchMemOpParams p{}; p.operation = wide ? CU_STREAM_MEM_OP_WRITE_VALUE_64 : CU_STREAM_MEM_OP_WRITE_VALUE_32; p.writeValue.address = address;
    if (wide) p.writeValue.value64 = value; else p.writeValue.value = static_cast<uint32_t>(value); return p;
}
CUstreamBatchMemOpParams memory_batch_wait(CUdeviceptr address, uint64_t value, bool wide = false, unsigned predicate = CU_STREAM_WAIT_VALUE_EQ) {
    CUstreamBatchMemOpParams p{}; p.operation = wide ? CU_STREAM_MEM_OP_WAIT_VALUE_64 : CU_STREAM_MEM_OP_WAIT_VALUE_32; p.waitValue.address = address; p.waitValue.flags = predicate;
    if (wide) p.waitValue.value64 = value; else p.waitValue.value = static_cast<uint32_t>(value); return p;
}
std::vector<CUstreamBatchMemOpParams> memory_batch_operations(const memory_batch_storage& storage, unsigned variant) {
    const auto base = storage.device.value + variant * 32;
    CUstreamBatchMemOpParams system{}, gpu{}; system.operation = gpu.operation = CU_STREAM_MEM_OP_BARRIER; gpu.memoryBarrier.flags = CU_STREAM_MEMORY_BARRIER_TYPE_GPU;
    return {memory_batch_write(base + 8, 0x81234567), memory_batch_wait(base + 8, 0x81234567), system,
        memory_batch_write(base + 16, 0xfedcba9876543210ull + variant, true), memory_batch_wait(base + 16, 0xfedcba9876543210ull + variant, true), gpu,
        memory_batch_write(base + 8, 0x76543210 + variant), memory_batch_wait(base + 8, 0x76543200, false, CU_STREAM_WAIT_VALUE_GEQ),
        memory_batch_write(base + 24, 0x100000000ull, true), memory_batch_wait(base + 24, 0x100000000ull, true, CU_STREAM_WAIT_VALUE_AND)};
}
void memory_batch_expected(std::vector<unsigned char>& bytes, unsigned variant) {
    const uint32_t narrow = 0x76543210 + variant; const uint64_t wide = 0xfedcba9876543210ull + variant, high = 0x100000000ull;
    std::memcpy(bytes.data() + variant * 32 + 8, &narrow, sizeof(narrow)); std::memcpy(bytes.data() + variant * 32 + 16, &wide, sizeof(wide)); std::memcpy(bytes.data() + variant * 32 + 24, &high, sizeof(high));
}
void memory_batch_ordering() {
    memory_batch_storage storage; memory_batch_stream stream;
    for (unsigned variant = 0; variant < 3; ++variant) {
        storage.reset(); auto ops = memory_batch_operations(storage, variant);
        CONFORMANCE_CUDA(cuStreamBatchMemOp(stream.value, static_cast<unsigned>(ops.size()), ops.data(), 0)); std::fill(ops.begin(), ops.end(), CUstreamBatchMemOpParams{});
        storage.read(stream.value, 0); stream.synchronize(); auto expected = storage.initial; memory_batch_expected(expected, variant); storage.expect(0, expected); storage.untouched(1);
    }
}
void memory_batch_ownership() {
    memory_batch_storage storage; memory_batch_stream stream; auto expected = storage.initial;
    std::array<std::vector<unsigned char>, 3> snapshots;
    for (unsigned variant = 0; variant < 3; ++variant) {
        {
            auto ops = memory_batch_operations(storage, variant); CONFORMANCE_CUDA(cuStreamBatchMemOp(stream.value, static_cast<unsigned>(ops.size()), ops.data(), 0));
            std::fill(ops.begin(), ops.end(), CUstreamBatchMemOpParams{});
        }
        storage.read(stream.value, variant); memory_batch_expected(expected, variant); snapshots[variant] = expected;
    }
    stream.synchronize(); for (unsigned i = 0; i < 3; ++i) storage.expect(i, snapshots[i]);
}
void memory_batch_boundaries() {
    memory_batch_storage storage; memory_batch_stream stream;
    for (unsigned count : {0u, 1u, 255u}) {
        storage.reset(); std::vector<CUstreamBatchMemOpParams> ops;
        for (unsigned i = 0; i < count; ++i) ops.push_back(memory_batch_write(storage.device.value + 124, 0x81234500u + i));
        CONFORMANCE_CUDA(cuStreamBatchMemOp(stream.value, count, count ? ops.data() : nullptr, 0)); std::fill(ops.begin(), ops.end(), CUstreamBatchMemOpParams{});
        storage.read(stream.value, 0); stream.synchronize(); auto expected = storage.initial;
        if (count) { const uint32_t last = 0x81234500u + count - 1; std::memcpy(expected.data() + 124, &last, sizeof(last)); }
        storage.expect(0, expected); storage.untouched(1);
    }
}
void memory_batch_validation() {
    memory_batch_storage storage; memory_batch_stream stream;
    for (unsigned invalid = 0; invalid < 13; ++invalid) {
        storage.reset(); std::vector<CUstreamBatchMemOpParams> ops{memory_batch_write(storage.device.value + 8, 0x12345678), memory_batch_write(storage.device.value + 16, 17)};
        unsigned count = 2, flags = 0; bool null = false;
        if (invalid == 0) ops[1].operation = static_cast<CUstreamBatchMemOpType>(99);
        if (invalid == 1) ops[1].writeValue.address = 0;
        if (invalid == 2) ops[1].writeValue.address = storage.device.value + 17;
        if (invalid == 3) ops[1].writeValue.address = storage.device.value + memory_batch_bytes;
        if (invalid == 4) ops[1] = memory_batch_write(storage.device.value + 124, 17, true);
        if (invalid == 5) ops[1].writeValue.flags = CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER;
        if (invalid == 6) ops[1].writeValue.flags = 2;
        if (invalid == 7) { ops[1] = memory_batch_wait(storage.device.value + 16, 17); ops[1].waitValue.flags = 4; }
        if (invalid == 8) { ops[1] = {}; ops[1].operation = CU_STREAM_MEM_OP_BARRIER; ops[1].memoryBarrier.flags = 2; }
        if (invalid == 9) null = true;
        if (invalid == 10) { ops.resize(256); count = 256; }
        if (invalid == 11) flags = 1;
        if (invalid == 12) { count = 0; flags = 1; }
        require(cuStreamBatchMemOp(stream.value, count, null ? nullptr : ops.data(), flags) == CUDA_ERROR_INVALID_VALUE, "invalid batch rejects before publishing its first valid operation");
        storage.read(stream.value, 0); stream.synchronize(); storage.expect(0, storage.initial); storage.untouched(1);
        ops = memory_batch_operations(storage, 1); CONFORMANCE_CUDA(cuStreamBatchMemOp(stream.value, static_cast<unsigned>(ops.size()), ops.data(), 0));
        std::fill(ops.begin(), ops.end(), CUstreamBatchMemOpParams{}); storage.read(stream.value, 1); stream.synchronize(); auto expected = storage.initial; memory_batch_expected(expected, 1); storage.expect(1, expected); storage.untouched(2);
    }
}
std::string memory_batch_case;
void execute_memory_batch() {
    if (memory_batch_case == "ordering") memory_batch_ordering();
    else if (memory_batch_case == "ownership") memory_batch_ownership();
    else if (memory_batch_case == "boundaries") memory_batch_boundaries();
    else if (memory_batch_case == "validation") memory_batch_validation();
    else require(false, "unknown stream memory batch case");
}
}
int main(int argc, char** argv) {
    if (argc != 3) return 1; cudavm::memory_batch_case = argv[2]; return cudavm::run_conformance(2, argv, cudavm::execute_memory_batch);
}
