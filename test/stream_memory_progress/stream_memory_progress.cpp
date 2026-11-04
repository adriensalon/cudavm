#include "conformance.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

namespace cudavm {
constexpr size_t memory_progress_source_bytes = 293, memory_progress_output_bytes = 336, memory_progress_slot_bytes = 368, memory_progress_host_offset = 13;
struct memory_progress_stream {
    CUstream value{}; memory_progress_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~memory_progress_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    void destroy() { CONFORMANCE_CUDA(cuStreamDestroy(value)); value = nullptr; }
};
struct memory_progress_event {
    CUevent value{}; memory_progress_event() { CONFORMANCE_CUDA(cuEventCreate(&value, CU_EVENT_DISABLE_TIMING)); }
    ~memory_progress_event() { if (value) cuEventDestroy(value); }
    void record(CUstream stream) const { CONFORMANCE_CUDA(cuEventRecord(value, stream)); }
    void complete() const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10); CUresult status;
        do { status = cuEventQuery(value); if (status != CUDA_ERROR_NOT_READY) break; std::this_thread::yield(); }
        while (std::chrono::steady_clock::now() < deadline);
        CONFORMANCE_CUDA(status);
    }
};
struct memory_progress_consumer {
    device_buffer output{memory_progress_output_bytes}; memory_progress_stream stream; memory_progress_event done;
};
struct memory_progress_host {
    void* value{}; size_t count;
    explicit memory_progress_host(size_t slots) : count(slots) { CONFORMANCE_CUDA(cuMemAllocHost(&value, slots * memory_progress_slot_bytes)); reset(); }
    ~memory_progress_host() { if (value) { cuCtxSynchronize(); cuMemFreeHost(value); } }
    void reset() { std::memset(value, 0xd3, count * memory_progress_slot_bytes); }
    void read(const memory_progress_consumer& consumer, unsigned index) const {
        CONFORMANCE_CUDA(cuMemcpyDtoHAsync(static_cast<unsigned char*>(value) + index * memory_progress_slot_bytes + memory_progress_host_offset,
            consumer.output.value, memory_progress_output_bytes, consumer.stream.value));
    }
    void expect(unsigned index, unsigned generation) const {
        std::vector<unsigned char> expected(memory_progress_slot_bytes, 0xd3);
        std::fill_n(expected.begin() + memory_progress_host_offset, memory_progress_output_bytes, 0xa5);
        const uint64_t marker = 0xdecafbad00000000ull + generation * 256 + index;
        std::memcpy(expected.data() + memory_progress_host_offset + 8, &marker, sizeof(marker));
        std::fill_n(expected.begin() + memory_progress_host_offset + 16, 97, static_cast<unsigned char>(0x40 + generation));
        require(std::memcmp(static_cast<unsigned char*>(value) + index * memory_progress_slot_bytes, expected.data(), expected.size()) == 0,
            "stream memory progress payload, marker, untouched device bytes and host guards");
    }
};
struct memory_progress_gate {
    std::mutex mutex; std::condition_variable changed; bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* argument) {
        auto& gate = *static_cast<memory_progress_gate*>(argument); std::unique_lock lock(gate.mutex); gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(10), [&] { return gate.released; })) gate.expired = true;
    }
    void await() { std::unique_lock lock(mutex); require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "memory progress producer gate entry"); }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
    void check() { std::lock_guard lock(mutex); require(entered && !expired, "memory progress producer gate did not expire"); }
};
struct memory_progress_cleanup { memory_progress_gate& gate; ~memory_progress_cleanup() { gate.release(); cuCtxSynchronize(); } };
CUresult memory_progress_publish(CUstream stream, CUdeviceptr address, uint64_t value, bool wide) {
    return wide ? cuStreamWriteValue64(stream, address, value, 0) : cuStreamWriteValue32(stream, address, static_cast<uint32_t>(value), 0);
}
void memory_progress_wait(CUstream stream, CUdeviceptr address, uint64_t value, bool wide, bool batch) {
    if (batch) {
        CUstreamBatchMemOpParams op{}; op.operation = wide ? CU_STREAM_MEM_OP_WAIT_VALUE_64 : CU_STREAM_MEM_OP_WAIT_VALUE_32;
        op.waitValue.address = address; op.waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
        if (wide) op.waitValue.value64 = value; else op.waitValue.value = static_cast<uint32_t>(value);
        CONFORMANCE_CUDA(cuStreamBatchMemOp(stream, 1, &op, 0)); op = {};
    } else if (wide) CONFORMANCE_CUDA(cuStreamWaitValue64(stream, address, value, CU_STREAM_WAIT_VALUE_EQ));
    else CONFORMANCE_CUDA(cuStreamWaitValue32(stream, address, static_cast<uint32_t>(value), CU_STREAM_WAIT_VALUE_EQ));
}
void memory_progress_round(unsigned count, bool chain, bool retirement, bool wide, bool batch, unsigned generation) {
    device_buffer source(memory_progress_source_bytes); memory_progress_stream producer; memory_progress_event produced;
    memory_progress_host host(count); std::vector<std::unique_ptr<memory_progress_consumer>> consumers;
    std::vector<unsigned char> initial(memory_progress_source_bytes, 0x91); std::fill_n(initial.begin() + 8, count * 8, 0);
    CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, initial.data(), initial.size()));
    for (unsigned i = 0; i < count; ++i) {
        auto consumer = std::make_unique<memory_progress_consumer>(); CONFORMANCE_CUDA(cuMemsetD8(consumer->output.value, 0xa5, memory_progress_output_bytes));
        consumer->done.record(consumer->stream.value); consumer->done.complete(); consumers.push_back(std::move(consumer));
    }
    produced.record(producer.value); produced.complete();
    memory_progress_gate gate; memory_progress_cleanup cleanup{gate};
    CONFORMANCE_CUDA(cuLaunchHostFunc(producer.value, memory_progress_gate::call, &gate)); gate.await();
    const uint64_t release = (wide ? 0x10000000000ull : 0) + 17 + generation;
    CONFORMANCE_CUDA(cuMemsetD8Async(source.value + 128, static_cast<unsigned char>(0x40 + generation), 97, producer.value));
    CONFORMANCE_CUDA(memory_progress_publish(producer.value, source.value + 8, release, wide)); produced.record(producer.value);
    for (unsigned i = 0; i < count; ++i) {
        auto& consumer = *consumers[i]; const auto flag = source.value + 8 + (chain ? i * 8 : 0);
        memory_progress_wait(consumer.stream.value, flag, release, wide, batch);
        CONFORMANCE_CUDA(cuStreamWaitEvent(consumer.stream.value, chain && i ? consumers[i - 1]->done.value : produced.value, 0));
        CONFORMANCE_CUDA(cuMemcpyDtoDAsync(consumer.output.value + 16, source.value + 128, 97, consumer.stream.value));
        const uint64_t marker = 0xdecafbad00000000ull + generation * 256 + i;
        CONFORMANCE_CUDA(cuStreamWriteValue64(consumer.stream.value, consumer.output.value + 8, marker, 0));
        if (chain && i + 1 < count) CONFORMANCE_CUDA(memory_progress_publish(consumer.stream.value, source.value + 8 + (i + 1) * 8, release, wide));
        host.read(consumer, i); consumer.done.record(consumer.stream.value);
    }
    require(cuEventQuery(produced.value) == CUDA_ERROR_NOT_READY, "producer publication is held by its own gate");
    for (auto& consumer : consumers) require(cuEventQuery(consumer->done.value) == CUDA_ERROR_NOT_READY, "memory consumers remain pending before publication");
    if (retirement) { for (auto& consumer : consumers) consumer->stream.destroy(); producer.destroy(); }
    gate.release(); produced.complete(); for (auto& consumer : consumers) consumer->done.complete(); gate.check();
    for (unsigned i = 0; i < count; ++i) host.expect(i, generation);
    auto expected = initial; std::fill_n(expected.begin() + 128, 97, static_cast<unsigned char>(0x40 + generation));
    const uint32_t narrow = static_cast<uint32_t>(release);
    for (unsigned i = 0; i < (chain ? count : 1u); ++i)
        std::memcpy(expected.data() + 8 + i * 8, wide ? static_cast<const void*>(&release) : static_cast<const void*>(&narrow), wide ? 8 : 4);
    std::vector<unsigned char> actual(memory_progress_source_bytes); CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), source.value, actual.size()));
    require(actual == expected, "progress producers preserve release widths, payload and every untouched source byte");
}
std::string memory_progress_case;
void execute_memory_progress() {
    require(memory_progress_case == "independent" || memory_progress_case == "fanout" || memory_progress_case == "chain" || memory_progress_case == "retirement", "unknown stream memory progress case");
    const unsigned count = memory_progress_case == "independent" ? 1 : memory_progress_case == "chain" ? 4 : 8;
    for (bool wide : {false, true}) for (bool batch : {false, true}) for (unsigned generation = 0; generation < 3; ++generation) {
        std::cout << "stream_memory_progress=" << memory_progress_case << " wide=" << wide << " batch=" << batch << " generation=" << generation << std::endl;
        memory_progress_round(count, memory_progress_case == "chain", memory_progress_case == "retirement", wide, batch, generation);
    }
}
}
int main(int argc, char** argv) {
    if (argc != 3) return 1; cudavm::memory_progress_case = argv[2]; return cudavm::run_conformance(2, argv, cudavm::execute_memory_progress);
}
