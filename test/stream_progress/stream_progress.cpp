#include "fixtures.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

namespace cudavm {
void check_stream_progress_runtime(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
#define STREAM_PROGRESS_RUNTIME(call) ::cudavm::check_stream_progress_runtime((call), #call)
constexpr size_t stream_progress_guard = 8, stream_progress_words = 224, stream_progress_slot_words = 240;
void* stream_progress_pointer(CUdeviceptr pointer) { return reinterpret_cast<void*>(static_cast<uintptr_t>(pointer)); }
struct stream_progress_stream {
    CUstream value{};
    stream_progress_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~stream_progress_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    void destroy() { CONFORMANCE_CUDA(cuStreamDestroy(value)); value = nullptr; }
};
struct stream_progress_event {
    CUevent value{}; stream_progress_event() { CONFORMANCE_CUDA(cuEventCreate(&value, CU_EVENT_DISABLE_TIMING)); }
    ~stream_progress_event() { if (value) cuEventDestroy(value); }
    void record(CUstream stream) const { CONFORMANCE_CUDA(cuEventRecord(value, stream)); }
    void complete() const {
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5); CUresult status;
        do { status = cuEventQuery(value); if (status != CUDA_ERROR_NOT_READY) break; std::this_thread::yield(); } while (std::chrono::steady_clock::now() < end);
        CONFORMANCE_CUDA(status);
    }
};
struct stream_progress_output {
    device_buffer device{stream_progress_words * sizeof(uint32_t)}; stream_progress_stream stream; stream_progress_event done;
    void* host{}; std::vector<uint32_t> initial;
    explicit stream_progress_output(unsigned seed) : initial(stream_progress_words) {
        for (size_t i = 0; i < initial.size(); ++i) initial[i] = static_cast<uint32_t>(0xfffffff0ull + uint64_t(i) * 0x1020305 + seed);
        CONFORMANCE_CUDA(cuMemAllocHost(&host, stream_progress_slot_words * sizeof(uint32_t)));
        try { CONFORMANCE_CUDA(cuMemcpyHtoD(device.value, initial.data(), initial.size() * sizeof(uint32_t))); }
        catch (...) { cuMemFreeHost(host); host = nullptr; throw; }
        std::memset(host, 0xd3, stream_progress_slot_words * sizeof(uint32_t));
    }
    ~stream_progress_output() { if (host) { cuCtxSynchronize(); cuMemFreeHost(host); } }
    void read(bool runtime) const {
        auto* output = static_cast<uint32_t*>(host) + stream_progress_guard;
        if (runtime) STREAM_PROGRESS_RUNTIME(cudaMemcpyAsync(output, stream_progress_pointer(device.value), stream_progress_words * sizeof(uint32_t), cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream.value)));
        else CONFORMANCE_CUDA(cuMemcpyDtoHAsync(output, device.value, stream_progress_words * sizeof(uint32_t), stream.value));
    }
    void expect(unsigned count, uint32_t add, bool produced) const {
        auto state = initial; if (produced) std::fill_n(state.begin() + stream_progress_guard, 192, 0x47474747);
        for (uint64_t i = 0; i < std::min(count, 192u); ++i) state[i + stream_progress_guard] = static_cast<uint32_t>(uint64_t(state[i + stream_progress_guard]) * 3 + add + i * 7);
        std::vector<uint32_t> expected(stream_progress_slot_words, 0xd3d3d3d3); std::copy(state.begin(), state.end(), expected.begin() + stream_progress_guard);
        expect_words({static_cast<uint32_t*>(host), stream_progress_slot_words}, expected, "independent/held stream kernel, inactive words and complete guards");
    }
};
struct stream_progress_kernel {
    CUfunction function; CUkernel kernel; bool runtime;
    stream_progress_kernel(const loaded_program& program, bool use_runtime) : function(program.function("replay_step")), kernel(program.library_kernel("replay_step")), runtime(use_runtime && kernel) {}
    void launch(const stream_progress_output& output, unsigned count, unsigned add) const {
        auto pointer = output.device.value + stream_progress_guard * sizeof(uint32_t); void* args[]{&pointer, &count, &add};
        if (runtime) STREAM_PROGRESS_RUNTIME(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(3), dim3(64), args, 0, reinterpret_cast<cudaStream_t>(output.stream.value)));
        else CONFORMANCE_CUDA(cuLaunchKernel(function, 3, 1, 1, 64, 1, 1, 0, output.stream.value, args, nullptr));
        pointer = 0; count = add = 0; std::fill(std::begin(args), std::end(args), nullptr);
    }
};
struct stream_progress_gate {
    std::mutex mutex; std::condition_variable changed; bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* argument) {
        auto& gate = *static_cast<stream_progress_gate*>(argument); std::unique_lock lock(gate.mutex); gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(10), [&] { return gate.released; })) gate.expired = true;
    }
    void await() { std::unique_lock lock(mutex); require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "stream progress gate entry"); }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
    void held() { std::lock_guard lock(mutex); require(entered && !released && !expired, "independent kernel completes while producer stays held"); }
    void check() { std::lock_guard lock(mutex); require(entered && !expired, "stream progress gate did not expire"); }
};
struct stream_progress_cleanup { stream_progress_gate& gate; ~stream_progress_cleanup() { gate.release(); cuCtxSynchronize(); } };
void stream_progress_round(const stream_progress_kernel& kernel, bool runtime, unsigned count, unsigned consumers_count, bool event_dependency, bool retirement) {
    stream_progress_stream producer; stream_progress_event produced; stream_progress_output independent(0x1234 + count);
    std::vector<std::unique_ptr<stream_progress_output>> consumers;
    for (unsigned i = 0; i < consumers_count; ++i) consumers.push_back(std::make_unique<stream_progress_output>(count + i * 1031));
    kernel.launch(independent, 0, 0); independent.done.record(independent.stream.value); independent.done.complete();
    const auto gated = event_dependency ? producer.value : consumers[0]->stream.value;
    stream_progress_gate gate; stream_progress_cleanup cleanup{gate};
    if (runtime) STREAM_PROGRESS_RUNTIME(cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(gated), stream_progress_gate::call, &gate));
    else CONFORMANCE_CUDA(cuLaunchHostFunc(gated, stream_progress_gate::call, &gate)); gate.await();
    for (const auto& consumer : consumers) {
        auto address = consumer->device.value + stream_progress_guard * sizeof(uint32_t);
        if (runtime) STREAM_PROGRESS_RUNTIME(cudaMemsetAsync(stream_progress_pointer(address), 0x47, 192 * sizeof(uint32_t), reinterpret_cast<cudaStream_t>(gated)));
        else CONFORMANCE_CUDA(cuMemsetD8Async(address, 0x47, 192 * sizeof(uint32_t), gated));
    }
    produced.record(gated);
    for (unsigned i = 0; i < consumers_count; ++i) {
        const auto& consumer = *consumers[i];
        if (event_dependency) {
            if (runtime) STREAM_PROGRESS_RUNTIME(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(consumer.stream.value), reinterpret_cast<cudaEvent_t>(produced.value), 0));
            else CONFORMANCE_CUDA(cuStreamWaitEvent(consumer.stream.value, produced.value, 0));
        }
        kernel.launch(consumer, count, 17 + i * 1031); consumer.read(runtime); consumer.done.record(consumer.stream.value);
    }
    const unsigned independent_count = count == 0 ? 211 : count == 97 ? 192 : 97;
    kernel.launch(independent, independent_count, 0xfffffff1); independent.read(runtime); independent.done.record(independent.stream.value);
    if (retirement) { producer.destroy(); independent.stream.destroy(); for (auto& consumer : consumers) consumer->stream.destroy(); }
    independent.done.complete(); gate.held(); independent.expect(independent_count, 0xfffffff1, false);
    require(cuEventQuery(produced.value) == CUDA_ERROR_NOT_READY, "held producer has not published its event");
    for (const auto& consumer : consumers) require(cuEventQuery(consumer->done.value) == CUDA_ERROR_NOT_READY, "dependent kernel remains held while independent work completes");
    gate.release(); produced.complete(); for (auto& consumer : consumers) consumer->done.complete(); gate.check();
    for (unsigned i = 0; i < consumers_count; ++i) consumers[i]->expect(count, 17 + i * 1031, true);
}
void execute_stream_progress(const loaded_program& program) {
    struct scenario { const char* name; unsigned consumers; bool events, retirement; };
    const std::array<scenario, 4> cases{{{"direct", 1, false, false}, {"events", 1, true, false}, {"fanout", 4, true, false}, {"retirement", 4, true, true}}};
    for (bool runtime : {false, true}) {
        stream_progress_kernel kernel(program, runtime);
        for (const auto& input : cases) for (unsigned count : {0u, 97u, 211u}) {
            std::cout << "stream_progress=" << input.name << " runtime=" << runtime << " runtime_kernel=" << kernel.runtime << " count=" << count << std::endl;
            stream_progress_round(kernel, runtime, count, input.consumers, input.events, input.retirement);
        }
    }
}
}
int main(int argc, char** argv) { return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_stream_progress); }
