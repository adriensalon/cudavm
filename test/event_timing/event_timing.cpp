#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

namespace cudavm {
void check_timing_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define TIMING_RUNTIME(call) ::cudavm::check_timing_runtime((call), #call)
struct timing_event {
    CUevent value{};
    timing_event(bool runtime, unsigned flags = 0) {
        if (runtime) {
            cudaEvent_t event{};
            TIMING_RUNTIME(cudaEventCreateWithFlags(&event, flags));
            value = reinterpret_cast<CUevent>(event);
        } else CONFORMANCE_CUDA(cuEventCreate(&value, flags));
    }
    ~timing_event() { if (value) cuEventDestroy(value); }
    cudaEvent_t runtime_value() const { return reinterpret_cast<cudaEvent_t>(value); }
    void record(CUstream stream, bool runtime) {
        if (runtime) TIMING_RUNTIME(cudaEventRecord(runtime_value(), reinterpret_cast<cudaStream_t>(stream)));
        else CONFORMANCE_CUDA(cuEventRecord(value, stream));
    }
    void synchronize(bool runtime) {
        if (runtime) TIMING_RUNTIME(cudaEventSynchronize(runtime_value()));
        else CONFORMANCE_CUDA(cuEventSynchronize(value));
        CONFORMANCE_CUDA(cuEventQuery(value));
        TIMING_RUNTIME(cudaEventQuery(runtime_value()));
    }
};
struct timing_stream {
    CUstream value{};
    explicit timing_stream(bool runtime) {
        if (runtime) {
            cudaStream_t stream{};
            TIMING_RUNTIME(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            value = reinterpret_cast<CUstream>(stream);
        } else CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING));
    }
    ~timing_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
};
struct timing_output {
    uint32_t before = 0xdeadbeef;
    float value = 123.0f;
    uint32_t after = 0xbad12345;
    void check() const { require(before == 0xdeadbeef && after == 0xbad12345, "elapsed-time float output width"); }
};
float timing_elapsed(CUevent start, CUevent end, CUresult driver = CUDA_SUCCESS,
                     cudaError_t runtime = cudaSuccess) {
    timing_output a, b;
    require(cuEventElapsedTime(&a.value, start, end) == driver, "Driver elapsed-time result");
    require(cudaEventElapsedTime(&b.value, reinterpret_cast<cudaEvent_t>(start),
        reinterpret_cast<cudaEvent_t>(end)) == runtime, "Runtime elapsed-time result");
    a.check(); b.check();
    if (driver != CUDA_SUCCESS) return 0;
    require(std::isfinite(a.value) && a.value >= 0, "finite nonnegative event interval");
    require(std::bit_cast<uint32_t>(a.value) == std::bit_cast<uint32_t>(b.value),
        "both facades observe the same completed timestamp pair");
    return a.value;
}
struct timing_delay {
    unsigned milliseconds;
    double observed = 0;
    bool completed = false;
    static void CUDA_CB call(void* argument) {
        auto& delay = *static_cast<timing_delay*>(argument);
        const auto begin = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(delay.milliseconds));
        delay.observed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        delay.completed = true;
    }
    void enqueue(CUstream stream, bool runtime) {
        if (runtime) TIMING_RUNTIME(cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), call, this));
        else CONFORMANCE_CUDA(cuLaunchHostFunc(stream, call, this));
    }
    void check(float elapsed) const {
        require(completed && observed >= milliseconds * 0.8, "ordered host delay completed");
        require(elapsed >= observed * 0.5, "GPU timestamp interval includes ordered delay in milliseconds");
    }
};
struct timing_readback {
    static constexpr size_t payload = 512, device_size = payload + 32, size = device_size + 32;
    void* value{};
    timing_readback() { CONFORMANCE_CUDA(cuMemAllocHost(&value, size)); std::memset(value, 0xd3, size); }
    ~timing_readback() { if (value) cuMemFreeHost(value); }
    void expect(bool second) const {
        std::vector<unsigned char> expected(size, 0xd3);
        std::fill_n(expected.begin() + 16, device_size, 0xa7);
        std::fill_n(expected.begin() + 32, payload, 0x39);
        std::fill_n(expected.begin() + 32 + 19, 41, 0x65);
        if (second) std::fill_n(expected.begin() + 32 + 173, 77, 0xb2);
        require(std::memcmp(value, expected.data(), size) == 0, "timed transfers, inactive bytes and guards");
    }
};
void timing_stage(CUstream stream, CUdeviceptr device, timing_readback& destination, bool second, bool runtime) {
    const auto offset = second ? 173u : 19u, count = second ? 77u : 41u;
    const unsigned char byte = second ? 0xb2 : 0x65;
    if (runtime) {
        TIMING_RUNTIME(cudaMemsetAsync(reinterpret_cast<void*>(static_cast<uintptr_t>(device + 16 + offset)),
            byte, count, reinterpret_cast<cudaStream_t>(stream)));
        CONFORMANCE_CUDA(cuMemcpyDtoHAsync(static_cast<unsigned char*>(destination.value) + 16,
            device, timing_readback::device_size, stream));
    } else {
        CONFORMANCE_CUDA(cuMemsetD8Async(device + 16 + offset, byte, count, stream));
        TIMING_RUNTIME(cudaMemcpyAsync(static_cast<unsigned char*>(destination.value) + 16,
            reinterpret_cast<void*>(static_cast<uintptr_t>(device)), timing_readback::device_size,
            cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream)));
    }
}
void event_timing_completed() {
    for (bool runtime : {false, true}) for (bool cross_stream : {false, true}) {
        device_buffer storage(timing_readback::device_size);
        timing_readback first, second;
        timing_delay first_delay{20}, second_delay{35};
        timing_event start(runtime), middle(!runtime, CU_EVENT_BLOCKING_SYNC), end(runtime);
        // Streams drain before the events, arguments and pinned destinations die.
        timing_stream producer(runtime), consumer(!runtime);
        CONFORMANCE_CUDA(cuMemsetD8(storage.value, 0xa7, timing_readback::device_size));
        CONFORMANCE_CUDA(cuMemsetD8(storage.value + 16, 0x39, timing_readback::payload));
        const auto begin = std::chrono::steady_clock::now();
        start.record(producer.value, !runtime);
        first_delay.enqueue(producer.value, runtime);
        timing_stage(producer.value, storage.value, first, false, runtime);
        middle.record(producer.value, runtime);
        auto tail = cross_stream ? consumer.value : producer.value;
        if (cross_stream) TIMING_RUNTIME(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(tail), middle.runtime_value(), 0));
        second_delay.enqueue(tail, !runtime);
        timing_stage(tail, storage.value, second, true, !runtime);
        end.record(tail, !runtime);
        end.synchronize(runtime);
        const auto wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        first.expect(false); second.expect(true);
        const auto a = timing_elapsed(start.value, middle.value);
        const auto b = timing_elapsed(middle.value, end.value);
        const auto total = timing_elapsed(start.value, end.value);
        first_delay.check(a); second_delay.check(b);
        require(total <= wall * 2 + 10, "device milliseconds bounded by enclosing host interval");
        require(std::abs(double(total) - a - b) <= std::max(0.01, double(total) * 0.00001),
            "adjacent timestamp intervals add to the complete interval");
        require(timing_elapsed(start.value, start.value) == 0 && timing_elapsed(end.value, end.value) == 0,
            "completed self intervals are zero");
        require(timing_elapsed(start.value, end.value) == total, "completed interval is stable");
    }
}
struct timing_gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* argument) {
        auto& gate = *static_cast<timing_gate*>(argument);
        std::unique_lock lock(gate.mutex);
        gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(15), [&] { return gate.released; })) gate.expired = true;
    }
    void await() {
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "timing gate entry deadline");
    }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
};
struct timing_gate_cleanup {
    timing_gate& gate;
    ~timing_gate_cleanup() { gate.release(); cuCtxSynchronize(); }
};
void event_timing_pending() {
    for (bool runtime : {false, true}) for (bool both_pending : {false, true}) {
        timing_gate gate;
        timing_event start(runtime), end(!runtime);
        timing_stream stream(runtime);
        timing_gate_cleanup cleanup{gate};
        if (!both_pending) { start.record(stream.value, runtime); start.synchronize(!runtime); }
        CONFORMANCE_CUDA(cuLaunchHostFunc(stream.value, timing_gate::call, &gate));
        if (both_pending) start.record(stream.value, !runtime);
        end.record(stream.value, runtime);
        gate.await();
        require(cuEventQuery(end.value) == CUDA_ERROR_NOT_READY
            && cudaEventQuery(end.runtime_value()) == cudaErrorNotReady, "timing end remains pending");
        if (both_pending) require(cuEventQuery(start.value) == CUDA_ERROR_NOT_READY, "timing start remains pending");
        timing_elapsed(start.value, end.value, CUDA_ERROR_NOT_READY, cudaErrorNotReady);
        gate.release(); end.synchronize(!runtime);
        require(!gate.expired, "timing gate did not expire");
        timing_elapsed(start.value, end.value);
    }
}
void event_timing_rerecord() {
    for (bool runtime : {false, true}) {
        timing_delay first{20}, later{35}, replacement{15};
        timing_event start(runtime), end(!runtime);
        timing_stream stream(runtime);
        start.record(stream.value, runtime);
        first.enqueue(stream.value, !runtime); end.record(stream.value, !runtime); end.synchronize(runtime);
        const auto old = timing_elapsed(start.value, end.value); first.check(old);
        later.enqueue(stream.value, runtime); end.record(stream.value, runtime); end.synchronize(!runtime);
        const auto latest = timing_elapsed(start.value, end.value);
        later.check(latest - old);
        start.record(stream.value, !runtime);
        replacement.enqueue(stream.value, runtime); end.record(stream.value, !runtime); end.synchronize(runtime);
        replacement.check(timing_elapsed(start.value, end.value));
        require(timing_elapsed(start.value, start.value) == 0, "re-recorded self interval");
    }
}
void event_timing_validation() {
    for (bool runtime : {false, true}) {
        timing_event a(runtime), b(!runtime), disabled(runtime, CU_EVENT_DISABLE_TIMING);
        timing_stream stream(!runtime);
        const auto invalid = [&](CUevent start, CUevent end) {
            timing_elapsed(start, end, CUDA_ERROR_INVALID_HANDLE, cudaErrorInvalidResourceHandle);
        };
        invalid(a.value, b.value);
        a.record(stream.value, runtime); a.synchronize(!runtime);
        invalid(a.value, b.value); invalid(b.value, a.value);
        b.record(stream.value, !runtime); disabled.record(stream.value, runtime); disabled.synchronize(!runtime);
        invalid(a.value, disabled.value); invalid(disabled.value, b.value); invalid(disabled.value, disabled.value);
        invalid(nullptr, b.value); invalid(a.value, nullptr);
        require(cuEventElapsedTime(nullptr, a.value, b.value) == CUDA_ERROR_INVALID_VALUE, "Driver null elapsed output");
        require(cudaEventElapsedTime(nullptr, a.runtime_value(), b.runtime_value()) == cudaErrorInvalidValue,
            "Runtime null elapsed output");
        timing_elapsed(a.value, b.value);
    }
}
std::string timing_case = "all";
void execute_event_timing() {
    bool selected = false;
    const auto run = [&](const char* name, auto execute) {
        if (timing_case == "all" || timing_case == name) { execute(); selected = true; }
    };
    run("completed", event_timing_completed); run("pending", event_timing_pending);
    run("rerecord", event_timing_rerecord); run("validation", event_timing_validation);
    require(selected, "unknown event_timing case: " + timing_case);
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::timing_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_event_timing);
}
