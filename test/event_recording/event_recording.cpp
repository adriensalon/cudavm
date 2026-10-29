#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <utility>

namespace cudavm {
void check_recording_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define RECORDING_RUNTIME(call) ::cudavm::check_recording_runtime((call), #call)
struct recording_event {
    CUevent value{};
    bool runtime;
    recording_event(bool use_runtime, unsigned flags) : runtime(use_runtime) {
        if (runtime) {
            cudaEvent_t event{};
            RECORDING_RUNTIME(cudaEventCreateWithFlags(&event, flags));
            value = reinterpret_cast<CUevent>(event);
        } else CONFORMANCE_CUDA(cuEventCreate(&value, flags));
    }
    ~recording_event() { if (value) cuEventDestroy(value); }
    cudaEvent_t runtime_value() const { return reinterpret_cast<cudaEvent_t>(value); }
    void record(CUstream stream, unsigned api) {
        if (api == 0) CONFORMANCE_CUDA(cuEventRecord(value, stream));
        else if (api == 1) RECORDING_RUNTIME(cudaEventRecord(runtime_value(), reinterpret_cast<cudaStream_t>(stream)));
        else if (api == 2) CONFORMANCE_CUDA(cuEventRecordWithFlags(value, stream, 0));
        else RECORDING_RUNTIME(cudaEventRecordWithFlags(runtime_value(), reinterpret_cast<cudaStream_t>(stream), 0));
    }
    void expect_ready(bool ready) const {
        require(cuEventQuery(value) == (ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY), "Driver event record state");
        require(cudaEventQuery(runtime_value()) == (ready ? cudaSuccess : cudaErrorNotReady), "Runtime event record state");
    }
    void synchronize(bool use_runtime) const {
        if (use_runtime) RECORDING_RUNTIME(cudaEventSynchronize(runtime_value()));
        else CONFORMANCE_CUDA(cuEventSynchronize(value));
    }
    void destroy() {
        if (runtime) CONFORMANCE_CUDA(cuEventDestroy(value));
        else RECORDING_RUNTIME(cudaEventDestroy(runtime_value()));
        value = nullptr;
    }
};
struct recording_stream {
    CUstream value{};
    recording_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~recording_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
};
struct recording_readback {
    static constexpr size_t payload_size = 512, size = payload_size + 32;
    void* value{};
    recording_readback() { CONFORMANCE_CUDA(cuMemAllocHost(&value, size)); std::memset(value, 0xd3, size); }
    ~recording_readback() { if (value) cuMemFreeHost(value); }
    void expect(std::span<const unsigned char> expected) const {
        require(expected.size() == size, "event readback oracle extent");
        const auto* bytes = static_cast<const unsigned char*>(value);
        for (size_t i = 0; i < size; ++i)
            if (bytes[i] != expected[i]) throw std::runtime_error("event readback payload/guard byte " + std::to_string(i));
    }
};
std::vector<unsigned char> recording_expected(size_t offset, size_t bytes, unsigned char value) {
    std::vector<unsigned char> result(recording_readback::size, 0xd3);
    std::fill_n(result.begin() + 16, recording_readback::payload_size, 0xa9);
    std::fill_n(result.begin() + 16 + offset, bytes, value);
    return result;
}
void recording_fill_readback(CUstream stream, CUdeviceptr device, recording_readback& destination,
                            size_t offset, size_t bytes, unsigned char value, bool runtime) {
    if (runtime) {
        RECORDING_RUNTIME(cudaMemsetAsync(reinterpret_cast<void*>(static_cast<uintptr_t>(device + offset)),
            value, bytes, reinterpret_cast<cudaStream_t>(stream)));
        CONFORMANCE_CUDA(cuMemcpyDtoHAsync(static_cast<unsigned char*>(destination.value) + 16,
            device, recording_readback::payload_size, stream));
    } else {
        CONFORMANCE_CUDA(cuMemsetD8Async(device + offset, value, bytes, stream));
        RECORDING_RUNTIME(cudaMemcpyAsync(static_cast<unsigned char*>(destination.value) + 16,
            reinterpret_cast<void*>(static_cast<uintptr_t>(device)), recording_readback::payload_size,
            cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream)));
    }
}
struct recording_gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* argument) {
        auto& gate = *static_cast<recording_gate*>(argument);
        std::unique_lock lock(gate.mutex);
        gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(15), [&] { return gate.released; })) gate.expired = true;
    }
    void await() {
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "event gate entry deadline");
    }
    void require_held() {
        std::lock_guard lock(mutex);
        require(entered && !released && !expired, "event completion does not wait for unrelated gated work");
    }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
};
struct recording_cleanup {
    recording_gate& gate;
    ~recording_cleanup() { gate.release(); cuCtxSynchronize(); }
};
void event_recording_unrecorded() {
    for (bool runtime : {false, true}) for (unsigned flags : {0u, 1u, 2u, 3u}) {
        recording_event event(runtime, flags);
        event.expect_ready(true);
        event.synchronize(false); event.synchronize(true);
        event.expect_ready(true);
        event.destroy();
    }
}
void event_recording_snapshot() {
    for (bool runtime : {false, true}) for (unsigned flags : {0u, 1u, 2u, 3u})
    for (unsigned api = 0; api < 4; ++api) {
        device_buffer device(recording_readback::payload_size);
        recording_readback first, later;
        recording_stream stream;
        recording_event event(runtime, flags);
        CONFORMANCE_CUDA(cuMemsetD8(device.value, 0xa9, recording_readback::payload_size));
        recording_fill_readback(stream.value, device.value, first, 17, 257, 0x5b, runtime);
        event.record(stream.value, api);
        recording_gate gate;
        recording_cleanup cleanup{gate};
        CONFORMANCE_CUDA(cuLaunchHostFunc(stream.value, recording_gate::call, &gate)); gate.await();
        recording_fill_readback(stream.value, device.value, later, 23, 73, 0x91, !runtime);
        event.expect_ready(true);
        event.synchronize(!runtime);
        gate.require_held();
        require(cuStreamQuery(stream.value) == CUDA_ERROR_NOT_READY, "work after recorded snapshot is still pending");
        const auto expected = recording_expected(17, 257, 0x5b);
        first.expect(expected);
        later.expect(std::vector<unsigned char>(recording_readback::size, 0xd3));
        event.expect_ready(true);
        gate.release();
        CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
        require(!gate.expired, "event snapshot gate deadline");
        auto changed = expected;
        std::fill_n(changed.begin() + 16 + 23, 73, 0x91);
        later.expect(changed); first.expect(expected);
        event.expect_ready(true);
        event.destroy();
    }
}
void event_recording_rerecord() {
    for (bool runtime : {false, true}) for (unsigned flags : {2u, 3u})
    for (unsigned api = 0; api < 4; ++api) {
        device_buffer first_device(recording_readback::payload_size), second_device(recording_readback::payload_size);
        recording_readback first, second;
        recording_stream blocked, independent;
        recording_event event(runtime, flags);
        CONFORMANCE_CUDA(cuMemsetD8(first_device.value, 0xa9, recording_readback::payload_size));
        CONFORMANCE_CUDA(cuMemsetD8(second_device.value, 0xa9, recording_readback::payload_size));
        recording_gate old_gate, new_gate;
        struct cleanup_gates {
            recording_gate& old_gate;
            recording_gate& new_gate;
            ~cleanup_gates() { old_gate.release(); new_gate.release(); cuCtxSynchronize(); }
        } cleanup{old_gate, new_gate};
        CONFORMANCE_CUDA(cuLaunchHostFunc(blocked.value, recording_gate::call, &old_gate)); old_gate.await();
        recording_fill_readback(blocked.value, first_device.value, first, 17, 257, 0x5b, runtime);
        event.record(blocked.value, api);
        event.expect_ready(false);
        // Snapshot the old record before replacing it. Entry to the new gate
        // then proves old work has completed, without requiring GPU progress
        // while any host function depends on unordered outstanding CUDA work.
        CONFORMANCE_CUDA(cuStreamWaitEvent(independent.value, event.value, 0));
        CONFORMANCE_CUDA(cuLaunchHostFunc(independent.value, recording_gate::call, &new_gate));
        recording_fill_readback(independent.value, second_device.value, second, 23, 73, 0x91, !runtime);
        event.record(independent.value, (api + 1) % 4);
        event.expect_ready(false);
        old_gate.release();
        new_gate.await();
        new_gate.require_held();
        first.expect(recording_expected(17, 257, 0x5b));
        second.expect(std::vector<unsigned char>(recording_readback::size, 0xd3));
        event.expect_ready(false);
        new_gate.release();
        event.synchronize(runtime);
        require(!old_gate.expired && !new_gate.expired, "re-record gate deadlines");
        event.expect_ready(true);
        first.expect(recording_expected(17, 257, 0x5b));
        second.expect(recording_expected(23, 73, 0x91));
        event.record(blocked.value, (api + 2) % 4);
        event.synchronize(!runtime);
        event.expect_ready(true);
        event.destroy();
    }
}
void event_recording_validation() {
    recording_stream stream;
    recording_event survivor(false, CU_EVENT_DISABLE_TIMING);
    require(cuEventCreate(nullptr, 0) == CUDA_ERROR_INVALID_VALUE, "null Driver event output");
    require(cudaEventCreateWithFlags(nullptr, 0) == cudaErrorInvalidValue, "null Runtime event output");
    for (unsigned flags : {8u, 0x80000000u}) {
        CUevent driver{};
        const auto result = cuEventCreate(&driver, flags);
        if (result == CUDA_SUCCESS) cuEventDestroy(driver);
        require(result == CUDA_ERROR_INVALID_VALUE, "invalid Driver event creation flags");
        cudaEvent_t runtime{};
        const auto status = cudaEventCreateWithFlags(&runtime, flags);
        if (status == cudaSuccess) cudaEventDestroy(runtime);
        require(status == cudaErrorInvalidValue, "invalid Runtime event creation flags");
        require(cuEventRecordWithFlags(survivor.value, stream.value, flags) == CUDA_ERROR_INVALID_VALUE, "invalid Driver event record flags");
        require(cudaEventRecordWithFlags(survivor.runtime_value(), reinterpret_cast<cudaStream_t>(stream.value), flags)
            == cudaErrorInvalidValue, "invalid Runtime event record flags");
    }
    require(cuEventQuery(nullptr) == CUDA_ERROR_INVALID_HANDLE, "null Driver event query");
    require(cudaEventQuery(nullptr) == cudaErrorInvalidResourceHandle, "null Runtime event query");
    require(cuEventRecord(nullptr, stream.value) == CUDA_ERROR_INVALID_HANDLE, "null Driver event record");
    require(cudaEventRecord(nullptr, reinterpret_cast<cudaStream_t>(stream.value)) == cudaErrorInvalidResourceHandle,
        "null Runtime event record");
    survivor.expect_ready(true);
    survivor.record(stream.value, 0); survivor.synchronize(true); survivor.expect_ready(true);
    survivor.destroy();
}
std::string_view event_recording_case = "all";
void execute_event_recording() {
    const std::array<std::pair<std::string_view, void (*)()>, 4> cases{{
        {"unrecorded", event_recording_unrecorded}, {"snapshot", event_recording_snapshot},
        {"rerecord", event_recording_rerecord}, {"validation", event_recording_validation}}};
    bool found = false;
    for (const auto& [name, execute] : cases) if (event_recording_case == "all" || event_recording_case == name) {
        found = true; std::cout << "event_recording=" << name << std::endl; execute();
    }
    require(found, "unknown event_recording case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::event_recording_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_event_recording);
}
