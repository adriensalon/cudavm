#include "test_framework.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>

namespace cudavm {
void check_transfer_runtime(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
#define TRANSFER_RUNTIME(call) ::cudavm::check_transfer_runtime((call), #call)
struct transfer_host {
    static constexpr size_t bytes = 809;
    std::vector<unsigned char> pageable; void* value{}; bool pinned;
    explicit transfer_host(bool use_pinned) : pinned(use_pinned) {
        if (pinned) CONFORMANCE_CUDA(cuMemAllocHost(&value, bytes)); else { pageable.resize(bytes); value = pageable.data(); }
    }
    ~transfer_host() { if (pinned && value) { cuCtxSynchronize(); cuMemFreeHost(value); } }
    unsigned char* data() const { return static_cast<unsigned char*>(value); }
    void set(const std::vector<unsigned char>& input) const { require(input.size() == bytes, "completion host extent"); std::memcpy(value, input.data(), bytes); }
    void expect(const std::vector<unsigned char>& expected, const std::string& context) const { require(std::memcmp(value, expected.data(), bytes) == 0, context); }
};
std::vector<unsigned char> transfer_pattern(unsigned seed) {
    std::vector<unsigned char> result(transfer_host::bytes);
    for (size_t i = 0; i < result.size(); ++i) result[i] = static_cast<unsigned char>(seed + 37 * i + (i >> 3));
    return result;
}
void* transfer_pointer(CUdeviceptr value) { return reinterpret_cast<void*>(static_cast<uintptr_t>(value)); }
void transfer_upload(bool runtime, CUdeviceptr to, const void* from, size_t bytes, CUstream stream = nullptr, bool asynchronous = false) {
    if (runtime) {
        if (asynchronous) TRANSFER_RUNTIME(cudaMemcpyAsync(transfer_pointer(to), from, bytes, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)));
        else TRANSFER_RUNTIME(cudaMemcpy(transfer_pointer(to), from, bytes, cudaMemcpyHostToDevice));
    } else {
        if (asynchronous) CONFORMANCE_CUDA(cuMemcpyHtoDAsync(to, from, bytes, stream)); else CONFORMANCE_CUDA(cuMemcpyHtoD(to, from, bytes));
    }
}
void transfer_download(bool runtime, void* to, CUdeviceptr from, size_t bytes, CUstream stream = nullptr, bool asynchronous = false) {
    if (runtime) {
        if (asynchronous) TRANSFER_RUNTIME(cudaMemcpyAsync(to, transfer_pointer(from), bytes, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream)));
        else TRANSFER_RUNTIME(cudaMemcpy(to, transfer_pointer(from), bytes, cudaMemcpyDeviceToHost));
    } else {
        if (asynchronous) CONFORMANCE_CUDA(cuMemcpyDtoHAsync(to, from, bytes, stream)); else CONFORMANCE_CUDA(cuMemcpyDtoH(to, from, bytes));
    }
}
void transfer_synchronous() {
    device_buffer device(transfer_host::bytes);
    for (bool runtime : {false, true}) for (bool pinned_source : {false, true}) for (bool pinned_destination : {false, true}) {
        transfer_host input(pinned_source), output(pinned_destination);
        for (size_t count : {size_t{1}, size_t{7}, size_t{127}, size_t{701}}) {
            const auto source = transfer_pattern(static_cast<unsigned>(count)), initial_device = transfer_pattern(91), initial_host = transfer_pattern(177);
            input.set(source); output.set(initial_host); CONFORMANCE_CUDA(cuMemcpyHtoD(device.value, initial_device.data(), initial_device.size()));
            transfer_upload(runtime, device.value + 31, input.data() + 17, count);
            // Pageable HtoD may retain staging, but cannot retain a live dependency
            // on caller bytes after return. Pinned synchronous HtoD is host complete.
            std::memset(input.value, 0x4f, transfer_host::bytes);
            transfer_download(!runtime, output.data() + 19, device.value + 11, 773);
            auto expected_device = initial_device, expected_output = initial_host;
            std::copy_n(source.begin() + 17, count, expected_device.begin() + 31);
            std::copy_n(expected_device.begin() + 11, 773, expected_output.begin() + 19);
            // No event, stream or context wait occurs between download and inspection.
            output.expect(expected_output, "synchronous download is host complete on return, with output guards");
            input.expect(std::vector<unsigned char>(transfer_host::bytes, 0x4f), "synchronous upload releases caller source bytes");
            std::vector<unsigned char> actual(transfer_host::bytes); CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), device.value, actual.size()));
            require(actual == expected_device, "synchronous upload staging and all device guards");
        }
    }
}
struct transfer_stream {
    CUstream value{};
    transfer_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~transfer_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
};
struct transfer_event {
    CUevent value{};
    transfer_event() { CONFORMANCE_CUDA(cuEventCreate(&value, CU_EVENT_DISABLE_TIMING)); }
    ~transfer_event() { if (value) cuEventDestroy(value); }
};
struct transfer_gate {
    std::mutex mutex; std::condition_variable changed; bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* argument) {
        auto& gate = *static_cast<transfer_gate*>(argument); std::unique_lock lock(gate.mutex);
        gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(10), [&] { return gate.released; })) gate.expired = true;
    }
    void await() { std::unique_lock lock(mutex); require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "completion gate entered"); }
    void check_held() { std::lock_guard lock(mutex); require(entered && !released && !expired, "pinned asynchronous copies returned while stream remained held"); }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
    void check_complete() { std::lock_guard lock(mutex); require(entered && released && !expired, "completion gate did not expire"); }
};
struct transfer_cleanup {
    transfer_gate& gate;
    ~transfer_cleanup() { gate.release(); cuCtxSynchronize(); }
};
void transfer_asynchronous(bool event_completion) {
    for (bool runtime : {false, true}) {
        transfer_host input(true), output(true); device_buffer first(transfer_host::bytes), second(transfer_host::bytes); transfer_stream stream; transfer_event completed;
        for (unsigned seed : {0u, 79u, 255u, 0xffffffffu}) {
            const auto source = transfer_pattern(seed), initial_first = transfer_pattern(33), initial_second = transfer_pattern(91), initial_host = transfer_pattern(177);
            input.set(source); output.set(initial_host);
            CONFORMANCE_CUDA(cuMemcpyHtoD(first.value, initial_first.data(), initial_first.size())); CONFORMANCE_CUDA(cuMemcpyHtoD(second.value, initial_second.data(), initial_second.size()));
            transfer_gate gate; transfer_cleanup cleanup{gate};
            if (runtime) TRANSFER_RUNTIME(cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream.value), transfer_gate::call, &gate));
            else CONFORMANCE_CUDA(cuLaunchHostFunc(stream.value, transfer_gate::call, &gate)); gate.await();
            transfer_upload(runtime, first.value + 31, input.data() + 17, 613, stream.value, true);
            if (runtime) TRANSFER_RUNTIME(cudaMemcpyAsync(transfer_pointer(second.value + 13), transfer_pointer(first.value + 31), 613, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream.value)));
            else CONFORMANCE_CUDA(cuMemcpyDtoDAsync(second.value + 13, first.value + 31, 613, stream.value));
            transfer_download(runtime, output.data() + 19, second.value + 7, 773, stream.value, true);
            if (runtime) TRANSFER_RUNTIME(cudaEventRecord(reinterpret_cast<cudaEvent_t>(completed.value), reinterpret_cast<cudaStream_t>(stream.value)));
            else CONFORMANCE_CUDA(cuEventRecord(completed.value, stream.value));
            gate.check_held(); output.expect(initial_host, "pending pinned download has not executed before prior stream gate");
            require(cuEventQuery(completed.value) == CUDA_ERROR_NOT_READY, "completion event is pending while stream is held");
            gate.release();
            if (event_completion) {
                if (runtime) CONFORMANCE_CUDA(cuEventSynchronize(completed.value)); else TRANSFER_RUNTIME(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(completed.value)));
            } else {
                if (runtime) CONFORMANCE_CUDA(cuStreamSynchronize(stream.value)); else TRANSFER_RUNTIME(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.value)));
            }
            // Event/stream completion alone permits inspecting the pinned result.
            auto a = initial_first, b = initial_second, h = initial_host;
            std::copy_n(source.begin() + 17, 613, a.begin() + 31); std::copy_n(a.begin() + 31, 613, b.begin() + 13); std::copy_n(b.begin() + 7, 773, h.begin() + 19);
            output.expect(h, "host observes complete payload and guards after specified completion point"); input.expect(source, "async source preserved through completion"); gate.check_complete();
            std::vector<unsigned char> actual(transfer_host::bytes); CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), first.value, actual.size())); require(actual == a, "async first device allocation guards");
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), second.value, actual.size())); require(actual == b, "async second device allocation guards");
            CONFORMANCE_CUDA(cuEventQuery(completed.value));
        }
    }
}
void transfer_host_completion() {
    for (bool asynchronous : {false, true}) for (bool pinned_source : {false, true}) for (bool pinned_destination : {false, true}) {
        transfer_host input(pinned_source), output(pinned_destination); transfer_stream stream;
        for (size_t count : {size_t{1}, size_t{7}, size_t{127}, size_t{701}}) {
            const auto source = transfer_pattern(static_cast<unsigned>(count)), initial = transfer_pattern(193); input.set(source); output.set(initial);
            if (asynchronous) TRANSFER_RUNTIME(cudaMemcpyAsync(output.data() + 29, input.data() + 17, count, cudaMemcpyHostToHost, reinterpret_cast<cudaStream_t>(stream.value)));
            else TRANSFER_RUNTIME(cudaMemcpy(output.data() + 29, input.data() + 17, count, cudaMemcpyHostToHost));
            std::memset(input.value, 0x4f, transfer_host::bytes); auto expected = initial; std::copy_n(source.begin() + 17, count, expected.begin() + 29);
            output.expect(expected, "Runtime host-to-host copy is complete on API return"); input.expect(std::vector<unsigned char>(transfer_host::bytes, 0x4f), "host copy source can be reused immediately");
        }
    }
}
std::string transfer_case = "all";
void execute_transfer_completion() {
    bool selected = false;
    const auto run = [&](const char* name, auto execute) { if (transfer_case == "all" || transfer_case == name) { std::cout << "case=" << name << std::endl; execute(); selected = true; } };
    run("synchronous", transfer_synchronous); run("asynchronous", [] { transfer_asynchronous(false); }); run("events", [] { transfer_asynchronous(true); }); run("host", transfer_host_completion);
    require(selected, "unknown transfer_completion case: " + transfer_case);
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::transfer_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_transfer_completion);
}
