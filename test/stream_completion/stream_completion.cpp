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
void check_completion_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define COMPLETION_RUNTIME(call) ::cudavm::check_completion_runtime((call), #call)
struct completion_stream {
    CUstream value{};
    bool runtime;
    completion_stream(bool use_runtime, unsigned flags) : runtime(use_runtime) {
        if (runtime) {
            cudaStream_t stream{};
            COMPLETION_RUNTIME(cudaStreamCreateWithFlags(&stream, flags));
            value = reinterpret_cast<CUstream>(stream);
        } else CONFORMANCE_CUDA(cuStreamCreate(&value, flags));
    }
    ~completion_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    cudaStream_t runtime_value() const { return reinterpret_cast<cudaStream_t>(value); }
    void destroy() {
        if (runtime) CONFORMANCE_CUDA(cuStreamDestroy(value));
        else COMPLETION_RUNTIME(cudaStreamDestroy(runtime_value()));
        value = nullptr;
    }
    void synchronize(bool use_runtime) {
        if (use_runtime) COMPLETION_RUNTIME(cudaStreamSynchronize(runtime_value()));
        else CONFORMANCE_CUDA(cuStreamSynchronize(value));
    }
    void expect_ready(bool ready) const {
        require(cuStreamQuery(value) == (ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY), "Driver stream completion state");
        require(cudaStreamQuery(runtime_value()) == (ready ? cudaSuccess : cudaErrorNotReady), "Runtime stream completion state");
    }
};
struct completion_host_buffer {
    static constexpr size_t size = 512;
    void* value{};
    completion_host_buffer() { CONFORMANCE_CUDA(cuMemAllocHost(&value, size)); }
    ~completion_host_buffer() { if (value) cuMemFreeHost(value); }
    unsigned char* data() const { return static_cast<unsigned char*>(value); }
    void expect(std::span<const unsigned char> expected, const std::string& operation) const {
        require(expected.size() == size, operation + ": host extent");
        for (size_t i = 0; i < size; ++i)
            if (data()[i] != expected[i]) throw std::runtime_error(operation + ": byte " + std::to_string(i));
    }
};
struct completion_event {
    CUevent value{};
    completion_event() { CONFORMANCE_CUDA(cuEventCreate(&value, CU_EVENT_DISABLE_TIMING)); }
    ~completion_event() { if (value) cuEventDestroy(value); }
};
struct completion_gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* argument) {
        auto& gate = *static_cast<completion_gate*>(argument);
        std::unique_lock lock(gate.mutex);
        gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(15), [&] { return gate.released; })) gate.expired = true;
    }
    void await() {
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "completion gate entry deadline");
    }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
};
struct completion_gate_cleanup {
    completion_gate& gate;
    ~completion_gate_cleanup() { gate.release(); cuCtxSynchronize(); }
};
void stream_completion_empty() {
    for (bool runtime : {false, true}) for (unsigned flags : {0u, unsigned(CU_STREAM_NON_BLOCKING)}) {
        completion_stream stream(runtime, flags);
        stream.expect_ready(true);
        stream.synchronize(false); stream.synchronize(true);
        stream.expect_ready(true);
        stream.destroy();
    }
}
void completion_transfer(completion_stream& stream, const completion_host_buffer& source,
                            completion_host_buffer& destination, CUdeviceptr device, size_t bytes) {
    constexpr size_t src = 17, dev = 23, dst = 31;
    if (stream.runtime) {
        COMPLETION_RUNTIME(cudaMemcpyAsync(reinterpret_cast<void*>(static_cast<uintptr_t>(device + dev)),
            source.data() + src, bytes, cudaMemcpyHostToDevice, stream.runtime_value()));
        CONFORMANCE_CUDA(cuMemcpyDtoHAsync(destination.data() + dst, device + dev, bytes, stream.value));
    } else {
        CONFORMANCE_CUDA(cuMemcpyHtoDAsync(device + dev, source.data() + src, bytes, stream.value));
        COMPLETION_RUNTIME(cudaMemcpyAsync(destination.data() + dst,
            reinterpret_cast<void*>(static_cast<uintptr_t>(device + dev)), bytes, cudaMemcpyDeviceToHost, stream.runtime_value()));
    }
}
void verify_completion_transfer(const completion_host_buffer& source, const completion_host_buffer& destination,
                                CUdeviceptr device, std::span<const unsigned char> input, size_t bytes) {
    source.expect(input, "completed upload source preservation");
    std::vector<unsigned char> expected(completion_host_buffer::size, 0xd3);
    std::copy_n(input.begin() + 17, bytes, expected.begin() + 31);
    destination.expect(expected, "completed readback payload and host guards");
    std::vector<unsigned char> actual(completion_host_buffer::size + 32, 0xe7), expected_device = actual;
    std::fill_n(expected_device.begin() + 16, completion_host_buffer::size, 0xa9);
    std::copy_n(input.begin() + 17, bytes, expected_device.begin() + 16 + 23);
    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data() + 16, device, completion_host_buffer::size));
    require(actual == expected_device, "completed device bytes and readback guards");
}
void stream_completion_pending() {
    for (bool runtime : {false, true}) for (unsigned flags : {0u, unsigned(CU_STREAM_NON_BLOCKING)})
    for (bool runtime_sync : {false, true}) {
        completion_host_buffer source, destination;
        device_buffer device(completion_host_buffer::size);
        completion_stream stream(runtime, flags);
        unsigned generation = 0;
        for (size_t bytes : {size_t(0), size_t(1), size_t(257), size_t(481)}) {
            std::vector<unsigned char> input(completion_host_buffer::size), untouched(input.size(), 0xd3);
            for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<unsigned char>(i * 73 + generation * 31);
            ++generation;
            std::memcpy(source.value, input.data(), input.size());
            std::memcpy(destination.value, untouched.data(), untouched.size());
            CONFORMANCE_CUDA(cuMemsetD8(device.value, 0xa9, completion_host_buffer::size));
            stream.expect_ready(true);
            completion_gate gate;
            completion_gate_cleanup cleanup{gate};
            if (runtime) COMPLETION_RUNTIME(cudaLaunchHostFunc(stream.runtime_value(), completion_gate::call, &gate));
            else CONFORMANCE_CUDA(cuLaunchHostFunc(stream.value, completion_gate::call, &gate));
            gate.await();
            completion_transfer(stream, source, destination, device.value, bytes);
            stream.expect_ready(false);
            destination.expect(untouched, "pending transfer retains initialized destination");
            gate.release();
            stream.synchronize(runtime_sync);
            require(!gate.expired, "stream completion gate deadline");
            stream.expect_ready(true);
            verify_completion_transfer(source, destination, device.value, input, bytes);
        }
        stream.destroy();
    }
}
void stream_completion_retirement() {
    for (bool runtime : {false, true}) for (unsigned flags : {0u, unsigned(CU_STREAM_NON_BLOCKING)}) {
        completion_host_buffer source, destination;
        device_buffer device(completion_host_buffer::size);
        completion_stream stream(runtime, flags);
        completion_event completed;
        std::vector<unsigned char> input(completion_host_buffer::size), untouched(input.size(), 0xd3);
        for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<unsigned char>((i * 73 + 19) ^ (i >> 4));
        std::memcpy(source.value, input.data(), input.size());
        std::memcpy(destination.value, untouched.data(), untouched.size());
        CONFORMANCE_CUDA(cuMemsetD8(device.value, 0xa9, completion_host_buffer::size));
        completion_gate gate;
        completion_gate_cleanup cleanup{gate};
        CONFORMANCE_CUDA(cuLaunchHostFunc(stream.value, completion_gate::call, &gate)); gate.await();
        completion_transfer(stream, source, destination, device.value, 257);
        CONFORMANCE_CUDA(cuEventRecord(completed.value, stream.value));
        stream.expect_ready(false);
        stream.destroy();
        // A blocking destroy would wait for gate expiry, violating retirement.
        {
            std::lock_guard lock(gate.mutex);
            require(!gate.expired && !gate.released, "destroy returns while accepted stream work is unfinished");
        }
        require(cuEventQuery(completed.value) == CUDA_ERROR_NOT_READY, "retired stream completion marker remains pending");
        destination.expect(untouched, "retired pending stream retains destination bytes");
        gate.release();
        if (runtime) CONFORMANCE_CUDA(cuEventSynchronize(completed.value));
        else COMPLETION_RUNTIME(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(completed.value)));
        require(!gate.expired, "retired stream callback state remains alive through completion");
        CONFORMANCE_CUDA(cuEventQuery(completed.value));
        verify_completion_transfer(source, destination, device.value, input, 257);
    }
}
void stream_completion_validation() {
    completion_stream survivor(false, CU_STREAM_NON_BLOCKING);
    require(cuStreamCreate(nullptr, 0) == CUDA_ERROR_INVALID_VALUE, "null Driver stream output");
    require(cudaStreamCreateWithFlags(nullptr, 0) == cudaErrorInvalidValue, "null Runtime stream output");
    for (unsigned flags : {2u, 0x80000000u}) {
        CUstream driver{};
        const auto result = cuStreamCreate(&driver, flags);
        if (result == CUDA_SUCCESS) cuStreamDestroy(driver);
        require(result == CUDA_ERROR_INVALID_VALUE, "invalid Driver stream flags");
        cudaStream_t runtime{};
        const auto status = cudaStreamCreateWithFlags(&runtime, flags);
        if (status == cudaSuccess) cudaStreamDestroy(runtime);
        require(status == cudaErrorInvalidValue, "invalid Runtime stream flags");
    }
    for (CUstream implicit : {static_cast<CUstream>(nullptr), CU_STREAM_LEGACY, CU_STREAM_PER_THREAD}) {
        require(cuStreamDestroy(implicit) == CUDA_ERROR_INVALID_HANDLE, "Driver implicit stream cannot be destroyed");
        require(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(implicit)) == cudaErrorInvalidResourceHandle,
            "Runtime implicit stream cannot be destroyed");
    }
    survivor.expect_ready(true);
    survivor.synchronize(false); survivor.synchronize(true);
    survivor.destroy();
}
std::string_view stream_completion_case = "all";
void execute_stream_completion() {
    const std::array<std::pair<std::string_view, void (*)()>, 4> cases{{
        {"empty", stream_completion_empty}, {"pending", stream_completion_pending},
        {"retirement", stream_completion_retirement}, {"validation", stream_completion_validation}}};
    bool found = false;
    for (const auto& [name, execute] : cases) if (stream_completion_case == "all" || stream_completion_case == name) {
        found = true; std::cout << "stream_completion=" << name << std::endl; execute();
    }
    require(found, "unknown stream_completion case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::stream_completion_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_stream_completion);
}
