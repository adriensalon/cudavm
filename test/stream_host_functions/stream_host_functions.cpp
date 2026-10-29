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
void check_host_function_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define HOST_FUNCTION_RUNTIME(call) ::cudavm::check_host_function_runtime((call), #call)
struct host_function_stream {
    CUstream value{};
    bool runtime;
    explicit host_function_stream(bool use_runtime) : runtime(use_runtime) {
        if (runtime) {
            cudaStream_t stream{};
            HOST_FUNCTION_RUNTIME(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            value = reinterpret_cast<CUstream>(stream);
        } else CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING));
    }
    ~host_function_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    cudaStream_t runtime_value() const { return reinterpret_cast<cudaStream_t>(value); }
    void destroy() {
        if (runtime) CONFORMANCE_CUDA(cuStreamDestroy(value));
        else HOST_FUNCTION_RUNTIME(cudaStreamDestroy(runtime_value()));
        value = nullptr;
    }
};
struct host_function_buffer {
    static constexpr size_t size = 544;
    void* value{};
    host_function_buffer() { CONFORMANCE_CUDA(cuMemAllocHost(&value, size)); }
    ~host_function_buffer() { if (value) cuMemFreeHost(value); }
    unsigned char* data() const { return static_cast<unsigned char*>(value); }
};
struct host_function_state {
    host_function_buffer source, first, second;
    device_buffer device{512};
    std::vector<unsigned char> expected_first, expected_second;
    std::array<unsigned, 4> trace{};
    unsigned next = 0;
    bool valid = true;
    CUstream stream{};
    struct argument { host_function_state* owner; unsigned stage; uint64_t token = 0x0123456789abcdef; };
    std::array<argument, 4> arguments{{{this, 0}, {this, 1}, {this, 2}, {this, 3}}};
    void initialize(CUstream target, unsigned generation) {
        stream = target; trace.fill(~0u); next = 0; valid = true;
        expected_first.assign(host_function_buffer::size, 0xd3);
        for (size_t i = 0; i < 512; ++i) expected_first[16 + i] = static_cast<unsigned char>((i * 73 + generation * 31) ^ (i >> 4));
        expected_second = expected_first;
        for (size_t i = 17; i < 274; ++i) expected_second[16 + i] ^= 0x5b;
        std::memcpy(source.value, expected_first.data(), expected_first.size());
        std::memset(first.value, 0xd3, host_function_buffer::size);
        std::memset(second.value, 0xd3, host_function_buffer::size);
    }
    static void observe(void* pointer, CUstream reported, bool callback_success) noexcept {
        auto& argument = *static_cast<host_function_state::argument*>(pointer);
        auto& state = *argument.owner;
        state.valid &= argument.token == 0x0123456789abcdef && reported == state.stream && callback_success;
        state.valid &= state.next == argument.stage && argument.stage < state.trace.size();
        if (state.next < state.trace.size()) state.trace[state.next] = argument.stage;
        ++state.next;
        if (argument.stage == 0) {
            state.valid &= std::equal(state.expected_first.begin(), state.expected_first.end(), state.first.data());
            state.valid &= std::equal(state.expected_first.begin(), state.expected_first.end(), state.source.data());
            for (size_t i = 17; i < 274; ++i) state.source.data()[16 + i] ^= 0x5b;
        } else {
            state.valid &= std::equal(state.expected_second.begin(), state.expected_second.end(), state.second.data());
            state.valid &= std::equal(state.expected_second.begin(), state.expected_second.end(), state.source.data());
        }
    }
    static void CUDA_CB driver_host(void* data) {
        auto& argument = *static_cast<host_function_state::argument*>(data);
        observe(data, argument.owner->stream, true);
    }
    static void CUDART_CB runtime_host(void* data) {
        auto& argument = *static_cast<host_function_state::argument*>(data);
        observe(data, argument.owner->stream, true);
    }
    static void CUDA_CB driver_callback(CUstream stream, CUresult status, void* data) { observe(data, stream, status == CUDA_SUCCESS); }
    static void CUDART_CB runtime_callback(cudaStream_t stream, cudaError_t status, void* data) { observe(data, reinterpret_cast<CUstream>(stream), status == cudaSuccess); }
    void enqueue() {
        CONFORMANCE_CUDA(cuMemcpyHtoDAsync(device.value, source.data() + 16, 512, stream));
        HOST_FUNCTION_RUNTIME(cudaMemcpyAsync(first.data() + 16, reinterpret_cast<void*>(static_cast<uintptr_t>(device.value)),
            512, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream)));
        CONFORMANCE_CUDA(cuLaunchHostFunc(stream, driver_host, &arguments[0]));
        HOST_FUNCTION_RUNTIME(cudaMemcpyAsync(reinterpret_cast<void*>(static_cast<uintptr_t>(device.value)), source.data() + 16,
            512, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)));
        CONFORMANCE_CUDA(cuMemcpyDtoHAsync(second.data() + 16, device.value, 512, stream));
        HOST_FUNCTION_RUNTIME(cudaStreamAddCallback(reinterpret_cast<cudaStream_t>(stream), runtime_callback, &arguments[1], 0));
        HOST_FUNCTION_RUNTIME(cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), runtime_host, &arguments[2]));
        CONFORMANCE_CUDA(cuStreamAddCallback(stream, driver_callback, &arguments[3], 0));
    }
    void verify() const {
        require(valid && next == 4 && trace == std::array<unsigned, 4>{0, 1, 2, 3}, "host functions/callbacks execute once in order with retained data and success status");
        require(std::equal(expected_first.begin(), expected_first.end(), first.data()), "first completed transfer and guards remain intact");
        require(std::equal(expected_second.begin(), expected_second.end(), second.data()), "host publication is visible to subsequent transfer with guards");
        require(std::equal(expected_second.begin(), expected_second.end(), source.data()), "host producer bytes and source guards");
        std::vector<unsigned char> actual(544, 0xd3);
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data() + 16, device.value, 512));
        require(actual == expected_second, "final device payload and readback guards");
    }
};
void stream_host_functions_ordering() {
    for (bool runtime : {false, true}) {
        host_function_state state;
        host_function_stream stream(runtime);
        struct drain { ~drain() { cuCtxSynchronize(); } } cleanup;
        for (unsigned generation = 0; generation < 3; ++generation) {
            state.initialize(stream.value, generation); state.enqueue();
            if (runtime) CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
            else HOST_FUNCTION_RUNTIME(cudaStreamSynchronize(stream.runtime_value()));
            state.verify();
        }
        stream.destroy();
    }
}
struct host_function_gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* data) {
        auto& gate = *static_cast<host_function_gate*>(data);
        std::unique_lock lock(gate.mutex);
        gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(15), [&] { return gate.released; })) gate.expired = true;
    }
    void await() {
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "host-function gate entry deadline");
    }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
};
void stream_host_functions_retirement() {
    for (bool runtime : {false, true}) {
        host_function_state state;
        host_function_stream stream(runtime);
        state.initialize(stream.value, 19);
        host_function_gate gate;
        struct release_and_drain {
            host_function_gate& gate;
            ~release_and_drain() { gate.release(); cuCtxSynchronize(); }
        } cleanup{gate};
        CONFORMANCE_CUDA(cuLaunchHostFunc(stream.value, host_function_gate::call, &gate)); gate.await();
        state.enqueue();
        stream.destroy();
        { std::lock_guard lock(gate.mutex); require(!gate.expired && !gate.released, "stream retirement preserves pending host functions"); }
        gate.release();
        CONFORMANCE_CUDA(cuCtxSynchronize());
        require(!gate.expired, "retained callback-state deadline");
        state.verify();
    }
}
void stream_host_functions_validation() {
    host_function_state state;
    host_function_stream stream(false);
    state.initialize(stream.value, 7);
    require(cuLaunchHostFunc(stream.value, nullptr, nullptr) == CUDA_ERROR_INVALID_VALUE, "null Driver host function");
    require(cudaLaunchHostFunc(stream.runtime_value(), nullptr, nullptr) == cudaErrorInvalidValue, "null Runtime host function");
    require(cuStreamAddCallback(stream.value, nullptr, nullptr, 0) == CUDA_ERROR_INVALID_VALUE, "null Driver callback");
    require(cudaStreamAddCallback(stream.runtime_value(), nullptr, nullptr, 0) == cudaErrorInvalidValue, "null Runtime callback");
    require(cuStreamAddCallback(stream.value, host_function_state::driver_callback, &state.arguments[0], 1) == CUDA_ERROR_INVALID_VALUE,
        "unsupported Driver callback flags");
    require(cudaStreamAddCallback(stream.runtime_value(), host_function_state::runtime_callback, &state.arguments[0], 1) == cudaErrorInvalidValue,
        "unsupported Runtime callback flags");
    struct drain { ~drain() { cuCtxSynchronize(); } } cleanup;
    state.enqueue(); CONFORMANCE_CUDA(cuStreamSynchronize(stream.value)); state.verify();
    stream.destroy();
}
std::string_view stream_host_functions_case = "all";
void execute_stream_host_functions() {
    const std::array<std::pair<std::string_view, void (*)()>, 3> cases{{
        {"ordering", stream_host_functions_ordering}, {"retirement", stream_host_functions_retirement},
        {"validation", stream_host_functions_validation}}};
    bool found = false;
    for (const auto& [name, execute] : cases) if (stream_host_functions_case == "all" || stream_host_functions_case == name) {
        found = true; std::cout << "stream_host_functions=" << name << std::endl; execute();
    }
    require(found, "unknown stream_host_functions case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::stream_host_functions_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_stream_host_functions);
}
