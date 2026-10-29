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
void check_dependency_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define DEPENDENCY_RUNTIME(call) ::cudavm::check_dependency_runtime((call), #call)
struct dependency_stream {
    CUstream value{};
    explicit dependency_stream(bool runtime) {
        if (runtime) {
            cudaStream_t stream{};
            DEPENDENCY_RUNTIME(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            value = reinterpret_cast<CUstream>(stream);
        } else CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING));
    }
    ~dependency_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    void wait(CUevent event, bool runtime) {
        if (runtime) DEPENDENCY_RUNTIME(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(value), reinterpret_cast<cudaEvent_t>(event), 0));
        else CONFORMANCE_CUDA(cuStreamWaitEvent(value, event, 0));
    }
    void synchronize(bool runtime) {
        if (runtime) DEPENDENCY_RUNTIME(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(value)));
        else CONFORMANCE_CUDA(cuStreamSynchronize(value));
    }
};
struct dependency_event {
    CUevent value{};
    bool runtime;
    explicit dependency_event(bool use_runtime) : runtime(use_runtime) {
        if (runtime) {
            cudaEvent_t event{};
            DEPENDENCY_RUNTIME(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            value = reinterpret_cast<CUevent>(event);
        } else CONFORMANCE_CUDA(cuEventCreate(&value, CU_EVENT_DISABLE_TIMING));
    }
    ~dependency_event() { if (value) cuEventDestroy(value); }
    void record(CUstream stream) {
        if (runtime) DEPENDENCY_RUNTIME(cudaEventRecord(reinterpret_cast<cudaEvent_t>(value), reinterpret_cast<cudaStream_t>(stream)));
        else CONFORMANCE_CUDA(cuEventRecord(value, stream));
    }
    void destroy() {
        if (runtime) CONFORMANCE_CUDA(cuEventDestroy(value));
        else DEPENDENCY_RUNTIME(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(value)));
        value = nullptr;
    }
};
struct dependency_storage {
    static constexpr size_t size = 512;
    device_buffer source{size}, destination{size};
    void* readback{};
    std::vector<unsigned char> input = std::vector<unsigned char>(size);
    dependency_storage() {
        CONFORMANCE_CUDA(cuMemAllocHost(&readback, size + 32));
        std::memset(readback, 0xd3, size + 32);
        for (size_t i = 0; i < size; ++i) input[i] = static_cast<unsigned char>((i * 73 + 19) ^ (i >> 4));
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), size));
        CONFORMANCE_CUDA(cuMemsetD8(destination.value, 0xa9, size));
    }
    ~dependency_storage() { if (readback) cuMemFreeHost(readback); }
    void produce(CUstream stream, size_t bytes) { CONFORMANCE_CUDA(cuMemsetD8Async(source.value + 17, 0x5b, bytes, stream)); }
    void consume(CUstream stream, size_t bytes, bool runtime) {
        if (runtime) DEPENDENCY_RUNTIME(cudaMemcpyAsync(reinterpret_cast<void*>(static_cast<uintptr_t>(destination.value + 23)),
            reinterpret_cast<void*>(static_cast<uintptr_t>(source.value + 17)), bytes, cudaMemcpyDeviceToDevice,
            reinterpret_cast<cudaStream_t>(stream)));
        else CONFORMANCE_CUDA(cuMemcpyDtoDAsync(destination.value + 23, source.value + 17, bytes, stream));
        CONFORMANCE_CUDA(cuMemcpyDtoHAsync(static_cast<unsigned char*>(readback) + 16, destination.value, size, stream));
    }
    void expect_readback(size_t bytes, bool produced) const {
        auto source_expected = input;
        if (produced) std::fill_n(source_expected.begin() + 17, bytes, 0x5b);
        std::vector<unsigned char> expected(size + 32, 0xd3);
        std::fill_n(expected.begin() + 16, size, 0xa9);
        std::copy_n(source_expected.begin() + 17, bytes, expected.begin() + 16 + 23);
        require(std::equal(expected.begin(), expected.end(), static_cast<unsigned char*>(readback)), "event-dependent payload and host/device guards");
    }
    void expect_untouched() const {
        const auto* bytes = static_cast<unsigned char*>(readback);
        require(std::all_of(bytes, bytes + size + 32, [](auto byte) { return byte == 0xd3; }), "pending event consumer leaves readback untouched");
    }
    void verify(size_t bytes, bool produced) const {
        expect_readback(bytes, produced);
        auto expected = input;
        if (produced) std::fill_n(expected.begin() + 17, bytes, 0x5b);
        std::vector<unsigned char> actual(size);
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), source.value, size));
        require(actual == expected, "event producer bytes and untouched source neighbors");
    }
};
struct dependency_gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* argument) {
        auto& gate = *static_cast<dependency_gate*>(argument);
        std::unique_lock lock(gate.mutex);
        gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(15), [&] { return gate.released; })) gate.expired = true;
    }
    void await() {
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "event dependency gate entry deadline");
    }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
};
struct dependency_cleanup {
    dependency_gate& first;
    dependency_gate& second;
    ~dependency_cleanup() { first.release(); second.release(); cuCtxSynchronize(); }
};
void event_dependencies_unrecorded() {
    for (bool runtime : {false, true}) for (bool runtime_wait : {false, true}) {
        dependency_storage storage;
        dependency_stream stream(runtime);
        dependency_event event(runtime);
        stream.wait(event.value, runtime_wait);
        storage.consume(stream.value, 257, !runtime);
        stream.synchronize(runtime);
        storage.verify(257, false);
        CONFORMANCE_CUDA(cuEventQuery(event.value));
        event.destroy();
    }
}
void event_dependencies_ordered(bool cross_stream) {
    for (bool runtime : {false, true}) for (bool runtime_wait : {false, true})
    for (size_t bytes : {size_t(0), size_t(1), size_t(257), size_t(489)}) {
        dependency_storage storage;
        dependency_stream producer(runtime), consumer(!runtime);
        dependency_event event(runtime);
        dependency_gate gate, unused;
        dependency_cleanup cleanup{gate, unused};
        if (cross_stream) { CONFORMANCE_CUDA(cuLaunchHostFunc(producer.value, dependency_gate::call, &gate)); gate.await(); }
        storage.produce(producer.value, bytes);
        event.record(producer.value);
        const auto target = cross_stream ? consumer.value : producer.value;
        if (runtime_wait) DEPENDENCY_RUNTIME(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(target), reinterpret_cast<cudaEvent_t>(event.value), 0));
        else CONFORMANCE_CUDA(cuStreamWaitEvent(target, event.value, 0));
        storage.consume(target, bytes, !runtime);
        if (cross_stream) {
            require(cuStreamQuery(producer.value) == CUDA_ERROR_NOT_READY && cuStreamQuery(consumer.value) == CUDA_ERROR_NOT_READY,
                "event-dependent producer and consumer are pending");
            storage.expect_untouched();
            gate.release();
            consumer.synchronize(runtime);
        } else producer.synchronize(!runtime);
        require(!gate.expired, "ordered event gate deadline");
        storage.verify(bytes, true);
        event.destroy();
    }
}
void event_dependencies_same_stream() { event_dependencies_ordered(false); }
void event_dependencies_cross_stream() { event_dependencies_ordered(true); }
void event_dependencies_generation() {
    for (bool runtime : {false, true}) for (bool runtime_wait : {false, true}) {
        dependency_storage storage;
        dependency_stream producer(runtime), consumer(!runtime);
        dependency_event event(runtime), consumed(!runtime);
        dependency_gate first, second;
        dependency_cleanup cleanup{first, second};
        CONFORMANCE_CUDA(cuLaunchHostFunc(producer.value, dependency_gate::call, &first)); first.await();
        storage.produce(producer.value, 257); event.record(producer.value);
        consumer.wait(event.value, runtime_wait);
        storage.consume(consumer.value, 257, runtime);
        consumed.record(consumer.value);
        CONFORMANCE_CUDA(cuLaunchHostFunc(consumer.value, dependency_gate::call, &second));
        event.record(consumer.value);
        first.release();
        second.await();
        CONFORMANCE_CUDA(cuEventQuery(consumed.value));
        require(cuEventQuery(event.value) == CUDA_ERROR_NOT_READY, "accepted wait uses old generation while newest record remains pending");
        storage.expect_readback(257, true);
        second.release();
        CONFORMANCE_CUDA(cuEventSynchronize(event.value));
        require(!first.expired && !second.expired, "wait generation gate deadlines");
        storage.verify(257, true);
        event.destroy(); consumed.destroy();
    }
}
void event_dependencies_retirement() {
    for (bool runtime : {false, true}) for (bool runtime_wait : {false, true}) {
        dependency_storage storage;
        dependency_stream producer(runtime), consumer(!runtime);
        dependency_event event(runtime);
        dependency_gate gate, unused;
        dependency_cleanup cleanup{gate, unused};
        CONFORMANCE_CUDA(cuLaunchHostFunc(producer.value, dependency_gate::call, &gate)); gate.await();
        storage.produce(producer.value, 257); event.record(producer.value);
        consumer.wait(event.value, runtime_wait);
        storage.consume(consumer.value, 257, !runtime);
        event.destroy();
        { std::lock_guard lock(gate.mutex); require(!gate.expired && !gate.released, "event destruction returns with accepted wait pending"); }
        storage.expect_untouched();
        gate.release(); consumer.synchronize(runtime);
        require(!gate.expired, "retired event wait gate deadline");
        storage.verify(257, true);
    }
}
void event_dependencies_validation() {
    dependency_stream stream(false);
    dependency_event event(false);
    require(cuStreamWaitEvent(stream.value, nullptr, 0) == CUDA_ERROR_INVALID_HANDLE, "null Driver wait event");
    require(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream.value), nullptr, 0) == cudaErrorInvalidResourceHandle,
        "null Runtime wait event");
    for (unsigned flags : {2u, 0x80000000u}) {
        require(cuStreamWaitEvent(stream.value, event.value, flags) == CUDA_ERROR_INVALID_VALUE, "invalid Driver wait flags");
        require(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream.value), reinterpret_cast<cudaEvent_t>(event.value), flags)
            == cudaErrorInvalidValue, "invalid Runtime wait flags");
    }
    stream.wait(event.value, true); stream.synchronize(false);
    CONFORMANCE_CUDA(cuEventQuery(event.value)); event.destroy();
}
std::string_view event_dependencies_case = "all";
void execute_event_dependencies() {
    const std::array<std::pair<std::string_view, void (*)()>, 6> cases{{
        {"unrecorded", event_dependencies_unrecorded}, {"same_stream", event_dependencies_same_stream},
        {"cross_stream", event_dependencies_cross_stream}, {"generation", event_dependencies_generation},
        {"retirement", event_dependencies_retirement}, {"validation", event_dependencies_validation}}};
    bool found = false;
    for (const auto& [name, execute] : cases) if (event_dependencies_case == "all" || event_dependencies_case == name) {
        found = true; std::cout << "event_dependencies=" << name << std::endl; execute();
    }
    require(found, "unknown event_dependencies case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::event_dependencies_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_event_dependencies);
}
