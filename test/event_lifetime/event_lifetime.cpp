#include "test_framework.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>

namespace cudavm {
void check_event_lifetime_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define EVENT_LIFETIME_RUNTIME(call) ::cudavm::check_event_lifetime_runtime((call), #call)
struct event_lifetime_gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* argument) {
        auto& gate = *static_cast<event_lifetime_gate*>(argument);
        std::unique_lock lock(gate.mutex); gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(15), [&] { return gate.released; })) gate.expired = true;
    }
    void await() {
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "event lifetime gate entry deadline");
    }
    void held() {
        std::lock_guard lock(mutex);
        require(entered && !released && !expired, "pending event retirement/completion does not wait for held producer");
    }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
    void completed() { std::lock_guard lock(mutex); require(!expired, "event lifetime gate completion deadline"); }
};
struct event_lifetime_cleanup {
    event_lifetime_gate& first;
    event_lifetime_gate& second;
    ~event_lifetime_cleanup() { first.release(); second.release(); cuCtxSynchronize(); }
};
struct event_lifetime_stream {
    CUstream value{};
    explicit event_lifetime_stream(bool runtime) {
        if (runtime) {
            cudaStream_t stream{}; EVENT_LIFETIME_RUNTIME(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            value = reinterpret_cast<CUstream>(stream);
        } else TEST_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING));
    }
    ~event_lifetime_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    cudaStream_t cast() const { return reinterpret_cast<cudaStream_t>(value); }
    void gate(event_lifetime_gate& state, bool runtime, bool await_entry = true) const {
        if (runtime) EVENT_LIFETIME_RUNTIME(cudaLaunchHostFunc(cast(), event_lifetime_gate::call, &state));
        else TEST_CUDA(cuLaunchHostFunc(value, event_lifetime_gate::call, &state));
        if (await_entry) state.await();
    }
    void wait(CUevent event, bool runtime) const {
        if (runtime) EVENT_LIFETIME_RUNTIME(cudaStreamWaitEvent(cast(), reinterpret_cast<cudaEvent_t>(event), 0));
        else TEST_CUDA(cuStreamWaitEvent(value, event, 0));
    }
    void synchronize(bool runtime) const {
        if (runtime) EVENT_LIFETIME_RUNTIME(cudaStreamSynchronize(cast()));
        else TEST_CUDA(cuStreamSynchronize(value));
    }
    void destroy(bool runtime) {
        if (runtime) EVENT_LIFETIME_RUNTIME(cudaStreamDestroy(cast()));
        else TEST_CUDA(cuStreamDestroy(value));
        value = nullptr;
    }
};
struct event_lifetime_event {
    CUevent value{};
    event_lifetime_event(bool runtime, unsigned flags) {
        if (runtime) {
            cudaEvent_t event{}; EVENT_LIFETIME_RUNTIME(cudaEventCreateWithFlags(&event, flags));
            value = reinterpret_cast<CUevent>(event);
        } else TEST_CUDA(cuEventCreate(&value, flags));
    }
    ~event_lifetime_event() { if (value) cuEventDestroy(value); }
    cudaEvent_t cast() const { return reinterpret_cast<cudaEvent_t>(value); }
    void record(const event_lifetime_stream& stream, bool runtime) const {
        if (runtime) EVENT_LIFETIME_RUNTIME(cudaEventRecord(cast(), stream.cast()));
        else TEST_CUDA(cuEventRecord(value, stream.value));
    }
    void destroy(bool runtime) {
        if (runtime) EVENT_LIFETIME_RUNTIME(cudaEventDestroy(cast()));
        else TEST_CUDA(cuEventDestroy(value));
        value = nullptr;
    }
};
struct event_lifetime_source {
    static constexpr size_t size = 512;
    device_buffer device{size};
    std::vector<unsigned char> expected = std::vector<unsigned char>(size);
    explicit event_lifetime_source(unsigned seed) {
        for (size_t i = 0; i < size; ++i) expected[i] = static_cast<unsigned char>(seed + i * 73 + (i >> 3));
        TEST_CUDA(cuMemcpyHtoD(device.value, expected.data(), size));
    }
    void publish(const event_lifetime_stream& stream, unsigned char value, bool runtime) {
        if (runtime) EVENT_LIFETIME_RUNTIME(cudaMemsetAsync(reinterpret_cast<void*>(static_cast<uintptr_t>(device.value + 17)), value, 257, stream.cast()));
        else TEST_CUDA(cuMemsetD8Async(device.value + 17, value, 257, stream.value));
        std::fill_n(expected.begin() + 17, 257, value);
    }
    void verify() const {
        std::vector<unsigned char> actual(size);
        TEST_CUDA(cuMemcpyDtoH(actual.data(), device.value, size));
        require(actual == expected, "event producer complete source bytes and guards");
    }
};
struct event_lifetime_readback {
    static constexpr size_t size = event_lifetime_source::size;
    device_buffer device{size};
    void* host{};
    event_lifetime_readback() {
        TEST_CUDA(cuMemsetD8(device.value, 0xa9, size));
        TEST_CUDA(cuMemAllocHost(&host, size + 32)); std::memset(host, 0xd3, size + 32);
    }
    ~event_lifetime_readback() { if (host) cuMemFreeHost(host); }
    void consume(const event_lifetime_stream& stream, const event_lifetime_source& source, size_t from, size_t to, size_t bytes, bool runtime) {
        if (runtime) EVENT_LIFETIME_RUNTIME(cudaMemcpyAsync(reinterpret_cast<void*>(static_cast<uintptr_t>(device.value + to)),
            reinterpret_cast<const void*>(static_cast<uintptr_t>(source.device.value + from)), bytes, cudaMemcpyDeviceToDevice, stream.cast()));
        else TEST_CUDA(cuMemcpyDtoDAsync(device.value + to, source.device.value + from, bytes, stream.value));
        TEST_CUDA(cuMemcpyDtoHAsync(static_cast<unsigned char*>(host) + 16, device.value, size, stream.value));
    }
    void untouched() const {
        const auto* data = static_cast<const unsigned char*>(host);
        require(std::all_of(data, data + size + 32, [](auto byte) { return byte == 0xd3; }), "pending event leaves entire readback untouched");
    }
    void verify(const event_lifetime_source& source, size_t from, size_t to, size_t bytes, bool check_device = true) const {
        std::vector<unsigned char> expected(size + 32, 0xd3);
        std::fill_n(expected.begin() + 16, size, 0xa9);
        std::copy_n(source.expected.begin() + from, bytes, expected.begin() + 16 + to);
        require(std::equal(expected.begin(), expected.end(), static_cast<const unsigned char*>(host)), "retained event exact consumer bytes and pinned guards");
        if (!check_device) return;
        std::vector<unsigned char> actual(size);
        TEST_CUDA(cuMemcpyDtoH(actual.data(), device.value, size));
        require(std::equal(actual.begin(), actual.end(), expected.begin() + 16), "retained event complete destination allocation");
    }
};
void event_lifetime_fanout() {
    for (bool runtime : {false, true}) for (unsigned flags : {0u, 1u, 2u, 3u}) {
        event_lifetime_source source(flags + 19);
        std::array<event_lifetime_readback, 4> readbacks;
        event_lifetime_stream producer(runtime);
        std::array<std::unique_ptr<event_lifetime_stream>, 4> consumers;
        for (size_t i = 0; i < consumers.size(); ++i) consumers[i] = std::make_unique<event_lifetime_stream>(i % 2 != 0);
        event_lifetime_event event(runtime, flags);
        event_lifetime_gate gate, unused;
        event_lifetime_cleanup cleanup{gate, unused};
        producer.gate(gate, runtime); source.publish(producer, 0x5b, !runtime); event.record(producer, runtime);
        for (size_t i = 0; i < consumers.size(); ++i) {
            consumers[i]->wait(event.value, i % 2 != 0);
            readbacks[i].consume(*consumers[i], source, 17 + i, 23 + i * 3, 241 - i * 37, i % 2 == 0);
        }
        require(cuEventQuery(event.value) == CUDA_ERROR_NOT_READY, "event fanout record remains pending");
        event.destroy(!runtime); gate.held(); producer.destroy(runtime); gate.held();
        for (size_t i = 0; i < consumers.size(); ++i) {
            require(cuStreamQuery(consumers[i]->value) == CUDA_ERROR_NOT_READY, "retired event retains each unfinished consumer");
            readbacks[i].untouched();
        }
        gate.release();
        for (size_t i = 0; i < consumers.size(); ++i) {
            consumers[i]->synchronize(i % 2 == 0);
            readbacks[i].verify(source, 17 + i, 23 + i * 3, 241 - i * 37);
        }
        source.verify(); gate.completed();
    }
}
void event_lifetime_generations() {
    for (bool runtime : {false, true}) for (unsigned flags : {0u, 1u, 2u, 3u}) {
        event_lifetime_source old_source(29), new_source(73);
        event_lifetime_readback old_result, new_result;
        event_lifetime_stream old_producer(runtime), old_consumer(!runtime), new_consumer(runtime);
        event_lifetime_event event(runtime, flags);
        event_lifetime_gate old_gate, new_gate;
        event_lifetime_cleanup cleanup{old_gate, new_gate};
        old_producer.gate(old_gate, runtime); old_source.publish(old_producer, 0x39, runtime);
        event.record(old_producer, !runtime); old_consumer.wait(event.value, runtime);
        old_result.consume(old_consumer, old_source, 17, 23, 257, !runtime);
        // The second callback follows the first generation's GPU/readback
        // work. Its entry proves those bytes complete without demanding
        // progress of unrelated CUDA work while a callback is held.
        old_consumer.gate(new_gate, !runtime, false); new_source.publish(old_consumer, 0x91, !runtime);
        event.record(old_consumer, runtime); new_consumer.wait(event.value, !runtime);
        new_result.consume(new_consumer, new_source, 19, 37, 229, runtime);
        event.destroy(!runtime); old_gate.held(); old_result.untouched(); new_result.untouched();
        old_gate.release(); new_gate.await(); new_gate.held();
        old_result.verify(old_source, 17, 23, 257, false); new_result.untouched();
        new_gate.release(); new_consumer.synchronize(!runtime); old_consumer.synchronize(runtime);
        old_result.verify(old_source, 17, 23, 257); new_result.verify(new_source, 19, 37, 229);
        old_source.verify(); new_source.verify(); old_gate.completed(); new_gate.completed();
    }
}
void event_lifetime_stream_retirement() {
    for (bool runtime : {false, true}) for (unsigned flags : {0u, 1u, 2u, 3u}) {
        event_lifetime_source source(137); event_lifetime_readback result;
        event_lifetime_stream producer(runtime); event_lifetime_event event(!runtime, flags);
        event_lifetime_gate gate, unused; event_lifetime_cleanup cleanup{gate, unused};
        producer.gate(gate, runtime); source.publish(producer, 0x6d, !runtime);
        result.consume(producer, source, 17, 23, 257, runtime); event.record(producer, !runtime);
        producer.destroy(!runtime); gate.held(); result.untouched();
        require(cuEventQuery(event.value) == CUDA_ERROR_NOT_READY && cudaEventQuery(event.cast()) == cudaErrorNotReady, "live event retains retired producer completion");
        gate.release();
        if (runtime) TEST_CUDA(cuEventSynchronize(event.value));
        else EVENT_LIFETIME_RUNTIME(cudaEventSynchronize(event.cast()));
        TEST_CUDA(cuEventQuery(event.value)); EVENT_LIFETIME_RUNTIME(cudaEventQuery(event.cast()));
        result.verify(source, 17, 23, 257); source.verify(); gate.completed(); event.destroy(runtime);
    }
}
std::string_view event_lifetime_case;
void execute_event_lifetime() {
    std::cout << "event_lifetime=" << event_lifetime_case << std::endl;
    if (event_lifetime_case == "fanout") event_lifetime_fanout();
    else if (event_lifetime_case == "generations") event_lifetime_generations();
    else if (event_lifetime_case == "stream_retirement") event_lifetime_stream_retirement();
    else throw std::runtime_error("unknown event_lifetime case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::event_lifetime_case = argv[2]; --argc; }
    return cudavm::run_test(argc, argv, cudavm::execute_event_lifetime);
}
