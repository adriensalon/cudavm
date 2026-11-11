#include "test_framework.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>

namespace cudavm {
void check_default_stream_runtime(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
#define DEFAULT_STREAM_RUNTIME(call) ::cudavm::check_default_stream_runtime((call), #call)
template<class T> struct default_stream_guarded {
    uint64_t prefix = 0x123456789abcdef0; T value{}; uint64_t suffix = 0xfedcba9876543210;
    void check() const { require(prefix == 0x123456789abcdef0 && suffix == 0xfedcba9876543210, "default stream public output guards"); }
};
template<class T> T default_stream_procedure(const char* name, bool per_thread) {
    void* address{}; CUdriverProcAddressQueryResult status{};
    TEST_CUDA(cuGetProcAddress(name, &address, 12000, per_thread ? CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM : CU_GET_PROC_ADDRESS_LEGACY_STREAM, &status));
    require(address && status == CU_GET_PROC_ADDRESS_SUCCESS, "default-stream procedure mode lookup"); return reinterpret_cast<T>(address);
}
struct default_stream_route {
    using fill_type = CUresult (CUDAAPI*)(CUdeviceptr, unsigned char, size_t, CUstream);
    using read_type = CUresult (CUDAAPI*)(void*, CUdeviceptr, size_t, CUstream);
    using host_type = CUresult (CUDAAPI*)(CUstream, CUhostFn, void*);
    using query_type = CUresult (CUDAAPI*)(CUstream);
    using id_type = CUresult (CUDAAPI*)(CUstream, unsigned long long*);
    bool runtime, per_thread; fill_type fill_call; read_type read_call; host_type host_call; query_type query_call, synchronize_call; id_type id_call;
    default_stream_route(bool use_runtime, bool thread_mode) : runtime(use_runtime), per_thread(thread_mode),
        fill_call(default_stream_procedure<fill_type>("cuMemsetD8Async", thread_mode)), read_call(default_stream_procedure<read_type>("cuMemcpyDtoHAsync", thread_mode)),
        host_call(default_stream_procedure<host_type>("cuLaunchHostFunc", thread_mode)), query_call(default_stream_procedure<query_type>("cuStreamQuery", thread_mode)),
        synchronize_call(default_stream_procedure<query_type>("cuStreamSynchronize", thread_mode)), id_call(default_stream_procedure<id_type>("cuStreamGetId", thread_mode)) {}
    CUstream selected() const { return runtime ? (per_thread ? CU_STREAM_PER_THREAD : CU_STREAM_LEGACY) : nullptr; }
    void fill(CUstream stream, CUdeviceptr address, unsigned char byte, size_t count) const {
        if (runtime) DEFAULT_STREAM_RUNTIME(cudaMemsetAsync(reinterpret_cast<void*>(static_cast<uintptr_t>(address)), byte, count, reinterpret_cast<cudaStream_t>(stream)));
        else TEST_CUDA(fill_call(address, byte, count, stream));
    }
    void read(CUstream stream, void* host, CUdeviceptr address, size_t bytes) const {
        if (runtime) DEFAULT_STREAM_RUNTIME(cudaMemcpyAsync(host, reinterpret_cast<void*>(static_cast<uintptr_t>(address)), bytes, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream)));
        else TEST_CUDA(read_call(host, address, bytes, stream));
    }
    void hold(CUstream stream, CUhostFn function, void* data) const {
        if (runtime) DEFAULT_STREAM_RUNTIME(cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), function, data));
        else TEST_CUDA(host_call(stream, function, data));
    }
    CUresult query(CUstream stream) const { return runtime ? static_cast<CUresult>(cudaStreamQuery(reinterpret_cast<cudaStream_t>(stream))) : query_call(stream); }
    void synchronize(CUstream stream) const {
        if (runtime) DEFAULT_STREAM_RUNTIME(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream))); else TEST_CUDA(synchronize_call(stream));
    }
    unsigned long long id(CUstream stream) const {
        default_stream_guarded<unsigned long long> output;
        if (runtime) DEFAULT_STREAM_RUNTIME(cudaStreamGetId(reinterpret_cast<cudaStream_t>(stream), &output.value)); else TEST_CUDA(id_call(stream, &output.value));
        output.check(); return output.value;
    }
};
struct default_stream_explicit {
    CUstream value{}; explicit default_stream_explicit(bool nonblocking) { TEST_CUDA(cuStreamCreate(&value, nonblocking ? CU_STREAM_NON_BLOCKING : CU_STREAM_DEFAULT)); }
    ~default_stream_explicit() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
};
struct default_stream_event {
    CUevent value{}; default_stream_event() { TEST_CUDA(cuEventCreate(&value, CU_EVENT_DISABLE_TIMING)); }
    ~default_stream_event() { if (value) cuEventDestroy(value); }
    void complete() const {
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5); CUresult status;
        do { status = cuEventQuery(value); if (status != CUDA_ERROR_NOT_READY) break; std::this_thread::yield(); } while (std::chrono::steady_clock::now() < end);
        TEST_CUDA(status);
    }
};
struct default_stream_storage {
    static constexpr size_t bytes = 293, slot_bytes = 337, offset = 19;
    device_buffer first{bytes}, second{bytes}; void* host{};
    default_stream_storage() {
        TEST_CUDA(cuMemAllocHost(&host, 2 * slot_bytes));
        try { TEST_CUDA(cuMemsetD8(first.value, 0x91, bytes)); TEST_CUDA(cuMemsetD8(second.value, 0x91, bytes)); }
        catch (...) { cuMemFreeHost(host); host = nullptr; throw; }
        std::memset(host, 0xd3, 2 * slot_bytes);
    }
    ~default_stream_storage() { if (host) { cuCtxSynchronize(); cuMemFreeHost(host); } }
    void produce(const default_stream_route& route, CUstream stream, bool other) const {
        route.fill(stream, (other ? second.value : first.value) + (other ? 11 : 23), other ? 0x71 : 0x47, other ? 211 : 197);
        route.read(stream, static_cast<unsigned char*>(host) + (other ? slot_bytes : 0) + offset, other ? second.value : first.value, bytes);
    }
    void expect(bool other) const {
        std::vector<unsigned char> expected(slot_bytes, 0xd3); std::fill_n(expected.begin() + offset, bytes, 0x91);
        std::fill_n(expected.begin() + offset + (other ? 11 : 23), other ? 211 : 197, other ? 0x71 : 0x47);
        require(std::memcmp(static_cast<unsigned char*>(host) + (other ? slot_bytes : 0), expected.data(), expected.size()) == 0,
            "default stream payload, implicit dependency and complete device/host guards");
    }
};
struct default_stream_gate {
    std::mutex mutex; std::condition_variable changed; bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* argument) {
        auto& gate = *static_cast<default_stream_gate*>(argument); std::unique_lock lock(gate.mutex); gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(10), [&] { return gate.released; })) gate.expired = true;
    }
    void await() { std::unique_lock lock(mutex); require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "default stream gate entry"); }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
    void check() { std::lock_guard lock(mutex); require(entered && !expired, "default stream gate did not expire"); }
};
struct default_stream_gate_cleanup { default_stream_gate& gate; ~default_stream_gate_cleanup() { gate.release(); cuCtxSynchronize(); } };
struct default_stream_thread_cleanup {
    default_stream_gate& gate; std::thread& worker;
    ~default_stream_thread_cleanup() { gate.release(); if (worker.joinable()) worker.join(); cuCtxSynchronize(); }
};
void default_stream_aliases() {
    default_stream_route legacy(false, false), per_thread(false, true), runtime_legacy(true, false), runtime_per_thread(true, true);
    const auto a = legacy.id(nullptr), b = per_thread.id(nullptr);
    require(a == legacy.id(CU_STREAM_LEGACY) && a == runtime_legacy.id(nullptr) && a == runtime_legacy.id(CU_STREAM_LEGACY), "all legacy aliases share stream identity");
    require(b == per_thread.id(CU_STREAM_PER_THREAD) && b == legacy.id(CU_STREAM_PER_THREAD) && b == runtime_per_thread.id(CU_STREAM_PER_THREAD), "per-thread null mode and explicit aliases agree");
    require(a != b, "legacy and per-thread defaults are distinct streams");
}
void default_stream_barriers() {
    for (bool runtime : {false, true}) for (bool per_thread_other : {false, true}) for (bool legacy_first : {false, true}) {
        std::cout << "default_stream barriers runtime=" << runtime << " per_thread_other=" << per_thread_other << " legacy_first=" << legacy_first << std::endl;
        default_stream_storage storage; default_stream_explicit blocking(false), independent(true); default_stream_event completed;
        default_stream_route route(runtime, false); const CUstream legacy = legacy_first ? CU_STREAM_LEGACY : nullptr, other = per_thread_other ? CU_STREAM_PER_THREAD : blocking.value;
        const auto producer = legacy_first ? legacy : other, consumer = legacy_first ? other : legacy;
        default_stream_gate gate; default_stream_gate_cleanup cleanup{gate}; route.hold(producer, default_stream_gate::call, &gate); gate.await();
        route.fill(producer, storage.first.value + 23, 0x47, 197);
        route.read(consumer, static_cast<unsigned char*>(storage.host) + default_stream_storage::offset, storage.first.value, default_stream_storage::bytes);
        storage.produce(route, independent.value, true); TEST_CUDA(cuEventRecord(completed.value, independent.value)); completed.complete(); storage.expect(true);
        const auto pending = route.query(consumer); gate.release(); route.synchronize(producer); route.synchronize(consumer); gate.check();
        require(pending == CUDA_ERROR_NOT_READY, "implicit default stream dependency holds the consumer while nonblocking work completes"); storage.expect(false);
    }
}
void default_stream_threading() {
    CUcontext context{}; TEST_CUDA(cuCtxGetCurrent(&context));
    for (bool runtime : {false, true}) for (bool per_thread : {false, true}) {
        std::cout << "default_stream threading runtime=" << runtime << " per_thread=" << per_thread << std::endl;
        default_stream_storage storage; default_stream_route route(runtime, per_thread); const auto main_id = route.id(route.selected());
        default_stream_gate gate; default_stream_gate_cleanup cleanup{gate}; route.hold(route.selected(), default_stream_gate::call, &gate); gate.await(); storage.produce(route, route.selected(), false);
        std::promise<unsigned long long> completion; auto future = completion.get_future();
        std::thread worker([&] {
            try {
                TEST_CUDA(cuCtxSetCurrent(context)); const auto id = route.id(route.selected()); storage.produce(route, route.selected(), true); route.synchronize(route.selected());
                TEST_CUDA(cuCtxSetCurrent(nullptr)); completion.set_value(id);
            } catch (...) { cuCtxSetCurrent(nullptr); completion.set_exception(std::current_exception()); }
        });
        default_stream_thread_cleanup joined{gate, worker};
        const bool finished = future.wait_for(per_thread ? std::chrono::milliseconds(5000) : std::chrono::milliseconds(30)) == std::future_status::ready;
        gate.release(); worker.join(); const auto worker_id = future.get(); route.synchronize(route.selected()); gate.check();
        require(per_thread ? finished && worker_id != main_id : !finished && worker_id == main_id, "default stream identity and work ordering across host threads");
        storage.expect(false); storage.expect(true);
    }
}
std::string default_stream_case;
void execute_default_stream() {
    if (default_stream_case == "aliases") default_stream_aliases();
    else if (default_stream_case == "barriers") default_stream_barriers();
    else if (default_stream_case == "threading") default_stream_threading();
    else require(false, "unknown default stream case");
}
}
int main(int argc, char** argv) {
    if (argc != 3) return 1; cudavm::default_stream_case = argv[2]; return cudavm::run_test(2, argv, cudavm::execute_default_stream);
}
