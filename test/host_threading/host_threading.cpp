#include "test_framework.hpp"
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
void check_threading_runtime(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
#define THREADING_RUNTIME(call) ::cudavm::check_threading_runtime((call), #call)
void* threading_pointer(CUdeviceptr value) { return reinterpret_cast<void*>(static_cast<uintptr_t>(value)); }
struct threading_rendezvous {
    std::mutex mutex; std::condition_variable changed; unsigned arrived = 0, generation = 0; bool cancelled = false;
    void cancel() { std::lock_guard lock(mutex); cancelled = true; changed.notify_all(); }
    void wait() {
        std::unique_lock lock(mutex); require(!cancelled, "worker rendezvous cancelled"); const auto before = generation;
        if (++arrived == 4) { arrived = 0; ++generation; changed.notify_all(); return; }
        if (!changed.wait_for(lock, std::chrono::seconds(15), [&] { return cancelled || generation != before; })) { cancelled = true; changed.notify_all(); }
        require(!cancelled, "worker rendezvous failed or exceeded its deadline");
    }
};
struct threading_binding {
    CUcontext value{}; bool owned;
    threading_binding(CUcontext shared, CUdevice device, bool independent) : owned(independent) {
        if (owned) CONFORMANCE_CUDA(cuCtxCreate(&value, nullptr, 0, device)); else { value = shared; CONFORMANCE_CUDA(cuCtxSetCurrent(value)); }
    }
    ~threading_binding() { if (owned) cuCtxDestroy(value); else cuCtxSetCurrent(nullptr); }
};
struct threading_stream {
    CUstream value{};
    explicit threading_stream(bool runtime) {
        if (runtime) { cudaStream_t result{}; THREADING_RUNTIME(cudaStreamCreateWithFlags(&result, cudaStreamNonBlocking)); value = reinterpret_cast<CUstream>(result); }
        else CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING));
    }
    ~threading_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
};
struct threading_event {
    CUevent value{};
    explicit threading_event(bool runtime) {
        if (runtime) { cudaEvent_t result{}; THREADING_RUNTIME(cudaEventCreateWithFlags(&result, cudaEventDisableTiming)); value = reinterpret_cast<CUevent>(result); }
        else CONFORMANCE_CUDA(cuEventCreate(&value, CU_EVENT_DISABLE_TIMING));
    }
    ~threading_event() { if (value) cuEventDestroy(value); }
};
struct threading_device {
    CUdeviceptr value{}; bool runtime;
    threading_device(size_t bytes, bool use_runtime) : runtime(use_runtime) {
        if (runtime) { void* result{}; THREADING_RUNTIME(cudaMalloc(&result, bytes)); value = reinterpret_cast<uintptr_t>(result); }
        else CONFORMANCE_CUDA(cuMemAlloc(&value, bytes));
    }
    ~threading_device() { if (value) { cuCtxSynchronize(); if (runtime) cuMemFree(value); else cudaFree(threading_pointer(value)); } }
};
struct threading_host {
    void* value{}; bool runtime; size_t bytes;
    threading_host(size_t size, bool use_runtime) : runtime(use_runtime), bytes(size) {
        if (runtime) THREADING_RUNTIME(cudaMallocHost(&value, bytes)); else CONFORMANCE_CUDA(cuMemAllocHost(&value, bytes));
    }
    ~threading_host() { if (value) { cuCtxSynchronize(); if (runtime) cuMemFreeHost(value); else cudaFreeHost(value); } }
    unsigned char* data() const { return static_cast<unsigned char*>(value); }
    void set(const std::vector<unsigned char>& source) { require(source.size() == bytes, "threaded host initialization extent"); std::memcpy(value, source.data(), bytes); }
    void expect(const std::vector<unsigned char>& expected, const std::string& context) const { require(std::memcmp(value, expected.data(), bytes) == 0, context); }
};
std::vector<unsigned char> threading_pattern(size_t bytes, unsigned seed) {
    std::vector<unsigned char> result(bytes);
    for (size_t i = 0; i < bytes; ++i) result[i] = static_cast<unsigned char>(seed + 37 * i + (i >> 3));
    return result;
}
void threading_upload(bool runtime, CUdeviceptr to, const void* from, size_t bytes, CUstream stream) {
    if (runtime) THREADING_RUNTIME(cudaMemcpyAsync(threading_pointer(to), from, bytes, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)));
    else CONFORMANCE_CUDA(cuMemcpyHtoDAsync(to, from, bytes, stream));
}
void threading_copy(bool runtime, CUdeviceptr to, CUdeviceptr from, size_t bytes, CUstream stream) {
    if (runtime) THREADING_RUNTIME(cudaMemcpyAsync(threading_pointer(to), threading_pointer(from), bytes, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream)));
    else CONFORMANCE_CUDA(cuMemcpyDtoDAsync(to, from, bytes, stream));
}
void threading_fill(bool runtime, CUdeviceptr to, unsigned char value, size_t bytes, CUstream stream) {
    if (runtime) THREADING_RUNTIME(cudaMemsetAsync(threading_pointer(to), value, bytes, reinterpret_cast<cudaStream_t>(stream)));
    else CONFORMANCE_CUDA(cuMemsetD8Async(to, value, bytes, stream));
}
void threading_read(bool runtime, void* to, CUdeviceptr from, size_t bytes, CUstream stream) {
    if (runtime) THREADING_RUNTIME(cudaMemcpyAsync(to, threading_pointer(from), bytes, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream)));
    else CONFORMANCE_CUDA(cuMemcpyDtoHAsync(to, from, bytes, stream));
}
void threading_record(bool runtime, CUevent event, CUstream stream) {
    if (runtime) THREADING_RUNTIME(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event), reinterpret_cast<cudaStream_t>(stream)));
    else CONFORMANCE_CUDA(cuEventRecord(event, stream));
}
void threading_complete(bool runtime, CUevent event) {
    if (runtime) THREADING_RUNTIME(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event))); else CONFORMANCE_CUDA(cuEventSynchronize(event));
}
void threading_join(std::array<std::thread, 4>& workers) { for (auto& worker : workers) if (worker.joinable()) worker.join(); }
void threading_independent(bool independent, bool errors) {
    CUcontext parent{}; CUdevice device{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&parent)); CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    std::array<unsigned long long, 4> context_ids{}, stream_ids{}, buffer_ids{};
    std::array<std::exception_ptr, 4> failures{}; std::array<std::thread, 4> workers; threading_rendezvous rendezvous;
    if (errors) {
        THREADING_RUNTIME(cudaGetLastError());
        require(cudaGetDeviceCount(nullptr) == cudaErrorInvalidValue, "parent has a known Runtime error before workers");
    }
    try {
        for (unsigned worker = 0; worker < workers.size(); ++worker) workers[worker] = std::thread([&, worker] {
            try {
                threading_binding binding(parent, device, independent); const bool runtime = (worker & 1) != 0;
                CONFORMANCE_CUDA(cuCtxGetId(binding.value, &context_ids[worker])); rendezvous.wait();
                cudaError_t expected_error = cudaSuccess;
                if (errors) {
                    THREADING_RUNTIME(cudaGetLastError());
                    if (worker & 1) { require(cudaSetDevice(-1) == cudaErrorInvalidDevice, "worker invalid-device setup"); expected_error = cudaErrorInvalidDevice; }
                    else { cudaStream_t rejected{}; require(cudaStreamCreateWithFlags(&rejected, 0xff) == cudaErrorInvalidValue, "worker invalid-flags setup"); expected_error = cudaErrorInvalidValue; }
                }
                for (unsigned round = 0; round < 5; ++round) {
                    const size_t bytes = 1013 + worker * 37; threading_host input(bytes, runtime), output(bytes, !runtime);
                    threading_device first(bytes, runtime), second(bytes, !runtime); threading_stream stream(runtime); threading_event event(!runtime);
                    CONFORMANCE_CUDA(cuStreamGetId(stream.value, &stream_ids[worker])); CONFORMANCE_CUDA(cuPointerGetAttribute(&buffer_ids[worker], CU_POINTER_ATTRIBUTE_BUFFER_ID, first.value));
                    rendezvous.wait();
                    for (unsigned other = 0; other < workers.size(); ++other) if (other != worker) {
                        require(stream_ids[worker] != stream_ids[other] && buffer_ids[worker] != buffer_ids[other], "concurrent live streams and buffers have distinct identities");
                        require((context_ids[worker] == context_ids[other]) == !independent, "worker context ownership identities");
                    }
                    const auto source = threading_pattern(bytes, 19 + worker * 53 + round * 79), initial_a = threading_pattern(bytes, 91), initial_b = threading_pattern(bytes, 147), initial_h = threading_pattern(bytes, 193);
                    input.set(source); output.set(initial_h); CONFORMANCE_CUDA(cuMemcpyHtoD(first.value, initial_a.data(), bytes)); CONFORMANCE_CUDA(cuMemcpyHtoD(second.value, initial_b.data(), bytes));
                    threading_upload(runtime, first.value + 31, input.data() + 17, 811, stream.value);
                    threading_fill(!runtime, first.value + 91, static_cast<unsigned char>(worker * 19 + round * 37), 97, stream.value);
                    threading_copy(runtime, second.value + 13, first.value + 31, 811, stream.value); threading_read(!runtime, output.data() + 19, second.value + 7, bytes - 29, stream.value);
                    threading_record(runtime, event.value, stream.value); threading_complete(!runtime, event.value);
                    auto a = initial_a, b = initial_b, h = initial_h;
                    std::copy_n(source.begin() + 17, 811, a.begin() + 31); std::fill_n(a.begin() + 91, 97, static_cast<unsigned char>(worker * 19 + round * 37));
                    std::copy_n(a.begin() + 31, 811, b.begin() + 13); std::copy_n(b.begin() + 7, bytes - 29, h.begin() + 19);
                    input.expect(source, "concurrent source storage and guards"); output.expect(h, "concurrent independent payload and host guards");
                    std::vector<unsigned char> actual(bytes); CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), first.value, bytes)); require(actual == a, "concurrent first device storage and guards");
                    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), second.value, bytes)); require(actual == b, "concurrent second device storage and guards");
                    CUcontext current{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&current)); require(current == binding.value, "worker context binding survives mixed calls");
                    int ordinal = -1; THREADING_RUNTIME(cudaGetDevice(&ordinal)); require(ordinal == device, "worker Runtime device matches bound context");
                    if (errors) require(cudaPeekAtLastError() == expected_error, "valid mixed work preserves each worker's own Runtime error");
                    rendezvous.wait();
                }
                if (errors) require(cudaGetLastError() == expected_error && cudaPeekAtLastError() == cudaSuccess, "worker get/clear remains thread-local");
            } catch (...) { failures[worker] = std::current_exception(); rendezvous.cancel(); }
        });
    } catch (...) { rendezvous.cancel(); threading_join(workers); throw; }
    threading_join(workers); for (const auto& failure : failures) if (failure) std::rethrow_exception(failure);
    CUcontext current{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&current)); require(current == parent, "parent binding survives concurrent worker contexts");
    if (errors) require(cudaPeekAtLastError() == cudaErrorInvalidValue && cudaGetLastError() == cudaErrorInvalidValue && cudaPeekAtLastError() == cudaSuccess, "worker errors and clears preserve parent last error");
}
struct threading_latch {
    std::mutex mutex; std::condition_variable changed; std::array<bool, 4> submitted{}; bool cancelled = false;
    void cancel() { std::lock_guard lock(mutex); cancelled = true; changed.notify_all(); }
    void publish(unsigned stage) { std::lock_guard lock(mutex); submitted[stage] = true; changed.notify_all(); }
    void wait(unsigned stage) {
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(15), [&] { return cancelled || submitted[stage]; }) && !cancelled, "cross-thread event record publication deadline");
    }
};
void threading_handoff() {
    constexpr size_t bytes = 1013; CUcontext context{}; CUdevice device{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&context)); CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    for (unsigned seed : {0u, 79u, 255u, 0xffffffffu}) {
        threading_host input(bytes, true), output(bytes, false); const auto source = threading_pattern(bytes, seed), initial_host = threading_pattern(bytes, 193);
        input.set(source); output.set(initial_host);
        std::array<std::unique_ptr<threading_device>, 4> storage; std::array<std::unique_ptr<threading_stream>, 4> streams;
        std::array<std::unique_ptr<threading_event>, 4> events; std::array<std::vector<unsigned char>, 4> expected;
        for (unsigned stage = 0; stage < 4; ++stage) {
            storage[stage] = std::make_unique<threading_device>(bytes, (stage & 1) != 0); streams[stage] = std::make_unique<threading_stream>((stage & 1) != 0); events[stage] = std::make_unique<threading_event>((stage & 1) == 0);
            expected[stage] = threading_pattern(bytes, 31 + stage * 19); CONFORMANCE_CUDA(cuMemcpyHtoD(storage[stage]->value, expected[stage].data(), bytes));
            if (stage == 0) std::copy_n(source.begin() + 17, 811, expected[stage].begin() + 23);
            else std::copy_n(expected[stage - 1].begin() + 23, 811, expected[stage].begin() + 23);
            std::fill_n(expected[stage].begin() + 61 + stage * 101, 31, static_cast<unsigned char>(seed + stage * 37));
        }
        threading_latch latch; std::array<std::thread, 4> workers; std::array<std::exception_ptr, 4> failures{};
        try {
            for (unsigned stage = 0; stage < 4; ++stage) workers[stage] = std::thread([&, stage] {
                try {
                    threading_binding binding(context, device, false); const bool runtime = (stage & 1) != 0; const auto stream = streams[stage]->value;
                    if (!stage) threading_upload(runtime, storage[stage]->value + 23, input.data() + 17, 811, stream);
                    else {
                        latch.wait(stage - 1);
                        if (runtime) THREADING_RUNTIME(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream), reinterpret_cast<cudaEvent_t>(events[stage - 1]->value), 0));
                        else CONFORMANCE_CUDA(cuStreamWaitEvent(stream, events[stage - 1]->value, 0));
                        threading_copy(runtime, storage[stage]->value + 23, storage[stage - 1]->value + 23, 811, stream);
                    }
                    threading_fill(!runtime, storage[stage]->value + 61 + stage * 101, static_cast<unsigned char>(seed + stage * 37), 31, stream);
                    if (stage == 3) threading_read(runtime, output.data() + 19, storage[stage]->value + 7, bytes - 29, stream);
                    threading_record(runtime, events[stage]->value, stream); latch.publish(stage); threading_complete(!runtime, events[stage]->value);
                } catch (...) { failures[stage] = std::current_exception(); latch.cancel(); }
            });
        } catch (...) { latch.cancel(); threading_join(workers); throw; }
        threading_join(workers); for (const auto& failure : failures) if (failure) std::rethrow_exception(failure);
        auto host_expected = initial_host; std::copy_n(expected[3].begin() + 7, bytes - 29, host_expected.begin() + 19);
        input.expect(source, "thread handoff preserves pinned source and guards"); output.expect(host_expected, "event-ordered four-thread handoff and host guards");
        for (unsigned stage = 0; stage < 4; ++stage) {
            std::vector<unsigned char> actual(bytes); CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), storage[stage]->value, bytes)); require(actual == expected[stage], "thread handoff complete intermediate allocation " + std::to_string(stage));
        }
    }
}
std::string threading_case = "all";
void execute_host_threading() {
    bool selected = false;
    const auto run = [&](const char* name, auto execute) { if (threading_case == "all" || threading_case == name) { std::cout << "case=" << name << std::endl; execute(); selected = true; } };
    run("shared", [] { threading_independent(false, false); }); run("contexts", [] { threading_independent(true, false); });
    run("errors", [] { threading_independent(false, true); }); run("handoff", threading_handoff);
    require(selected, "unknown host_threading case: " + threading_case);
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::threading_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_host_threading);
}
