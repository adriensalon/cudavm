// H01.lifetime/H01.primary: cuDevicePrimaryCtxReset (current ABI) and
// cudaDeviceReset on each available backend. Exact primary flags/active state,
// balanced Driver retains, generation IDs and guarded bytes establish reset,
// reactivation and explicit-context isolation. Pending work uses a portable
// host allocation owned by the surviving explicit context. Reset resources
// are never queried, freed or otherwise reused. Legacy reset, registered
// modules, async/pool allocations, fatal faults and multi-device reset remain
// open obligations. There are no compiler inputs or capability-based passes.
#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <mutex>
#include <set>
#include <thread>

namespace cudavm {
void check_reset_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define RESET_RUNTIME(call) ::cudavm::check_reset_runtime((call), #call)
CUcontext reset_current() { CUcontext value{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&value)); return value; }
void reset_expect_state(CUdevice device, unsigned flags, bool active) {
    unsigned actual_flags = ~0u; int actual_active = -1;
    CONFORMANCE_CUDA(cuDevicePrimaryCtxGetState(device, &actual_flags, &actual_active));
    require(actual_flags == flags && actual_active == int(active), "reset primary flags and active state");
}
struct reset_retains {
    CUdevice device;
    unsigned count = 0;
    explicit reset_retains(CUdevice selected) : device(selected) {}
    ~reset_retains() { while (count) { cuDevicePrimaryCtxRelease(device); --count; } }
    CUcontext retain() { CUcontext value{}; CONFORMANCE_CUDA(cuDevicePrimaryCtxRetain(&value, device)); ++count; return value; }
    void release() { CONFORMANCE_CUDA(cuDevicePrimaryCtxRelease(device)); --count; }
};
void reset_expect_bytes(CUdeviceptr pointer, size_t size, unsigned seed, bool runtime) {
    std::vector<unsigned char> actual(size + 32, 0xd3), expected = actual;
    for (size_t i = 0; i < size; ++i) expected[i + 16] = static_cast<unsigned char>((i * 73 + seed) ^ (i >> 3));
    if (runtime) RESET_RUNTIME(cudaMemcpy(actual.data() + 16, reinterpret_cast<void*>(static_cast<uintptr_t>(pointer)), size, cudaMemcpyDeviceToHost));
    else CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data() + 16, pointer, size));
    require(actual == expected, "reset generation payload and readback guards");
}
void reset_write_bytes(CUdeviceptr pointer, size_t size, unsigned seed, bool runtime) {
    std::vector<unsigned char> input(size);
    for (size_t i = 0; i < size; ++i) input[i] = static_cast<unsigned char>((i * 73 + seed) ^ (i >> 3));
    if (runtime) RESET_RUNTIME(cudaMemcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(pointer)), input.data(), size, cudaMemcpyHostToDevice));
    else CONFORMANCE_CUDA(cuMemcpyHtoD(pointer, input.data(), size));
}
void reset_on_fresh_thread(void (*execute)(CUdevice)) {
    CUdevice device{}; CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    const auto parent = reset_current();
    std::exception_ptr error;
    std::jthread worker([&] {
        struct cleanup { ~cleanup() { cudaDeviceReset(); cuCtxSetCurrent(nullptr); } } cleanup;
        try {
            require(reset_current() == nullptr, "reset worker starts without a current context");
            reset_expect_state(device, 0, false);
            execute(device);
            RESET_RUNTIME(cudaDeviceReset());
            CONFORMANCE_CUDA(cuCtxSetCurrent(nullptr));
            reset_expect_state(device, 0, false);
            require(cuDevicePrimaryCtxRelease(device) == CUDA_ERROR_INVALID_CONTEXT, "reset balances all Runtime/Driver retains");
        } catch (...) { error = std::current_exception(); }
    });
    worker.join();
    require(reset_current() == parent, "worker reset preserves parent explicit context");
    if (error) std::rethrow_exception(error);
}
void device_reset_inactive(CUdevice device) {
    for (unsigned flags : {unsigned(CU_CTX_SCHED_SPIN), unsigned(CU_CTX_SCHED_YIELD), unsigned(CU_CTX_SCHED_BLOCKING_SYNC)}) {
        CONFORMANCE_CUDA(cuDevicePrimaryCtxSetFlags(device, flags)); reset_expect_state(device, flags, false);
        for (unsigned repeat = 0; repeat < 3; ++repeat) {
            CONFORMANCE_CUDA(cuDevicePrimaryCtxReset(device)); reset_expect_state(device, 0, false);
            require(reset_current() == nullptr, "inactive Driver reset does not bind a context");
            RESET_RUNTIME(cudaDeviceReset()); reset_expect_state(device, 0, false);
            require(reset_current() == nullptr, "inactive Runtime reset does not bind a context");
        }
    }
}
void device_reset_retains(CUdevice device) {
    reset_retains retains(device);
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        CONFORMANCE_CUDA(cuDevicePrimaryCtxSetFlags(device, CU_CTX_SCHED_YIELD));
        const auto primary = retains.retain();
        require(retains.retain() == primary && retains.retain() == primary, "retains share one primary generation");
        CONFORMANCE_CUDA(cuCtxSetCurrent(primary));
        unsigned long long before{}; CONFORMANCE_CUDA(cuCtxGetId(nullptr, &before));
        CUdeviceptr storage{}; CONFORMANCE_CUDA(cuMemAlloc(&storage, 523));
        reset_write_bytes(storage, 523, 31 + repeat, false); reset_expect_bytes(storage, 523, 31 + repeat, false);
        CONFORMANCE_CUDA(cuDevicePrimaryCtxReset(device)); reset_expect_state(device, 0, false);
        require(reset_current() == primary, "Driver reset leaves the primary binding on its host thread");
        CONFORMANCE_CUDA(cuCtxSetCurrent(nullptr));
        retains.release(); retains.release(); reset_expect_state(device, 0, false);
        const auto reactivated = retains.retain();
        CONFORMANCE_CUDA(cuCtxSetCurrent(reactivated));
        unsigned long long after{}; CONFORMANCE_CUDA(cuCtxGetId(nullptr, &after));
        require(after != before, "reactivation creates a fresh context generation");
        reset_expect_state(device, 0, true);
        { device_buffer fresh(523); reset_write_bytes(fresh.value, 523, 97 + repeat, false); reset_expect_bytes(fresh.value, 523, 97 + repeat, false); }
        CONFORMANCE_CUDA(cuCtxSetCurrent(nullptr));
        retains.release(); reset_expect_state(device, 0, true);
        retains.release(); reset_expect_state(device, 0, false);
    }
}
void device_reset_runtime(CUdevice device) {
    std::set<unsigned long long> generations;
    for (unsigned repeat = 0; repeat < 4; ++repeat) {
        RESET_RUNTIME(cudaSetDevice(device));
        unsigned long long id{}; CONFORMANCE_CUDA(cuCtxGetId(nullptr, &id));
        require(generations.insert(id).second, "Runtime reset reinitializes a fresh primary generation");
        reset_retains external(device); external.retain();
        void* storage{}; cudaStream_t stream{}; cudaEvent_t event{}; cudaArray_t array{};
        RESET_RUNTIME(cudaMalloc(&storage, 557));
        RESET_RUNTIME(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        RESET_RUNTIME(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        cudaChannelFormatDesc channel{8, 0, 0, 0, cudaChannelFormatKindUnsigned};
        RESET_RUNTIME(cudaMallocArray(&array, &channel, 17, 3, 0));
        reset_write_bytes(reinterpret_cast<uintptr_t>(storage), 557, 17 + repeat, true);
        RESET_RUNTIME(cudaEventRecord(event, stream)); RESET_RUNTIME(cudaEventSynchronize(event));
        reset_expect_bytes(reinterpret_cast<uintptr_t>(storage), 557, 17 + repeat, false);
        RESET_RUNTIME(cudaDeviceReset()); reset_expect_state(device, 0, false);
        // The external retain survives Runtime reset, but resources do not.
        external.release(); reset_expect_state(device, 0, false);
        CONFORMANCE_CUDA(cuCtxSetCurrent(nullptr));
        int selected = -1; RESET_RUNTIME(cudaGetDevice(&selected));
        require(selected == device && reset_current() == nullptr, "reset preserves Runtime device selection without query reactivation");
    }
}
void device_reset_interoperation(CUdevice device) {
    RESET_RUNTIME(cudaSetDevice(device));
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        void* storage{}; RESET_RUNTIME(cudaMalloc(&storage, 389));
        unsigned long long before{}; CONFORMANCE_CUDA(cuCtxGetId(nullptr, &before));
        reset_write_bytes(reinterpret_cast<uintptr_t>(storage), 389, 41 + repeat, true);
        CONFORMANCE_CUDA(cuDevicePrimaryCtxReset(device)); reset_expect_state(device, 0, false);
        // An ordinary Runtime allocation reactivates its retained primary.
        void* fresh{}; RESET_RUNTIME(cudaMalloc(&fresh, 601));
        unsigned long long after{}; CONFORMANCE_CUDA(cuCtxGetId(nullptr, &after));
        require(after != before, "Runtime allocation recovers after Driver primary reset");
        reset_write_bytes(reinterpret_cast<uintptr_t>(fresh), 601, 109 + repeat, false);
        reset_expect_bytes(reinterpret_cast<uintptr_t>(fresh), 601, 109 + repeat, true);
        RESET_RUNTIME(cudaFree(fresh));
        reset_expect_state(device, 0, true);
    }
}
void device_reset_explicit() {
    CUdevice device{}; CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    const auto original = reset_current();
    device_buffer storage(557); reset_write_bytes(storage.value, 557, 71, false);
    CUstream stream{}; CUevent event{};
    CONFORMANCE_CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CONFORMANCE_CUDA(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
    struct cleanup { CUstream stream; CUevent event; ~cleanup() { cuStreamDestroy(stream); cuEventDestroy(event); } } cleanup{stream, event};
    reset_retains primary(device); primary.retain();
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        CONFORMANCE_CUDA(cuDevicePrimaryCtxReset(device));
        RESET_RUNTIME(cudaDeviceReset());
        require(reset_current() == original, "Driver and Runtime reset preserve an explicit binding");
        reset_expect_state(device, 0, false);
        RESET_RUNTIME(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event), reinterpret_cast<cudaStream_t>(stream)));
        RESET_RUNTIME(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event)));
        reset_expect_bytes(storage.value, 557, 71 + repeat, true);
        reset_write_bytes(storage.value, 557, 72 + repeat, true);
    }
    reset_expect_bytes(storage.value, 557, 74, false);
    primary.release();
}
struct reset_gate {
    std::mutex mutex; std::condition_variable changed;
    bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* argument) {
        auto& gate = *static_cast<reset_gate*>(argument);
        std::unique_lock lock(gate.mutex); gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(10), [&] { return gate.released; })) gate.expired = true;
    }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
};
void device_reset_pending() {
    CUdevice device{}; CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    const auto original = reset_current();
    void* output{}; CONFORMANCE_CUDA(cuMemHostAlloc(&output, 601, CU_MEMHOSTALLOC_PORTABLE));
    struct host_cleanup { void* value; ~host_cleanup() { cuMemFreeHost(value); } } free_host{output};
    for (bool runtime : {false, true}) {
        std::memset(output, 0xd3, 601);
        reset_retains primary(device); const auto owner = primary.retain();
        CONFORMANCE_CUDA(cuCtxPushCurrent(owner));
        struct binding_cleanup { bool active = true; ~binding_cleanup() { if (active) cuCtxSetCurrent(nullptr); } } binding;
        CUdeviceptr storage{}; CUstream stream{};
        CONFORMANCE_CUDA(cuMemAlloc(&storage, 557)); CONFORMANCE_CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        reset_gate gate;
        bool retired = false;
        struct release_work {
            reset_gate& gate; bool& retired;
            ~release_work() { gate.release(); if (!retired) cuCtxSynchronize(); }
        } release{gate, retired};
        CONFORMANCE_CUDA(cuLaunchHostFunc(stream, reset_gate::call, &gate));
        { std::unique_lock lock(gate.mutex); require(gate.changed.wait_for(lock, std::chrono::seconds(5), [&] { return gate.entered; }), "reset work gate entry deadline"); }
        CONFORMANCE_CUDA(cuMemsetD8Async(storage, 0xa9, 557, stream));
        CONFORMANCE_CUDA(cuMemsetD8Async(storage + 23, 0x71, 389, stream));
        CONFORMANCE_CUDA(cuMemcpyDtoHAsync(static_cast<unsigned char*>(output) + 19, storage, 557, stream));
        // The helper only releases a host gate; no second host thread uses CUDA
        // while reset is in progress.
        std::jthread releaser([&] {
            std::unique_lock lock(gate.mutex);
            gate.changed.wait_for(lock, std::chrono::milliseconds(25), [&] { return gate.released; });
            gate.released = true; gate.changed.notify_all();
        });
        if (runtime) RESET_RUNTIME(cudaDeviceReset()); else CONFORMANCE_CUDA(cuDevicePrimaryCtxReset(device));
        retired = true;
        releaser.join(); reset_expect_state(device, 0, false);
        std::vector<unsigned char> expected(601, 0xd3);
        std::fill_n(expected.begin() + 19, 557, 0xa9); std::fill_n(expected.begin() + 42, 389, 0x71);
        require(!gate.expired && std::memcmp(output, expected.data(), expected.size()) == 0, "reset completes accepted transfers and preserves surviving host guards");
        require(reset_current() == owner, "reset retains the host thread's primary binding");
        CONFORMANCE_CUDA(cuCtxSetCurrent(nullptr)); binding.active = false;
        require(reset_current() == original, "clearing reset primary exposes surviving explicit context");
        primary.release();
    }
}
std::string reset_case = "all";
void execute_device_reset() {
    bool selected = false;
    const auto worker = [&](const char* name, auto execute) { if (reset_case == "all" || reset_case == name) { reset_on_fresh_thread(execute); selected = true; } };
    worker("inactive", device_reset_inactive); worker("retains", device_reset_retains);
    worker("runtime", device_reset_runtime); worker("interoperation", device_reset_interoperation);
    if (reset_case == "all" || reset_case == "explicit") { device_reset_explicit(); selected = true; }
    if (reset_case == "all" || reset_case == "pending") { device_reset_pending(); selected = true; }
    require(selected, "unknown device_reset case: " + reset_case);
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::reset_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_device_reset);
}
