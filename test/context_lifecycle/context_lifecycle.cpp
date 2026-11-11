// H01.lifetime: explicit Driver contexts, on the first available endpoint of
// each backend, without compiler inputs. Creation covers the public v4 entry
// and procedure-address v2/v3 ABIs with no affinity/advanced parameters. The
// oracle checks exact current bindings, device/flags/API-version consistency,
// process-unique IDs, independent initialized storage, and complete guarded
// readback after synchronization or destruction with accepted work pending.
// No destroyed context/resource is reused. Primary reset, affinity, green
// contexts, imported modules, fatal errors and other host ABIs remain open.
#include "test_framework.hpp"
#include <cudaTypedefs.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <mutex>
#include <set>
#include <thread>

namespace cudavm {
CUcontext lifecycle_current() {
    CUcontext value{};
    TEST_CUDA(cuCtxGetCurrent(&value));
    return value;
}
template<class Function> Function lifecycle_procedure(const char* name, int version) {
    void* pointer{};
    CUdriverProcAddressQueryResult status{};
    TEST_CUDA(cuGetProcAddress(name, &pointer, version, CU_GET_PROC_ADDRESS_DEFAULT, &status));
    require(pointer && status == CU_GET_PROC_ADDRESS_SUCCESS, "context ABI procedure lookup");
    return reinterpret_cast<Function>(pointer);
}
struct lifecycle_context {
    CUcontext value{}, preceding = lifecycle_current();
    explicit lifecycle_context(CUdevice device, unsigned flags = 0, unsigned abi = 4) {
        if (abi == 2) TEST_CUDA(lifecycle_procedure<PFN_cuCtxCreate_v3020>("cuCtxCreate", 3020)(&value, flags, device));
        else if (abi == 3) TEST_CUDA(lifecycle_procedure<PFN_cuCtxCreate_v11040>("cuCtxCreate", 11040)(&value, nullptr, 0, flags, device));
        else TEST_CUDA(cuCtxCreate(&value, nullptr, flags, device));
        require(value && value != preceding && lifecycle_current() == value, "creation pushes a distinct current context");
    }
    ~lifecycle_context() { if (value) cuCtxDestroy(value); }
    void detach() {
        CUcontext popped{};
        TEST_CUDA(cuCtxPopCurrent(&popped));
        require(popped == value && lifecycle_current() == preceding, "detach exposes preceding context");
    }
    void destroy() { TEST_CUDA(cuCtxDestroy(value)); value = nullptr; }
};
void context_lifecycle_creation() {
    CUdevice device{}; TEST_CUDA(cuCtxGetDevice(&device));
    const auto original = lifecycle_current();
    std::set<unsigned long long> identities;
    unsigned long long original_id{}; TEST_CUDA(cuCtxGetId(original, &original_id)); identities.insert(original_id);
    for (unsigned abi : {2u, 3u, 4u})
    for (unsigned flags : {unsigned(CU_CTX_SCHED_AUTO), unsigned(CU_CTX_SCHED_SPIN),
                           unsigned(CU_CTX_SCHED_YIELD), unsigned(CU_CTX_SCHED_BLOCKING_SYNC)}) {
        lifecycle_context context(device, flags, abi);
        struct guarded_id { unsigned long long prefix = 0x123456789abcdef0, value{}, suffix = 0xfedcba9876543210; } id;
        TEST_CUDA(cuCtxGetId(context.value, &id.value));
        require(id.prefix == 0x123456789abcdef0 && id.suffix == 0xfedcba9876543210, "context ID output width");
        require(identities.insert(id.value).second, "context IDs are unique across live and retired contexts");
        unsigned long long current_id{}; TEST_CUDA(cuCtxGetId(nullptr, &current_id));
        CUdevice actual{}; unsigned actual_flags = ~0u, explicit_version = 0, current_version = 0;
        TEST_CUDA(cuCtxGetDevice(&actual)); TEST_CUDA(cuCtxGetFlags(&actual_flags));
        TEST_CUDA(cuCtxGetApiVersion(context.value, &explicit_version));
        TEST_CUDA(cuCtxGetApiVersion(nullptr, &current_version));
        require(actual == device && (actual_flags & CU_CTX_SCHED_MASK) == flags && current_id == id.value,
            "created context device, scheduling flags and current identity");
        require(explicit_version >= 3020 && explicit_version == current_version, "current and explicit context API versions");
        context.destroy();
        require(lifecycle_current() == original, "destroying current context exposes runner context");
    }
}
void lifecycle_expect_bytes(CUdeviceptr pointer, std::span<const unsigned char> expected) {
    std::vector<unsigned char> actual(expected.size() + 32, 0xd3), guarded = actual;
    std::copy(expected.begin(), expected.end(), guarded.begin() + 16);
    TEST_CUDA(cuMemcpyDtoH(actual.data() + 16, pointer, expected.size()));
    require(actual == guarded, "independent context payload and readback guards");
}
void context_lifecycle_ownership() {
    CUdevice device{}; TEST_CUDA(cuCtxGetDevice(&device));
    const auto original = lifecycle_current();
    std::vector<unsigned char> parent_expected(389), child_expected(521);
    for (size_t i = 0; i < parent_expected.size(); ++i) parent_expected[i] = static_cast<unsigned char>(i * 73 + 19);
    for (size_t i = 0; i < child_expected.size(); ++i) child_expected[i] = static_cast<unsigned char>((i * 31 + 41) ^ (i >> 3));
    device_buffer parent(parent_expected.size());
    TEST_CUDA(cuMemcpyHtoD(parent.value, parent_expected.data(), parent_expected.size()));
    for (bool detached : {false, true}) {
        lifecycle_context context(device);
        // These resources are deliberately retired by their owning context.
        CUdeviceptr child{}; CUstream stream{}; CUevent event{}; CUarray array{};
        TEST_CUDA(cuMemAlloc(&child, child_expected.size()));
        TEST_CUDA(cuMemcpyHtoD(child, child_expected.data(), child_expected.size()));
        TEST_CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        TEST_CUDA(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
        CUDA_ARRAY_DESCRIPTOR descriptor{17, 3, CU_AD_FORMAT_UNSIGNED_INT8, 1};
        TEST_CUDA(cuArrayCreate(&array, &descriptor));
        CUcontext allocation_owner{}, stream_owner{};
        TEST_CUDA(cuPointerGetAttribute(&allocation_owner, CU_POINTER_ATTRIBUTE_CONTEXT, child));
        TEST_CUDA(cuStreamGetCtx(stream, &stream_owner));
        require(allocation_owner == context.value && stream_owner == context.value, "resources report their explicit owning context");
        lifecycle_expect_bytes(child, child_expected);
        TEST_CUDA(cuEventRecord(event, stream)); TEST_CUDA(cuEventSynchronize(event));
        if (detached) context.detach();
        context.destroy();
        require(lifecycle_current() == original, "detached/current context teardown preserves unrelated binding");
        lifecycle_expect_bytes(parent.value, parent_expected);
        lifecycle_context replacement(device);
        { device_buffer fresh(child_expected.size());
          TEST_CUDA(cuMemcpyHtoD(fresh.value, child_expected.data(), child_expected.size()));
          lifecycle_expect_bytes(fresh.value, child_expected); }
        replacement.destroy();
    }
}
struct lifecycle_host {
    static constexpr size_t bytes = 601;
    void* value{};
    lifecycle_host() { TEST_CUDA(cuMemHostAlloc(&value, bytes, CU_MEMHOSTALLOC_PORTABLE)); std::memset(value, 0xd3, bytes); }
    ~lifecycle_host() { if (value) cuMemFreeHost(value); }
};
struct lifecycle_gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* argument) {
        auto& gate = *static_cast<lifecycle_gate*>(argument);
        std::unique_lock lock(gate.mutex);
        gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(10), [&] { return gate.released; })) gate.expired = true;
    }
    void await() {
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "context work gate entry deadline");
    }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
};
void lifecycle_pending(bool teardown, bool explicit_sync) {
    CUdevice device{}; TEST_CUDA(cuCtxGetDevice(&device));
    const auto original = lifecycle_current();
    lifecycle_host output;
    lifecycle_context context(device);
    auto synchronize = lifecycle_procedure<PFN_cuCtxSynchronize_v13000>("cuCtxSynchronize", 13000);
    lifecycle_gate gate;
    struct release_work {
        lifecycle_gate& gate; lifecycle_context& context; PFN_cuCtxSynchronize_v13000 synchronize;
        ~release_work() { gate.release(); if (context.value) synchronize(context.value); }
    } release{gate, context, synchronize};
    CUdeviceptr storage{}; TEST_CUDA(cuMemAlloc(&storage, 557));
    CUstream stream{}; TEST_CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    TEST_CUDA(cuLaunchHostFunc(stream, lifecycle_gate::call, &gate)); gate.await();
    TEST_CUDA(cuMemsetD8Async(storage, 0xa9, 557, stream));
    TEST_CUDA(cuMemsetD8Async(storage + 23, 0x71, 389, stream));
    TEST_CUDA(cuMemcpyDtoHAsync(static_cast<unsigned char*>(output.value) + 19, storage, 557, stream));
    context.detach();
    std::exception_ptr error;
    std::mutex mutex; std::condition_variable changed;
    bool started = false, completed = false;
    std::jthread worker([&] {
        { std::lock_guard lock(mutex); started = true; changed.notify_all(); }
        try {
            if (teardown) { TEST_CUDA(cuCtxDestroy(context.value)); context.value = nullptr; }
            else if (explicit_sync) TEST_CUDA(synchronize(context.value));
            else {
                TEST_CUDA(cuCtxPushCurrent(context.value));
                TEST_CUDA(cuCtxSynchronize());
                CUcontext popped{}; TEST_CUDA(cuCtxPopCurrent(&popped));
                require(popped == context.value, "synchronizing worker balances context binding");
            }
        } catch (...) { error = std::current_exception(); }
        { std::lock_guard lock(mutex); completed = true; changed.notify_all(); }
    });
    // Unwind must release the callback before joining the API worker.
    struct release_worker { lifecycle_gate& gate; ~release_worker() { gate.release(); } } release_before_join{gate};
    {
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return started; }), "context API worker entry deadline");
        require(!changed.wait_for(lock, std::chrono::milliseconds(25), [&] { return completed; }),
            "context synchronization/teardown waits for accepted unfinished work");
    }
    gate.release(); worker.join();
    if (error) std::rethrow_exception(error);
    require(!gate.expired && lifecycle_current() == original, "bounded context wait preserves caller binding");
    std::vector<unsigned char> expected(lifecycle_host::bytes, 0xd3);
    std::fill_n(expected.begin() + 19, 557, 0xa9);
    std::fill_n(expected.begin() + 19 + 23, 389, 0x71);
    require(std::memcmp(output.value, expected.data(), expected.size()) == 0, "context completion publishes exact payload and all host guards");
    if (!teardown) {
        TEST_CUDA(cuCtxPushCurrent(context.value));
        std::vector<unsigned char> device_expected(expected.begin() + 19, expected.begin() + 19 + 557);
        lifecycle_expect_bytes(storage, device_expected);
        TEST_CUDA(cuMemFree(storage)); TEST_CUDA(cuStreamDestroy(stream));
        context.detach(); context.destroy();
    }
}
void context_lifecycle_synchronization() { lifecycle_pending(false, false); lifecycle_pending(false, true); }
void context_lifecycle_teardown() { lifecycle_pending(true, false); }
void context_lifecycle_validation() {
    CUdevice device{}; TEST_CUDA(cuCtxGetDevice(&device));
    const auto original = lifecycle_current();
    require(cuCtxCreate(nullptr, nullptr, 0, device) == CUDA_ERROR_INVALID_VALUE, "null context output");
    CUcontext context{};
    require(cuCtxCreate(&context, nullptr, 0, -1) == CUDA_ERROR_INVALID_DEVICE, "invalid context device");
    for (unsigned flags : {3u, 0x80000000u})
        require(cuCtxCreate(&context, nullptr, flags, device) == CUDA_ERROR_INVALID_VALUE, "invalid scheduling/unknown context flags");
    require(cuCtxGetId(original, nullptr) == CUDA_ERROR_INVALID_VALUE, "null context identity output");
    require(cuCtxGetApiVersion(original, nullptr) == CUDA_ERROR_INVALID_VALUE, "null context API-version output");
    require(lifecycle_current() == original, "rejected context operations preserve current binding");
    lifecycle_context valid(device); valid.destroy();
    require(lifecycle_current() == original, "valid creation remains possible after rejected requests");
}
std::string lifecycle_case = "all";
void execute_context_lifecycle() {
    bool selected = false;
    const auto run = [&](const char* name, auto execute) {
        if (lifecycle_case == "all" || lifecycle_case == name) { execute(); selected = true; }
    };
    run("creation", context_lifecycle_creation); run("ownership", context_lifecycle_ownership);
    run("synchronization", context_lifecycle_synchronization); run("teardown", context_lifecycle_teardown);
    run("validation", context_lifecycle_validation);
    require(selected, "unknown context_lifecycle case: " + lifecycle_case);
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::lifecycle_case = argv[2]; --argc; }
    return cudavm::run_test(argc, argv, cudavm::execute_context_lifecycle);
}
