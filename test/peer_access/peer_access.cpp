#include "conformance.hpp"
#include <cudaTypedefs.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

namespace cudavm {
std::string_view peer_case;
constexpr uint32_t peer_before = 0x19375928, peer_after = 0xfedcba01;
constexpr size_t peer_guard = 9, peer_count = 1031, peer_total = peer_guard + peer_count + 11;
void peer_runtime(cudaError_t status) { require(status == cudaSuccess, std::string("peer Runtime: ") + cudaGetErrorString(status)); }
struct peer_current {
    CUcontext previous{};
    explicit peer_current(CUcontext value) { CONFORMANCE_CUDA(cuCtxGetCurrent(&previous)); CONFORMANCE_CUDA(cuCtxSetCurrent(value)); }
    ~peer_current() { cuCtxSetCurrent(previous); }
};
struct peer_context {
    CUcontext value{};
    int device;
    bool primary;
    peer_context(int ordinal, bool use_primary = false) : device(ordinal), primary(use_primary) {
        if (primary) CONFORMANCE_CUDA(cuDevicePrimaryCtxRetain(&value, device));
        else {
            CONFORMANCE_CUDA(cuCtxCreate(&value, nullptr, 0, device)); CUcontext popped{};
            CONFORMANCE_CUDA(cuCtxPopCurrent(&popped)); require(popped == value, "peer context creation restores the preceding stack");
        }
    }
    ~peer_context() { if (value) { if (primary) cuDevicePrimaryCtxRelease(device); else cuCtxDestroy(value); } }
    void current() const { CONFORMANCE_CUDA(cuCtxSetCurrent(value)); }
};
struct peer_buffer {
    CUcontext owner{};
    CUdeviceptr value{};
    explicit peer_buffer(CUcontext context) : owner(context) { peer_current scope(owner); CONFORMANCE_CUDA(cuMemAlloc(&value, peer_total * sizeof(uint32_t))); }
    ~peer_buffer() { if (value) { CUcontext previous{}; cuCtxGetCurrent(&previous); cuCtxSetCurrent(owner); cuMemFree(value); cuCtxSetCurrent(previous); } }
    void write(std::span<const uint32_t> words, bool runtime = false) const {
        peer_current scope(owner);
        if (runtime) peer_runtime(cudaMemcpy(reinterpret_cast<void*>(uintptr_t(value)), words.data(), words.size_bytes(), cudaMemcpyHostToDevice));
        else CONFORMANCE_CUDA(cuMemcpyHtoD(value, words.data(), words.size_bytes()));
    }
    void expect(std::span<const uint32_t> expected, bool runtime = false) const {
        peer_current scope(owner); std::vector<uint32_t> actual(expected.size() + 8, 0x8d7a3e91);
        if (runtime) peer_runtime(cudaMemcpy(actual.data() + 4, reinterpret_cast<void*>(uintptr_t(value)), expected.size_bytes(), cudaMemcpyDeviceToHost));
        else CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data() + 4, value, expected.size_bytes()));
        for (size_t i = 0; i < actual.size(); ++i) {
            const auto wanted = i < 4 || i >= expected.size() + 4 ? 0x8d7a3e91 : expected[i - 4];
            require(actual[i] == wanted, "peer full payload/output guards at word " + std::to_string(i));
        }
    }
};
std::vector<uint32_t> peer_words(unsigned seed) {
    std::vector<uint32_t> result(peer_total);
    for (size_t i = 0; i < result.size(); ++i) result[i] = uint32_t((i * 0x9e3779b9u) ^ (seed * 0x317593u) ^ (i >> 3));
    return result;
}
constexpr char peer_ptx[] = R"ptx(.version 6.0
.target sm_60
.address_size 64
.visible .entry peer_update(.param .u64 remote, .param .u64 local, .param .u32 count, .param .u32 salt) {
    .reg .pred %p;
    .reg .b32 %r<10>;
    .reg .b64 %d<7>;
    ld.param.u64 %d0, [remote]; ld.param.u64 %d1, [local];
    ld.param.u32 %r0, [count]; ld.param.u32 %r1, [salt];
    mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p, %r5, %r0; @%p bra done;
    mul.wide.u32 %d2, %r5, 4; add.u64 %d3, %d0, %d2; add.u64 %d4, %d1, %d2;
    ld.global.u32 %r6, [%d3]; xor.b32 %r7, %r6, %r1; add.u32 %r8, %r7, %r5;
    st.global.u32 [%d3], %r8; st.global.u32 [%d4], %r6;
done: ret;
})ptx";
void peer_expected(std::vector<uint32_t>& remote, std::vector<uint32_t>& local, uint32_t salt) {
    for (size_t i = 0; i < peer_count; ++i) {
        local[i + peer_guard] = remote[i + peer_guard]; remote[i + peer_guard] = (remote[i + peer_guard] ^ salt) + uint32_t(i);
    }
}
void peer_launch(const loaded_program& program, const peer_buffer& remote, const peer_buffer& local, uint32_t salt, CUstream stream = nullptr) {
    auto r = remote.value + peer_guard * sizeof(uint32_t), l = local.value + peer_guard * sizeof(uint32_t); unsigned count = peer_count;
    void* parameters[]{&r, &l, &count, &salt};
    CONFORMANCE_CUDA(cuLaunchKernel(program.function("peer_update"), unsigned((peer_count + 63) / 64), 1, 1, 64, 1, 1, 0, stream, parameters, nullptr));
}
int peer_devices() { int count{}; CONFORMANCE_CUDA(cuDeviceGetCount(&count)); return count; }
int peer_can(int source, int target) {
    struct { uint32_t before = peer_before; int value = -1; uint32_t after = peer_after; } driver, runtime;
    CONFORMANCE_CUDA(cuDeviceCanAccessPeer(&driver.value, source, target)); peer_runtime(cudaDeviceCanAccessPeer(&runtime.value, source, target));
    require(driver.before == peer_before && driver.after == peer_after && runtime.before == peer_before && runtime.after == peer_after, "peer capability output width");
    require((driver.value == 0 || driver.value == 1) && driver.value == runtime.value, "Driver/Runtime peer capability agreement"); return driver.value;
}
void peer_topology() {
    CUcontext original{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&original)); const auto count = peer_devices();
    for (int source = 0; source < count; ++source) for (int target = 0; target < count; ++target) {
        const auto accessible = peer_can(source, target);
        if (source == target) { require(!accessible, "self device is not a peer topology edge"); continue; }
        for (int attr = 1; attr <= 4; ++attr) {
            struct { uint32_t before = peer_before; int value = -1; uint32_t after = peer_after; } d, r;
            CONFORMANCE_CUDA(cuDeviceGetP2PAttribute(&d.value, static_cast<CUdevice_P2PAttribute>(attr), source, target));
            peer_runtime(cudaDeviceGetP2PAttribute(&r.value, static_cast<cudaDeviceP2PAttr>(attr), source, target));
            require(d.before == peer_before && d.after == peer_after && r.before == peer_before && r.after == peer_after, "peer attribute exact output extent");
            require(d.value == r.value && d.value >= 0 && (attr == 1 || d.value <= 1), "peer attribute value and facade agreement");
            if (attr == CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED) require(d.value == accessible, "peer access attribute agrees with directed topology");
        }
    }
    CUcontext after{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&after)); require(after == original, "topology queries preserve current context");
}
void peer_validation() {
    auto count = peer_devices(); CUcontext original{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&original)); CUdevice device{}; CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    int output{}; unsigned capability{}; CUatomicOperation op = CU_ATOMIC_OPERATION_INTEGER_ADD;
    cudaAtomicOperation runtime_op = cudaAtomicOperationIntegerAdd;
    require(cuDeviceCanAccessPeer(nullptr, device, device) == CUDA_ERROR_INVALID_VALUE && cudaDeviceCanAccessPeer(nullptr, device, device) == cudaErrorInvalidValue, "null peer topology outputs");
    for (int invalid : {-1, count}) {
        require(cuDeviceCanAccessPeer(&output, invalid, device) == CUDA_ERROR_INVALID_DEVICE && cuDeviceCanAccessPeer(&output, device, invalid) == CUDA_ERROR_INVALID_DEVICE, "Driver invalid topology endpoints");
        require(cudaDeviceCanAccessPeer(&output, invalid, device) == cudaErrorInvalidDevice && cudaDeviceCanAccessPeer(&output, device, invalid) == cudaErrorInvalidDevice, "Runtime invalid topology endpoints");
        require(cuDeviceGetP2PAttribute(&output, CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED, invalid, device) == CUDA_ERROR_INVALID_DEVICE, "invalid attribute device");
        require(cudaDeviceGetP2PAttribute(&output, cudaDevP2PAttrAccessSupported, device, invalid) == cudaErrorInvalidDevice, "Runtime invalid attribute device");
        require(cudaDeviceEnablePeerAccess(invalid, 0) == cudaErrorInvalidDevice && cudaDeviceDisablePeerAccess(invalid) == cudaErrorInvalidDevice, "Runtime invalid peer endpoint controls");
    }
    require(cuDeviceGetP2PAttribute(&output, CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED, device, device) == CUDA_ERROR_INVALID_DEVICE, "link attributes reject self device");
    require(cudaDeviceGetP2PAttribute(&output, cudaDevP2PAttrAccessSupported, device, device) == cudaErrorInvalidDevice, "Runtime link attributes reject self device");
    require(cuDeviceGetP2PAttribute(nullptr, CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED, device, device) == CUDA_ERROR_INVALID_VALUE, "null link attribute output");
    require(cudaDeviceGetP2PAttribute(nullptr, cudaDevP2PAttrAccessSupported, device, device) == cudaErrorInvalidValue, "Runtime null link attribute output");
    require(cuDeviceGetP2PAttribute(&output, static_cast<CUdevice_P2PAttribute>(99), device, device) == CUDA_ERROR_INVALID_VALUE, "invalid peer attribute");
    require(cudaDeviceGetP2PAttribute(&output, static_cast<cudaDeviceP2PAttr>(99), device, device) == cudaErrorInvalidValue, "Runtime invalid peer attribute");
    require(cuDeviceGetP2PAtomicCapabilities(nullptr, &op, 1, device, device) == CUDA_ERROR_INVALID_VALUE && cuDeviceGetP2PAtomicCapabilities(&capability, nullptr, 1, device, device) == CUDA_ERROR_INVALID_VALUE && cuDeviceGetP2PAtomicCapabilities(&capability, &op, 0, device, device) == CUDA_ERROR_INVALID_VALUE, "invalid atomic capability arrays");
    require(cudaDeviceGetP2PAtomicCapabilities(nullptr, &runtime_op, 1, device, device) == cudaErrorInvalidValue && cudaDeviceGetP2PAtomicCapabilities(&capability, nullptr, 1, device, device) == cudaErrorInvalidValue && cudaDeviceGetP2PAtomicCapabilities(&capability, &runtime_op, 0, device, device) == cudaErrorInvalidValue, "Runtime invalid atomic capability arrays");
    op = CU_ATOMIC_OPERATION_MAX; runtime_op = static_cast<cudaAtomicOperation>(CU_ATOMIC_OPERATION_MAX);
    require(cuDeviceGetP2PAtomicCapabilities(&capability, &op, 1, device, device) == CUDA_ERROR_INVALID_VALUE && cudaDeviceGetP2PAtomicCapabilities(&capability, &runtime_op, 1, device, device) == cudaErrorInvalidValue, "invalid atomic operation IDs");
    op = CU_ATOMIC_OPERATION_INTEGER_ADD; runtime_op = cudaAtomicOperationIntegerAdd;
    require(cuDeviceGetP2PAtomicCapabilities(&capability, &op, 1, device, device) == CUDA_ERROR_INVALID_DEVICE && cudaDeviceGetP2PAtomicCapabilities(&capability, &runtime_op, 1, device, device) == cudaErrorInvalidDevice, "atomic links reject identical devices");
    require(cuCtxEnablePeerAccess(original, 0) == CUDA_ERROR_INVALID_CONTEXT && cuCtxDisablePeerAccess(original) == CUDA_ERROR_INVALID_CONTEXT, "self context controls");
    require(cudaDeviceEnablePeerAccess(device, 0) == cudaErrorInvalidDevice && cudaDeviceDisablePeerAccess(device) == cudaErrorInvalidDevice, "Runtime self device controls");
    peer_context other(device); CONFORMANCE_CUDA(cuCtxSetCurrent(original));
    require(cuCtxEnablePeerAccess(other.value, 1) == CUDA_ERROR_INVALID_VALUE && cudaDeviceEnablePeerAccess(device, 1) == cudaErrorInvalidValue, "reserved peer flags");
    require(cuCtxEnablePeerAccess(nullptr, 0) == CUDA_ERROR_INVALID_CONTEXT && cuCtxDisablePeerAccess(nullptr) == CUDA_ERROR_INVALID_CONTEXT, "missing peer context");
    CONFORMANCE_CUDA(cuCtxSetCurrent(nullptr));
    auto enable = cuCtxEnablePeerAccess(other.value, 0), disable = cuCtxDisablePeerAccess(other.value);
    CONFORMANCE_CUDA(cuCtxSetCurrent(original)); require(enable == CUDA_ERROR_INVALID_CONTEXT && disable == CUDA_ERROR_INVALID_CONTEXT, "Driver peer controls require a current context");
    peer_buffer repaired(original); const auto expected = peer_words(13); repaired.write(expected); repaired.expect(expected, true);
}
void peer_unsupported() {
    const auto count = peer_devices();
    // Separate contexts on the same endpoint also provide an inaccessible
    // control on a host with only one native device.
    for (int source = 0; source < count; ++source) for (int target = 0; target < count; ++target) if (!peer_can(source, target)) {
        peer_context a(source), b(target); a.current();
        require(cuCtxEnablePeerAccess(b.value, 0) == CUDA_ERROR_PEER_ACCESS_UNSUPPORTED, "inaccessible context pair cannot enable");
        require(cuCtxDisablePeerAccess(b.value) == CUDA_ERROR_PEER_ACCESS_NOT_ENABLED, "rejected enable leaves the edge disabled");
        if (source != target) {
            require(cudaDeviceEnablePeerAccess(target, 0) == cudaErrorPeerAccessUnsupported, "Runtime inaccessible device pair cannot enable");
            require(cudaDeviceDisablePeerAccess(target) == cudaErrorPeerAccessNotEnabled, "Runtime rejected edge remains disabled");
        }
    }
}
void peer_atomics() {
    auto count = peer_devices(); require_capability(count > 1, "peer atomic capability queries require distinct native devices");
    constexpr unsigned n = CU_ATOMIC_OPERATION_MAX;
    for (int source = 0; source < count; ++source) for (int target = 0; target < count; ++target) if (source != target) {
        int partial{}, runtime_partial{};
        CONFORMANCE_CUDA(cuDeviceGetP2PAttribute(&partial, CU_DEVICE_P2P_ATTRIBUTE_ONLY_PARTIAL_NATIVE_ATOMIC_SUPPORTED, source, target));
        peer_runtime(cudaDeviceGetP2PAttribute(&runtime_partial, cudaDevP2PAttrOnlyPartialNativeAtomicSupported, source, target));
        require(partial == runtime_partial && (partial == 0 || partial == 1), "partial native atomic link metadata");
        std::array<CUatomicOperation, n> operations{}; std::array<cudaAtomicOperation, n> runtime_operations{};
        for (unsigned i = 0; i < n; ++i) { operations[i] = static_cast<CUatomicOperation>(i); runtime_operations[i] = static_cast<cudaAtomicOperation>(i); }
        std::array<unsigned, n + 2> driver{}, runtime{}; driver.fill(peer_before); runtime.fill(peer_before);
        CONFORMANCE_CUDA(cuDeviceGetP2PAtomicCapabilities(driver.data() + 1, operations.data(), n, source, target));
        peer_runtime(cudaDeviceGetP2PAtomicCapabilities(runtime.data() + 1, runtime_operations.data(), n, source, target));
        require(driver.front() == peer_before && driver.back() == peer_before && runtime.front() == peer_before && runtime.back() == peer_before, "atomic capability batch extent guards");
        require(driver == runtime, "atomic operation bitmasks agree through both facades");
        std::reverse(operations.begin(), operations.end()); std::array<unsigned, n> reversed{};
        CONFORMANCE_CUDA(cuDeviceGetP2PAtomicCapabilities(reversed.data(), operations.data(), n, source, target));
        for (unsigned i = 0; i < n; ++i) {
            unsigned single{}; auto op = static_cast<CUatomicOperation>(i);
            CONFORMANCE_CUDA(cuDeviceGetP2PAtomicCapabilities(&single, &op, 1, source, target));
            require(single == driver[i + 1] && single == reversed[n - 1 - i] && !(single & ~127u), "single and reordered atomic capability association");
        }
    }
}
void peer_fixture_device(int device) {
    int unified{}, major{}; CONFORMANCE_CUDA(cuDeviceGetAttribute(&unified, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device));
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    require_capability(unified && major >= 6, "peer execution fixture requires UVA and sm_60 or newer");
}
std::vector<std::pair<int, int>> peer_pairs() {
    std::vector<std::pair<int, int>> result; const auto count = peer_devices();
    for (int source = 0; source < count; ++source) for (int target = 0; target < count; ++target)
        if (source != target && peer_can(source, target)) result.emplace_back(source, target);
    require_capability(!result.empty(), "peer dereference requires an advertised directed pair of distinct native devices"); return result;
}
void peer_local() {
    CUdevice device{}; CUcontext context{}; CONFORMANCE_CUDA(cuCtxGetDevice(&device)); CONFORMANCE_CUDA(cuCtxGetCurrent(&context)); peer_fixture_device(device);
    for (bool library : {false, true}) {
        loaded_program program(peer_ptx, library); peer_buffer remote(context), local(context); auto r = peer_words(17), l = peer_words(23);
        remote.write(r); local.write(l, true); peer_launch(program, remote, local, 0x93278451); CONFORMANCE_CUDA(cuCtxSynchronize());
        peer_expected(r, l, 0x93278451); remote.expect(r, true); local.expect(l);
    }
}
void peer_visibility() {
    for (auto [source, target] : peer_pairs()) {
        peer_fixture_device(source); peer_fixture_device(target);
        for (bool runtime : {false, true}) for (bool library : {false, true}) {
            peer_context a(source), b(target, runtime); a.current(); loaded_program program(peer_ptx, library);
            peer_buffer before(b.value), local(a.value); auto r = peer_words(29), l = peer_words(31); before.write(r, runtime); local.write(l, !runtime);
            require(cuCtxDisablePeerAccess(b.value) == CUDA_ERROR_PEER_ACCESS_NOT_ENABLED, "fresh peer edge is disabled");
            if (runtime) peer_runtime(cudaDeviceEnablePeerAccess(target, 0)); else CONFORMANCE_CUDA(cuCtxEnablePeerAccess(b.value, 0));
            require(cuCtxEnablePeerAccess(b.value, 0) == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED, "duplicate Driver enable");
            if (runtime) require(cudaDeviceEnablePeerAccess(target, 0) == cudaErrorPeerAccessAlreadyEnabled, "Runtime observes Driver peer state");
            { peer_current reverse(b.value); require(cuCtxDisablePeerAccess(a.value) == CUDA_ERROR_PEER_ACCESS_NOT_ENABLED, "peer edge is unidirectional"); }
            peer_buffer after(b.value); auto r2 = peer_words(37); after.write(r2);
            for (auto* remote : {&before, &after}) {
                auto& expected = remote == &before ? r : r2;
                peer_launch(program, *remote, local, 0xc1375928); CONFORMANCE_CUDA(cuCtxSynchronize()); peer_expected(expected, l, 0xc1375928);
                remote->expect(expected, true); local.expect(l);
            }
            CONFORMANCE_CUDA(cuCtxDisablePeerAccess(b.value));
            require(cuCtxDisablePeerAccess(b.value) == CUDA_ERROR_PEER_ACCESS_NOT_ENABLED, "repeated disable");
            if (runtime) {
                require(cudaDeviceDisablePeerAccess(target) == cudaErrorPeerAccessNotEnabled, "Runtime observes disabled Driver edge");
                CONFORMANCE_CUDA(cuCtxEnablePeerAccess(b.value, 0)); peer_runtime(cudaDeviceDisablePeerAccess(target)); peer_runtime(cudaDeviceEnablePeerAccess(target, 0));
            } else CONFORMANCE_CUDA(cuCtxEnablePeerAccess(b.value, 0));
            peer_launch(program, before, local, 0x57a31928); CONFORMANCE_CUDA(cuCtxSynchronize()); peer_expected(r, l, 0x57a31928); before.expect(r); after.expect(r2); local.expect(l, true);
            CONFORMANCE_CUDA(cuCtxDisablePeerAccess(b.value));
        }
    }
}
struct peer_gate {
    std::mutex mutex; std::condition_variable changed; bool entered = false, open = false;
    static void CUDA_CB callback(void* data) {
        auto& gate = *static_cast<peer_gate*>(data); std::unique_lock lock(gate.mutex); gate.entered = true; gate.changed.notify_all();
        gate.changed.wait(lock, [&] { return gate.open; });
    }
    void wait() { std::unique_lock lock(mutex); require(changed.wait_for(lock, std::chrono::seconds(10), [&] { return entered; }), "peer host gate starts"); }
    void release() { std::lock_guard lock(mutex); open = true; changed.notify_all(); }
    ~peer_gate() { release(); }
};
void peer_lifetime() {
    for (auto [source, target] : peer_pairs()) for (bool destroy : {false, true}) {
        peer_fixture_device(source); peer_fixture_device(target); peer_context a(source), b(target); a.current();
        loaded_program program(peer_ptx, destroy); peer_buffer remote(b.value), local(a.value); auto r = peer_words(43), l = peer_words(47);
        remote.write(r); local.write(l); CONFORMANCE_CUDA(cuCtxEnablePeerAccess(b.value, 0));
        CUstream stream{}; CONFORMANCE_CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING)); peer_gate gate;
        CONFORMANCE_CUDA(cuLaunchHostFunc(stream, peer_gate::callback, &gate));
        try { peer_launch(program, remote, local, 0x712893a5, stream); gate.wait(); }
        catch (...) { gate.release(); cuStreamSynchronize(stream); cuStreamDestroy(stream); throw; }
        std::atomic<bool> started{}, done{}; CUresult status = CUDA_ERROR_UNKNOWN;
        std::thread retirement([&] {
            auto context_status = cuCtxSetCurrent(b.value); started.store(true, std::memory_order_release);
            status = context_status == CUDA_SUCCESS ? (destroy ? cuCtxDestroy(b.value) : cuMemFree(remote.value)) : context_status;
            done.store(true, std::memory_order_release);
        });
        try {
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (!started.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
            require(started.load(std::memory_order_acquire), "peer retirement caller starts");
            // The gate forbids completion; this is a bounded observation of a
            // host API, without requiring any particular GPU execution time.
            auto observation = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
            while (!done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < observation) std::this_thread::yield();
            require(!done.load(std::memory_order_acquire), "peer retirement waits for already accepted foreign work");
        } catch (...) { gate.release(); retirement.join(); cuStreamSynchronize(stream); cuStreamDestroy(stream); throw; }
        gate.release(); retirement.join(); remote.value = 0; if (destroy) b.value = nullptr;
        CONFORMANCE_CUDA(status); CONFORMANCE_CUDA(cuStreamSynchronize(stream)); CONFORMANCE_CUDA(cuStreamDestroy(stream));
        peer_expected(r, l, 0x712893a5); local.expect(l, true);
        if (!destroy) CONFORMANCE_CUDA(cuCtxDisablePeerAccess(b.value));
        peer_context replacement(target); a.current(); peer_buffer fresh(replacement.value); auto f = peer_words(53); fresh.write(f);
        CONFORMANCE_CUDA(cuCtxEnablePeerAccess(replacement.value, 0)); peer_launch(program, fresh, local, 0x193871a5); CONFORMANCE_CUDA(cuCtxSynchronize());
        peer_expected(f, l, 0x193871a5); fresh.expect(f); local.expect(l); CONFORMANCE_CUDA(cuCtxDisablePeerAccess(replacement.value));
    }
}
void peer_limits() {
    auto pairs = peer_pairs(); bool exhausted = false;
    for (int source = 0; source < peer_devices(); ++source) {
        peer_context a(source); a.current(); std::vector<std::unique_ptr<peer_context>> peers; std::vector<CUcontext> enabled;
        for (auto [from, target] : pairs) if (from == source) {
            peers.push_back(std::make_unique<peer_context>(target)); a.current(); auto handle = peers.back()->value;
            auto status = cuCtxEnablePeerAccess(handle, 0);
            if (status == CUDA_ERROR_TOO_MANY_PEERS) {
                exhausted = true; require(!enabled.empty(), "peer limit admits a nonempty neighbor set");
                for (auto active : enabled) require(cuCtxEnablePeerAccess(active, 0) == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED, "failed admission preserves active mappings");
                peer_fixture_device(source); peer_fixture_device(peers.front()->device);
                loaded_program program(peer_ptx, false); peer_buffer local(a.value), remote(enabled.front());
                auto l = peer_words(61), r = peer_words(67); local.write(l); remote.write(r);
                peer_launch(program, remote, local, 0x193785a1); CONFORMANCE_CUDA(cuCtxSynchronize());
                peer_expected(r, l, 0x193785a1); remote.expect(r); local.expect(l);
                require(cuCtxDisablePeerAccess(handle) == CUDA_ERROR_PEER_ACCESS_NOT_ENABLED, "rejected peer has no mapping");
                CONFORMANCE_CUDA(cuCtxDisablePeerAccess(enabled.back())); enabled.pop_back();
                CONFORMANCE_CUDA(cuCtxEnablePeerAccess(handle, 0)); enabled.push_back(handle); break;
            }
            CONFORMANCE_CUDA(status); enabled.push_back(handle);
        }
        for (auto handle : enabled) CONFORMANCE_CUDA(cuCtxDisablePeerAccess(handle));
    }
    require_capability(exhausted, "available topology does not exhaust native peer connection admission");
}
void peer_discovery() {
    for (auto [name, version] : std::array<std::pair<const char*, int>, 5>{{{"cuDeviceCanAccessPeer", 4000}, {"cuCtxEnablePeerAccess", 4000}, {"cuCtxDisablePeerAccess", 4000}, {"cuDeviceGetP2PAttribute", 8000}, {"cuDeviceGetP2PAtomicCapabilities", 13000}}}) {
        void* address{}; CUdriverProcAddressQueryResult result{};
        CONFORMANCE_CUDA(cuGetProcAddress(name, &address, version, 0, &result)); require(address && result == CU_GET_PROC_ADDRESS_SUCCESS, "peer procedure introduction boundary");
        CONFORMANCE_CUDA(cuGetProcAddress(name, &address, version - 1, 0, &result)); require(!address && result == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "peer procedure pre-introduction lookup");
    }
    void* address{}; CONFORMANCE_CUDA(cuGetProcAddress("cuDeviceCanAccessPeer", &address, 4000, 0, nullptr));
    require(reinterpret_cast<PFN_cuDeviceCanAccessPeer_v4000>(address)(nullptr, 0, 0) == CUDA_ERROR_NOT_INITIALIZED, "cold peer procedure behavior");
}
void execute_peer_access() {
    CUcontext original{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&original)); peer_current restore(original);
    if (peer_case == "topology") peer_topology();
    else if (peer_case == "validation") peer_validation();
    else if (peer_case == "unsupported") peer_unsupported();
    else if (peer_case == "atomics") peer_atomics();
    else if (peer_case == "local") peer_local();
    else if (peer_case == "visibility") peer_visibility();
    else if (peer_case == "lifetime") peer_lifetime();
    else if (peer_case == "limits") peer_limits();
    else throw std::runtime_error("unknown peer_access case");
}
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "discovery") return cudavm::run_host_conformance(argc, argv, cudavm::peer_discovery);
    if (argc == 3) { cudavm::peer_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_peer_access);
}
