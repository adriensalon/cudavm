#include "test_framework.hpp"
#include <cudaTypedefs.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <utility>

namespace cudavm {
std::string_view copy_case;
constexpr size_t copy_guard = 37;
CUresult peer_copy_ptds_driver(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, bool);
CUresult peer_copy_ptds_driver_3d(const CUDA_MEMCPY3D_PEER*, bool);
cudaError_t peer_copy_ptds_runtime(void*, int, const void*, int, size_t);
cudaError_t peer_copy_ptds_runtime_3d(const cudaMemcpy3DPeerParms*, bool);
CUresult peer_copy_ptds_gate(CUhostFn, void*);
CUresult peer_copy_ptds_sync();
void copy_runtime(cudaError_t status) { require(status == cudaSuccess, std::string("peer copy Runtime: ") + cudaGetErrorString(status)); }
struct copy_current {
    CUcontext previous{};
    explicit copy_current(CUcontext context) { CONFORMANCE_CUDA(cuCtxGetCurrent(&previous)); CONFORMANCE_CUDA(cuCtxSetCurrent(context)); }
    ~copy_current() { cuCtxSetCurrent(previous); }
};
struct copy_context {
    CUcontext value{};
    int device;
    explicit copy_context(int ordinal) : device(ordinal) {
        CONFORMANCE_CUDA(cuCtxCreate(&value, nullptr, 0, device)); CUcontext popped{};
        CONFORMANCE_CUDA(cuCtxPopCurrent(&popped)); require(popped == value, "peer copy context creation stack");
    }
    ~copy_context() { if (value) cuCtxDestroy(value); }
    void current() const { CONFORMANCE_CUDA(cuCtxSetCurrent(value)); }
};
std::vector<unsigned char> copy_pattern(size_t bytes, unsigned seed) {
    std::vector<unsigned char> result(bytes);
    for (size_t i = 0; i < bytes; ++i) result[i] = static_cast<unsigned char>((i * 73 + seed * 37) ^ (i >> 7));
    return result;
}
struct copy_buffer {
    CUcontext owner;
    int device;
    CUdeviceptr value{};
    std::vector<unsigned char> expected;
    copy_buffer(const copy_context& context, size_t bytes, unsigned seed) : owner(context.value), device(context.device), expected(copy_pattern(bytes, seed)) {
        copy_current scope(owner); CONFORMANCE_CUDA(cuMemAlloc(&value, bytes)); CONFORMANCE_CUDA(cuMemcpyHtoD(value, expected.data(), expected.size()));
    }
    ~copy_buffer() { if (value) { copy_current scope(owner); cuMemFree(value); } }
    void expect(bool runtime = false) const { copy_current scope(owner); virtual_memory_expect(value, expected, "peer copy complete allocation/padding/guards", runtime); }
    void* pointer(size_t offset = 0) const { return reinterpret_cast<void*>(uintptr_t(value + offset)); }
};
struct copy_stream {
    CUcontext owner;
    CUstream value{};
    explicit copy_stream(const copy_context& context) : owner(context.value) { copy_current scope(owner); CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~copy_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    void sync() const { CONFORMANCE_CUDA(cuStreamSynchronize(value)); }
};
struct copy_event {
    CUevent value{};
    explicit copy_event(const copy_context& owner) { copy_current scope(owner.value); CONFORMANCE_CUDA(cuEventCreate(&value, CU_EVENT_DISABLE_TIMING)); }
    ~copy_event() { if (value) cuEventDestroy(value); }
};
struct copy_gate {
    CUcontext owner{};
    std::mutex mutex; std::condition_variable changed; bool entered = false, open = false;
    static void CUDA_CB callback(void* data) {
        auto& gate = *static_cast<copy_gate*>(data); std::unique_lock lock(gate.mutex); gate.entered = true; gate.changed.notify_all();
        gate.changed.wait(lock, [&] { return gate.open; });
    }
    void wait() { std::unique_lock lock(mutex); require(changed.wait_for(lock, std::chrono::seconds(10), [&] { return entered; }), "peer copy host gate starts"); }
    void queue(CUstream stream) {
        CONFORMANCE_CUDA(cuStreamGetCtx(stream, &owner));
        copy_current scope(owner);
        CONFORMANCE_CUDA(cuLaunchHostFunc(stream, callback, this));
    }
    void release() { std::lock_guard lock(mutex); open = true; changed.notify_all(); }
    ~copy_gate() {
        release();
        // The callback must leave its mutex before this storage is destroyed,
        // including when an assertion unwinds before explicit stream cleanup.
        if (owner) { CUcontext previous{}; cuCtxGetCurrent(&previous); cuCtxSetCurrent(owner); cuCtxSynchronize(); cuCtxSetCurrent(previous); }
    }
};
void copy_returns(const copy_context& context, const std::function<void()>& action, const std::function<void()>& cleanup) {
    std::atomic<bool> done{}; std::exception_ptr error;
    std::thread caller([&] { try { context.current(); action(); } catch (...) { error = std::current_exception(); } done.store(true, std::memory_order_release); });
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    const bool returned = done.load(std::memory_order_acquire);
    if (!returned) cleanup(); caller.join();
    require(returned, "device peer copy returns while participating host work is gated"); if (error) std::rethrow_exception(error);
}
void copy_linear(bool runtime, bool async, copy_buffer& to, size_t destination, const copy_buffer& from, size_t source, size_t bytes, CUstream stream = nullptr) {
    CUcontext before{}, after{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&before));
    if (runtime) {
        if (async) copy_runtime(cudaMemcpyPeerAsync(to.pointer(destination), to.device, from.pointer(source), from.device, bytes, reinterpret_cast<cudaStream_t>(stream)));
        else copy_runtime(cudaMemcpyPeer(to.pointer(destination), to.device, from.pointer(source), from.device, bytes));
    } else if (async) CONFORMANCE_CUDA(cuMemcpyPeerAsync(to.value + destination, to.owner, from.value + source, from.owner, bytes, stream));
    else CONFORMANCE_CUDA(cuMemcpyPeer(to.value + destination, to.owner, from.value + source, from.owner, bytes));
    CONFORMANCE_CUDA(cuCtxGetCurrent(&after)); require(before == after, "peer copy preserves caller current context");
    std::copy_n(from.expected.begin() + source, bytes, to.expected.begin() + destination);
}
cudaMemcpy3DPeerParms copy_runtime_volume(const CUDA_MEMCPY3D_PEER& p, int source_device, int destination_device, size_t element) {
    cudaMemcpy3DPeerParms result{}; result.srcDevice = source_device; result.dstDevice = destination_device;
    if (p.srcMemoryType == CU_MEMORYTYPE_ARRAY) result.srcArray = reinterpret_cast<cudaArray_t>(p.srcArray);
    else result.srcPtr = {reinterpret_cast<void*>(uintptr_t(p.srcDevice)), p.srcPitch, p.srcPitch, p.srcHeight};
    if (p.dstMemoryType == CU_MEMORYTYPE_ARRAY) result.dstArray = reinterpret_cast<cudaArray_t>(p.dstArray);
    else result.dstPtr = {reinterpret_cast<void*>(uintptr_t(p.dstDevice)), p.dstPitch, p.dstPitch, p.dstHeight};
    result.srcPos = {p.srcMemoryType == CU_MEMORYTYPE_ARRAY ? p.srcXInBytes / element : p.srcXInBytes, p.srcY, p.srcZ};
    result.dstPos = {p.dstMemoryType == CU_MEMORYTYPE_ARRAY ? p.dstXInBytes / element : p.dstXInBytes, p.dstY, p.dstZ};
    result.extent = {p.WidthInBytes / element, p.Height, p.Depth}; return result;
}
void copy_volume(bool runtime, bool async, const CUDA_MEMCPY3D_PEER& p, int source, int destination, size_t element, CUstream stream = nullptr, bool ptds = false) {
    if (runtime) {
        auto descriptor = copy_runtime_volume(p, source, destination, element);
        if (ptds) copy_runtime(peer_copy_ptds_runtime_3d(&descriptor, async));
        else if (async) copy_runtime(cudaMemcpy3DPeerAsync(&descriptor, reinterpret_cast<cudaStream_t>(stream)));
        else copy_runtime(cudaMemcpy3DPeer(&descriptor));
        std::memset(&descriptor, 0x8d, sizeof(descriptor));
    } else if (ptds) CONFORMANCE_CUDA(peer_copy_ptds_driver_3d(&p, async));
    else if (async) CONFORMANCE_CUDA(cuMemcpy3DPeerAsync(&p, stream));
    else CONFORMANCE_CUDA(cuMemcpy3DPeer(&p));
}
void copy_expected(std::vector<unsigned char>& to, const std::vector<unsigned char>& from, const CUDA_MEMCPY3D_PEER& p, size_t to_base, size_t from_base,
                   size_t to_pitch, size_t to_height, size_t from_pitch, size_t from_height) {
    for (size_t z = 0; z < p.Depth; ++z) for (size_t y = 0; y < p.Height; ++y) {
        auto source = from_base + ((z + p.srcZ) * from_height + y + p.srcY) * from_pitch + p.srcXInBytes;
        auto destination = to_base + ((z + p.dstZ) * to_height + y + p.dstY) * to_pitch + p.dstXInBytes;
        std::copy_n(from.begin() + source, p.WidthInBytes, to.begin() + destination);
    }
}
void copy_linear_pair(const copy_context& source, const copy_context& destination, const copy_context& executing, bool large) {
    const size_t bytes = large ? 1024 * 1024 + 313 : 8197;
    copy_buffer a(source, bytes, 19), b(destination, bytes, 31); copy_stream stream(executing); executing.current();
    if (source.device != destination.device) {
        require(cudaMemcpyPeer(b.pointer(), destination.device, a.pointer(), destination.device, 1) == cudaErrorInvalidDevice, "Runtime source pointer/device mismatch");
        require(cudaMemcpyPeerAsync(b.pointer(), source.device, a.pointer(), source.device, 1, reinterpret_cast<cudaStream_t>(stream.value)) == cudaErrorInvalidDevice, "Runtime destination pointer/device mismatch");
    }
    for (bool reverse : {false, true}) for (bool runtime : {false, true}) for (bool async : {false, true}) {
        auto& to = reverse ? a : b; auto& from = reverse ? b : a;
        for (size_t count : {size_t(0), size_t(1), size_t(17), size_t(255), size_t(4099), large ? size_t(1024 * 1024 + 37) : size_t(6113)}) {
            copy_linear(runtime, async, to, 71, from, 43, count, stream.value); if (async) stream.sync(); to.expect(runtime); from.expect(!runtime);
        }
    }
}
CUDA_MEMCPY3D_PEER copy_pitched(const copy_buffer& to, const copy_buffer& from) {
    CUDA_MEMCPY3D_PEER p{}; p.srcMemoryType = p.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    p.srcContext = from.owner; p.dstContext = to.owner; p.srcDevice = from.value + copy_guard; p.dstDevice = to.value + copy_guard;
    p.srcPitch = 83; p.srcHeight = 7; p.dstPitch = 101; p.dstHeight = 8;
    p.srcXInBytes = 7; p.srcY = 2; p.srcZ = 1; p.dstXInBytes = 11; p.dstY = 3; p.dstZ = 2;
    p.WidthInBytes = 23; p.Height = 3; p.Depth = 2; return p;
}
void copy_geometry_pair(const copy_context& source, const copy_context& destination, const copy_context& executing) {
    copy_buffer from(source, copy_guard * 2 + 83 * 7 * 5, 43), to(destination, copy_guard * 2 + 101 * 8 * 6, 59);
    copy_stream stream(executing); executing.current();
    for (bool runtime : {false, true}) for (bool async : {false, true}) {
        auto p = copy_pitched(to, from); p.dstXInBytes += runtime ? 29 : 0; p.dstZ -= async ? 1 : 0;
        copy_expected(to.expected, from.expected, p, copy_guard, copy_guard, 101, 8, 83, 7);
        copy_volume(runtime, async, p, source.device, destination.device, 1, stream.value); std::memset(&p, 0x8d, sizeof(p));
        if (async) stream.sync(); to.expect(runtime); from.expect(!runtime);
    }
}
struct copy_array {
    CUcontext owner;
    int device;
    CUarray value{};
    size_t width, height, depth, element;
    std::vector<unsigned char> expected;
    copy_array(const copy_context& context, size_t w, size_t h, size_t d, CUarray_format format, unsigned channels, size_t bytes_per_element, unsigned seed)
        : owner(context.value), device(context.device), width(w), height(h), depth(d), element(bytes_per_element), expected(copy_pattern(w * h * d * bytes_per_element, seed)) {
        copy_current scope(owner); CUDA_ARRAY3D_DESCRIPTOR descriptor{}; descriptor.Width = w; descriptor.Height = h; descriptor.Depth = d; descriptor.Format = format; descriptor.NumChannels = channels;
        CONFORMANCE_CUDA(cuArray3DCreate(&value, &descriptor)); transfer(expected.data(), true);
    }
    ~copy_array() { if (value) { copy_current scope(owner); cuArrayDestroy(value); } }
    void transfer(void* host, bool upload) const {
        copy_current scope(owner); CUDA_MEMCPY3D p{}; p.WidthInBytes = width * element; p.Height = height; p.Depth = depth;
        if (upload) { p.srcMemoryType = CU_MEMORYTYPE_HOST; p.srcHost = host; p.srcPitch = width * element; p.srcHeight = height; p.dstMemoryType = CU_MEMORYTYPE_ARRAY; p.dstArray = value; }
        else { p.dstMemoryType = CU_MEMORYTYPE_HOST; p.dstHost = host; p.dstPitch = width * element; p.dstHeight = height; p.srcMemoryType = CU_MEMORYTYPE_ARRAY; p.srcArray = value; }
        CONFORMANCE_CUDA(cuMemcpy3D(&p));
    }
    void expect() const {
        std::vector<unsigned char> actual(expected.size() + 32, 0xd7); transfer(actual.data() + 16, false);
        for (size_t i = 0; i < actual.size(); ++i) require(actual[i] == (i < 16 || i >= expected.size() + 16 ? 0xd7 : expected[i - 16]), "peer copy complete array/borders at byte " + std::to_string(i));
    }
};
void copy_arrays_pair(const copy_context& source, const copy_context& destination, const copy_context& executing, bool all_formats) {
    for (unsigned format = 0; format < (all_formats ? 3u : 1u); ++format) {
        const size_t element = format == 0 ? 1 : format == 1 ? 4 : 16;
        auto kind = format == 0 ? CU_AD_FORMAT_UNSIGNED_INT8 : format == 1 ? CU_AD_FORMAT_UNSIGNED_INT16 : CU_AD_FORMAT_UNSIGNED_INT32;
        auto channels = format == 0 ? 1u : format == 1 ? 2u : 4u;
        copy_array from(source, 19, 6, 4, kind, channels, element, 71), to(destination, 23, 7, 5, kind, channels, element, 89);
        const auto pitch = element * 29 + 17; copy_buffer linear_source(source, copy_guard * 2 + pitch * 8 * 6, 97), linear_to(destination, copy_guard * 2 + pitch * 8 * 6, 103);
        copy_stream stream(executing); executing.current();
        for (unsigned route = 0; route < 3; ++route) for (bool runtime : {false, true}) for (bool async : {false, true}) {
            CUDA_MEMCPY3D_PEER p{}; p.srcContext = source.value; p.dstContext = destination.value;
            p.srcMemoryType = route == 2 ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_ARRAY; p.srcArray = from.value;
            p.dstMemoryType = route == 1 ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_ARRAY; p.dstArray = to.value;
            p.srcDevice = linear_source.value + copy_guard; p.dstDevice = linear_to.value + copy_guard;
            p.srcPitch = p.dstPitch = pitch; p.srcHeight = p.dstHeight = 8;
            p.srcXInBytes = route == 2 ? 7 : element * 2; p.srcY = 1; p.srcZ = 1;
            p.dstXInBytes = route == 1 ? 11 : element * 3; p.dstY = 2; p.dstZ = async ? 1 : 2;
            p.WidthInBytes = element * 7; p.Height = 3; p.Depth = 2;
            if (source.device != destination.device && route == 0 && !runtime && !async) {
                auto wrong = copy_runtime_volume(p, source.device, destination.device, element); wrong.srcDevice = destination.device;
                require(cudaMemcpy3DPeer(&wrong) == cudaErrorInvalidDevice, "Runtime source array/device mismatch");
                wrong = copy_runtime_volume(p, source.device, destination.device, element); wrong.dstDevice = source.device;
                require(cudaMemcpy3DPeerAsync(&wrong, reinterpret_cast<cudaStream_t>(stream.value)) == cudaErrorInvalidDevice, "Runtime destination array/device mismatch");
            }
            auto& expected_to = route == 1 ? linear_to.expected : to.expected; const auto& expected_from = route == 2 ? linear_source.expected : from.expected;
            copy_expected(expected_to, expected_from, p, route == 1 ? copy_guard : 0, route == 2 ? copy_guard : 0,
                route == 1 ? pitch : to.width * element, route == 1 ? 8 : to.height,
                route == 2 ? pitch : from.width * element, route == 2 ? 8 : from.height);
            copy_volume(runtime, async, p, source.device, destination.device, element, stream.value); std::memset(&p, 0x8d, sizeof(p));
            if (async) stream.sync(); to.expect(); from.expect(); linear_source.expect(); linear_to.expect();
        }
    }
}
void copy_streams(int device) {
    copy_context a(device), b(device), c(device); copy_buffer from(a, 1024, 107), to(b, 1024, 109);
    copy_stream sa(a), sb(b), sc(c); c.current();
    for (auto* stream : {&sa, &sb, &sc}) for (bool runtime : {false, true}) { copy_linear(runtime, true, to, 73, from, 47, 113, stream->value); stream->sync(); to.expect(runtime); from.expect(!runtime); }
    CONFORMANCE_CUDA(cuCtxSetCurrent(nullptr));
    auto status = cuMemcpyPeerAsync(to.value + 19, b.value, from.value + 7, a.value, 37, sa.value);
    auto synchronized = cuStreamSynchronize(sa.value); CUcontext after{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&after)); c.current();
    CONFORMANCE_CUDA(status); CONFORMANCE_CUDA(synchronized); require(!after, "explicit Driver peer stream needs no current context");
    std::copy_n(from.expected.begin() + 7, 37, to.expected.begin() + 19); to.expect(); from.expect();
}
void copy_ptds(int device) {
    copy_context a(device), b(device), c(device); copy_buffer from(a, 1024, 113), to(b, 1024, 127); c.current();
    CONFORMANCE_CUDA(peer_copy_ptds_driver(to.value + 37, b.value, from.value + 43, a.value, 151, false));
    std::copy_n(from.expected.begin() + 43, 151, to.expected.begin() + 37); to.expect(); from.expect();
    CUDA_MEMCPY3D_PEER p{}; p.srcMemoryType = p.dstMemoryType = CU_MEMORYTYPE_DEVICE; p.srcDevice = from.value; p.dstDevice = to.value;
    p.srcContext = a.value; p.dstContext = b.value; p.srcPitch = p.dstPitch = 1024; p.srcXInBytes = 79; p.dstXInBytes = 257; p.WidthInBytes = 137; p.Height = p.Depth = 1;
    for (bool runtime : {false, true}) { copy_volume(runtime, false, p, device, device, 1, nullptr, true); std::copy_n(from.expected.begin() + 79, 137, to.expected.begin() + 257); to.expect(); }
    copy_gate gate; gate.owner = c.value; CUresult accepted{};
    std::thread other([&] { accepted = cuCtxSetCurrent(c.value); if (accepted == CUDA_SUCCESS) accepted = peer_copy_ptds_gate(copy_gate::callback, &gate); }); other.join(); CONFORMANCE_CUDA(accepted);
    try {
        gate.wait();
        for (bool runtime : {false, true}) {
            copy_returns(c, [&] {
                if (runtime) copy_runtime(peer_copy_ptds_runtime(to.pointer(451), device, from.pointer(5), device, 71));
                else CONFORMANCE_CUDA(peer_copy_ptds_driver(to.value + 451, b.value, from.value + 5, a.value, 71, true));
                CONFORMANCE_CUDA(peer_copy_ptds_sync());
            }, [&] { gate.release(); });
            std::copy_n(from.expected.begin() + 5, 71, to.expected.begin() + 451);
        }
        for (bool runtime : {false, true}) {
            copy_returns(c, [&] { copy_volume(runtime, true, p, device, device, 1, nullptr, true); CONFORMANCE_CUDA(peer_copy_ptds_sync()); }, [&] { gate.release(); });
            std::copy_n(from.expected.begin() + 79, 137, to.expected.begin() + 257);
        }
        to.expect(); from.expect();
    } catch (...) { gate.release(); throw; }
    gate.release();
}
struct copy_host {
    CUcontext owner;
    void* value{};
    size_t bytes;
    copy_host(const copy_context& context, size_t size) : owner(context.value), bytes(size) {
        copy_current scope(owner); CONFORMANCE_CUDA(cuMemAllocHost(&value, bytes + 32)); std::memset(value, 0xa7, bytes + 32);
    }
    ~copy_host() { if (value) cuMemFreeHost(value); }
    void* payload() const { return static_cast<unsigned char*>(value) + 16; }
    void expect(const std::vector<unsigned char>& expected) const {
        const auto* actual = static_cast<const unsigned char*>(value);
        require(expected.size() == bytes, "peer snapshot oracle extent");
        for (size_t i = 0; i < bytes + 32; ++i) require(actual[i] == (i < 16 || i >= bytes + 16 ? 0xa7 : expected[i - 16]), "peer ordered snapshot/host guards at byte " + std::to_string(i));
    }
};
void copy_plain_ordering(int device) {
    for (bool runtime : {false, true}) for (unsigned gated = 0; gated < 3; ++gated) {
        copy_context a(device), b(device), c(device); copy_buffer from(a, 4099, 131), to(b, 4099, 137), marker(c, 257, 139);
        copy_stream sa(a), sb(b), sc(c); copy_host snapshot(b, to.expected.size()); copy_event finished(b); copy_gate gate;
        copy_stream* blocked[]{&sa, &sb, &sc};
        gate.queue(blocked[gated]->value);
        try {
            gate.wait();
            { copy_current scope(a.value); CONFORMANCE_CUDA(cuMemsetD8Async(from.value + 43, 0x51, 1001, sa.value)); }
            std::fill_n(from.expected.begin() + 43, 1001, 0x51);
            { copy_current scope(b.value); CONFORMANCE_CUDA(cuMemsetD8Async(to.value + 71, 0x23, 1001, sb.value)); }
            std::fill_n(to.expected.begin() + 71, 1001, 0x23);
            c.current(); copy_returns(c, [&] { copy_linear(runtime, false, to, 71, from, 43, 1001); }, [&] { gate.release(); });
            { copy_current scope(a.value); CONFORMANCE_CUDA(cuMemsetD8Async(from.value + 43, 0x96, 1001, sa.value)); }
            std::fill_n(from.expected.begin() + 43, 1001, 0x96);
            { copy_current scope(b.value); CONFORMANCE_CUDA(cuMemcpyDtoHAsync(snapshot.payload(), to.value, to.expected.size(), sb.value)); CONFORMANCE_CUDA(cuEventRecord(finished.value, sb.value)); }
            CONFORMANCE_CUDA(cuMemsetD8Async(marker.value + 17, 0x83, 127, sc.value)); std::fill_n(marker.expected.begin() + 17, 127, 0x83);
            auto observation = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
            while (std::chrono::steady_clock::now() < observation) {
                require(cuEventQuery(finished.value) == CUDA_ERROR_NOT_READY, "future endpoint work remains behind the gated plain peer copy"); std::this_thread::yield();
            }
            gate.release(); sa.sync(); sb.sync(); sc.sync(); snapshot.expect(to.expected); to.expect(runtime); from.expect(!runtime); marker.expect();
        } catch (...) { gate.release(); throw; }
    }
}
void copy_async_ordering(int device) {
    for (bool runtime : {false, true}) {
        copy_context a(device), b(device), c(device); copy_buffer from(a, 4099, 149), to(b, 4099, 151), snapshot(c, 4099, 157);
        copy_stream sa(a), sb(b), sc(c); copy_event produced(a); copy_host host(c, to.expected.size()); copy_gate source_gate, destination_gate;
        source_gate.queue(sa.value); destination_gate.queue(sb.value);
        try {
            source_gate.wait(); destination_gate.wait();
            { copy_current scope(a.value); CONFORMANCE_CUDA(cuMemsetD8Async(from.value + 43, 0x67, 1001, sa.value)); CONFORMANCE_CUDA(cuEventRecord(produced.value, sa.value)); }
            std::fill_n(from.expected.begin() + 43, 1001, 0x67); c.current(); CONFORMANCE_CUDA(cuStreamWaitEvent(sc.value, produced.value, 0));
            copy_returns(c, [&] { copy_linear(runtime, true, to, 71, from, 43, 1001, sc.value); }, [&] { source_gate.release(); destination_gate.release(); });
            source_gate.release();
            copy_returns(c, [&] {
                copy_linear(!runtime, true, snapshot, 0, to, 0, to.expected.size(), sc.value);
                CONFORMANCE_CUDA(cuMemcpyDtoHAsync(host.payload(), snapshot.value, snapshot.expected.size(), sc.value)); sc.sync();
            }, [&] { destination_gate.release(); });
            host.expect(to.expected); from.expect(); destination_gate.release(); to.expect(runtime); snapshot.expect(!runtime);
        } catch (...) { source_gate.release(); destination_gate.release(); throw; }
    }
}
void copy_retirement(int device) {
    for (unsigned route = 0; route < 5; ++route) {
        const bool array = route == 2 || route == 3, runtime = route == 1 || route == 3, destroy_context = route == 4;
        copy_context a(device), b(device), c(device); copy_buffer from(a, 4099, 163), to(b, 4099, 167);
        copy_array source_array(a, 19, 6, 4, CU_AD_FORMAT_UNSIGNED_INT16, 2, 4, 173), destination_array(b, 23, 7, 5, CU_AD_FORMAT_UNSIGNED_INT16, 2, 4, 179);
        copy_stream stream(c); copy_gate gate; c.current(); gate.queue(stream.value);
        if (array) {
            CUDA_MEMCPY3D_PEER p{}; p.srcMemoryType = p.dstMemoryType = CU_MEMORYTYPE_ARRAY; p.srcArray = source_array.value; p.dstArray = destination_array.value;
            p.srcContext = a.value; p.dstContext = b.value; p.srcXInBytes = 8; p.srcY = p.srcZ = 1; p.dstXInBytes = 12; p.dstY = p.dstZ = 2; p.WidthInBytes = 28; p.Height = 3; p.Depth = 2;
            copy_expected(destination_array.expected, source_array.expected, p, 0, 0, 23 * 4, 7, 19 * 4, 6);
            copy_volume(runtime, true, p, device, device, 4, stream.value);
        } else copy_linear(runtime, true, to, 71, from, 43, 1001, stream.value);
        gate.wait(); std::atomic<bool> started{}, done{}; int status = -1;
        std::thread retirement([&] {
            status = int(cuCtxSetCurrent(a.value)); started.store(true, std::memory_order_release);
            if (!status) {
                if (destroy_context) status = int(cuCtxDestroy(a.value));
                else if (array) status = runtime ? int(cudaFreeArray(reinterpret_cast<cudaArray_t>(source_array.value))) : int(cuArrayDestroy(source_array.value));
                else status = runtime ? int(cudaFree(from.pointer())) : int(cuMemFree(from.value));
            }
            done.store(true, std::memory_order_release);
        });
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!started.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
        auto observation = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
        while (!done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < observation) std::this_thread::yield();
        const bool pending = started.load(std::memory_order_acquire) && !done.load(std::memory_order_acquire);
        gate.release(); retirement.join();
        if (!status) { if (array || destroy_context) source_array.value = nullptr; if (!array || destroy_context) from.value = 0; if (destroy_context) a.value = nullptr; }
        require(pending, "source retirement waits for the gated accepted peer transfer"); require(status == 0, "source peer retirement succeeds, status=" + std::to_string(status));
        stream.sync(); if (array) destination_array.expect(); else to.expect(runtime);
    }
}
void copy_validation(int device) {
    copy_context a(device), b(device), c(device); copy_buffer from(a, 4099, 181), to(b, 4099, 191); copy_stream stream(c); c.current();
    int count{}; CONFORMANCE_CUDA(cuDeviceGetCount(&count));
    for (int invalid : {-1, count}) {
        require(cudaMemcpyPeer(to.pointer(), invalid, from.pointer(), device, 1) == cudaErrorInvalidDevice, "invalid Runtime destination device");
        require(cudaMemcpyPeerAsync(to.pointer(), device, from.pointer(), invalid, 1, reinterpret_cast<cudaStream_t>(stream.value)) == cudaErrorInvalidDevice, "invalid Runtime source device");
    }
    require(cuMemcpyPeer(to.value, nullptr, from.value, a.value, 1) == CUDA_ERROR_INVALID_CONTEXT, "missing peer destination context");
    require(cuMemcpyPeerAsync(to.value, b.value, from.value, nullptr, 1, stream.value) == CUDA_ERROR_INVALID_CONTEXT, "missing peer source context");
    require(cuMemcpyPeer(to.value, a.value, from.value, a.value, 1) == CUDA_ERROR_INVALID_CONTEXT, "declared pointer context ownership");
    require(cuMemcpyPeer(to.value, b.value, from.value + from.expected.size(), a.value, 1) == CUDA_ERROR_INVALID_VALUE, "source copy range bound");
    require(cudaMemcpyPeer(to.pointer(to.expected.size()), device, from.pointer(), device, 1) == cudaErrorInvalidValue, "Runtime destination range bound");
    require(cuMemcpyPeer(from.value + 1, a.value, from.value, a.value, 257) == CUDA_ERROR_INVALID_VALUE, "overlapping peer ranges");
    require(cudaMemcpyPeer(from.pointer(1), device, from.pointer(), device, 257) == cudaErrorInvalidValue, "Runtime overlapping peer ranges");
    auto invalid_stream = reinterpret_cast<CUstream>(uintptr_t(0x123));
    require(cuMemcpyPeerAsync(to.value, b.value, from.value, a.value, 1, invalid_stream) == CUDA_ERROR_INVALID_HANDLE, "invalid peer stream");
    require(cudaMemcpyPeerAsync(to.pointer(), device, from.pointer(), device, 1, reinterpret_cast<cudaStream_t>(invalid_stream)) == cudaErrorInvalidResourceHandle, "Runtime invalid peer stream");
    require(cuMemcpy3DPeer(nullptr) == CUDA_ERROR_INVALID_VALUE && cuMemcpy3DPeerAsync(nullptr, stream.value) == CUDA_ERROR_INVALID_VALUE, "missing Driver peer descriptors");
    require(cudaMemcpy3DPeer(nullptr) == cudaErrorInvalidValue && cudaMemcpy3DPeerAsync(nullptr, reinterpret_cast<cudaStream_t>(stream.value)) == cudaErrorInvalidValue, "missing Runtime peer descriptors");
    auto repaired = copy_pitched(to, from);
    for (bool source : {false, true}) {
        auto p = repaired; if (source) p.srcLOD = 1; else p.dstLOD = 1;
        require(cuMemcpy3DPeer(&p) == CUDA_ERROR_INVALID_VALUE && cuMemcpy3DPeerAsync(&p, stream.value) == CUDA_ERROR_INVALID_VALUE, "peer copy LOD must be zero");
        p = repaired; if (source) p.srcMemoryType = static_cast<CUmemorytype>(0); else p.dstMemoryType = static_cast<CUmemorytype>(0);
        require(cuMemcpy3DPeer(&p) == CUDA_ERROR_INVALID_VALUE, "invalid peer memory type");
        p = repaired; if (source) p.srcPitch = 3; else p.dstPitch = 3;
        require(cuMemcpy3DPeer(&p) == CUDA_ERROR_INVALID_VALUE, "Driver pitch is too small");
        auto runtime = copy_runtime_volume(p, device, device, 1);
        require(cudaMemcpy3DPeer(&runtime) == cudaErrorInvalidPitchValue, "Runtime peer pitch error");
    }
    auto overflow = repaired; overflow.srcXInBytes = SIZE_MAX;
    require(cuMemcpy3DPeer(&overflow) == CUDA_ERROR_INVALID_VALUE, "peer geometry arithmetic overflow");
    copy_array small(a, 19, 6, 4, CU_AD_FORMAT_UNSIGNED_INT8, 1, 1, 193), wide(b, 23, 7, 5, CU_AD_FORMAT_UNSIGNED_INT16, 1, 2, 197);
    cudaMemcpy3DPeerParms incompatible{}; incompatible.srcArray = reinterpret_cast<cudaArray_t>(small.value); incompatible.dstArray = reinterpret_cast<cudaArray_t>(wide.value);
    incompatible.srcDevice = incompatible.dstDevice = device; incompatible.extent = {7, 2, 2};
    require(cudaMemcpy3DPeer(&incompatible) == cudaErrorInvalidValue, "Runtime incompatible peer array element sizes");
    incompatible.srcPtr = {from.pointer(), 19, 19, 6};
    require(cudaMemcpy3DPeerAsync(&incompatible, reinterpret_cast<cudaStream_t>(stream.value)) == cudaErrorInvalidValue, "multiple Runtime peer source endpoints");
    small.expect(); wide.expect(); to.expect(); from.expect();
    copy_linear(false, false, to, 71, from, 43, 1001); to.expect(true); from.expect();
    copy_expected(to.expected, from.expected, repaired, copy_guard, copy_guard, 101, 8, 83, 7);
    copy_volume(true, true, repaired, device, device, 1, stream.value); stream.sync(); to.expect(); from.expect(true);
}
std::string copy_device_name(int device) { char name[256]{}; CONFORMANCE_CUDA(cuDeviceGetName(name, sizeof(name), device)); return name; }
void copy_devices(bool mixed) {
    int count{}; CONFORMANCE_CUDA(cuDeviceGetCount(&count)); unsigned exercised = 0;
    for (int source = 0; source < count; ++source) for (int destination = 0; destination < count; ++destination) {
        if (source == destination) continue;
        const auto a_name = copy_device_name(source), b_name = copy_device_name(destination);
        if (mixed && a_name.substr(a_name.rfind(" (")) == b_name.substr(b_name.rfind(" ("))) continue;
        std::cout << "peer copy source=" << a_name << " destination=" << b_name << std::endl;
        copy_context a(source), b(destination), c(source); c.current();
        copy_linear_pair(a, b, c, mixed); copy_geometry_pair(a, b, c); copy_arrays_pair(a, b, c, false); ++exercised;
    }
    require_capability(exercised != 0, mixed ? "peer mixed profile requires different backend endpoints" : "peer device profile requires distinct devices in this backend");
}
void copy_discovery() {
    for (const char* name : {"cuMemcpyPeer", "cuMemcpyPeerAsync", "cuMemcpy3DPeer", "cuMemcpy3DPeerAsync"}) for (auto flags : {CU_GET_PROC_ADDRESS_LEGACY_STREAM, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM}) {
        const int version = flags == CU_GET_PROC_ADDRESS_LEGACY_STREAM ? 4000 : 7000; void* address{}; CUdriverProcAddressQueryResult result{};
        CONFORMANCE_CUDA(cuGetProcAddress(name, &address, version, flags, &result)); require(address && result == CU_GET_PROC_ADDRESS_SUCCESS, "peer copy procedure introduction");
        if (flags == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) {
            void* legacy{}; CONFORMANCE_CUDA(cuGetProcAddress(name, &legacy, 4000, CU_GET_PROC_ADDRESS_LEGACY_STREAM, nullptr));
            require(address != legacy, "per-thread peer copy procedure has its own entry");
            CONFORMANCE_CUDA(cuGetProcAddress(name, &address, 6999, flags, &result)); require(!address && result == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "pre-PTDS request cannot substitute legacy stream semantics");
            cudaDriverEntryPointQueryResult runtime_result{};
            copy_runtime(cudaGetDriverEntryPointByVersion(name, &address, 6999, flags, &runtime_result));
            require(!address && runtime_result == cudaDriverEntryPointVersionNotSufficent, "Runtime peer PTDS introduction boundary");
        }
        CONFORMANCE_CUDA(cuGetProcAddress(name, &address, 3999, flags, &result)); require(!address && result == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "peer copy procedure version boundary");
    }
    void* address{}; CONFORMANCE_CUDA(cuGetProcAddress("cuMemcpyPeer", &address, 4000, CU_GET_PROC_ADDRESS_LEGACY_STREAM, nullptr));
    require(reinterpret_cast<PFN_cuMemcpyPeer_v4000>(address)(0, nullptr, 0, nullptr, 0) == CUDA_ERROR_NOT_INITIALIZED, "cold peer copy procedure");
}
void execute_peer_copy() {
    CUcontext original{}; CUdevice device{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&original)); CONFORMANCE_CUDA(cuCtxGetDevice(&device)); copy_current restore(original);
    if (copy_case == "linear" || copy_case == "geometry" || copy_case == "arrays") {
        copy_context a(device), b(device), c(device); c.current();
        if (copy_case == "linear") copy_linear_pair(a, b, c, true);
        else if (copy_case == "geometry") copy_geometry_pair(a, b, c);
        else copy_arrays_pair(a, b, c, true);
    } else if (copy_case == "streams") copy_streams(device);
    else if (copy_case == "ptds") copy_ptds(device);
    else if (copy_case == "ordering") { copy_plain_ordering(device); copy_async_ordering(device); }
    else if (copy_case == "retirement") copy_retirement(device);
    else if (copy_case == "validation") copy_validation(device);
    else if (copy_case == "devices") copy_devices(false);
    else if (copy_case == "mixed") copy_devices(true);
    else throw std::runtime_error("unknown peer_copy case");
}
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "discovery") return cudavm::run_host_conformance(argc, argv, cudavm::copy_discovery);
    if (argc == 3) { cudavm::copy_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_peer_copy);
}
