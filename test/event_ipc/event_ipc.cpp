#include "test_framework.hpp"
#include <cudaTypedefs.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <type_traits>

namespace cudavm {
std::string_view event_ipc_case;
const char* event_ipc_executable{};
std::unique_ptr<test_peer> event_ipc_peer;
constexpr size_t event_ipc_bytes = 1024;
void event_ipc_runtime(cudaError_t status) { require(status == cudaSuccess, std::string("event IPC Runtime: ") + cudaGetErrorString(status)); }
template<class T> struct event_ipc_output {
    std::array<unsigned char, 16> before; T value{}; std::array<unsigned char, 16> after;
    event_ipc_output() { before.fill(0x63); after.fill(0xa7); }
    T get() const {
        require(std::all_of(before.begin(), before.end(), [](auto v) { return v == 0x63; })
            && std::all_of(after.begin(), after.end(), [](auto v) { return v == 0xa7; }), "IPC output exact extents and guards"); return value;
    }
};
bool event_ipc_supported() {
    CUdevice device{}; int supported{}, uva{}, runtime{};
    TEST_CUDA(cuCtxGetDevice(&device));
    TEST_CUDA(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED, device));
    TEST_CUDA(cuDeviceGetAttribute(&uva, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device));
    event_ipc_runtime(cudaDeviceGetAttribute(&runtime, cudaDevAttrIpcEventSupport, device));
    require(runtime == supported, "Driver/Runtime IPC event capability agreement");
    return supported && uva;
}
struct event_ipc_event {
    CUevent value{};
    explicit event_ipc_event(bool runtime, unsigned flags = CU_EVENT_INTERPROCESS | CU_EVENT_DISABLE_TIMING) {
        event_ipc_output<CUevent> output;
        if (runtime) {
            cudaEvent_t event{}; event_ipc_runtime(cudaEventCreateWithFlags(&event, flags)); output.value = reinterpret_cast<CUevent>(event);
        } else TEST_CUDA(cuEventCreate(&output.value, flags));
        value = output.get(); require(value, "event creation publishes a handle");
    }
    explicit event_ipc_event(CUipcEventHandle record, bool runtime) {
        event_ipc_output<CUevent> output;
        if (runtime) {
            cudaIpcEventHandle_t handle{}; std::memcpy(&handle, &record, sizeof(handle));
            event_ipc_output<cudaEvent_t> result; event_ipc_runtime(cudaIpcOpenEventHandle(&result.value, handle)); output.value = reinterpret_cast<CUevent>(result.get());
        } else TEST_CUDA(cuIpcOpenEventHandle(&output.value, record));
        value = output.get(); require(value, "event import publishes a handle");
    }
    ~event_ipc_event() { if (value) cuEventDestroy(value); }
    void close(bool runtime) {
        if (runtime) event_ipc_runtime(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(value)));
        else TEST_CUDA(cuEventDestroy(value)); value = nullptr;
    }
    CUipcEventHandle export_handle(bool runtime) const {
        event_ipc_output<CUipcEventHandle> result;
        if (runtime) {
            event_ipc_output<cudaIpcEventHandle_t> output; event_ipc_runtime(cudaIpcGetEventHandle(&output.value, reinterpret_cast<cudaEvent_t>(value)));
            auto handle = output.get(); std::memcpy(&result.value, &handle, sizeof(handle));
        } else TEST_CUDA(cuIpcGetEventHandle(&result.value, value));
        return result.get();
    }
    void record(CUstream stream, bool runtime) const {
        if (runtime) event_ipc_runtime(cudaEventRecord(reinterpret_cast<cudaEvent_t>(value), reinterpret_cast<cudaStream_t>(stream)));
        else TEST_CUDA(cuEventRecord(value, stream));
    }
    void wait(CUstream stream, bool runtime) const {
        if (runtime) event_ipc_runtime(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream), reinterpret_cast<cudaEvent_t>(value), 0));
        else TEST_CUDA(cuStreamWaitEvent(stream, value, 0));
    }
};
struct event_ipc_stream {
    CUstream value{};
    event_ipc_stream() { TEST_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~event_ipc_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    void wait() const { TEST_CUDA(cuStreamSynchronize(value)); }
    void close() { TEST_CUDA(cuStreamDestroy(value)); value = nullptr; }
};
unsigned char event_ipc_byte(size_t i, unsigned seed) {
    return i < 16 || i >= event_ipc_bytes - 16 ? 0xa7 : static_cast<unsigned char>((i * 73 + seed * 43) ^ (i >> 3));
}
struct event_ipc_host {
    unsigned char* value{};
    explicit event_ipc_host(unsigned seed = 0) {
        void* result{}; TEST_CUDA(cuMemAllocHost(&result, event_ipc_bytes + 32)); value = static_cast<unsigned char*>(result);
        std::fill_n(value, event_ipc_bytes + 32, 0x63);
        if (seed) for (size_t i = 0; i < event_ipc_bytes; ++i) value[16 + i] = event_ipc_byte(i, seed);
    }
    ~event_ipc_host() { if (value) cuMemFreeHost(value); }
    void verify(unsigned seed) const {
        for (size_t i = 0; i < event_ipc_bytes + 32; ++i)
            require(value[i] == (i < 16 || i >= event_ipc_bytes + 16 ? 0x63 : event_ipc_byte(i - 16, seed)), "IPC event complete payload/guard byte=" + std::to_string(i));
    }
};
void event_ipc_write(CUdeviceptr pointer, event_ipc_host& source, CUstream stream, bool runtime) {
    if (runtime) event_ipc_runtime(cudaMemcpyAsync(reinterpret_cast<void*>(uintptr_t(pointer)), source.value + 16, event_ipc_bytes,
        cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)));
    else TEST_CUDA(cuMemcpyHtoDAsync(pointer, source.value + 16, event_ipc_bytes, stream));
}
void event_ipc_read(event_ipc_host& target, CUdeviceptr pointer, CUstream stream, bool runtime) {
    if (runtime) event_ipc_runtime(cudaMemcpyAsync(target.value + 16, reinterpret_cast<void*>(uintptr_t(pointer)), event_ipc_bytes,
        cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream)));
    else TEST_CUDA(cuMemcpyDtoHAsync(target.value + 16, pointer, event_ipc_bytes, stream));
}
struct event_ipc_gate {
    std::mutex mutex; std::condition_variable changed;
    bool entered = false, released = false, expired = false;
    CUstream stream{};
    std::thread watchdog;
    explicit event_ipc_gate(CUstream target) : stream(target), watchdog([this] {
        std::unique_lock lock(mutex);
        if (!changed.wait_for(lock, std::chrono::seconds(20), [&] { return released; })) { expired = true; released = true; }
        lock.unlock(); changed.notify_all();
    }) {}
    void release() { { std::lock_guard lock(mutex); released = true; } changed.notify_all(); }
    ~event_ipc_gate() { release(); if (watchdog.joinable()) watchdog.join(); if (stream) cuStreamSynchronize(stream); }
    static void CUDA_CB callback(void* pointer) {
        auto& gate = *static_cast<event_ipc_gate*>(pointer); std::unique_lock lock(gate.mutex);
        gate.entered = true; gate.changed.notify_all(); gate.changed.wait(lock, [&] { return gate.released; });
    }
    void block() {
        TEST_CUDA(cuLaunchHostFunc(stream, callback, this)); std::unique_lock lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "bounded IPC producer callback entry");
    }
};
void event_ipc_validation() {
    CUevent output{}; require(cuEventCreate(&output, CU_EVENT_INTERPROCESS) == CUDA_ERROR_INVALID_VALUE, "IPC event requires disabled timing");
    require(cudaEventCreateWithFlags(nullptr, cudaEventDisableTiming | cudaEventInterprocess) == cudaErrorInvalidValue, "null Runtime event output");
    event_ipc_event ordinary(false, CU_EVENT_DISABLE_TIMING); CUipcEventHandle handle{};
    require(cuIpcGetEventHandle(nullptr, ordinary.value) == CUDA_ERROR_INVALID_VALUE, "null Driver IPC event output");
    require(cuIpcGetEventHandle(&handle, ordinary.value) == CUDA_ERROR_INVALID_HANDLE, "ordinary event cannot export IPC");
    require(cudaIpcGetEventHandle(nullptr, reinterpret_cast<cudaEvent_t>(ordinary.value)) == cudaErrorInvalidValue, "null Runtime IPC event output");
    require(cuIpcOpenEventHandle(nullptr, {}) == CUDA_ERROR_INVALID_VALUE, "null Driver IPC event import output");
    require(cudaIpcOpenEventHandle(nullptr, {}) == cudaErrorInvalidValue, "null Runtime IPC event import output");
    event_ipc_stream stream; ordinary.record(stream.value, false); stream.wait(); TEST_CUDA(cuEventSynchronize(ordinary.value));
}
void event_ipc_local() {
    require_capability(event_ipc_supported(), "IPC event execution requires advertised IPC and UVA support");
    for (bool runtime : {false, true}) {
        event_ipc_event event(runtime); event.export_handle(runtime); TEST_CUDA(cuEventQuery(event.value));
        TEST_CUDA(cuEventSynchronize(event.value)); event_ipc_stream stream;
        event_ipc_gate gate(stream.value); gate.block(); event.record(stream.value, runtime);
        require(cuEventQuery(event.value) == CUDA_ERROR_NOT_READY, "published record remains pending behind a callback");
        require(cudaEventQuery(reinterpret_cast<cudaEvent_t>(event.value)) == cudaErrorNotReady, "Runtime observes pending IPC generation");
        event.export_handle(!runtime); gate.release(); stream.wait();
        event_ipc_runtime(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event.value))); require(!gate.expired, "IPC record returns before producer release");
        float elapsed{}; require(cuEventElapsedTime(&elapsed, event.value, event.value) == CUDA_ERROR_INVALID_HANDLE, "IPC events have no timing");
        CUevent bad{}; auto status = cuIpcOpenEventHandle(&bad, {});
        require(status == CUDA_ERROR_INVALID_HANDLE || status == CUDA_ERROR_INVALID_VALUE || status == CUDA_ERROR_MAP_FAILED, "malformed IPC event handle is rejected");
    }
}
struct event_ipc_message {
    unsigned kind{}, mode{}, runtime{};
    CUuuid uuid{};
    CUipcEventHandle event{};
    CUipcMemHandle memory{};
};
void event_ipc_send(test_peer& peer, event_ipc_message message) { peer.send(std::as_bytes(std::span(&message, 1))); }
event_ipc_message event_ipc_receive(test_peer& peer) { event_ipc_message result{}; peer.receive(std::as_writable_bytes(std::span(&result, 1))); return result; }
void event_ipc_importer() {
    require_capability(event_ipc_supported(), "IPC event importer requires advertised IPC and UVA support");
    auto& peer = *event_ipc_peer; auto hello = event_ipc_receive(peer); bool runtime = !hello.runtime;
    CUdevice device{}; CUuuid uuid{}; TEST_CUDA(cuCtxGetDevice(&device)); TEST_CUDA(cuDeviceGetUuid(&uuid, device));
    require(std::memcmp(&uuid, &hello.uuid, sizeof(uuid)) == 0, "IPC peers select the same native device");
    event_ipc_event event(hello.event, runtime); CUdeviceptr memory{};
    if (runtime) {
        cudaIpcMemHandle_t record{}; std::memcpy(&record, &hello.memory, sizeof(record)); void* pointer{};
        event_ipc_runtime(cudaIpcOpenMemHandle(&pointer, record, cudaIpcMemLazyEnablePeerAccess)); memory = reinterpret_cast<uintptr_t>(pointer);
    } else TEST_CUDA(cuIpcOpenMemHandle(&memory, hello.memory, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));
    struct import_cleanup { CUdeviceptr pointer; ~import_cleanup() { if (pointer) cuIpcCloseMemHandle(pointer); } } cleanup{memory};
    event_ipc_stream first, second; event_ipc_host first_result, second_result;
    event_ipc_send(peer, {2}); require(event_ipc_receive(peer).kind == 3, "first record publication handoff");
    event.wait(first.value, runtime); event_ipc_read(first_result, memory, first.value, runtime);
    require(cuEventQuery(event.value) == CUDA_ERROR_NOT_READY, "imported event observes blocked producer");
    if (hello.mode == 2) { event.close(runtime); first.close(); }
    event_ipc_send(peer, {4});
    if (hello.mode == 1) {
        require(event_ipc_receive(peer).kind == 5, "newer blocked record publication");
        event.wait(second.value, !runtime); event_ipc_read(second_result, memory, second.value, !runtime);
        first.wait(); first_result.verify(19);
        require(cuStreamQuery(second.value) == CUDA_ERROR_NOT_READY, "newer captured generation remains blocked");
        require(cuEventQuery(event.value) == CUDA_ERROR_NOT_READY, "query observes the newer record while the older wait completes");
        event_ipc_send(peer, {6});
    }
    require(event_ipc_receive(peer).kind == 7, "producer released the older generation");
    if (hello.mode == 2) TEST_CUDA(cuCtxSynchronize()); else first.wait();
    first_result.verify(19);
    if (hello.mode == 1) { second.wait(); second_result.verify(37); }
    if (event.value) event.close(runtime);
    TEST_CUDA(cuIpcCloseMemHandle(memory)); cleanup.pointer = 0; event_ipc_send(peer, {8});
}
void event_ipc_processes(unsigned mode) {
    require_capability(event_ipc_supported(), "IPC event sharing requires advertised IPC and UVA support");
    for (bool runtime : {false, true}) {
        device_buffer memory(event_ipc_bytes); event_ipc_event event(runtime); event_ipc_stream first, second;
        event_ipc_host original(11), source(19), newer(37), final_result;
        TEST_CUDA(cuMemcpyHtoD(memory.value, original.value + 16, event_ipc_bytes));
        event_ipc_message hello{}; hello.kind = 1; hello.mode = mode; hello.runtime = runtime;
        CUdevice device{}; TEST_CUDA(cuCtxGetDevice(&device)); TEST_CUDA(cuDeviceGetUuid(&hello.uuid, device));
        hello.event = event.export_handle(runtime); TEST_CUDA(cuIpcGetMemHandle(&hello.memory, memory.value));
        test_peer peer(event_ipc_executable, "cuda", "importer"); event_ipc_send(peer, hello);
        require(event_ipc_receive(peer).kind == 2, "independent importer opened real handles");
        event_ipc_gate gate(first.value); gate.block(); event_ipc_write(memory.value, source, first.value, runtime);
        event.record(first.value, runtime); event_ipc_send(peer, {3});
        require(event_ipc_receive(peer).kind == 4, "importer captured first wait before later record");
        std::unique_ptr<event_ipc_gate> later;
        if (mode == 1) {
            later = std::make_unique<event_ipc_gate>(second.value); later->block();
            event_ipc_write(memory.value, newer, second.value, !runtime); event.record(second.value, !runtime);
            event_ipc_send(peer, {5}); gate.release();
            require(event_ipc_receive(peer).kind == 6, "older wait completes while the latest generation stays blocked");
            later->release();
        }
        gate.release(); event_ipc_send(peer, {7}); require(event_ipc_receive(peer).kind == 8, "importer closed handles before original retirement");
        require(peer.wait() == 0, "independent IPC event peer exits successfully");
        first.wait(); second.wait(); require(!gate.expired && (!later || !later->expired), "IPC generations make progress without watchdog release");
        event_ipc_read(final_result, memory.value, first.value, runtime); first.wait(); final_result.verify(mode == 1 ? 37 : 19);
        original.verify(11); source.verify(19); newer.verify(37); event.close(!runtime);
    }
}
void event_ipc_discovery() {
    static_assert(std::is_same_v<decltype(&cuIpcGetEventHandle), PFN_cuIpcGetEventHandle_v4010>);
    static_assert(std::is_same_v<decltype(&cuIpcOpenEventHandle), PFN_cuIpcOpenEventHandle_v4010>);
    static_assert(sizeof(CUipcEventHandle) == 64 && sizeof(cudaIpcEventHandle_t) == 64);
    for (const char* name : {"cuIpcGetEventHandle", "cuIpcOpenEventHandle"}) {
        void* driver{}; CUdriverProcAddressQueryResult query{};
        TEST_CUDA(cuGetProcAddress(name, &driver, 4009, 0, &query)); require(!driver && query == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "IPC event introduction boundary");
        TEST_CUDA(cuGetProcAddress(name, &driver, 4010, 0, &query)); require(driver && query == CU_GET_PROC_ADDRESS_SUCCESS, "IPC event introduction lookup");
        void* runtime{}; cudaDriverEntryPointQueryResult other{};
        event_ipc_runtime(cudaGetDriverEntryPointByVersion(name, &runtime, 4010, 0, &other));
        require(runtime == driver && other == cudaDriverEntryPointSuccess, "Runtime discovers Driver IPC event entry");
    }
}
void execute_event_ipc() {
    if (event_ipc_case == "validation") event_ipc_validation();
    else if (event_ipc_case == "local") event_ipc_local();
    else if (event_ipc_case == "sharing") event_ipc_processes(0);
    else if (event_ipc_case == "generations") event_ipc_processes(1);
    else if (event_ipc_case == "retirement") event_ipc_processes(2);
    else if (event_ipc_case == "importer") event_ipc_importer();
    else throw std::runtime_error("unknown event IPC case");
}
}
int main(int argc, char** argv) {
    cudavm::event_ipc_executable = argv[0];
    if (argc == 2 && std::string_view(argv[1]) == "discovery") return cudavm::run_host_test(argc, argv, cudavm::event_ipc_discovery);
    if (argc == 7 && std::string_view(argv[3]) == "--peer") {
        cudavm::event_ipc_peer = std::make_unique<cudavm::test_peer>(cudavm::test_peer::connect(argv[4], argv[5], argv[6])); argc = 3;
    }
    if (argc == 3) { cudavm::event_ipc_case = argv[2]; --argc; }
    return cudavm::run_test(argc, argv, cudavm::execute_event_ipc);
}
