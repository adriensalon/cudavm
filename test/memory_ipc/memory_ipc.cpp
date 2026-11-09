#include "test_framework.hpp"
#include <cudaTypedefs.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>

namespace cudavm {
std::string_view ipc_case;
const char* ipc_executable{};
std::unique_ptr<conformance_peer> ipc_peer;
constexpr size_t ipc_bytes = 8197, ipc_guard = 37;
constexpr uint64_t ipc_before = 0x19375928abcdef01, ipc_after = 0xfedcba0129183745;
void* ipc_pointer(CUdeviceptr value) { return reinterpret_cast<void*>(uintptr_t(value)); }
void ipc_runtime(cudaError_t result) { require(result == cudaSuccess, std::string("memory IPC Runtime: ") + cudaGetErrorString(result)); }
void ipc_prerequisite() {
    CUdevice device{}; int uva{}, supported{};
    CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&uva, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device));
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED, device));
    require_capability(uva && supported, "memory IPC requires advertised unified addressing and IPC support");
}
struct ipc_original {
    CUdeviceptr value{};
    void allocate(bool runtime) {
        if (runtime) { void* pointer{}; ipc_runtime(cudaMalloc(&pointer, ipc_bytes)); value = reinterpret_cast<uintptr_t>(pointer); }
        else CONFORMANCE_CUDA(cuMemAlloc(&value, ipc_bytes));
    }
    void close(bool runtime) {
        if (runtime) ipc_runtime(cudaFree(ipc_pointer(value))); else CONFORMANCE_CUDA(cuMemFree(value));
        value = 0;
    }
    ~ipc_original() { if (value) cuMemFree(value); }
};
CUipcMemHandle ipc_record(CUdeviceptr pointer, bool runtime) {
    struct { uint64_t before = ipc_before; CUipcMemHandle value{}; uint64_t after = ipc_after; } output;
    if (runtime) {
        struct { uint64_t before = ipc_before; cudaIpcMemHandle_t value{}; uint64_t after = ipc_after; } result;
        ipc_runtime(cudaIpcGetMemHandle(&result.value, ipc_pointer(pointer)));
        require(result.before == ipc_before && result.after == ipc_after, "Runtime IPC record exact output guards");
        static_assert(sizeof(result.value) == sizeof(output.value)); std::memcpy(&output.value, &result.value, sizeof(output.value));
    } else CONFORMANCE_CUDA(cuIpcGetMemHandle(&output.value, pointer));
    require(output.before == ipc_before && output.after == ipc_after, "Driver IPC record exact output guards");
    return output.value;
}
CUdeviceptr ipc_open(CUipcMemHandle record, bool runtime, bool legacy = false) {
    struct { uint64_t before = ipc_before; CUdeviceptr value{}; uint64_t after = ipc_after; } output;
    if (runtime) {
        struct { uintptr_t before = uintptr_t(ipc_before); void* value{}; uintptr_t after = uintptr_t(ipc_after); } result;
        cudaIpcMemHandle_t copy{}; std::memcpy(&copy, &record, sizeof(copy));
        ipc_runtime(cudaIpcOpenMemHandle(&result.value, copy, cudaIpcMemLazyEnablePeerAccess)); output.value = reinterpret_cast<uintptr_t>(result.value);
        require(result.before == uintptr_t(ipc_before) && result.after == uintptr_t(ipc_after), "Runtime IPC pointer exact output guards");
    } else if (legacy) {
        void* procedure{}; CONFORMANCE_CUDA(cuGetProcAddress("cuIpcOpenMemHandle", &procedure, 4010, 0, nullptr));
        CONFORMANCE_CUDA(reinterpret_cast<PFN_cuIpcOpenMemHandle_v11000>(procedure)(&output.value, record, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));
    } else CONFORMANCE_CUDA(cuIpcOpenMemHandle(&output.value, record, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));
    require(output.before == ipc_before && output.after == ipc_after && output.value, "Driver IPC pointer exact output guards");
    return output.value;
}
struct ipc_imported {
    CUdeviceptr value{}; unsigned references{};
    void open(CUipcMemHandle record, bool runtime, bool legacy = false) {
        auto result = ipc_open(record, runtime, legacy);
        if (references) require(result == value, "v2 repeated open preserves imported address");
        value = result; ++references;
    }
    void close(bool runtime) {
        require(references != 0, "test owns an IPC reference");
        if (runtime) ipc_runtime(cudaIpcCloseMemHandle(ipc_pointer(value))); else CONFORMANCE_CUDA(cuIpcCloseMemHandle(value));
        if (!--references) value = 0;
    }
    ~ipc_imported() { while (references) { --references; cuIpcCloseMemHandle(value); } }
};
void ipc_metadata() {
    std::array<ipc_original, 2> allocations;
    for (unsigned i = 0; i < allocations.size(); ++i) {
        allocations[i].allocate(i == 0);
        const auto expected = virtual_memory_pattern(ipc_bytes, 113 + i); virtual_memory_write(allocations[i].value, expected, i != 0);
        struct { unsigned before = 0x19375928, value{}, after = 0xfedcba01; } capability;
        CONFORMANCE_CUDA(cuPointerGetAttribute(&capability.value, CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE, allocations[i].value));
        require(capability.before == 0x19375928 && capability.after == 0xfedcba01 && capability.value == 1, "ordinary allocation IPC capability and exact output");
        auto first = ipc_record(allocations[i].value, false), second = ipc_record(allocations[i].value, true);
        // Records are opaque. Repeated exports are independently valid, without
        // imposing byte equality on the driver's representation.
        (void)first; (void)second;
        virtual_memory_expect(allocations[i].value, expected, "record exports preserve complete original bytes", i == 0);
    }
    for (unsigned i = 0; i < allocations.size(); ++i) allocations[i].close(i != 0);
    ipc_original first; first.allocate(false); const auto address = first.value; const auto old_record = ipc_record(first.value, false); first.close(false);
    ipc_original replacement; replacement.allocate(true); const auto new_record = ipc_record(replacement.value, true);
    if (replacement.value == address) require(std::memcmp(&old_record, &new_record, sizeof(old_record)) != 0, "reused allocation address has a unique opaque IPC handle");
    const auto expected = virtual_memory_pattern(ipc_bytes, 139); virtual_memory_write(replacement.value, expected);
    virtual_memory_expect(replacement.value, expected, "new allocation remains usable after IPC handle reuse check"); replacement.close(true);
}
void ipc_validation() {
    ipc_original allocation; allocation.allocate(false); const auto expected = virtual_memory_pattern(ipc_bytes, 107);
    virtual_memory_write(allocation.value, expected); CUipcMemHandle empty{}, output{}; CUdeviceptr pointer{};
    require(cuIpcGetMemHandle(nullptr, allocation.value) == CUDA_ERROR_INVALID_VALUE, "missing Driver IPC export output");
    require(cuIpcGetMemHandle(&output, allocation.value + 1) == CUDA_ERROR_INVALID_VALUE, "IPC export requires exact ordinary allocation base");
    require(cuIpcGetMemHandle(&output, 0) == CUDA_ERROR_INVALID_VALUE, "IPC export null pointer");
    require(cuIpcOpenMemHandle(nullptr, empty, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS) == CUDA_ERROR_INVALID_VALUE, "missing Driver import output");
    for (unsigned flags : {0u, 2u, 3u}) {
        require(cuIpcOpenMemHandle(&pointer, empty, flags) == CUDA_ERROR_INVALID_VALUE, "Driver IPC requires defined flags");
        void* result{}; cudaIpcMemHandle_t record{};
        require(cudaIpcOpenMemHandle(&result, record, flags) == cudaErrorInvalidValue, "Runtime IPC requires defined flags");
    }
    require(cuIpcOpenMemHandle(&pointer, empty, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS) == CUDA_ERROR_INVALID_VALUE, "malformed opaque record rejected by native importer");
    require(cuIpcCloseMemHandle(allocation.value) == CUDA_ERROR_INVALID_VALUE && cuIpcCloseMemHandle(0) == CUDA_ERROR_INVALID_VALUE, "ordinary and null pointers cannot close imports");
    cudaIpcMemHandle_t runtime_record{};
    require(cudaIpcGetMemHandle(nullptr, ipc_pointer(allocation.value)) == cudaErrorInvalidValue, "missing Runtime IPC export output");
    require(cudaIpcGetMemHandle(&runtime_record, ipc_pointer(allocation.value + 1)) == cudaErrorInvalidValue, "Runtime export exact base");
    require(cudaIpcOpenMemHandle(nullptr, runtime_record, cudaIpcMemLazyEnablePeerAccess) == cudaErrorInvalidValue, "missing Runtime import output");
    require(cudaIpcCloseMemHandle(ipc_pointer(allocation.value)) == cudaErrorInvalidValue, "Runtime ordinary allocation is not an import");
    cudaGetLastError();
    (void)ipc_record(allocation.value, true); virtual_memory_expect(allocation.value, expected, "repaired IPC operations preserve bytes");
    allocation.close(true);
}
enum class ipc_profile : unsigned { sharing, references, retirement, context_release, legacy };
struct ipc_message { unsigned kind{}, phase{}; ipc_profile profile{}; CUuuid uuid{}; size_t bytes{}; };
void ipc_send(conformance_peer& peer, const ipc_message& message) { peer.send(std::as_bytes(std::span(&message, 1))); }
ipc_message ipc_receive(conformance_peer& peer) { ipc_message message{}; peer.receive(std::as_writable_bytes(std::span(&message, 1))); return message; }
ipc_message ipc_hello(ipc_profile profile) {
    ipc_message message{1, 0, profile, {}, ipc_bytes}; CUdevice device{};
    CONFORMANCE_CUDA(cuCtxGetDevice(&device)); CONFORMANCE_CUDA(cuDeviceGetUuid(&message.uuid, device)); return message;
}
size_t ipc_patch_begin(unsigned phase, bool importer) { return ipc_guard + phase * 911 + (importer ? 317 : 0); }
void ipc_patch(std::vector<unsigned char>& expected, unsigned phase, bool importer) {
    const auto begin = ipc_patch_begin(phase, importer);
    for (size_t i = begin; i < begin + 257; ++i) expected[i] = static_cast<unsigned char>((i * 31 + phase * 19 + (importer ? 137 : 53)) ^ (i >> 5));
}
void ipc_write_patch(CUdeviceptr pointer, std::vector<unsigned char>& expected, unsigned phase, bool importer, bool runtime) {
    ipc_patch(expected, phase, importer); const auto begin = ipc_patch_begin(phase, importer);
    virtual_memory_write(pointer + begin, std::span<const unsigned char>(expected).subspan(begin, 257), runtime);
    CONFORMANCE_CUDA(cuCtxSynchronize());
}
struct ipc_gate {
    CUcontext owner{}; std::mutex mutex; std::condition_variable changed; bool entered{}, released{}, expired{};
    static void CUDA_CB callback(void* data) {
        auto& gate = *static_cast<ipc_gate*>(data); std::unique_lock lock(gate.mutex); gate.entered = true; gate.changed.notify_all();
        gate.expired = !gate.changed.wait_for(lock, std::chrono::seconds(15), [&] { return gate.released; });
    }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
    void queue(CUstream stream) {
        CONFORMANCE_CUDA(cuCtxGetCurrent(&owner)); CONFORMANCE_CUDA(cuLaunchHostFunc(stream, callback, this));
        std::unique_lock lock(mutex); require(changed.wait_for(lock, std::chrono::seconds(10), [&] { return entered; }), "IPC retirement host gate starts");
    }
    ~ipc_gate() {
        release(); if (owner) { CUcontext previous{}; cuCtxGetCurrent(&previous); if (cuCtxSetCurrent(owner) == CUDA_SUCCESS) cuCtxSynchronize(); cuCtxSetCurrent(previous); }
    }
};
void ipc_pending(ipc_imported& mapping, bool destroy_context) {
    CUcontext context{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&context));
    CUstream stream{}; CONFORMANCE_CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    ipc_gate gate; gate.queue(stream);
    CONFORMANCE_CUDA(cuMemsetD8Async(mapping.value + ipc_guard, 0xd3, 111, stream));
    std::atomic<bool> started{}, done{}; CUresult result = CUDA_ERROR_UNKNOWN; cudaError_t runtime_result = cudaSuccess;
    std::thread retire([&] {
        cuCtxSetCurrent(context); started.store(true, std::memory_order_release);
        if (destroy_context) result = cuCtxDestroy(context);
        else { runtime_result = cudaIpcCloseMemHandle(ipc_pointer(mapping.value)); result = CUDA_SUCCESS; }
        done.store(true, std::memory_order_release);
    });
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!started.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const bool blocked = !done.load(std::memory_order_acquire);
    // Queries stay callable while retirement waits for the gated command.
    int version{}; const auto query = cuDriverGetVersion(&version);
    int runtime_version{}; const auto runtime_query = destroy_context ? cudaSuccess : cudaDriverGetVersion(&runtime_version);
    gate.release(); retire.join(); CONFORMANCE_CUDA(result); ipc_runtime(runtime_result); CONFORMANCE_CUDA(query); ipc_runtime(runtime_query);
    { std::lock_guard lock(gate.mutex); require(!gate.expired, "unrelated API queries complete before the host gate deadline"); }
    require(blocked && version > 0, "IPC retirement waits for accepted work without holding the Driver lock");
    if (destroy_context) { mapping.value = 0; mapping.references = 0; gate.owner = nullptr; }
    else { mapping.value = 0; mapping.references = 0; CONFORMANCE_CUDA(cuStreamDestroy(stream)); }
}
void ipc_concurrent_closes(ipc_imported& mapping, CUipcMemHandle record) {
    mapping.open(record, true); require(mapping.references == 2, "two owned references before concurrent closes");
    CUcontext context{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&context)); const auto pointer = mapping.value;
    std::atomic<bool> ready{}, start{}; CUresult driver_result = CUDA_ERROR_UNKNOWN;
    std::thread close_driver([&] {
        cuCtxSetCurrent(context); ready.store(true, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
        driver_result = cuIpcCloseMemHandle(pointer);
    });
    while (!ready.load(std::memory_order_acquire)) std::this_thread::yield();
    start.store(true, std::memory_order_release); const auto runtime_result = cudaIpcCloseMemHandle(ipc_pointer(pointer)); close_driver.join();
    if (driver_result == CUDA_SUCCESS && runtime_result == cudaSuccess) { mapping.references = 0; mapping.value = 0; }
    CONFORMANCE_CUDA(driver_result); ipc_runtime(runtime_result);
    mapping.open(record, false); // No closed pointer is dereferenced.
}
void ipc_importer() {
    auto& peer = *ipc_peer; const auto hello = ipc_receive(peer); const auto local = ipc_hello(hello.profile);
    require(hello.kind == 1 && hello.bytes == ipc_bytes && std::memcmp(&hello.uuid, &local.uuid, sizeof(CUuuid)) == 0, "fresh IPC importer physical device identity and extents");
    CUcontext original{}, dedicated{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&original));
    if (hello.profile == ipc_profile::context_release) {
        CUdevice device{}; CONFORMANCE_CUDA(cuCtxGetDevice(&device)); CONFORMANCE_CUDA(cuCtxCreate(&dedicated, nullptr, 0, device));
        CUcontext popped{}; CONFORMANCE_CUDA(cuCtxPopCurrent(&popped)); require(popped == dedicated, "importer dedicated context creation stack");
        CONFORMANCE_CUDA(cuCtxSetCurrent(dedicated));
    }
    std::array<CUipcMemHandle, 2> records{}; std::array<ipc_imported, 2> pointers; std::array<std::vector<unsigned char>, 2> expected;
    for (unsigned i = 0; i < pointers.size(); ++i) {
        peer.receive(std::as_writable_bytes(std::span(&records[i], 1)));
        auto copy = records[i]; pointers[i].open(copy, hello.profile != ipc_profile::legacy && i == 0, hello.profile == ipc_profile::legacy);
        std::memset(&copy, 0x8d, sizeof(copy)); expected[i] = virtual_memory_pattern(ipc_bytes, 113 + i);
        virtual_memory_expect(pointers[i].value, expected[i], ("fresh importer entire initial allocation " + std::to_string(i)).c_str(), i == 0);
        require(cuMemFree(pointers[i].value) == CUDA_ERROR_INVALID_VALUE, "ordinary free rejects imported mapping");
        require(cuMemFreeAsync(pointers[i].value, nullptr) == CUDA_ERROR_INVALID_VALUE, "async free rejects imported mapping");
        require(cudaFree(ipc_pointer(pointers[i].value)) == cudaErrorInvalidValue, "Runtime ordinary free rejects imported mapping");
        require(cudaFreeAsync(ipc_pointer(pointers[i].value), nullptr) == cudaErrorInvalidValue, "Runtime async free rejects imported mapping"); cudaGetLastError();
        virtual_memory_expect(pointers[i].value, expected[i], "rejected frees preserve imported mapping");
    }
    if (hello.profile == ipc_profile::references || hello.profile == ipc_profile::context_release) {
        for (unsigned i = 0; i < pointers.size(); ++i) pointers[i].open(records[i], i != 0);
        if (hello.profile == ipc_profile::references) for (unsigned i = 0; i < pointers.size(); ++i) {
            pointers[i].close(i == 0); virtual_memory_expect(pointers[i].value, expected[i], "one v2 close preserves remaining reference");
        }
    }
    ipc_send(peer, {2});
    for (unsigned phase = 0; phase < 3; ++phase) {
        const auto message = ipc_receive(peer); require(message.kind == 3 && message.phase == phase, "ordered IPC writer handoff");
        for (unsigned i = 0; i < pointers.size(); ++i) {
            ipc_patch(expected[i], phase, false); virtual_memory_expect(pointers[i].value, expected[i], "importer sees completed original writes", i != 0);
            ipc_write_patch(pointers[i].value, expected[i], phase, true, i == 0);
            virtual_memory_expect(pointers[i].value, expected[i], "importer preserves every untouched byte", i == 0);
        }
        ipc_send(peer, {4, phase});
    }
    require(ipc_receive(peer).kind == 5, "IPC retirement handoff");
    if (hello.profile == ipc_profile::references) for (unsigned i = 0; i < pointers.size(); ++i) {
        pointers[i].close(i != 0); pointers[i].open(records[i], i != 0);
        virtual_memory_expect(pointers[i].value, expected[i], "reopen after final close preserves original allocation");
        ipc_concurrent_closes(pointers[i], records[i]);
        virtual_memory_expect(pointers[i].value, expected[i], "reopen after concurrent matching closes preserves original bytes");
    }
    if (hello.profile == ipc_profile::retirement) ipc_pending(pointers[0], false);
    if (hello.profile == ipc_profile::context_release) {
        ipc_pending(pointers[0], true); pointers[1].value = 0; pointers[1].references = 0; CONFORMANCE_CUDA(cuCtxSetCurrent(original));
        std::fill_n(expected[0].begin() + ipc_guard, 111, 0xd3);
        pointers[0].open(records[0], true);
        virtual_memory_expect(pointers[0].value, expected[0], "another context imports after the prior importing context retires"); pointers[0].close(false);
    } else for (unsigned i = 0; i < pointers.size(); ++i) if (pointers[i].references) pointers[i].close(i == 0);
    ipc_send(peer, {6});
}
void ipc_processes(ipc_profile profile) {
    std::array<ipc_original, 2> originals; std::array<std::vector<unsigned char>, 2> expected;
    for (unsigned i = 0; i < originals.size(); ++i) {
        originals[i].allocate(i != 0); expected[i] = virtual_memory_pattern(ipc_bytes, 113 + i); virtual_memory_write(originals[i].value, expected[i], i == 0);
    }
    CONFORMANCE_CUDA(cuCtxSynchronize()); conformance_peer peer(ipc_executable, "cuda", "importer"); ipc_send(peer, ipc_hello(profile));
    for (unsigned i = 0; i < originals.size(); ++i) {
        (void)ipc_record(originals[i].value, i != 0); auto record = ipc_record(originals[i].value, i == 0);
        peer.send(std::as_bytes(std::span(&record, 1)));
        for (unsigned j = 0; j < originals.size(); ++j) virtual_memory_expect(originals[j].value, expected[j], ("original after export " + std::to_string(i) + " allocation " + std::to_string(j)).c_str());
    }
    require(ipc_receive(peer).kind == 2, "independent importer opened original allocations");
    for (unsigned phase = 0; phase < 3; ++phase) {
        for (unsigned i = 0; i < originals.size(); ++i) ipc_write_patch(originals[i].value, expected[i], phase, false, i == 0);
        ipc_send(peer, {3, phase}); const auto reply = ipc_receive(peer); require(reply.kind == 4 && reply.phase == phase, "independent importer completed writes");
        for (unsigned i = 0; i < originals.size(); ++i) {
            ipc_patch(expected[i], phase, true); virtual_memory_expect(originals[i].value, expected[i], "exporter verifies imported writes and all guards", i != 0);
        }
    }
    ipc_send(peer, {5}); require(ipc_receive(peer).kind == 6, "importer retired all native references before original free");
    require(peer.wait() == 0, "independent IPC importer exits successfully");
    if (profile == ipc_profile::retirement || profile == ipc_profile::context_release) std::fill_n(expected[0].begin() + ipc_guard, 111, 0xd3);
    for (unsigned i = 0; i < originals.size(); ++i) {
        virtual_memory_expect(originals[i].value, expected[i], "original bytes survive legal importer retirement", i == 0); originals[i].close(i == 0);
    }
    ipc_original replacement; replacement.allocate(true); const auto pattern = virtual_memory_pattern(ipc_bytes, 127);
    virtual_memory_write(replacement.value, pattern); virtual_memory_expect(replacement.value, pattern, "ordinary allocation use after all IPC retirement"); replacement.close(false);
}
void ipc_discovery() {
    static_assert(std::is_same_v<decltype(&cuIpcGetMemHandle), PFN_cuIpcGetMemHandle_v4010>);
    static_assert(std::is_same_v<decltype(&cuIpcOpenMemHandle), PFN_cuIpcOpenMemHandle_v11000>);
    static_assert(std::is_same_v<decltype(&cuIpcCloseMemHandle), PFN_cuIpcCloseMemHandle_v4010>);
    static_assert(sizeof(CUipcMemHandle) == 64 && sizeof(cudaIpcMemHandle_t) == 64);
    for (const char* name : {"cuIpcGetMemHandle", "cuIpcOpenMemHandle", "cuIpcCloseMemHandle"}) {
        void* address{}; CUdriverProcAddressQueryResult status{};
        CONFORMANCE_CUDA(cuGetProcAddress(name, &address, 4010, 0, &status)); require(address && status == CU_GET_PROC_ADDRESS_SUCCESS, "IPC procedure introduction");
        CONFORMANCE_CUDA(cuGetProcAddress(name, &address, 4009, 0, &status)); require(!address && status == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "IPC procedure introduction boundary");
        cudaDriverEntryPointQueryResult runtime_status{};
        ipc_runtime(cudaGetDriverEntryPointByVersion(name, &address, 4010, 0, &runtime_status)); require(address && runtime_status == cudaDriverEntryPointSuccess, "Runtime discovers Driver IPC procedure");
    }
    void* legacy{}; void* modern{}; void* previous{};
    CONFORMANCE_CUDA(cuGetProcAddress("cuIpcOpenMemHandle", &legacy, 4010, 0, nullptr));
    CONFORMANCE_CUDA(cuGetProcAddress("cuIpcOpenMemHandle", &previous, 10999, 0, nullptr));
    CONFORMANCE_CUDA(cuGetProcAddress("cuIpcOpenMemHandle", &modern, 11000, 0, nullptr));
    require(legacy == previous && modern != legacy, "IPC legacy and v2 ABI selection");
    CUdeviceptr pointer{}; CUipcMemHandle record{};
    require(reinterpret_cast<PFN_cuIpcOpenMemHandle_v11000>(modern)(&pointer, record, 1) == CUDA_ERROR_NOT_INITIALIZED, "cold discovered IPC open");
}
void execute_memory_ipc() {
    ipc_prerequisite();
    if (ipc_case == "validation") ipc_validation();
    else if (ipc_case == "metadata") ipc_metadata();
    else if (ipc_case == "sharing") ipc_processes(ipc_profile::sharing);
    else if (ipc_case == "references") ipc_processes(ipc_profile::references);
    else if (ipc_case == "retirement") ipc_processes(ipc_profile::retirement);
    else if (ipc_case == "context_release") ipc_processes(ipc_profile::context_release);
    else if (ipc_case == "legacy") ipc_processes(ipc_profile::legacy);
    else if (ipc_case == "importer") ipc_importer();
    else throw std::runtime_error("unknown memory_ipc case");
}
}
int main(int argc, char** argv) {
    cudavm::ipc_executable = argv[0];
    if (argc == 2 && std::string_view(argv[1]) == "discovery") return cudavm::run_host_conformance(argc, argv, cudavm::ipc_discovery);
    if (argc == 7 && std::string_view(argv[3]) == "--peer") {
        cudavm::ipc_peer = std::make_unique<cudavm::conformance_peer>(cudavm::conformance_peer::connect(argv[4], argv[5], argv[6])); argc = 3;
    }
    if (argc == 3) { cudavm::ipc_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_memory_ipc);
}
