#include "test_framework.hpp"
#include <cudaTypedefs.h>
#include <cstring>
#include <type_traits>
#include <utility>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace cudavm {
std::string_view sharing_case;
const char* sharing_executable{};
std::unique_ptr<test_peer> sharing_peer;
constexpr uint64_t sharing_before = 0x19375928abcdef01, sharing_after = 0xfedcba0129183745;
constexpr size_t sharing_bytes = 8197;
#ifdef _WIN32
constexpr auto sharing_type = CU_MEM_HANDLE_TYPE_WIN32;
#else
constexpr auto sharing_type = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif
void sharing_runtime(cudaError_t status) {
    require(status == cudaSuccess, std::string("pool sharing Runtime: ") + cudaGetErrorString(status));
}
struct sharing_os_handle {
#ifdef _WIN32
    HANDLE value{};
    ~sharing_os_handle() { if (value) CloseHandle(value); }
    void* argument() const { return value; }
    uintptr_t transport() const { return reinterpret_cast<uintptr_t>(value); }
    void alive() const { DWORD flags{}; require(value && GetHandleInformation(value, &flags), "caller owns pool OS reference"); }
    void close() { require(CloseHandle(value), "close pool OS reference"); value = nullptr; }
#else
    int value = -1;
    ~sharing_os_handle() { if (value >= 0) ::close(value); }
    void* argument() const { return reinterpret_cast<void*>(uintptr_t(value)); }
    uintptr_t transport() const { return uintptr_t(value); }
    void alive() const { require(value >= 0 && fcntl(value, F_GETFD) >= 0, "caller owns pool OS descriptor"); }
    void close() { require(::close(value) == 0, "close pool OS descriptor"); value = -1; }
#endif
    sharing_os_handle() = default;
    sharing_os_handle(const sharing_os_handle&) = delete;
    sharing_os_handle(sharing_os_handle&& other) noexcept : value(std::exchange(other.value,
#ifdef _WIN32
        nullptr
#else
        -1
#endif
    )) {}
};
struct sharing_pool {
    CUmemoryPool value{};
    sharing_pool() = default;
    sharing_pool(const sharing_pool&) = delete;
    ~sharing_pool() { if (value) cuMemPoolDestroy(value); }
    cudaMemPool_t runtime() const { return reinterpret_cast<cudaMemPool_t>(value); }
    void create(bool exported, bool use_runtime) {
        CUdevice device{}; TEST_CUDA(cuCtxGetDevice(&device));
#ifdef _WIN32
        SECURITY_ATTRIBUTES security{sizeof(security), nullptr, FALSE};
#endif
        if (use_runtime) {
            cudaMemPoolProps properties{}; properties.allocType = cudaMemAllocationTypePinned;
            properties.location = {cudaMemLocationTypeDevice, device};
            properties.handleTypes = exported ? static_cast<cudaMemAllocationHandleType>(sharing_type) : cudaMemHandleTypeNone;
#ifdef _WIN32
            properties.win32SecurityAttributes = exported ? &security : nullptr;
#endif
            cudaMemPool_t output{}; sharing_runtime(cudaMemPoolCreate(&output, &properties)); value = reinterpret_cast<CUmemoryPool>(output);
        } else {
            CUmemPoolProps properties{}; properties.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
            properties.location = {CU_MEM_LOCATION_TYPE_DEVICE, device}; properties.handleTypes = exported ? sharing_type : CU_MEM_HANDLE_TYPE_NONE;
#ifdef _WIN32
            properties.win32SecurityAttributes = exported ? &security : nullptr;
#endif
            TEST_CUDA(cuMemPoolCreate(&value, &properties));
        }
    }
    void import(sharing_os_handle& os, bool use_runtime) {
        struct { uint64_t before = sharing_before; CUmemoryPool value{}; uint64_t after = sharing_after; } output;
        if (use_runtime) {
            // Keep the public Runtime output type, including adjacent guards.
            struct { uint64_t before = sharing_before; cudaMemPool_t value{}; uint64_t after = sharing_after; } result;
            sharing_runtime(cudaMemPoolImportFromShareableHandle(&result.value, os.argument(), static_cast<cudaMemAllocationHandleType>(sharing_type), 0));
            value = reinterpret_cast<CUmemoryPool>(result.value);
            require(result.before == sharing_before && result.after == sharing_after, "Runtime pool import output guards");
        } else {
            TEST_CUDA(cuMemPoolImportFromShareableHandle(&output.value, os.argument(), sharing_type, 0)); value = output.value;
            require(output.before == sharing_before && output.after == sharing_after, "Driver pool import output guards");
        }
        os.alive(); require(value, "import publishes a pool");
    }
    void close(bool use_runtime) {
        if (use_runtime) sharing_runtime(cudaMemPoolDestroy(runtime())); else TEST_CUDA(cuMemPoolDestroy(value));
        value = nullptr;
    }
};
struct sharing_pointer {
    CUdeviceptr value{};
    ~sharing_pointer() { if (value) cuMemFree(value); }
    void allocate(const sharing_pool& pool, bool runtime, size_t bytes = sharing_bytes) {
        if (runtime) {
            void* pointer{}; sharing_runtime(cudaMallocFromPoolAsync(&pointer, bytes, pool.runtime(), nullptr)); value = reinterpret_cast<uintptr_t>(pointer);
        } else TEST_CUDA(cuMemAllocFromPoolAsync(&value, bytes, pool.value, nullptr));
        TEST_CUDA(cuStreamSynchronize(nullptr)); require(value, "allocated pointer");
    }
    void import(const sharing_pool& pool, const CUmemPoolPtrExportData& data, bool runtime) {
        auto copy = data;
        if (runtime) {
            cudaMemPoolPtrExportData record{}; std::memcpy(&record, &copy, sizeof(record));
            struct { uint64_t before = sharing_before; void* value{}; uint64_t after = sharing_after; } output;
            sharing_runtime(cudaMemPoolImportPointer(&output.value, pool.runtime(), &record)); value = reinterpret_cast<uintptr_t>(output.value);
            require(output.before == sharing_before && output.after == sharing_after, "Runtime pointer import guards");
        } else {
            struct { uint64_t before = sharing_before; CUdeviceptr value{}; uint64_t after = sharing_after; } output;
            TEST_CUDA(cuMemPoolImportPointer(&output.value, pool.value, &copy)); value = output.value;
            require(output.before == sharing_before && output.after == sharing_after, "Driver pointer import guards");
        }
        require(value, "imported pointer");
    }
    void close(bool runtime, bool asynchronous) {
        if (runtime) {
            auto pointer = reinterpret_cast<void*>(uintptr_t(value));
            sharing_runtime(asynchronous ? cudaFreeAsync(pointer, nullptr) : cudaFree(pointer));
            if (asynchronous) sharing_runtime(cudaStreamSynchronize(nullptr));
        } else {
            TEST_CUDA(asynchronous ? cuMemFreeAsync(value, nullptr) : cuMemFree(value));
            if (asynchronous) TEST_CUDA(cuStreamSynchronize(nullptr));
        }
        value = 0;
    }
};
sharing_os_handle sharing_export(const sharing_pool& pool, bool runtime) {
    using guard = std::conditional_t<sizeof(sharing_os_handle::value) == 4, uint32_t, uint64_t>;
    struct { guard before = guard(sharing_before); decltype(sharing_os_handle::value) value{}; guard after = guard(sharing_after); } output;
    if (runtime) sharing_runtime(cudaMemPoolExportToShareableHandle(&output.value, pool.runtime(), static_cast<cudaMemAllocationHandleType>(sharing_type), 0));
    else TEST_CUDA(cuMemPoolExportToShareableHandle(&output.value, pool.value, sharing_type, 0));
    sharing_os_handle result; result.value = output.value;
    require(output.before == guard(sharing_before) && output.after == guard(sharing_after), "pool OS export exact output guards"); result.alive(); return result;
}
CUmemPoolPtrExportData sharing_record(CUdeviceptr pointer, bool runtime) {
    struct { uint64_t before = sharing_before; CUmemPoolPtrExportData value{}; uint64_t after = sharing_after; } output;
    if (runtime) {
        struct { uint64_t before = sharing_before; cudaMemPoolPtrExportData value{}; uint64_t after = sharing_after; } result;
        sharing_runtime(cudaMemPoolExportPointer(&result.value, reinterpret_cast<void*>(uintptr_t(pointer))));
        require(result.before == sharing_before && result.after == sharing_after, "Runtime opaque record output guards"); std::memcpy(&output.value, &result.value, sizeof(output.value));
    } else TEST_CUDA(cuMemPoolExportPointer(&output.value, pointer));
    require(output.before == sharing_before && output.after == sharing_after, "Driver opaque record output guards"); return output.value;
}
void sharing_prerequisite(bool exported) {
    CUdevice device{}; int supported{}; TEST_CUDA(cuCtxGetDevice(&device));
    TEST_CUDA(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, device));
    require_capability(supported != 0, "memory pool support is required");
    if (exported) {
        TEST_CUDA(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES, device));
        require_capability((supported & sharing_type) != 0, "pool sharing needs the advertised platform handle type");
    }
}
void sharing_restrictions(const sharing_pool& pool) {
    CUdevice device{}; TEST_CUDA(cuCtxGetDevice(&device));
    CUdeviceptr pointer{}; void* runtime_pointer{};
    require(cuMemAllocFromPoolAsync(&pointer, 32, pool.value, nullptr) == CUDA_ERROR_INVALID_VALUE, "imported pool cannot allocate");
    require(cudaMallocFromPoolAsync(&runtime_pointer, 32, pool.runtime(), nullptr) == cudaErrorInvalidValue, "Runtime imported pool cannot allocate");
    require(cuDeviceSetMemPool(device, pool.value) == CUDA_ERROR_INVALID_VALUE, "imported pool cannot become current");
    require(cudaDeviceSetMemPool(device, pool.runtime()) == cudaErrorInvalidValue, "Runtime imported pool cannot become current");
}
void sharing_validation() {
    for (bool runtime : {false, true}) {
        sharing_pool pool; pool.create(false, runtime); sharing_pointer pointer; pointer.allocate(pool, runtime);
        auto expected = virtual_memory_pattern(sharing_bytes, 19); virtual_memory_write(pointer.value, expected, runtime);
        decltype(sharing_os_handle::value) os{}; CUmemPoolPtrExportData record{}; cudaMemPoolPtrExportData runtime_record{};
        CUmemoryPool output{}; cudaMemPool_t runtime_pool{}; CUdeviceptr address{}; void* runtime_address{};
        require(cuMemPoolExportToShareableHandle(nullptr, pool.value, sharing_type, 0) == CUDA_ERROR_INVALID_VALUE, "missing pool export output");
        require(cudaMemPoolExportToShareableHandle(nullptr, pool.runtime(), static_cast<cudaMemAllocationHandleType>(sharing_type), 0) == cudaErrorInvalidValue, "Runtime missing pool export output");
        for (auto type : {CU_MEM_HANDLE_TYPE_NONE, sharing_type}) {
            require(cuMemPoolExportToShareableHandle(&os, pool.value, type, 0) == CUDA_ERROR_INVALID_VALUE, "private pool export is invalid");
            require(cudaMemPoolExportToShareableHandle(&os, pool.runtime(), static_cast<cudaMemAllocationHandleType>(type), 0) == cudaErrorInvalidValue, "Runtime private pool export is invalid");
        }
        require(cuMemPoolExportToShareableHandle(&os, pool.value, sharing_type, 1ull << 32) == CUDA_ERROR_INVALID_VALUE, "Driver reserved high flag bits");
        require(cudaMemPoolExportToShareableHandle(&os, pool.runtime(), static_cast<cudaMemAllocationHandleType>(sharing_type), 1) == cudaErrorInvalidValue, "Runtime reserved flags");
        require(cuMemPoolImportFromShareableHandle(nullptr, nullptr, sharing_type, 0) == CUDA_ERROR_INVALID_VALUE, "missing pool import output");
        require(cudaMemPoolImportFromShareableHandle(nullptr, nullptr, static_cast<cudaMemAllocationHandleType>(sharing_type), 0) == cudaErrorInvalidValue, "Runtime missing pool import output");
        require(cuMemPoolImportFromShareableHandle(&output, nullptr, CU_MEM_HANDLE_TYPE_NONE, 0) == CUDA_ERROR_INVALID_VALUE, "invalid pool import type");
        require(cudaMemPoolImportFromShareableHandle(&runtime_pool, nullptr, cudaMemHandleTypeNone, 0) == cudaErrorInvalidValue, "Runtime invalid pool import type");
        require(cuMemPoolImportFromShareableHandle(&output, nullptr, sharing_type, 1) == CUDA_ERROR_INVALID_VALUE, "pool import reserved flags");
        require(cudaMemPoolImportFromShareableHandle(&runtime_pool, nullptr, static_cast<cudaMemAllocationHandleType>(sharing_type), 1) == cudaErrorInvalidValue, "Runtime pool import reserved flags");
        require(cuMemPoolExportPointer(nullptr, pointer.value) == CUDA_ERROR_INVALID_VALUE, "missing pointer record output");
        require(cudaMemPoolExportPointer(nullptr, reinterpret_cast<void*>(uintptr_t(pointer.value))) == cudaErrorInvalidValue, "Runtime missing pointer record output");
        require(cuMemPoolExportPointer(&record, pointer.value) == CUDA_ERROR_INVALID_VALUE, "private pool pointer export");
        require(cudaMemPoolExportPointer(&runtime_record, reinterpret_cast<void*>(uintptr_t(pointer.value))) == cudaErrorInvalidValue, "Runtime private pool pointer export");
        require(cuMemPoolImportPointer(&address, pool.value, &record) == CUDA_ERROR_INVALID_VALUE, "pointer import needs imported pool");
        require(cudaMemPoolImportPointer(&runtime_address, pool.runtime(), &runtime_record) == cudaErrorInvalidValue, "Runtime pointer import needs imported pool");
        require(cuMemPoolImportPointer(nullptr, pool.value, &record) == CUDA_ERROR_INVALID_VALUE && cuMemPoolImportPointer(&address, pool.value, nullptr) == CUDA_ERROR_INVALID_VALUE, "pointer import missing output or record");
        require(cudaMemPoolImportPointer(nullptr, pool.runtime(), &runtime_record) == cudaErrorInvalidValue && cudaMemPoolImportPointer(&runtime_address, pool.runtime(), nullptr) == cudaErrorInvalidValue, "Runtime pointer import missing output or record");
        device_buffer ordinary(63);
        require(cuMemPoolExportPointer(&record, ordinary.value) == CUDA_ERROR_INVALID_VALUE, "ordinary allocation has no pool record");
        require(cudaMemPoolExportPointer(&runtime_record, reinterpret_cast<void*>(uintptr_t(ordinary.value))) == cudaErrorInvalidValue, "Runtime ordinary allocation has no pool record");
        CUmemoryPool default_pool{}; TEST_CUDA(cuDeviceGetDefaultMemPool(&default_pool, virtual_memory_properties().location.id));
        require(cuMemPoolExportToShareableHandle(&os, default_pool, sharing_type, 0) == CUDA_ERROR_INVALID_VALUE, "default pool is not exportable");
        virtual_memory_expect(pointer.value, expected, "failed sharing operations preserve complete bytes", !runtime);
        TEST_CUDA(cuMemsetD8(pointer.value + 31, 0x71, 257)); std::fill_n(expected.begin() + 31, 257, 0x71);
        virtual_memory_expect(pointer.value, expected, "repaired ordinary pool operations preserve guards", runtime);
        pointer.close(!runtime, true);
    }
}
void sharing_roundtrip() {
    for (bool runtime : {false, true}) {
        sharing_pool source; source.create(true, runtime); auto os = sharing_export(source, !runtime);
        CUmemoryPool rejected{}; cudaMemPool_t runtime_rejected{};
        require(cuMemPoolImportFromShareableHandle(&rejected, os.argument(), sharing_type, 1) == CUDA_ERROR_INVALID_VALUE, "valid OS handle survives rejected Driver import");
        require(cudaMemPoolImportFromShareableHandle(&runtime_rejected, os.argument(), static_cast<cudaMemAllocationHandleType>(sharing_type), 1) == cudaErrorInvalidValue, "valid OS handle survives rejected Runtime import"); os.alive();
        sharing_pool imported; imported.import(os, runtime); os.close(); sharing_restrictions(imported);
        std::array<sharing_pointer, 2> originals;
        for (unsigned i = 0; i < originals.size(); ++i) {
            originals[i].allocate(source, runtime); auto expected = virtual_memory_pattern(sharing_bytes, 41 + i);
            virtual_memory_write(originals[i].value, expected, runtime); TEST_CUDA(cuCtxSynchronize());
            (void)sharing_record(originals[i].value, !runtime);
            virtual_memory_expect(originals[i].value, expected, "pool and pointer export preserve complete original allocation", !runtime);
        }
        source.close(!runtime); imported.close(runtime);
        for (unsigned i = 0; i < originals.size(); ++i) {
            auto expected = virtual_memory_pattern(sharing_bytes, 41 + i);
            TEST_CUDA(cuMemsetD8(originals[i].value + 37, 0xa3 + i, 997)); std::fill_n(expected.begin() + 37, 997, 0xa3 + i);
            virtual_memory_expect(originals[i].value, expected, "original allocation survives exported pool handle destruction", runtime);
            originals[i].close(!runtime, i == 0);
        }
    }
}
struct sharing_message {
    uint64_t kind{}, phase{}, retirement{};
    std::array<unsigned char, 16> uuid{};
};
void sharing_send(test_peer& peer, const sharing_message& message) { peer.send(std::as_bytes(std::span(&message, 1))); }
sharing_message sharing_receive(test_peer& peer) { sharing_message result{}; peer.receive(std::as_writable_bytes(std::span(&result, 1))); return result; }
sharing_message sharing_hello(bool retirement) {
    sharing_message result{1, 0, uint64_t(retirement)}; CUuuid uuid{}; auto device = virtual_memory_properties().location.id;
    TEST_CUDA(cuDeviceGetUuid(&uuid, device)); std::memcpy(result.uuid.data(), uuid.bytes, sizeof(uuid.bytes)); return result;
}
void sharing_patch(std::vector<unsigned char>& bytes, unsigned phase, bool importer) {
    auto offset = (importer ? 4099 : 43) + phase * 157; auto count = 307 + phase * 29;
    std::fill_n(bytes.begin() + offset, count, (importer ? 113 : 37) + phase * 17);
}
void sharing_write_patch(CUdeviceptr pointer, std::vector<unsigned char>& expected, unsigned phase, bool importer) {
    sharing_patch(expected, phase, importer); auto offset = (importer ? 4099 : 43) + phase * 157; auto count = 307 + phase * 29;
    if (importer) sharing_runtime(cudaMemset(reinterpret_cast<void*>(uintptr_t(pointer + offset)), (importer ? 113 : 37) + phase * 17, count));
    else TEST_CUDA(cuMemsetD8(pointer + offset, 37 + phase * 17, count));
    TEST_CUDA(cuCtxSynchronize());
}
void sharing_importer() {
    require(sharing_peer != nullptr, "independent importer peer"); auto& peer = *sharing_peer;
    const auto hello = sharing_receive(peer); require(hello.kind == 1 && hello.uuid == sharing_hello(hello.retirement != 0).uuid, "same physical device UUID");
    sharing_os_handle os;
#ifdef _WIN32
    os.value = reinterpret_cast<HANDLE>(peer.receive_handle());
#else
    os.value = int(peer.receive_handle());
#endif
    sharing_pool pool; pool.import(os, true); os.close(); sharing_restrictions(pool);
    std::array<sharing_pointer, 2> pointers; std::array<std::vector<unsigned char>, 2> expected;
    for (unsigned i = 0; i < pointers.size(); ++i) {
        CUmemPoolPtrExportData record{}; peer.receive(std::as_writable_bytes(std::span(&record, 1)));
        pointers[i].import(pool, record, i == 0); std::memset(&record, 0x8d, sizeof(record)); expected[i] = virtual_memory_pattern(sharing_bytes, 73 + i);
        virtual_memory_expect(pointers[i].value, expected[i], "fresh importer complete initial bytes", true);
    }
    if (hello.retirement) pool.close(false);
    sharing_send(peer, {2});
    for (unsigned phase = 0; phase < 3; ++phase) {
        const auto request = sharing_receive(peer); require(request.kind == 3 && request.phase == phase, "ordered pool sharing handoff");
        for (unsigned i = 0; i < pointers.size(); ++i) {
            sharing_patch(expected[i], phase, false); virtual_memory_expect(pointers[i].value, expected[i], "importer sees completed exporter writes");
            sharing_write_patch(pointers[i].value, expected[i], phase, true); virtual_memory_expect(pointers[i].value, expected[i], "importer preserves all guards", true);
        }
        sharing_send(peer, {4, phase});
    }
    require(sharing_receive(peer).kind == 5, "pointer retirement request");
    for (unsigned i = 0; i < pointers.size(); ++i) pointers[i].close(i == 0, i != 0);
    sharing_send(peer, {6}); // Both imported frees have completed before exporter retirement.
}
void sharing_processes(bool retirement) {
    sharing_pool pool; pool.create(true, false); std::array<sharing_pointer, 2> pointers;
    std::array<std::vector<unsigned char>, 2> expected;
    for (unsigned i = 0; i < pointers.size(); ++i) {
        pointers[i].allocate(pool, i == 0); expected[i] = virtual_memory_pattern(sharing_bytes, 73 + i); virtual_memory_write(pointers[i].value, expected[i], i != 0);
    }
    TEST_CUDA(cuCtxSynchronize()); auto os = sharing_export(pool, true); test_peer peer(sharing_executable, "cuda", "importer");
    sharing_send(peer, sharing_hello(retirement)); peer.send_handle(os.transport());
    for (unsigned i = 0; i < pointers.size(); ++i) { auto record = sharing_record(pointers[i].value, i == 0); peer.send(std::as_bytes(std::span(&record, 1))); }
    require(sharing_receive(peer).kind == 2, "importer owns live pointers"); os.close(); if (retirement) pool.close(true);
    for (unsigned phase = 0; phase < 3; ++phase) {
        for (unsigned i = 0; i < pointers.size(); ++i) sharing_write_patch(pointers[i].value, expected[i], phase, false);
        sharing_send(peer, {3, phase}); const auto reply = sharing_receive(peer); require(reply.kind == 4 && reply.phase == phase, "importer completed GPU writes");
        for (unsigned i = 0; i < pointers.size(); ++i) {
            sharing_patch(expected[i], phase, true); virtual_memory_expect(pointers[i].value, expected[i], "exporter sees independent importer bytes and guards", true);
        }
    }
    sharing_send(peer, {5}); require(sharing_receive(peer).kind == 6, "imported frees complete before original frees");
    require(peer.wait() == 0, "importer exits successfully after pointer retirement");
    for (unsigned i = 0; i < pointers.size(); ++i) {
        virtual_memory_expect(pointers[i].value, expected[i], "exporter bytes survive legal importer retirement"); pointers[i].close(i != 0, true);
    }
    if (!retirement) {
        sharing_pointer replacement; replacement.allocate(pool, true); const auto bytes = virtual_memory_pattern(sharing_bytes, 97);
        virtual_memory_write(replacement.value, bytes); virtual_memory_expect(replacement.value, bytes, "pool reuse after all shared pointers retire"); replacement.close(false, false);
    }
}
void sharing_discovery() {
    for (const char* name : {"cuMemPoolExportToShareableHandle", "cuMemPoolImportFromShareableHandle", "cuMemPoolExportPointer", "cuMemPoolImportPointer"}) {
        void* address{}; CUdriverProcAddressQueryResult status{};
        TEST_CUDA(cuGetProcAddress(name, &address, 11020, 0, &status)); require(address && status == CU_GET_PROC_ADDRESS_SUCCESS, "pool sharing procedure introduction");
        TEST_CUDA(cuGetProcAddress(name, &address, 11019, 0, &status)); require(!address && status == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "pool sharing version boundary");
    }
    void* address{}; TEST_CUDA(cuGetProcAddress("cuMemPoolImportFromShareableHandle", &address, 11020, 0, nullptr));
    require(reinterpret_cast<PFN_cuMemPoolImportFromShareableHandle_v11020>(address)(nullptr, nullptr, sharing_type, 0) == CUDA_ERROR_NOT_INITIALIZED, "cold discovered sharing API");
}
void execute_memory_pool_sharing() {
    sharing_prerequisite(sharing_case != "validation");
    if (sharing_case == "validation") sharing_validation();
    else if (sharing_case == "roundtrip") sharing_roundtrip();
    else if (sharing_case == "sharing") sharing_processes(false);
    else if (sharing_case == "retirement") sharing_processes(true);
    else if (sharing_case == "importer") sharing_importer();
    else throw std::runtime_error("unknown memory_pool_sharing case");
}
}
int main(int argc, char** argv) {
    cudavm::sharing_executable = argv[0];
    if (argc == 2 && std::string_view(argv[1]) == "discovery") return cudavm::run_host_test(argc, argv, cudavm::sharing_discovery);
    if (argc == 7 && std::string_view(argv[3]) == "--peer") {
        cudavm::sharing_peer = std::make_unique<cudavm::test_peer>(cudavm::test_peer::connect(argv[4], argv[5], argv[6])); argc = 3;
    }
    if (argc == 3) { cudavm::sharing_case = argv[2]; --argc; }
    return cudavm::run_test(argc, argv, cudavm::execute_memory_pool_sharing);
}
