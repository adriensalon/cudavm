#include "conformance.hpp"
#include <cudaTypedefs.h>
#include <atomic>
#include <cstring>
#include <thread>
#include <type_traits>
#include <utility>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winternl.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace cudavm {
std::string_view handles_case;
const char* handles_executable{};
std::unique_ptr<conformance_peer> handles_peer;
constexpr uint64_t handles_before = 0x19375928abcdef01, handles_after = 0xfedcba0129183745;
struct physical_reference {
    CUmemGenericAllocationHandle value{};
    explicit physical_reference(CUmemGenericAllocationHandle handle = 0) : value(handle) {}
    physical_reference(physical_reference&& other) noexcept : value(std::exchange(other.value, 0)) {}
    physical_reference(const physical_reference&) = delete;
    ~physical_reference() { if (value) cuMemRelease(value); }
    void close() { CONFORMANCE_CUDA(cuMemRelease(value)); value = 0; }
};
physical_reference handles_retain(CUdeviceptr pointer, CUmemGenericAllocationHandle expected) {
    struct { uint64_t before = handles_before; CUmemGenericAllocationHandle value{}; uint64_t after = handles_after; } output;
    CONFORMANCE_CUDA(cuMemRetainAllocationHandle(&output.value, reinterpret_cast<void*>(uintptr_t(pointer))));
    physical_reference reference(output.value);
    require(output.before == handles_before && output.after == handles_after && output.value == expected, "retained handle identity and exact output guards");
    return reference;
}
CUmemAllocationProp handles_properties(CUmemGenericAllocationHandle handle, const CUmemAllocationProp& expected, bool imported = false) {
    struct { uint64_t before = handles_before; CUmemAllocationProp value{}; uint64_t after = handles_after; } output;
    CONFORMANCE_CUDA(cuMemGetAllocationPropertiesFromHandle(&output.value, handle));
    require(output.before == handles_before && output.after == handles_after, "allocation properties exact output guards");
    const auto& value = output.value;
    require(value.type == expected.type && value.location.type == expected.location.type && value.location.id == expected.location.id &&
        value.allocFlags.compressionType == 0 && value.allocFlags.gpuDirectRDMACapable == 0 && value.allocFlags.usage == 0,
        "physical allocation defined properties");
    if (!imported) require(value.requestedHandleTypes == expected.requestedHandleTypes, "physical allocation requested handle types");
    return value;
}
struct handle_configuration {
    CUmemAllocationProp properties = virtual_memory_properties();
#ifdef _WIN32
    OBJECT_ATTRIBUTES attributes{};
    static constexpr auto type = CU_MEM_HANDLE_TYPE_WIN32;
#else
    static constexpr auto type = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif
    handle_configuration() {
        int available{};
#ifdef _WIN32
        CONFORMANCE_CUDA(cuDeviceGetAttribute(&available, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, properties.location.id));
        attributes.Length = sizeof(attributes); properties.win32HandleMetaData = &attributes;
#else
        CONFORMANCE_CUDA(cuDeviceGetAttribute(&available, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, properties.location.id));
#endif
        require_capability(available != 0, "VMM OS sharing requires the advertised platform handle type");
        properties.requestedHandleTypes = type;
    }
};
struct physical_os_handle {
#ifdef _WIN32
    HANDLE value{};
    ~physical_os_handle() { if (value) CloseHandle(value); }
    void* argument() const { return value; }
    uintptr_t transport() const { return reinterpret_cast<uintptr_t>(value); }
    void alive() const { DWORD flags{}; require(value && GetHandleInformation(value, &flags), "caller still owns its Win32 handle"); }
    void close() { require(CloseHandle(value), "close owned Win32 memory handle"); value = nullptr; }
    physical_os_handle duplicate() const {
        physical_os_handle result; require(DuplicateHandle(GetCurrentProcess(), value, GetCurrentProcess(), &result.value, 0, FALSE, DUPLICATE_SAME_ACCESS), "duplicate memory handle"); return result;
    }
#else
    int value = -1;
    ~physical_os_handle() { if (value >= 0) ::close(value); }
    void* argument() const { return reinterpret_cast<void*>(uintptr_t(value)); }
    uintptr_t transport() const { return uintptr_t(value); }
    void alive() const { require(value >= 0 && fcntl(value, F_GETFD) >= 0, "caller still owns its POSIX descriptor"); }
    void close() { require(::close(value) == 0, "close owned POSIX memory descriptor"); value = -1; }
    physical_os_handle duplicate() const {
        physical_os_handle result; result.value = fcntl(value, F_DUPFD_CLOEXEC, 0); require(result.value >= 0, "duplicate memory descriptor"); return result;
    }
#endif
    physical_os_handle() = default;
    physical_os_handle(physical_os_handle&& other) noexcept : value(std::exchange(other.value, decltype(value){})) {
#ifndef _WIN32
        other.value = -1;
#endif
    }
    physical_os_handle(const physical_os_handle&) = delete;
    static physical_os_handle receive(conformance_peer& peer) {
        physical_os_handle result;
#ifdef _WIN32
        result.value = reinterpret_cast<HANDLE>(peer.receive_handle());
#else
        result.value = int(peer.receive_handle());
#endif
        result.alive(); return result;
    }
};
physical_os_handle handles_export(CUmemGenericAllocationHandle handle) {
    using guard_type = std::conditional_t<sizeof(physical_os_handle::value) == sizeof(uint32_t), uint32_t, uint64_t>;
    struct { guard_type before = guard_type(handles_before); decltype(physical_os_handle::value) value{}; guard_type after = guard_type(handles_after); } output;
    CONFORMANCE_CUDA(cuMemExportToShareableHandle(&output.value, handle, handle_configuration::type, 0));
    physical_os_handle result; result.value = output.value;
    require(output.before == guard_type(handles_before) && output.after == guard_type(handles_after), "OS export exact output width");
    result.alive(); return result;
}
physical_reference handles_import(const physical_os_handle& source) {
    struct { uint64_t before = handles_before; CUmemGenericAllocationHandle value{}; uint64_t after = handles_after; } output;
    CONFORMANCE_CUDA(cuMemImportFromShareableHandle(&output.value, source.argument(), handle_configuration::type));
    physical_reference result(output.value);
    require(output.before == handles_before && output.after == handles_after && output.value, "OS import exact output guards");
    source.alive(); return result;
}
void handles_metadata() {
    const auto expected = virtual_memory_properties(); auto caller = expected;
    const auto bytes = virtual_memory_granularity(expected) * 3;
    virtual_physical_memory physical(bytes, caller), independent(bytes, expected);
    caller.type = static_cast<CUmemAllocationType>(0); caller.location.id = -2;
    handles_properties(physical.value, expected); handles_properties(independent.value, expected);
    virtual_memory_reservation reserved(bytes); virtual_memory_mapping mapping(reserved.value, bytes, physical.value); mapping.grant(expected.location);
    const auto input = virtual_memory_pattern(bytes, 31); virtual_memory_write(mapping.value, input);
    for (unsigned i = 0; i < 3; ++i) { handles_properties(physical.value, expected); virtual_memory_expect(mapping.value, input, "property queries preserve all backing bytes"); }
}
void handles_retains(bool resurrection) {
    const auto properties = virtual_memory_properties(); const auto granularity = virtual_memory_granularity(properties), bytes = granularity * 3;
    virtual_physical_memory physical(bytes, properties); const auto identity = physical.value;
    virtual_memory_reservation first(bytes), second(bytes), replacement(bytes);
    virtual_memory_mapping a(first.value, bytes, identity), b(second.value, bytes, identity);
    a.grant(properties.location); b.grant(properties.location);
    const auto expected = virtual_memory_pattern(bytes, 57); virtual_memory_write(a.value, expected);
    std::vector<physical_reference> references;
    if (!resurrection) for (auto address : {a.value, a.value + 1, a.value + granularity, a.value + bytes - 1, b.value + bytes / 2})
        references.push_back(handles_retain(address, identity));
    physical.close();
    if (resurrection) {
        CUcontext original{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&original)); CONFORMANCE_CUDA(cuCtxSetCurrent(nullptr));
        try { references.push_back(handles_retain(b.value + bytes - 1, identity)); }
        catch (...) { cuCtxSetCurrent(original); throw; }
        CONFORMANCE_CUDA(cuCtxSetCurrent(original));
    }
    for (size_t i = 1; i < references.size(); ++i) references[i].close();
    a.close(); b.close();
    handles_properties(references.front().value, properties);
    virtual_memory_mapping restored(replacement.value, bytes, references.front().value); restored.grant(properties.location);
    references.front().close();
    virtual_memory_expect(restored.value, expected, "retained backing survives complete unmap/remap", true);
    CONFORMANCE_CUDA(cuMemsetD8(restored.value + 17, 0x9b, 511));
    auto changed = expected; std::fill_n(changed.begin() + 17, 511, 0x9b);
    virtual_memory_expect(restored.value, changed, "last mapping owns writable physical storage after all generic releases");
}
void handles_threading() {
    const auto properties = virtual_memory_properties(); const auto bytes = virtual_memory_granularity(properties) * 2;
    virtual_physical_memory physical(bytes, properties); virtual_memory_reservation reserved(bytes);
    virtual_memory_mapping mapping(reserved.value, bytes, physical.value); mapping.grant(properties.location);
    const auto expected = virtual_memory_pattern(bytes, 87); virtual_memory_write(mapping.value, expected);
    std::atomic<bool> started{false}; std::array<std::exception_ptr, 4> errors{}; std::vector<std::jthread> workers; workers.reserve(errors.size());
    for (size_t index = 0; index < errors.size(); ++index) workers.emplace_back([&, index](std::stop_token stop) {
        while (!started.load(std::memory_order_acquire)) { if (stop.stop_requested()) return; std::this_thread::yield(); }
        try {
            for (unsigned round = 0; round < 32; ++round) {
                auto reference = handles_retain(mapping.value + (index * 71 + round * 37) % bytes, physical.value);
                handles_properties(reference.value, properties);
            }
        } catch (...) { errors[index] = std::current_exception(); }
    });
    started.store(true, std::memory_order_release); for (auto& worker : workers) worker.join();
    for (auto error : errors) if (error) std::rethrow_exception(error);
    physical.close(); virtual_memory_expect(mapping.value, expected, "threaded reference operations preserve full payload and guards", true);
}
void handles_roundtrip() {
    handle_configuration config; const auto bytes = virtual_memory_granularity(config.properties) * 2;
    virtual_physical_memory physical(bytes, config.properties); virtual_memory_reservation original(bytes), first(bytes), second(bytes);
    virtual_memory_mapping initial(original.value, bytes, physical.value); initial.grant(config.properties.location);
    auto expected = virtual_memory_pattern(bytes, 47); virtual_memory_write(initial.value, expected);
    auto exported = handles_export(physical.value); auto duplicate = exported.duplicate();
    require(cuMemImportFromShareableHandle(nullptr, exported.argument(), config.type) == CUDA_ERROR_INVALID_VALUE, "failed import does not consume OS reference"); exported.alive();
    auto imported_a = handles_import(exported), imported_b = handles_import(duplicate);
    auto imported_properties = handles_properties(imported_a.value, config.properties, true);
    handles_properties(imported_b.value, config.properties, true);
    if (imported_properties.requestedHandleTypes & config.type) {
        auto reexported = handles_export(imported_a.value); auto reimported = handles_import(reexported);
        handles_properties(reimported.value, config.properties, true);
    } else {
        decltype(physical_os_handle::value) unused{};
        require(cuMemExportToShareableHandle(&unused, imported_a.value, config.type, 0) == CUDA_ERROR_INVALID_VALUE, "imported handle does not advertise re-export permission");
    }
    exported.close(); duplicate.close(); initial.close(); physical.close();
    virtual_memory_mapping a(first.value, bytes, imported_a.value), b(second.value, bytes, imported_b.value);
    a.grant(config.properties.location); b.grant(config.properties.location);
    auto retained = handles_retain(b.value + 13, imported_b.value);
    imported_a.close(); imported_b.close(); retained.close();
    virtual_memory_expect(a.value, expected, "imported backing survives all OS and creating references", true);
    virtual_runtime(cudaMemset(reinterpret_cast<void*>(uintptr_t(a.value + 29)), 0x5d, bytes - 73));
    std::fill(expected.begin() + 29, expected.end() - 44, 0x5d);
    virtual_memory_expect(b.value, expected, "independent imports share actual physical bytes and guards");
}
void handles_validation() {
    const auto properties = virtual_memory_properties(); const auto bytes = virtual_memory_granularity(properties);
    CUmemGenericAllocationHandle output{}; CUmemAllocationProp queried{};
    require(cuMemCreate(nullptr, bytes, &properties, 0) == CUDA_ERROR_INVALID_VALUE, "missing physical allocation output");
    require(cuMemCreate(&output, bytes, nullptr, 0) == CUDA_ERROR_INVALID_VALUE, "missing physical allocation properties");
    require(cuMemCreate(&output, 0, &properties, 0) == CUDA_ERROR_INVALID_VALUE, "empty physical allocation");
    require(cuMemCreate(&output, bytes + 1, &properties, 0) == CUDA_ERROR_INVALID_VALUE, "physical allocation granularity");
    require(cuMemCreate(&output, bytes, &properties, 1) == CUDA_ERROR_INVALID_VALUE, "reserved physical allocation flags");
    auto invalid = properties; invalid.location.id = -2;
    require(cuMemCreate(&output, bytes, &invalid, 0) == CUDA_ERROR_INVALID_DEVICE, "invalid physical allocation device");
    virtual_physical_memory physical(bytes, properties); virtual_memory_reservation reserved(bytes);
    virtual_memory_mapping mapping(reserved.value, bytes, physical.value); mapping.grant(properties.location);
    const auto expected = virtual_memory_pattern(bytes, 73); virtual_memory_write(mapping.value, expected);
    device_buffer ordinary(64);
    require(cuMemRetainAllocationHandle(nullptr, reinterpret_cast<void*>(uintptr_t(mapping.value))) == CUDA_ERROR_INVALID_VALUE, "missing retain output");
    require(cuMemRetainAllocationHandle(&output, reinterpret_cast<void*>(uintptr_t(ordinary.value))) == CUDA_ERROR_INVALID_VALUE, "retain rejects ordinary allocations");
    require(cuMemGetAllocationPropertiesFromHandle(nullptr, physical.value) == CUDA_ERROR_INVALID_VALUE, "missing properties output");
    decltype(physical_os_handle::value) os{};
    require(cuMemExportToShareableHandle(nullptr, physical.value, handle_configuration::type, 0) == CUDA_ERROR_INVALID_VALUE, "missing OS export output");
    require(cuMemExportToShareableHandle(&os, physical.value, handle_configuration::type, 1) == CUDA_ERROR_INVALID_VALUE, "reserved OS export flags");
    require(cuMemExportToShareableHandle(&os, physical.value, CU_MEM_HANDLE_TYPE_NONE, 0) == CUDA_ERROR_INVALID_VALUE, "invalid OS export type");
    require(cuMemExportToShareableHandle(&os, physical.value, handle_configuration::type, 0) == CUDA_ERROR_INVALID_VALUE, "non-exportable backing");
    require(cuMemImportFromShareableHandle(&output, nullptr, CU_MEM_HANDLE_TYPE_NONE) == CUDA_ERROR_INVALID_VALUE, "invalid OS import type");
    auto repaired = handles_retain(mapping.value + bytes - 1, physical.value);
    CONFORMANCE_CUDA(cuMemGetAllocationPropertiesFromHandle(&queried, repaired.value));
    virtual_memory_expect(mapping.value, expected, "argument repair preserves complete physical backing");
    repaired.close(); mapping.close();
    require(cuMemRetainAllocationHandle(&output, reinterpret_cast<void*>(uintptr_t(reserved.value))) == CUDA_ERROR_INVALID_VALUE, "retain rejects unbacked reservation");
}
struct handles_message {
    uint64_t kind{}, bytes{}, seed{}, phase{};
    std::array<unsigned char, 16> uuid{};
    std::array<uint64_t, 2> reserved{};
};
static_assert(sizeof(handles_message) == 64);
handles_message handles_hello(size_t bytes, unsigned seed) {
    handles_message message{1, bytes, seed}; CUuuid uuid{}; CUdevice device{};
    CONFORMANCE_CUDA(cuCtxGetDevice(&device)); CONFORMANCE_CUDA(cuDeviceGetUuid(&uuid, device));
    std::memcpy(message.uuid.data(), uuid.bytes, sizeof(uuid.bytes)); return message;
}
void handles_send(conformance_peer& peer, const handles_message& message) { peer.send(std::as_bytes(std::span(&message, 1))); }
handles_message handles_receive(conformance_peer& peer) { handles_message message{}; peer.receive(std::as_writable_bytes(std::span(&message, 1))); return message; }
void handles_identity(const handles_message& message, const handle_configuration& config) {
    require(message.kind == 1 && message.bytes == virtual_memory_granularity(config.properties) * 2 && message.seed <= UINT32_MAX, "peer allocation metadata");
    require(message.uuid == handles_hello(message.bytes, unsigned(message.seed)).uuid, "peer participants use the same actual device UUID");
}
void handles_patch(CUdeviceptr pointer, std::vector<unsigned char>& expected, unsigned phase, bool importer) {
    const size_t offset = importer ? expected.size() / 2 + phase * 211 : 64 + phase * 173;
    const size_t count = importer ? 383 + phase * 43 : 257 + phase * 37;
    const auto pattern = static_cast<unsigned char>((importer ? 91 : 17) + phase * 31);
    if (importer) virtual_runtime(cudaMemset(reinterpret_cast<void*>(uintptr_t(pointer + offset)), pattern, count));
    else CONFORMANCE_CUDA(cuMemsetD8(pointer + offset, pattern, count));
    std::fill_n(expected.begin() + offset, count, pattern); CONFORMANCE_CUDA(cuCtxSynchronize());
}
void handles_expected_patch(std::vector<unsigned char>& expected, unsigned phase, bool importer) {
    const size_t offset = importer ? expected.size() / 2 + phase * 211 : 64 + phase * 173;
    std::fill_n(expected.begin() + offset, importer ? 383 + phase * 43 : 257 + phase * 37,
        static_cast<unsigned char>((importer ? 91 : 17) + phase * 31));
}
void handles_importer() {
    require(handles_peer != nullptr, "importer has a real control peer"); handle_configuration config;
    auto& peer = *handles_peer; const auto hello = handles_receive(peer); handles_identity(hello, config);
    auto os = physical_os_handle::receive(peer); auto imported = handles_import(os); os.close();
    virtual_memory_reservation reserved(hello.bytes); virtual_memory_mapping mapping(reserved.value, hello.bytes, imported.value); mapping.grant(config.properties.location);
    auto expected = virtual_memory_pattern(hello.bytes, unsigned(hello.seed)); virtual_memory_expect(mapping.value, expected, "independent importer initial complete backing");
    imported.close(); handles_send(peer, handles_message{2});
    for (unsigned phase = 0; phase < 3; ++phase) {
        const auto command = handles_receive(peer); require(command.kind == 3 && command.phase == phase, "ordered importer phase");
        handles_expected_patch(expected, phase, false);
        virtual_memory_expect(mapping.value, expected, "importer sees exporter's completed GPU writes", true);
        handles_patch(mapping.value, expected, phase, true);
        virtual_memory_expect(mapping.value, expected, "importer complete bytes and preserved guards");
        handles_send(peer, handles_message{4, 0, 0, phase});
    }
    require(handles_receive(peer).kind == 9, "importer completion handoff");
}
void handles_sharing() {
    handle_configuration config; const auto bytes = virtual_memory_granularity(config.properties) * 2;
    virtual_physical_memory physical(bytes, config.properties); virtual_memory_reservation reserved(bytes);
    virtual_memory_mapping mapping(reserved.value, bytes, physical.value); mapping.grant(config.properties.location);
    auto expected = virtual_memory_pattern(bytes, 43); virtual_memory_write(mapping.value, expected);
    auto os = handles_export(physical.value); conformance_peer peer(handles_executable, "cuda", "importer");
    handles_send(peer, handles_hello(bytes, 43)); peer.send_handle(os.transport());
    require(handles_receive(peer).kind == 2, "independent importer owns its live mapping"); os.close(); physical.close();
    for (unsigned phase = 0; phase < 3; ++phase) {
        handles_patch(mapping.value, expected, phase, false);
        handles_send(peer, handles_message{3, 0, 0, phase});
        const auto reply = handles_receive(peer); require(reply.kind == 4 && reply.phase == phase, "completed importer GPU phase");
        handles_expected_patch(expected, phase, true);
        virtual_memory_expect(mapping.value, expected, "exporter sees independent process writes with all guards", true);
    }
    handles_send(peer, handles_message{9}); require(peer.wait() == 0, "importer exits successfully");
    handles_patch(mapping.value, expected, 3, false);
    virtual_memory_expect(mapping.value, expected, "exporter mapping survives importer process exit");
}
void handles_exporter() {
    require(handles_peer != nullptr, "exporter has a real control peer"); handle_configuration config;
    const auto bytes = virtual_memory_granularity(config.properties) * 2;
    virtual_physical_memory physical(bytes, config.properties); virtual_memory_reservation reserved(bytes);
    virtual_memory_mapping mapping(reserved.value, bytes, physical.value); mapping.grant(config.properties.location);
    const auto expected = virtual_memory_pattern(bytes, 67); virtual_memory_write(mapping.value, expected);
    auto os = handles_export(physical.value); handles_send(*handles_peer, handles_hello(bytes, 67)); handles_peer->send_handle(os.transport());
    require(handles_receive(*handles_peer).kind == 2, "importer acquired allocation before exporter exit");
}
void handles_exporter_exit() {
    handle_configuration config; conformance_peer peer(handles_executable, "cuda", "exporter");
    const auto hello = handles_receive(peer); handles_identity(hello, config);
    auto os = physical_os_handle::receive(peer); auto imported = handles_import(os); os.close();
    virtual_memory_reservation reserved(hello.bytes); virtual_memory_mapping mapping(reserved.value, hello.bytes, imported.value); mapping.grant(config.properties.location);
    auto expected = virtual_memory_pattern(hello.bytes, unsigned(hello.seed)); virtual_memory_expect(mapping.value, expected, "fresh exporter initial shared payload");
    handles_send(peer, handles_message{2}); require(peer.wait() == 0, "exporter process has actually exited");
    imported.close(); virtual_memory_expect(mapping.value, expected, "importer bytes survive exporter process and all generic references", true);
    handles_patch(mapping.value, expected, 2, true);
    virtual_memory_expect(mapping.value, expected, "importer owns writable backing after actual exporter exit");
}
void handles_discovery() {
    for (const char* name : {"cuMemCreate", "cuMemRelease", "cuMemExportToShareableHandle", "cuMemImportFromShareableHandle", "cuMemGetAllocationPropertiesFromHandle", "cuMemRetainAllocationHandle"}) {
        const int version = std::string_view(name) == "cuMemRetainAllocationHandle" ? 11000 : 10020;
        void* address{}; CUdriverProcAddressQueryResult status{};
        CONFORMANCE_CUDA(cuGetProcAddress(name, &address, version, 0, &status)); require(address && status == CU_GET_PROC_ADDRESS_SUCCESS, "physical handle procedure version");
        CONFORMANCE_CUDA(cuGetProcAddress(name, &address, version - 1, 0, &status)); require(!address && status == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "physical handle introduction boundary");
    }
    void* address{}; CONFORMANCE_CUDA(cuGetProcAddress("cuMemImportFromShareableHandle", &address, 10020, 0, nullptr));
    require(reinterpret_cast<PFN_cuMemImportFromShareableHandle_v10020>(address)(nullptr, nullptr, CU_MEM_HANDLE_TYPE_NONE) == CUDA_ERROR_NOT_INITIALIZED, "discovered import preserves cold Driver behavior");
}
void execute_virtual_memory_handles() {
    require_virtual_memory();
    if (handles_case == "metadata") handles_metadata();
    else if (handles_case == "retains") handles_retains(false);
    else if (handles_case == "resurrection") handles_retains(true);
    else if (handles_case == "threading") handles_threading();
    else if (handles_case == "roundtrip") handles_roundtrip();
    else if (handles_case == "validation") handles_validation();
    else if (handles_case == "sharing") handles_sharing();
    else if (handles_case == "exporter_exit") handles_exporter_exit();
    else if (handles_case == "importer") handles_importer();
    else if (handles_case == "exporter") handles_exporter();
    else throw std::runtime_error("unknown virtual_memory_handles case");
}
}
int main(int argc, char** argv) {
    cudavm::handles_executable = argv[0];
    if (argc == 2 && std::string_view(argv[1]) == "discovery") return cudavm::run_host_conformance(argc, argv, cudavm::handles_discovery);
    if (argc == 7 && std::string_view(argv[3]) == "--peer") {
        cudavm::handles_peer = std::make_unique<cudavm::conformance_peer>(cudavm::conformance_peer::connect(argv[4], argv[5], argv[6])); argc = 3;
    }
    if (argc == 3) { cudavm::handles_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_virtual_memory_handles);
}
