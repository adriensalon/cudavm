#include "test_framework.hpp"
#include <cudaTypedefs.h>

namespace cudavm {
std::string_view alias_case;
void alias_buffers(bool submaps, bool released) {
    const auto properties = virtual_memory_properties();
    const auto granularity = virtual_memory_granularity(properties), bytes = granularity * 3;
    virtual_physical_memory physical(bytes, properties), separate(bytes, properties);
    virtual_memory_reservation first(bytes), second(submaps ? bytes * 2 : bytes), independent(bytes);
    virtual_memory_mapping a(first.value, bytes, physical.value), b(second.value, submaps ? granularity : bytes, physical.value),
        other(independent.value, bytes, separate.value);
    a.grant(properties.location); b.grant(properties.location); other.grant(properties.location);
    auto expected = virtual_memory_pattern(bytes, 17), untouched = virtual_memory_pattern(bytes, 29);
    virtual_memory_write(a.value, expected); virtual_memory_write(other.value, untouched);
    if (released) physical.close();
    virtual_memory_expect(b.value, {expected.data(), b.bytes}, "initial physical alias", true);
    for (bool runtime : {false, true}) {
        const size_t offset = 19 + runtime * 23, count = b.bytes - offset - 31;
        const auto pattern = static_cast<unsigned char>(runtime ? 0xd9 : 0x37);
        if (runtime) virtual_runtime(cudaMemset(reinterpret_cast<void*>(uintptr_t(b.value + offset)), pattern, count));
        else TEST_CUDA(cuMemsetD8(b.value + offset, pattern, count));
        std::fill_n(expected.begin() + offset, count, pattern);
        virtual_memory_expect(a.value, expected, "full backing after alias write", runtime);
        virtual_memory_expect(b.value, {expected.data(), b.bytes}, "written alias and guards", !runtime);
        virtual_memory_expect(other.value, untouched, "independent physical storage", runtime);
    }
    if (submaps) {
        virtual_memory_mapping adjacent(second.value + granularity, bytes, physical.value);
        adjacent.grant(properties.location);
        const auto changed = virtual_memory_pattern(bytes, 93);
        virtual_memory_write(adjacent.value, changed, true);
        virtual_memory_expect(a.value, changed, "adjacent mapping shares complete physical backing");
        virtual_memory_expect(b.value, {changed.data(), b.bytes}, "prefix alias after adjacent mapping write");
    }
    b.close();
    virtual_memory_expect(a.value, submaps ? virtual_memory_pattern(bytes, 93) : expected, "surviving mapping after sibling unmap");
}
void alias_remap() {
    const auto properties = virtual_memory_properties(); const auto bytes = virtual_memory_granularity(properties) * 2;
    virtual_physical_memory physical_a(bytes, properties), physical_b(bytes, properties);
    virtual_memory_reservation fixed(bytes), other_a(bytes), other_b(bytes);
    virtual_memory_mapping a(other_a.value, bytes, physical_a.value), b(other_b.value, bytes, physical_b.value);
    a.grant(properties.location); b.grant(properties.location);
    const auto expected_a = virtual_memory_pattern(bytes, 13), expected_b = virtual_memory_pattern(bytes, 87);
    virtual_memory_write(a.value, expected_a); virtual_memory_write(b.value, expected_b, true);
    for (unsigned phase = 0; phase < 4; ++phase) {
        virtual_memory_mapping mapping(fixed.value, bytes, phase & 1 ? physical_b.value : physical_a.value);
        unsigned long long access = 99; TEST_CUDA(cuMemGetAccess(&access, &properties.location, mapping.value));
        require(access == CU_MEM_ACCESS_FLAGS_PROT_NONE, "remapped VA starts without access");
        mapping.grant(properties.location);
        virtual_memory_expect(mapping.value, phase & 1 ? expected_b : expected_a, "fixed VA selects its new physical backing", phase & 1);
        virtual_memory_expect(a.value, expected_a, "original backing survives remap");
        virtual_memory_expect(b.value, expected_b, "alternate backing survives remap");
    }
}
void alias_ordering() {
    const auto properties = virtual_memory_properties(); const auto bytes = virtual_memory_granularity(properties) * 2;
    virtual_physical_memory physical(bytes, properties);
    virtual_memory_reservation first(bytes), second(bytes);
    virtual_memory_mapping a(first.value, bytes, physical.value), b(second.value, bytes, physical.value);
    a.grant(properties.location); b.grant(properties.location);
    device_buffer snapshot(bytes); auto expected = virtual_memory_pattern(bytes, 41); virtual_memory_write(a.value, expected);
    CUstream producer{}, consumer{}; CUevent event{};
    TEST_CUDA(cuStreamCreate(&producer, CU_STREAM_NON_BLOCKING));
    struct stream_cleanup { CUstream value{}; ~stream_cleanup() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } } } clean_producer{producer}, clean_consumer;
    TEST_CUDA(cuStreamCreate(&consumer, CU_STREAM_NON_BLOCKING)); clean_consumer.value = consumer;
    TEST_CUDA(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
    struct event_cleanup { CUevent value; ~event_cleanup() { cuEventDestroy(value); } } clean_event{event};
    for (unsigned phase = 0; phase < 3; ++phase) {
        const size_t offset = 23 + phase * 73, count = bytes - offset - 47;
        const auto pattern = static_cast<unsigned char>(phase * 37 + 51);
        TEST_CUDA(cuMemsetD8Async(a.value + offset, pattern, count, producer));
        virtual_runtime(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event), reinterpret_cast<cudaStream_t>(producer)));
        TEST_CUDA(cuStreamWaitEvent(consumer, event, 0));
        virtual_runtime(cudaMemcpyAsync(reinterpret_cast<void*>(uintptr_t(snapshot.value)), reinterpret_cast<void*>(uintptr_t(b.value)), bytes,
            cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(consumer)));
        TEST_CUDA(cuStreamSynchronize(consumer));
        std::fill_n(expected.begin() + offset, count, pattern);
        virtual_memory_expect(snapshot.value, expected, "event-ordered snapshot through other alias");
        virtual_memory_expect(b.value, expected, "event-ordered complete alias backing", true);
    }
}
void alias_contexts() {
    CUcontext original{}, separate{}; CUdevice device{};
    TEST_CUDA(cuCtxGetCurrent(&original)); TEST_CUDA(cuCtxGetDevice(&device));
    TEST_CUDA(cuCtxCreate(&separate, nullptr, 0, device));
    try {
        const auto properties = virtual_memory_properties(); const auto bytes = virtual_memory_granularity(properties) * 2;
        virtual_physical_memory physical(bytes, properties);
        virtual_memory_reservation first(bytes), second(bytes);
        virtual_memory_mapping a(first.value, bytes, physical.value), b(second.value, bytes, physical.value);
        a.grant(properties.location); b.grant(properties.location);
        auto expected = virtual_memory_pattern(bytes, 97); virtual_memory_write(a.value, expected);
        TEST_CUDA(cuCtxSetCurrent(original));
        TEST_CUDA(cuCtxDestroy(separate)); separate = nullptr;
        virtual_memory_expect(b.value, expected, "process mapping survives creating context retirement", true);
        virtual_runtime(cudaMemset(reinterpret_cast<void*>(uintptr_t(b.value + 13)), 0x5b, bytes - 37));
        std::fill(expected.begin() + 13, expected.end() - 24, 0x5b);
        virtual_memory_expect(a.value, expected, "retired-context alias remains writable from original context");
        physical.close();
        virtual_memory_expect(b.value, expected, "retired context and released handle preserve mapping");
    } catch (...) { cuCtxSetCurrent(original); if (separate) cuCtxDestroy(separate); throw; }
    TEST_CUDA(cuCtxSetCurrent(original));
}
void alias_validation() {
    const auto properties = virtual_memory_properties(); const auto granularity = virtual_memory_granularity(properties), bytes = granularity * 2;
    virtual_physical_memory physical(bytes, properties); virtual_memory_reservation reserved(bytes * 3), extra(bytes);
    virtual_memory_mapping mapping(reserved.value, bytes, physical.value); mapping.grant(properties.location);
    const auto expected = virtual_memory_pattern(bytes, 63); virtual_memory_write(mapping.value, expected);
    for (auto result : {cuMemMap(extra.value, bytes, 0, physical.value, 1),
                        cuMemMap(extra.value, bytes, granularity, physical.value, 0),
                        cuMemMap(extra.value + 1, granularity, 0, physical.value, 0),
                        cuMemMap(extra.value, granularity + 1, 0, physical.value, 0),
                        cuMemMap(extra.value, 0, 0, physical.value, 0),
                        cuMemMap(reserved.value + granularity, bytes, 0, physical.value, 0),
                        cuMemUnmap(reserved.value, granularity),
                        cuMemUnmap(reserved.value + granularity, granularity),
                        cuMemAddressFree(reserved.value, reserved.bytes), cuMemFree(reserved.value)})
        require(result == CUDA_ERROR_INVALID_VALUE, "rejected mapping argument preserves existing backing");
    virtual_memory_expect(mapping.value, expected, "all backing bytes survive invalid mapping/unmap/free");
    virtual_memory_mapping repaired(extra.value, bytes, physical.value); repaired.grant(properties.location);
    virtual_memory_expect(repaired.value, expected, "repaired physical alias mapping", true);
}
void alias_contextless() {
    const auto properties = virtual_memory_properties(); CUcontext original{};
    TEST_CUDA(cuCtxGetCurrent(&original)); TEST_CUDA(cuCtxSetCurrent(nullptr));
    try {
        const auto bytes = virtual_memory_granularity(properties) * 2;
        virtual_physical_memory physical(bytes, properties); virtual_memory_reservation first(bytes), second(bytes);
        virtual_memory_mapping a(first.value, bytes, physical.value), b(second.value, bytes, physical.value);
        a.grant(properties.location); b.grant(properties.location);
        CUcontext current = original; TEST_CUDA(cuCtxGetCurrent(&current));
        require(!current, "VMM creation and mapping do not bind a public context");
        TEST_CUDA(cuCtxSetCurrent(original));
        const auto expected = virtual_memory_pattern(bytes, 73);
        virtual_memory_write(a.value, expected); virtual_memory_expect(b.value, expected, "context-free VMM maps execute in an explicit context", true);
        TEST_CUDA(cuCtxSetCurrent(nullptr));
        a.close(); b.close(); physical.close();
    } catch (...) { cuCtxSetCurrent(original); throw; }
    TEST_CUDA(cuCtxSetCurrent(original));
}
void alias_discovery() {
    for (const char* name : {"cuMemCreate", "cuMemRelease", "cuMemMap", "cuMemUnmap", "cuMemSetAccess", "cuMemGetAccess", "cuMemGetAllocationPropertiesFromHandle", "cuMemRetainAllocationHandle"}) {
        const int version = std::string_view(name) == "cuMemRetainAllocationHandle" ? 11000 : 10020;
        void* address{}; CUdriverProcAddressQueryResult status{};
        TEST_CUDA(cuGetProcAddress(name, &address, version, 0, &status));
        require(address && status == CU_GET_PROC_ADDRESS_SUCCESS, "virtual memory procedure introduction version");
        TEST_CUDA(cuGetProcAddress(name, &address, version - 1, 0, &status));
        require(!address && status == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "virtual memory procedure predates introduction");
    }
    void* address{}; TEST_CUDA(cuGetProcAddress("cuMemMap", &address, 10020, 0, nullptr));
    require(reinterpret_cast<PFN_cuMemMap_v10020>(address)(0, 0, 0, 0, 0) == CUDA_ERROR_NOT_INITIALIZED, "discovered mapping preserves cold Driver behavior");
}
void execute_virtual_memory_aliases() {
    require_virtual_memory();
    if (alias_case == "aliases") alias_buffers(false, false);
    else if (alias_case == "submaps") alias_buffers(true, false);
    else if (alias_case == "released") alias_buffers(false, true);
    else if (alias_case == "remap") alias_remap();
    else if (alias_case == "ordering") alias_ordering();
    else if (alias_case == "contexts") alias_contexts();
    else if (alias_case == "validation") alias_validation();
    else if (alias_case == "contextless") alias_contextless();
    else throw std::runtime_error("unknown virtual_memory_aliases case");
}
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "discovery") return cudavm::run_host_test(argc, argv, cudavm::alias_discovery);
    if (argc == 3) { cudavm::alias_case = argv[2]; --argc; }
    return cudavm::run_test(argc, argv, cudavm::execute_virtual_memory_aliases);
}
