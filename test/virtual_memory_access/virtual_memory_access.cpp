#include "test_framework.hpp"
#include <cudaTypedefs.h>
#include <cstring>

namespace cudavm {
std::string_view access_case;
unsigned long long access_query(CUdeviceptr pointer, CUmemLocation location) {
    struct { uint64_t before = 0x19375928abcdef01; unsigned long long value = 99; uint64_t after = 0xfedcba0129183745; } output;
    TEST_CUDA(cuMemGetAccess(&output.value, &location, pointer));
    require(output.before == 0x19375928abcdef01 && output.after == 0xfedcba0129183745, "access query exact-width output guards");
    return output.value;
}
void access_expect(CUdeviceptr pointer, size_t bytes, CUmemLocation location, CUmemAccess_flags flags) {
    for (auto offset : {size_t(0), size_t(1), bytes / 2, bytes - 1})
        require(access_query(pointer + offset, location) == flags, "complete mapped range permission state");
}
void access_states(bool aliases) {
    const auto properties = virtual_memory_properties(); const auto bytes = virtual_memory_granularity(properties) * 2;
    virtual_physical_memory physical(bytes, properties), independent(bytes, properties);
    virtual_memory_reservation first(bytes), second(bytes);
    virtual_memory_mapping a(first.value, bytes, physical.value), b(second.value, bytes, aliases ? physical.value : independent.value);
    access_expect(a.value, bytes, properties.location, CU_MEM_ACCESS_FLAGS_PROT_NONE);
    access_expect(b.value, bytes, properties.location, CU_MEM_ACCESS_FLAGS_PROT_NONE);
    a.grant(properties.location); b.grant(properties.location);
    auto expected_a = virtual_memory_pattern(bytes, 17), expected_b = aliases ? expected_a : virtual_memory_pattern(bytes, 93);
    virtual_memory_write(a.value, expected_a); virtual_memory_write(b.value, expected_b, true);
    const std::array<CUmemAccess_flags, 7> states{CU_MEM_ACCESS_FLAGS_PROT_READ, CU_MEM_ACCESS_FLAGS_PROT_READ,
        CU_MEM_ACCESS_FLAGS_PROT_NONE, CU_MEM_ACCESS_FLAGS_PROT_READWRITE, CU_MEM_ACCESS_FLAGS_PROT_READ,
        CU_MEM_ACCESS_FLAGS_PROT_NONE, CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
    unsigned phase = 0;
    for (auto flags : states) {
        a.grant(properties.location, flags);
        access_expect(a.value, bytes, properties.location, flags);
        access_expect(b.value, bytes, properties.location, CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
        const size_t offset = 23 + phase * 37, count = bytes - offset - 41;
        const auto pattern = static_cast<unsigned char>(0x35 + phase * 29);
        virtual_runtime(cudaMemset(reinterpret_cast<void*>(uintptr_t(b.value + offset)), pattern, count));
        std::fill_n(expected_b.begin() + offset, count, pattern);
        if (aliases) expected_a = expected_b;
        if (flags != CU_MEM_ACCESS_FLAGS_PROT_NONE) virtual_memory_expect(a.value, expected_a, "legal access after permission transition", phase & 1);
        virtual_memory_expect(b.value, expected_b, "other mapping permission and full guarded bytes", !(phase & 1));
        if (flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE) {
            TEST_CUDA(cuMemsetD8(a.value + 13, 0x97, bytes - 31));
            std::fill(expected_a.begin() + 13, expected_a.end() - 18, 0x97);
            if (aliases) expected_b = expected_a;
            virtual_memory_expect(a.value, expected_a, "restored write permission and guards");
        }
        ++phase;
    }
}
void access_spans() {
    const auto properties = virtual_memory_properties(); const auto bytes = virtual_memory_granularity(properties);
    virtual_physical_memory physical_a(bytes, properties), physical_b(bytes, properties), physical_c(bytes, properties);
    virtual_memory_reservation reserved(bytes * 3);
    virtual_memory_mapping a(reserved.value, bytes, physical_a.value), b(reserved.value + bytes, bytes, physical_b.value),
        c(reserved.value + bytes * 2, bytes, physical_c.value);
    CUmemAccessDesc access{properties.location, CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
    TEST_CUDA(cuMemSetAccess(reserved.value, bytes * 3, &access, 1));
    const auto expected_a = virtual_memory_pattern(bytes, 23), expected_b = virtual_memory_pattern(bytes, 51), expected_c = virtual_memory_pattern(bytes, 91);
    virtual_memory_write(a.value, expected_a); virtual_memory_write(b.value, expected_b); virtual_memory_write(c.value, expected_c);
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
    TEST_CUDA(cuMemSetAccess(reserved.value, bytes * 2, &access, 1));
    access_expect(a.value, bytes, properties.location, access.flags); access_expect(b.value, bytes, properties.location, access.flags);
    access_expect(c.value, bytes, properties.location, CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
    virtual_memory_expect(c.value, expected_c, "neighbor access and payload survive range protection", true);
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
    TEST_CUDA(cuMemSetAccess(reserved.value, bytes * 3, &access, 1));
    virtual_memory_expect(a.value, expected_a, "first adjacent mapping after access restoration");
    virtual_memory_expect(b.value, expected_b, "second adjacent mapping after access restoration", true);
    virtual_memory_expect(c.value, expected_c, "third adjacent mapping after access restoration");
}
void access_validation() {
    const auto properties = virtual_memory_properties(); const auto bytes = virtual_memory_granularity(properties) * 2;
    virtual_physical_memory physical(bytes, properties); virtual_memory_reservation reserved(bytes);
    virtual_memory_mapping mapping(reserved.value, bytes, physical.value); mapping.grant(properties.location);
    auto expected = virtual_memory_pattern(bytes, 71); virtual_memory_write(mapping.value, expected);
    mapping.grant(properties.location, CU_MEM_ACCESS_FLAGS_PROT_READ);
    CUmemAccessDesc valid{properties.location, CU_MEM_ACCESS_FLAGS_PROT_READWRITE}, invalid = valid;
    require(cuMemSetAccess(mapping.value, bytes, nullptr, 1) == CUDA_ERROR_INVALID_VALUE, "missing access descriptors");
    invalid.flags = static_cast<CUmemAccess_flags>(2);
    std::array<CUmemAccessDesc, 2> values{valid, invalid};
    require(cuMemSetAccess(mapping.value, bytes, values.data(), values.size()) == CUDA_ERROR_INVALID_VALUE, "whole access descriptor array validated before mutation");
    access_expect(mapping.value, bytes, properties.location, CU_MEM_ACCESS_FLAGS_PROT_READ);
    invalid = valid; invalid.location.type = CU_MEM_LOCATION_TYPE_INVALID;
    require(cuMemSetAccess(mapping.value, bytes, &invalid, 1) == CUDA_ERROR_INVALID_VALUE, "invalid access location type");
    invalid = valid; invalid.location.id = -2;
    require(cuMemSetAccess(mapping.value, bytes, &invalid, 1) == CUDA_ERROR_INVALID_DEVICE, "invalid access location device");
    require(cuMemSetAccess(mapping.value + 1, bytes - 1, &valid, 1) == CUDA_ERROR_INVALID_VALUE, "unaligned access range");
    require(cuMemSetAccess(mapping.value, bytes / 2, &valid, 1) == CUDA_ERROR_INVALID_VALUE, "incomplete mapping access range");
    require(cuMemSetAccess(mapping.value, 0, &valid, 1) == CUDA_ERROR_INVALID_VALUE, "empty access range");
    unsigned long long flags{}; device_buffer ordinary(64);
    require(cuMemGetAccess(nullptr, &properties.location, mapping.value) == CUDA_ERROR_INVALID_VALUE, "missing access query output");
    require(cuMemGetAccess(&flags, nullptr, mapping.value) == CUDA_ERROR_INVALID_VALUE, "missing access query location");
    require(cuMemGetAccess(&flags, &properties.location, ordinary.value) == CUDA_ERROR_INVALID_VALUE, "access query requires VMM mapping");
    access_expect(mapping.value, bytes, properties.location, CU_MEM_ACCESS_FLAGS_PROT_READ);
    virtual_memory_expect(mapping.value, expected, "all bytes and permissions survive rejected access arguments", true);
    mapping.grant(properties.location);
    TEST_CUDA(cuMemsetD8(mapping.value + 17, 0x8b, bytes - 49));
    std::fill(expected.begin() + 17, expected.end() - 32, 0x8b);
    virtual_memory_expect(mapping.value, expected, "repaired protection permits writes");
}
void access_contextless() {
    const auto properties = virtual_memory_properties(); const auto bytes = virtual_memory_granularity(properties);
    virtual_physical_memory physical(bytes, properties); virtual_memory_reservation reserved(bytes);
    virtual_memory_mapping mapping(reserved.value, bytes, physical.value); mapping.grant(properties.location);
    const auto expected = virtual_memory_pattern(bytes, 83); virtual_memory_write(mapping.value, expected);
    CUcontext original{}; TEST_CUDA(cuCtxGetCurrent(&original));
    unsigned before_flags{}, after_flags{}; int before_active{}, after_active{};
    TEST_CUDA(cuDevicePrimaryCtxGetState(properties.location.id, &before_flags, &before_active));
    TEST_CUDA(cuCtxSetCurrent(nullptr));
    try {
        for (auto flags : {CU_MEM_ACCESS_FLAGS_PROT_READ, CU_MEM_ACCESS_FLAGS_PROT_NONE, CU_MEM_ACCESS_FLAGS_PROT_READWRITE}) {
            mapping.grant(properties.location, flags); access_expect(mapping.value, bytes, properties.location, flags);
        }
        CUcontext current = original; TEST_CUDA(cuCtxGetCurrent(&current)); require(!current, "access control leaves current context null");
        TEST_CUDA(cuDevicePrimaryCtxGetState(properties.location.id, &after_flags, &after_active));
        require(before_flags == after_flags && before_active == after_active, "access control preserves public primary context state");
    } catch (...) { cuCtxSetCurrent(original); throw; }
    TEST_CUDA(cuCtxSetCurrent(original));
    virtual_memory_expect(mapping.value, expected, "context-free access transitions preserve all initialized bytes");
}
constexpr char access_ptx[] = R"ptx(
.version 6.0
.target sm_60
.address_size 64
.visible .entry access_probe(.param .u64 output) {
 .reg .b64 %out; .reg .b32 %value;
 ld.param.u64 %out,[output]; mov.u32 %value,0x19375928;
 st.global.u32 [%out],%value; ret;
}
.visible .entry access_touch(.param .u64 pointer, .param .u64 output, .param .u32 mode) {
 .reg .b64 %ptr,%out; .reg .b32 %mode,%value; .reg .pred %write;
 ld.param.u64 %ptr,[pointer]; ld.param.u64 %out,[output]; ld.param.u32 %mode,[mode];
 setp.ne.u32 %write,%mode,0; @%write bra STORE;
 ld.global.u32 %value,[%ptr]; st.global.u32 [%out],%value; ret;
STORE:
 mov.u32 %value,0x3719a5c9; st.global.u32 [%ptr],%value; st.global.u32 [%out],%value; ret;
}
)ptx";
struct access_program {
    CUmodule module{}; CUlibrary library{};
    explicit access_program(bool use_library) {
        if (use_library) TEST_CUDA(cuLibraryLoadData(&library, access_ptx, nullptr, nullptr, 0, nullptr, nullptr, 0));
        else TEST_CUDA(cuModuleLoadData(&module, access_ptx));
    }
    ~access_program() { if (library) cuLibraryUnload(library); if (module) cuModuleUnload(module); }
    CUfunction function(const char* name) const {
        CUfunction result{};
        if (library) { CUkernel kernel{}; TEST_CUDA(cuLibraryGetKernel(&kernel, library, name)); TEST_CUDA(cuKernelGetFunction(&result, kernel)); }
        else TEST_CUDA(cuModuleGetFunction(&result, module, name));
        return result;
    }
};
void access_execution_probe() {
    access_program program(false); device_buffer output(sizeof(uint32_t)); uint32_t actual{};
    TEST_CUDA(cuMemsetD32(output.value, 0, 1)); void* arguments[]{&output.value};
    TEST_CUDA(cuLaunchKernel(program.function("access_probe"), 1, 1, 1, 1, 1, 1, 0, nullptr, arguments, nullptr));
    TEST_CUDA(cuMemcpyDtoH(&actual, output.value, sizeof(actual)));
    require(actual == 0x19375928, "independent native protection execution probe");
}
void access_fault() {
    require_virtual_memory();
    const auto properties = virtual_memory_properties(); const auto bytes = virtual_memory_granularity(properties);
    // Disposable process ownership: no SDK destructors run after the fault.
    auto* physical = new virtual_physical_memory(bytes, properties);
    auto* reserved = new virtual_memory_reservation(bytes);
    auto* mapping = new virtual_memory_mapping(reserved->value, bytes, physical->value); mapping->grant(properties.location);
    auto expected = virtual_memory_pattern(bytes, 39); virtual_memory_write(mapping->value, expected);
    auto* program = new access_program(access_case.ends_with("library"));
    CUstream stream{}; TEST_CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CUdeviceptr output{}; TEST_CUDA(cuMemAlloc(&output, 8 * sizeof(uint32_t)));
    std::array<uint32_t, 8> guarded; guarded.fill(0xa1b2c3d4);
    TEST_CUDA(cuMemcpyHtoD(output, guarded.data(), sizeof(guarded)));
    auto pointer = mapping->value + 64, result = output + 4 * sizeof(uint32_t); unsigned mode = 0;
    void* arguments[]{&pointer, &result, &mode}; const auto function = program->function("access_touch");
    for (mode = 0; mode < 2; ++mode) {
        TEST_CUDA(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, stream, arguments, nullptr));
        TEST_CUDA(cuStreamSynchronize(stream));
        if (mode) { const uint32_t value = 0x3719a5c9; std::memcpy(expected.data() + 64, &value, sizeof(value)); }
        std::memcpy(&guarded[4], expected.data() + 64, sizeof(uint32_t));
        std::array<uint32_t, 8> actual{}; TEST_CUDA(cuMemcpyDtoH(actual.data(), output, sizeof(actual)));
        require(actual == guarded, "guarded legal read/write controls precede prohibited access");
        virtual_memory_expect(mapping->value, expected, "complete legal control payload and guards");
    }
    mapping->grant(properties.location, access_case.starts_with("readonly") ? CU_MEM_ACCESS_FLAGS_PROT_READ : CU_MEM_ACCESS_FLAGS_PROT_NONE);
    mode = access_case.starts_with("none_read") ? 0 : 1;
    TEST_CUDA(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, stream, arguments, nullptr));
    const auto status = cuStreamSynchronize(stream);
    require(status == CUDA_ERROR_ILLEGAL_ADDRESS, "protected access reports illegal address, got " + std::to_string(status));
    require(cudaDeviceSynchronize() == cudaErrorIllegalAddress, "Runtime translates persistent protected-access fault");
}
void access_discovery() {
    for (const char* name : {"cuMemGetAccess", "cuMemSetAccess"}) {
        void* address{}; CUdriverProcAddressQueryResult status{};
        TEST_CUDA(cuGetProcAddress(name, &address, 10020, 0, &status));
        require(address && status == CU_GET_PROC_ADDRESS_SUCCESS, "access procedure introduction version");
        TEST_CUDA(cuGetProcAddress(name, &address, 10019, 0, &status));
        require(!address && status == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "access procedure rejects earlier ABI");
    }
    void* address{}; TEST_CUDA(cuGetProcAddress("cuMemSetAccess", &address, 10020, 0, nullptr));
    require(reinterpret_cast<PFN_cuMemSetAccess_v10020>(address)(0, 0, nullptr, 0) == CUDA_ERROR_NOT_INITIALIZED, "discovered access procedure preserves cold behavior");
}
void execute_virtual_memory_access() {
    require_virtual_memory();
    if (access_case == "states") access_states(false);
    else if (access_case == "aliases") access_states(true);
    else if (access_case == "spans") access_spans();
    else if (access_case == "validation") access_validation();
    else if (access_case == "contextless") access_contextless();
    else throw std::runtime_error("unknown virtual_memory_access case");
}
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "discovery") return cudavm::run_host_test(argc, argv, cudavm::access_discovery);
    if (argc == 3) { cudavm::access_case = argv[2]; --argc; }
    if (cudavm::access_case.starts_with("none_") || cudavm::access_case.starts_with("readonly_"))
        cudavm::run_isolated_fault_test(argc, argv, cudavm::access_fault, cudavm::access_execution_probe);
    return cudavm::run_test(argc, argv, cudavm::execute_virtual_memory_access);
}
