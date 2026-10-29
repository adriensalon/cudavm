#include "conformance.hpp"
#include <algorithm>
#include <iostream>
#include <utility>

namespace cudavm {
struct primary_ownership {
    CUdevice device{};
    CUcontext original{}, value{};
    unsigned retains = 0;
    primary_ownership() {
        CONFORMANCE_CUDA(cuCtxGetCurrent(&original));
        CONFORMANCE_CUDA(cuCtxGetDevice(&device));
        expect_state(0, false);
    }
    ~primary_ownership() {
        cuCtxSetCurrent(original);
        while (retains) { cuDevicePrimaryCtxRelease(device); --retains; }
    }
    void expect_binding() const {
        CUcontext current{};
        CONFORMANCE_CUDA(cuCtxGetCurrent(&current));
        require(current == original, "primary retain/release/query preserves explicit current context");
    }
    void expect_state(unsigned expected_flags, bool expected_active) const {
        unsigned flags = ~0u;
        int active = -1;
        CONFORMANCE_CUDA(cuDevicePrimaryCtxGetState(device, &flags, &active));
        require(flags == expected_flags && active == int(expected_active), "primary configurable flags and active state");
        expect_binding();
    }
    void retain() {
        CUcontext result{};
        CONFORMANCE_CUDA(cuDevicePrimaryCtxRetain(&result, device));
        ++retains;
        require(result != nullptr && result != original, "retained primary is distinct from explicit context");
        if (value && retains > 1) require(result == value, "simultaneous retains share one primary context");
        value = result;
        expect_binding();
    }
    void release() {
        require(retains != 0, "fixture owns a retain to release");
        CONFORMANCE_CUDA(cuDevicePrimaryCtxRelease(device)); --retains;
        expect_binding();
    }
    void bind() const {
        CONFORMANCE_CUDA(cuCtxPushCurrent(value));
        CUcontext current{};
        CONFORMANCE_CUDA(cuCtxGetCurrent(&current));
        require(current == value, "primary context binds explicitly");
    }
    void unbind() const {
        CUcontext popped{};
        CONFORMANCE_CUDA(cuCtxPopCurrent(&popped));
        require(popped == value, "pop returns bound primary context");
        expect_binding();
    }
};
void primary_context_retains() {
    primary_ownership primary;
    for (unsigned generation = 0; generation < 3; ++generation) {
        for (unsigned reference = 0; reference < 4; ++reference) {
            primary.retain(); primary.expect_state(0, true);
        }
        for (unsigned reference = 0; reference < 4; ++reference) {
            primary.release(); primary.expect_state(0, reference != 3);
        }
    }
}
void primary_expect_bound_flags(const primary_ownership& primary, unsigned expected) {
    primary.bind();
    unsigned flags = ~0u;
    CUdevice device{};
    CONFORMANCE_CUDA(cuCtxGetFlags(&flags));
    CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    require((flags & CU_CTX_SCHED_MASK) == expected && device == primary.device, "bound primary scheduling flags and device");
    primary.unbind();
}
void primary_context_flags() {
    primary_ownership primary;
    const std::array<unsigned, 4> modes{CU_CTX_SCHED_AUTO, CU_CTX_SCHED_SPIN, CU_CTX_SCHED_YIELD, CU_CTX_SCHED_BLOCKING_SYNC};
    for (size_t i = 0; i < modes.size(); ++i) {
        CONFORMANCE_CUDA(cuDevicePrimaryCtxSetFlags(primary.device, modes[i]));
        primary.expect_state(modes[i], false);
        primary.retain(); primary.expect_state(modes[i], true);
        primary_expect_bound_flags(primary, modes[i]);
        const auto changed = modes[(i + 1) % modes.size()];
        CONFORMANCE_CUDA(cuDevicePrimaryCtxSetFlags(primary.device, changed));
        primary.expect_state(changed, true);
        primary_expect_bound_flags(primary, changed);
        primary.release(); primary.expect_state(0, false);
        primary.retain(); primary.expect_state(0, true);
        primary_expect_bound_flags(primary, CU_CTX_SCHED_AUTO);
        primary.release(); primary.expect_state(0, false);
    }
}
void primary_expect_bytes(CUdeviceptr pointer, std::span<const unsigned char> expected, const std::string& operation) {
    std::vector<unsigned char> actual(expected.size() + 32, 0xd3);
    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data() + 16, pointer, expected.size()));
    require(std::equal(expected.begin(), expected.end(), actual.begin() + 16), operation + ": every payload byte");
    for (size_t i = 0; i < actual.size(); ++i)
        if (i < 16 || i >= 16 + expected.size()) require(actual[i] == 0xd3, operation + ": host readback guard");
}
void primary_context_storage() {
    device_buffer explicit_storage(257);
    std::vector<unsigned char> explicit_expected(257), primary_expected(523);
    for (size_t i = 0; i < explicit_expected.size(); ++i) explicit_expected[i] = static_cast<unsigned char>(i * 73 + 19);
    for (size_t i = 0; i < primary_expected.size(); ++i) primary_expected[i] = static_cast<unsigned char>((i * 41 + 31) ^ (i >> 4));
    CONFORMANCE_CUDA(cuMemcpyHtoD(explicit_storage.value, explicit_expected.data(), explicit_expected.size()));
    primary_ownership primary;
    primary.retain(); primary.retain();
    primary.bind();
    CUdeviceptr primary_storage{};
    CONFORMANCE_CUDA(cuMemAlloc(&primary_storage, primary_expected.size()));
    CONFORMANCE_CUDA(cuMemcpyHtoD(primary_storage, primary_expected.data(), primary_expected.size()));
    primary.unbind();
    primary.release(); primary.expect_state(0, true);
    primary_expect_bytes(explicit_storage.value, explicit_expected, "explicit storage while primary remains retained");
    primary.bind();
    primary_expect_bytes(primary_storage, primary_expected, "primary storage after intermediate release");
    CONFORMANCE_CUDA(cuMemFree(primary_storage));
    primary.unbind();
    primary.release(); primary.expect_state(0, false);
    primary_expect_bytes(explicit_storage.value, explicit_expected, "explicit storage survives primary retirement");
}
void primary_context_validation() {
    primary_ownership primary;
    require(cuDevicePrimaryCtxRelease(primary.device) == CUDA_ERROR_INVALID_CONTEXT, "cannot release an unretained primary");
    require(cuDevicePrimaryCtxRetain(nullptr, primary.device) == CUDA_ERROR_INVALID_VALUE, "null primary retain output");
    unsigned flags = 0; int active = 0;
    require(cuDevicePrimaryCtxGetState(primary.device, nullptr, &active) == CUDA_ERROR_INVALID_VALUE, "null primary flags output");
    require(cuDevicePrimaryCtxGetState(primary.device, &flags, nullptr) == CUDA_ERROR_INVALID_VALUE, "null primary active output");
    CUcontext invalid{};
    require(cuDevicePrimaryCtxRetain(&invalid, -1) == CUDA_ERROR_INVALID_DEVICE, "invalid retain device");
    require(cuDevicePrimaryCtxGetState(-1, &flags, &active) == CUDA_ERROR_INVALID_DEVICE, "invalid state device");
    require(cuDevicePrimaryCtxRelease(-1) == CUDA_ERROR_INVALID_DEVICE, "invalid release device");
    require(cuDevicePrimaryCtxSetFlags(-1, 0) == CUDA_ERROR_INVALID_DEVICE, "invalid primary flags device");
    primary.expect_state(0, false);
    primary.retain();
    for (unsigned invalid_flags : {3u, 0x80000000u}) {
        require(cuDevicePrimaryCtxSetFlags(primary.device, invalid_flags) == CUDA_ERROR_INVALID_VALUE, "invalid primary scheduling/flag bits");
        primary.expect_state(0, true);
    }
    require(cuCtxDestroy(primary.value) == CUDA_ERROR_INVALID_CONTEXT, "primary context ownership requires release");
    primary.expect_state(0, true);
    primary_expect_bound_flags(primary, CU_CTX_SCHED_AUTO);
    primary.release(); primary.expect_state(0, false);
}
std::string_view primary_context_case = "all";
void execute_primary_context() {
    const std::array<std::pair<std::string_view, void (*)()>, 4> cases{{
        {"retains", primary_context_retains}, {"flags", primary_context_flags},
        {"storage", primary_context_storage}, {"validation", primary_context_validation}}};
    bool found = false;
    for (const auto& [name, execute] : cases) if (primary_context_case == "all" || primary_context_case == name) {
        found = true; std::cout << "primary_context=" << name << std::endl; execute();
    }
    require(found, "unknown primary_context case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::primary_context_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_primary_context);
}
