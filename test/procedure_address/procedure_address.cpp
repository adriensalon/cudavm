#include "conformance.hpp"
#include <cudaTypedefs.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <exception>
#include <thread>
#include <utility>

namespace cudavm {
template<class T> struct procedure_output {
    static_assert(alignof(T) <= 16);
    std::array<unsigned char, 16> before;
    T value{};
    std::array<unsigned char, 16> after;
    procedure_output() { before.fill(0xe3); after.fill(0x59); }
    T get() const {
        require(std::all_of(before.begin(), before.end(), [](auto byte) { return byte == 0xe3; })
            && std::all_of(after.begin(), after.end(), [](auto byte) { return byte == 0x59; }), "procedure output has exact byte guards");
        return value;
    }
};
void procedure_runtime(cudaError_t result) {
    require(result == cudaSuccess, std::string("Runtime procedure lookup: ") + cudaGetErrorString(result));
}
std::pair<void*, CUdriverProcAddressQueryResult> procedure_query(const char* name, int version, cuuint64_t flags, bool runtime = false, bool unversioned = false) {
    procedure_output<void*> address;
    address.value = reinterpret_cast<void*>(uintptr_t{1});
    CUdriverProcAddressQueryResult status{};
    if (runtime) {
        procedure_output<cudaDriverEntryPointQueryResult> result;
        result.value = static_cast<cudaDriverEntryPointQueryResult>(12345);
        if (unversioned) procedure_runtime(cudaGetDriverEntryPoint(name, &address.value, flags, &result.value));
        else procedure_runtime(cudaGetDriverEntryPointByVersion(name, &address.value, static_cast<unsigned>(version), flags, &result.value));
        switch (result.get()) {
        case cudaDriverEntryPointSuccess: status = CU_GET_PROC_ADDRESS_SUCCESS; break;
        case cudaDriverEntryPointSymbolNotFound: status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND; break;
        case cudaDriverEntryPointVersionNotSufficent: status = CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT; break;
        default: require(false, "Runtime lookup writes a defined query status");
        }
    } else {
        procedure_output<CUdriverProcAddressQueryResult> result;
        result.value = static_cast<CUdriverProcAddressQueryResult>(12345);
        CONFORMANCE_CUDA(cuGetProcAddress(name, &address.value, version, flags, &result.value));
        status = result.get();
    }
    return {address.get(), status};
}
template<class Signature> Signature procedure_resolve(const char* name, int version, cuuint64_t flags = CU_GET_PROC_ADDRESS_LEGACY_STREAM, bool runtime = false) {
    const auto [address, status] = procedure_query(name, version, flags, runtime);
    require(address && status == CU_GET_PROC_ADDRESS_SUCCESS, std::string("resolved public procedure: ") + name);
    return reinterpret_cast<Signature>(address);
}
void procedure_uninitialized() {
    int count = -1;
    require(cuDeviceGetCount(&count) == CUDA_ERROR_NOT_INITIALIZED, "procedure lookup does not initialize Driver");
}
void procedure_discovery() {
    for (bool runtime : {false, true}) {
        const auto [address, status] = procedure_query("cuMemcpyPeer", 6999, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, runtime);
        require(!address && status == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "per-thread introduction does not fall back to legacy semantics");
    }
    procedure_uninitialized();
    for (bool runtime : {false, true}) for (cuuint64_t flags : {cuuint64_t{0}, cuuint64_t{1}, cuuint64_t{2}}) {
        const auto version = procedure_resolve<PFN_cuDriverGetVersion_v2020>("cuDriverGetVersion", 2020, flags, runtime);
        procedure_output<int> value; CONFORMANCE_CUDA(version(&value.value));
        require(value.get() == CUDA_VERSION, "returned version function is callable before initialization");
        const auto initialize = procedure_resolve<PFN_cuInit_v2000>("cuInit", 2000, flags, runtime);
        require(initialize(1) == CUDA_ERROR_INVALID_VALUE, "returned initialization function validates flags without initializing");
        for (const auto* name : {"missing_conformance_procedure", "cudaMalloc", "cuMemAlloc_v2", ""}) {
            const auto [address, status] = procedure_query(name, CUDA_VERSION, flags, runtime);
            require(address == nullptr && status == CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND, "unknown/base-name mismatch has null pointer and symbol-not-found status");
        }
        for (int version_requested : {0, 1999}) {
            const auto [address, status] = procedure_query("cuInit", version_requested, flags, runtime);
            require(address == nullptr && status == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "known symbol requested before introduction reports version-not-sufficient");
        }
    }
    for (bool runtime : {false, true}) {
        const auto old_query = procedure_resolve<PFN_cuGetProcAddress_v11030>("cuGetProcAddress", 11030, 0, runtime);
        procedure_output<void*> address;
        CONFORMANCE_CUDA(old_query("cuInit", &address.value, 2000, 0));
        require(address.get() != nullptr, "legacy four-argument discovery returns callable symbol");
        require(old_query("missing_conformance_procedure", &address.value, CUDA_VERSION, 0) == CUDA_ERROR_NOT_FOUND
            && address.get() == nullptr, "legacy discovery reports missing symbol through return status");
        const auto new_query = procedure_resolve<PFN_cuGetProcAddress_v12000>("cuGetProcAddress", 12000, 0, runtime);
        procedure_output<CUdriverProcAddressQueryResult> status;
        CONFORMANCE_CUDA(new_query("cuInit", &address.value, 2000, 0, &status.value));
        require(address.get() != nullptr && status.get() == CU_GET_PROC_ADDRESS_SUCCESS, "five-argument discovery callable through returned pointer");
    }
    const auto [address, status] = procedure_query("cuDriverGetVersion", CUDA_VERSION, 0, true, true);
    require(address && status == CU_GET_PROC_ADDRESS_SUCCESS, "deprecated Runtime lookup uses pinned version");
    procedure_output<int> version;
    CONFORMANCE_CUDA(reinterpret_cast<PFN_cuDriverGetVersion_v2020>(address)(&version.value));
    require(version.get() == CUDA_VERSION, "deprecated Runtime lookup returns correct callable ABI");
    procedure_uninitialized();
}
void procedure_validation() {
    procedure_uninitialized();
    procedure_output<void*> address;
    require(cuGetProcAddress(nullptr, &address.value, 2000, 0, nullptr) == CUDA_ERROR_INVALID_VALUE
        && cuGetProcAddress("cuInit", nullptr, 2000, 0, nullptr) == CUDA_ERROR_INVALID_VALUE, "Driver lookup rejects null required arguments");
    for (int version : {-1, CUDA_VERSION + 10})
        require(cuGetProcAddress("cuInit", &address.value, version, 0, nullptr) == CUDA_ERROR_INVALID_VALUE, "Driver lookup rejects out-of-range version");
    require(cuGetProcAddress("cuInit", &address.value, 2000, cuuint64_t{1} << 63, nullptr) == CUDA_ERROR_INVALID_VALUE, "Driver lookup rejects unknown flags");
    require(cudaGetDriverEntryPointByVersion(nullptr, &address.value, 2000, 0, nullptr) == cudaErrorInvalidValue
        && cudaGetDriverEntryPointByVersion("cuInit", nullptr, 2000, 0, nullptr) == cudaErrorInvalidValue, "Runtime lookup rejects null required arguments");
    require(cudaGetDriverEntryPointByVersion("cuInit", &address.value, CUDA_VERSION + 10, 0, nullptr) == cudaErrorInvalidValue
        && cudaGetDriverEntryPointByVersion("cuInit", &address.value, 2000, 1ull << 63, nullptr) == cudaErrorInvalidValue, "Runtime lookup rejects invalid version and flags");
    CONFORMANCE_CUDA(cuGetProcAddress("cuInit", &address.value, 2000, 0, nullptr));
    require(address.get() != nullptr, "optional Driver query status may be null after rejected calls");
    procedure_runtime(cudaGetDriverEntryPointByVersion("cuInit", &address.value, 2000, 0, nullptr));
    require(address.get() != nullptr, "optional Runtime query status may be null");
    require(cudaGetLastError() == cudaErrorInvalidValue, "clear expected lookup validation error");
    procedure_uninitialized();
}
void procedure_threading() {
    procedure_uninitialized();
    std::array<std::exception_ptr, 4> errors{};
    std::array<std::jthread, 4> workers;
    for (unsigned worker = 0; worker < workers.size(); ++worker) workers[worker] = std::jthread([&, worker] {
        try {
            for (unsigned repeat = 0; repeat < 64; ++repeat) {
                const auto version = procedure_resolve<PFN_cuDriverGetVersion_v2020>("cuDriverGetVersion", 2020,
                    (repeat % 3), (worker & 1) != 0);
                procedure_output<int> value; CONFORMANCE_CUDA(version(&value.value));
                require(value.get() == CUDA_VERSION, "concurrent independent query outputs retain callable ABI");
            }
        } catch (...) { errors[worker] = std::current_exception(); }
    });
    for (auto& worker : workers) worker.join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
    procedure_uninitialized();
}
void procedure_calls() {
    for (bool runtime : {false, true}) for (cuuint64_t flags : {cuuint64_t{1}, cuuint64_t{2}}) {
        auto allocate = procedure_resolve<PFN_cuMemAlloc_v3020>("cuMemAlloc", 3020, flags, runtime);
        auto release = procedure_resolve<PFN_cuMemFree_v3020>("cuMemFree", 3020, flags, !runtime);
        auto upload = procedure_resolve<PFN_cuMemcpyHtoD_v3020>("cuMemcpyHtoD", 7000, flags, runtime);
        auto download = procedure_resolve<PFN_cuMemcpyDtoH_v3020>("cuMemcpyDtoH", 7000, flags, !runtime);
        auto fill = procedure_resolve<PFN_cuMemsetD8_v3020>("cuMemsetD8", 7000, flags, runtime);
        procedure_output<CUdeviceptr> output;
        CONFORMANCE_CUDA(allocate(&output.value, 2051)); const auto pointer = output.get();
        struct retire { CUdeviceptr value; PFN_cuMemFree_v3020 release; ~retire() { if (value) release(value); } } memory{pointer, release};
        require(pointer != 0, "resolved allocator publishes storage");
        std::vector<unsigned char> expected(2051), actual(2051 + 32, 0xc7);
        for (size_t i = 0; i < expected.size(); ++i) expected[i] = static_cast<unsigned char>(i * 29 + (i >> 5) * 73 + 17);
        CONFORMANCE_CUDA(upload(pointer, expected.data(), expected.size()));
        CONFORMANCE_CUDA(fill(pointer + 37, 0x6d, 1991));
        std::fill(expected.begin() + 37, expected.begin() + 2028, 0x6d);
        CONFORMANCE_CUDA(download(actual.data() + 16, pointer, expected.size()));
        for (size_t i = 0; i < actual.size(); ++i)
            require(actual[i] == (i >= 16 && i < expected.size() + 16 ? expected[i - 16] : 0xc7), "resolved cross-surface memory operations share registry and preserve byte guards");
        CONFORMANCE_CUDA(release(pointer)); memory.value = 0;
    }
}
void procedure_versions() {
    CUdevice device{}; CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    struct contexts {
        CUcontext parent{}; std::array<CUcontext, 3> values{};
        contexts() { CONFORMANCE_CUDA(cuCtxGetCurrent(&parent)); }
        ~contexts() { for (auto i = values.rbegin(); i != values.rend(); ++i) if (*i) cuCtxDestroy(*i); cuCtxSetCurrent(parent); }
    } owners;
    CONFORMANCE_CUDA(procedure_resolve<PFN_cuCtxCreate_v3020>("cuCtxCreate", 11030)(&owners.values[0], 0, device));
    CONFORMANCE_CUDA(procedure_resolve<PFN_cuCtxCreate_v11040>("cuCtxCreate", 12040, 0, true)(&owners.values[1], nullptr, 0, 0, device));
    CONFORMANCE_CUDA(procedure_resolve<PFN_cuCtxCreate_v12050>("cuCtxCreate", 12050)(&owners.values[2], nullptr, 0, device));
    procedure_output<CUdevice> result;
    CONFORMANCE_CUDA(procedure_resolve<PFN_cuCtxGetDevice_v2000>("cuCtxGetDevice", 12990)(&result.value));
    require(result.get() == device, "pre-13 context-device ABI uses current context");
    for (const auto context : owners.values) {
        CONFORMANCE_CUDA(procedure_resolve<PFN_cuCtxGetDevice_v13000>("cuCtxGetDevice", 13000, 0, true)(&result.value, context));
        require(result.get() == device, "13.0 context-device ABI accepts explicit context");
        CONFORMANCE_CUDA(procedure_resolve<PFN_cuCtxSynchronize_v13000>("cuCtxSynchronize", 13000)(context));
        CUcontext current{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&current));
        require(current == owners.values.back(), "explicit-context discovered calls preserve current binding");
    }
}
void procedure_streams() {
    const auto legacy = procedure_resolve<PFN_cuStreamGetId_v12000>("cuStreamGetId", 12000, 1);
    const auto per_thread = procedure_resolve<PFN_cuStreamGetId_v12000_ptsz>("cuStreamGetId", 12000, 2, true);
    procedure_output<unsigned long long> first, second, explicit_legacy, selected;
    CONFORMANCE_CUDA(legacy(nullptr, &first.value)); CONFORMANCE_CUDA(per_thread(nullptr, &second.value));
    CONFORMANCE_CUDA(per_thread(CU_STREAM_LEGACY, &explicit_legacy.value));
    require(first.get() != second.get() && first.get() == explicit_legacy.get(), "lookup flags select distinct legacy/PTDS defaults and preserve explicit legacy alias");
    for (bool runtime : {false, true}) {
        CONFORMANCE_CUDA(procedure_resolve<PFN_cuStreamGetId_v12000>("cuStreamGetId", 12000, 0, runtime)(nullptr, &selected.value));
#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
        require(selected.get() == second.get(), "zero flags select caller PTDS mode");
#else
        require(selected.get() == first.get(), "zero flags select caller legacy mode");
#endif
    }
    CUstream stream{}; CONFORMANCE_CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    struct retire { CUstream value; ~retire() { cuStreamDestroy(value); } } owner{stream};
    CONFORMANCE_CUDA(legacy(stream, &selected.value)); CONFORMANCE_CUDA(per_thread(stream, &explicit_legacy.value));
    require(selected.get() == explicit_legacy.get() && selected.get() != first.get() && selected.get() != second.get(), "explicit stream identity survives either lookup mode");
}
std::string_view procedure_address_case = "all";
void execute_procedure_host() {
    if (procedure_address_case == "discovery") procedure_discovery();
    else if (procedure_address_case == "validation") procedure_validation();
    else if (procedure_address_case == "threading") procedure_threading();
    else require(false, "unknown host procedure_address case");
}
void execute_procedure_device() {
    if (procedure_address_case == "calls") procedure_calls();
    else if (procedure_address_case == "versions") procedure_versions();
    else if (procedure_address_case == "streams") procedure_streams();
    else require(false, "unknown device procedure_address case");
}
}
int main(int argc, char** argv) {
    if (argc == 2) {
        cudavm::procedure_address_case = argv[1];
        return cudavm::run_host_conformance(argc, argv, cudavm::execute_procedure_host);
    }
    if (argc == 3) { cudavm::procedure_address_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_procedure_device);
}
