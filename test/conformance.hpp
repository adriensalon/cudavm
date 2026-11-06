#pragma once
#ifndef CUDAVM_CONFORMANCE_HOST_ONLY
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#ifdef CUDAVM_CONFORMANCE_REGISTRATION
#include <cuda_runtime_api.h>
// Private compiler ABI declarations belong to the registration fixtures.
extern "C" {
void** CUDARTAPI __cudaRegisterFatBinary(void*);
void CUDARTAPI __cudaRegisterFatBinaryEnd(void**);
void CUDARTAPI __cudaUnregisterFatBinary(void**);
void CUDARTAPI __cudaRegisterFunction(void**, const char*, char*, const char*, int, uint3*, uint3*, dim3*, dim3*, int*);
void CUDARTAPI __cudaRegisterVar(void**, char*, char*, const char*, int, size_t, int, int);
void CUDARTAPI __cudaRegisterManagedVar(void**, void**, char*, const char*, int, size_t, int, int);
char CUDARTAPI __cudaInitModule(void**);
}
#endif

namespace cudavm {
void require(bool condition, const std::string& message);
// Host/compiler contracts share reporting without GPU initialization or Driver
// references. HOST_ONLY registration selects this subset of the common runner.
int run_host_conformance(int argc, char** argv, void (*execute)());

struct conformance_child_result { int status; std::string output; };
// Run the same executable with <backend> <case> --worker and inherited stdout
// capture. The worker has a bounded deadline and propagates skip/failure status.
conformance_child_result run_conformance_child(const char* executable, const char* backend, const char* case_name,
                                               bool capture_stderr = false);

// A fresh copy of this executable receives <backend> <case> --peer <read>
// <write> <partner>. Control bytes and real OS handles use separate operations;
// GPU payloads never pass through this channel. One thread owns each endpoint.
class conformance_peer {
    struct state;
    std::unique_ptr<state> storage;
    explicit conformance_peer(std::unique_ptr<state> value);
public:
    conformance_peer(const char* executable, const char* backend, const char* case_name);
    static conformance_peer connect(std::string_view read, std::string_view write, std::string_view partner);
    ~conformance_peer();
    conformance_peer(conformance_peer&&) noexcept;
    conformance_peer(const conformance_peer&) = delete;
    void send(std::span<const std::byte> bytes);
    void receive(std::span<std::byte> bytes);
    void send_handle(uintptr_t handle);
    uintptr_t receive_handle();
    int wait();
};

struct fixture_data {
    std::string_view source, ptx, nvvm;
    std::span<const unsigned char> cubin;
    unsigned architecture = 75;
    unsigned minimum_ptx = 0;
};

#ifndef CUDAVM_CONFORMANCE_HOST_ONLY
// Only an absent declared device prerequisite is unavailable evidence. API or
// execution failures on an applicable device must still fail the test.
void require_capability(bool available, const std::string& reason);
void check_driver(CUresult result, const char* operation);
#define CONFORMANCE_CUDA(call) ::cudavm::check_driver((call), #call)

#ifdef CUDAVM_CONFORMANCE_REGISTRATION
struct registration_wrapper { uint32_t magic = 0x466243b1, version = 1; const void* image{}; const void* auxiliary{}; };
// Independently encode one uncompressed, 64-bit NVIDIA PTX fatbinary member.
std::vector<unsigned char> registration_ptx_image(std::string_view source, unsigned major, unsigned minor, unsigned architecture);
#endif

class device_buffer {
public:
    CUdeviceptr value{};
    explicit device_buffer(size_t size) { CONFORMANCE_CUDA(cuMemAlloc(&value, size)); }
    ~device_buffer() { if (value) cuMemFree(value); }
    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;
};

class loaded_program {
    CUmodule module{};
    CUlibrary library{};
public:
    loaded_program(const void* image, bool use_library);
    ~loaded_program();
    loaded_program(const loaded_program&) = delete;
    loaded_program& operator=(const loaded_program&) = delete;
    CUfunction function(const char* name) const;
    // Contextless kernel routes apply to library loads; module loads return null.
    CUkernel library_kernel(const char* name) const;
    CUdeviceptr global(const char* name, size_t expected_size) const;
};

// Format-independent Driver API contracts need a device, but no shader compiler
// or executable-input probe. A boundary slice that owns executable input may
// supply an independent probe; slice failures never trigger device fallback.
int run_conformance(int argc, char** argv, void (*execute)(), unsigned context_flags = 0,
                    void (*probe)() = nullptr);
// Fatal-device witnesses own a disposable CTest process. Report and exit without
// synchronizing or destroying a faulted context, even when an assertion fails.
[[noreturn]] void run_isolated_fault_conformance(int argc, char** argv, void (*execute)(),
                                               void (*probe)() = nullptr);

// Execution slices reuse one oracle unchanged by every importer and loader.
// Context flags declare prerequisites such as host mapping for the slice itself.
int run_conformance(int argc, char** argv, const fixture_data& fixtures,
                    void (*execute)(const loaded_program&), unsigned load_repetitions = 1,
                    unsigned context_flags = 0);
void expect_words(std::span<const uint32_t> actual, std::span<const uint32_t> expected,
                  const std::string& context);

inline void virtual_runtime(cudaError_t result) {
    require(result == cudaSuccess, std::string("virtual memory Runtime: ") + cudaGetErrorString(result));
}
inline CUmemAllocationProp virtual_memory_properties() {
    CUmemAllocationProp properties{}; properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    CONFORMANCE_CUDA(cuCtxGetDevice(&properties.location.id));
    return properties;
}
inline size_t virtual_memory_granularity(const CUmemAllocationProp& properties) {
    size_t result{}; CONFORMANCE_CUDA(cuMemGetAllocationGranularity(&result, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    require(result && result <= std::numeric_limits<size_t>::max() / 16, "bounded virtual memory granularity");
    return result;
}
struct virtual_physical_memory {
    CUmemGenericAllocationHandle value{};
    size_t bytes;
    virtual_physical_memory(size_t size, const CUmemAllocationProp& properties) : bytes(size) {
        struct { uint64_t before = 0x19375928abcdef01; CUmemGenericAllocationHandle value{}; uint64_t after = 0xfedcba0129183745; } output;
        CONFORMANCE_CUDA(cuMemCreate(&output.value, bytes, &properties, 0));
        value = output.value;
        require(value && output.before == 0x19375928abcdef01 && output.after == 0xfedcba0129183745, "physical allocation output guards");
    }
    ~virtual_physical_memory() { if (value) cuMemRelease(value); }
    virtual_physical_memory(const virtual_physical_memory&) = delete;
    void close() { CONFORMANCE_CUDA(cuMemRelease(value)); value = 0; }
};
struct virtual_memory_reservation {
    CUdeviceptr value{};
    size_t bytes;
    explicit virtual_memory_reservation(size_t size) : bytes(size) { CONFORMANCE_CUDA(cuMemAddressReserve(&value, bytes, 0, 0, 0)); }
    ~virtual_memory_reservation() { if (value) cuMemAddressFree(value, bytes); }
    virtual_memory_reservation(const virtual_memory_reservation&) = delete;
};
struct virtual_memory_mapping {
    CUdeviceptr value{};
    size_t bytes;
    virtual_memory_mapping(CUdeviceptr pointer, size_t size, CUmemGenericAllocationHandle physical) : value(pointer), bytes(size) {
        CONFORMANCE_CUDA(cuMemMap(value, bytes, 0, physical, 0));
    }
    ~virtual_memory_mapping() { if (value) cuMemUnmap(value, bytes); }
    virtual_memory_mapping(const virtual_memory_mapping&) = delete;
    void grant(CUmemLocation location, CUmemAccess_flags flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE) const {
        CUmemAccessDesc access{location, flags}; CONFORMANCE_CUDA(cuMemSetAccess(value, bytes, &access, 1));
    }
    void close() { CONFORMANCE_CUDA(cuMemUnmap(value, bytes)); value = 0; }
};
inline std::vector<unsigned char> virtual_memory_pattern(size_t bytes, unsigned seed) {
    std::vector<unsigned char> values(bytes);
    for (size_t i = 0; i < bytes; ++i) values[i] = static_cast<unsigned char>((i * 73 + seed * 37) ^ (i >> 7));
    return values;
}
inline void virtual_memory_write(CUdeviceptr pointer, std::span<const unsigned char> values, bool runtime = false) {
    if (runtime) virtual_runtime(cudaMemcpy(reinterpret_cast<void*>(uintptr_t(pointer)), values.data(), values.size(), cudaMemcpyHostToDevice));
    else CONFORMANCE_CUDA(cuMemcpyHtoD(pointer, values.data(), values.size()));
}
inline void virtual_memory_expect(CUdeviceptr pointer, std::span<const unsigned char> expected, const char* label, bool runtime = false) {
    std::vector<unsigned char> values(expected.size() + 32, 0xa7);
    if (runtime) virtual_runtime(cudaMemcpy(values.data() + 16, reinterpret_cast<void*>(uintptr_t(pointer)), expected.size(), cudaMemcpyDeviceToHost));
    else CONFORMANCE_CUDA(cuMemcpyDtoH(values.data() + 16, pointer, expected.size()));
    for (size_t i = 0; i < values.size(); ++i) {
        const auto wanted = i < 16 || i >= expected.size() + 16 ? 0xa7 : expected[i - 16];
        if (values[i] != wanted) require(false, std::string(label) + " byte " + std::to_string(i));
    }
}
inline void require_virtual_memory() {
    CUdevice device{}; int supported{};
    CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));
    require_capability(supported != 0, "virtual memory requires advertised VMM support");
}
#endif
}
