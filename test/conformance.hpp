#pragma once
#ifndef CUDAVM_CONFORMANCE_HOST_ONLY
#include <cuda.h>
#endif
#include <array>
#include <cstdint>
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
#endif
}
