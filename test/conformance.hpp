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

namespace cudavm {
void require(bool condition, const std::string& message);
// Host/compiler contracts share reporting without GPU initialization or Driver
// references. HOST_ONLY registration selects this subset of the common runner.
int run_host_conformance(int argc, char** argv, void (*execute)());

#ifndef CUDAVM_CONFORMANCE_HOST_ONLY
// Only an absent declared device prerequisite is unavailable evidence. API or
// execution failures on an applicable device must still fail the test.
void require_capability(bool available, const std::string& reason);
void check_driver(CUresult result, const char* operation);
#define CONFORMANCE_CUDA(call) ::cudavm::check_driver((call), #call)

struct fixture_data {
    std::string_view source, ptx, nvvm;
    std::span<const unsigned char> cubin;
};

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

// Execution slices reuse one oracle unchanged by every importer and loader.
// Context flags declare prerequisites such as host mapping for the slice itself.
int run_conformance(int argc, char** argv, const fixture_data& fixtures,
                    void (*execute)(const loaded_program&), unsigned load_repetitions = 1,
                    unsigned context_flags = 0);
void expect_words(std::span<const uint32_t> actual, std::span<const uint32_t> expected,
                  const std::string& context);
#endif
}
