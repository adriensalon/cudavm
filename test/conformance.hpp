#pragma once
#include <cuda.h>
#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cudavm {
void require(bool condition, const std::string& message);
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
};

// Each slice supplies one oracle, reused unchanged by every importer and loader.
int run_conformance(int argc, char** argv, const fixture_data& fixtures,
                    void (*execute)(const loaded_program&));
void expect_words(std::span<const uint32_t> actual, std::span<const uint32_t> expected,
                  const std::string& context);
}
