#pragma once

#ifdef CUDAVM_PERF_CUDA
#include <cuda.h>
#endif
#ifdef CUDAVM_PERF_NVRTC
#include <nvrtc.h>
#endif
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cudavm {

struct perf_options {
    std::string_view slice, backend, case_name;
    size_t warmup = 3;
    size_t samples = 10;
    size_t iterations = 1;
};

struct perf_fixture_data {
    std::string_view source, ptx, nvvm;
    std::span<const unsigned char> cubin;
};

struct perf_result {
    std::string device;
    std::string device_uuid;
    std::string device_luid, device_pci_address;
    unsigned device_node_mask = 0;
    int device_ordinal = -1;
    std::vector<std::pair<std::string, std::string>> parameters;
    // Host wall time in milliseconds per operation, retaining every sample.
    std::vector<double> milliseconds;
};

struct perf_unavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void perf_require(bool condition, const std::string& message);
void perf_expect_bytes(std::span<const unsigned char> actual, std::span<const unsigned char> expected,
                       std::string_view context);
std::vector<unsigned char> perf_input_bytes(size_t bytes, uint32_t seed);

#ifdef CUDAVM_PERF_CUDA
struct perf_driver_error : std::runtime_error {
    CUresult result;
    perf_driver_error(CUresult status, const std::string& message);
};
void perf_check_driver(CUresult result, const char* operation);
#define PERF_CUDA(call) ::cudavm::perf_check_driver((call), #call)

struct perf_device {
    CUdevice value{};
    int ordinal{};
    std::string name, uuid, luid, pci_address;
    unsigned node_mask{};
    void describe(perf_result& result) const;
};
perf_device select_perf_device(std::string_view backend, void (*probe)() = nullptr);
void require_perf_execution();

struct perf_context {
    CUcontext value{};
    explicit perf_context(CUdevice device);
    ~perf_context();
    perf_context(const perf_context&) = delete;
    perf_context& operator=(const perf_context&) = delete;
    void close();
};

struct perf_primary_context {
    CUdevice device{};
    CUcontext value{};
    explicit perf_primary_context(CUdevice selected);
    ~perf_primary_context();
    perf_primary_context(const perf_primary_context&) = delete;
    perf_primary_context& operator=(const perf_primary_context&) = delete;
    void close();
};

struct perf_device_buffer {
    CUdeviceptr value{};
    explicit perf_device_buffer(size_t bytes);
    ~perf_device_buffer();
    perf_device_buffer(const perf_device_buffer&) = delete;
    perf_device_buffer& operator=(const perf_device_buffer&) = delete;
    void close();
};

struct perf_stream {
    CUstream value{};
    explicit perf_stream(unsigned flags = CU_STREAM_NON_BLOCKING);
    ~perf_stream();
    perf_stream(const perf_stream&) = delete;
    perf_stream& operator=(const perf_stream&) = delete;
    void synchronize() const;
    void close();
};

struct perf_event {
    CUevent value{};
    explicit perf_event(unsigned flags = 0);
    ~perf_event();
    perf_event(const perf_event&) = delete;
    perf_event& operator=(const perf_event&) = delete;
    void record(CUstream stream) const;
    void synchronize() const;
    void close();
};

struct perf_host_buffer {
    void* value{};
    size_t bytes{};
    explicit perf_host_buffer(size_t size);
    ~perf_host_buffer();
    perf_host_buffer(const perf_host_buffer&) = delete;
    perf_host_buffer& operator=(const perf_host_buffer&) = delete;
    unsigned char* data() const;
    void close();
};

struct perf_module {
    CUmodule value{};
    explicit perf_module(const void* image);
    ~perf_module();
    perf_module(const perf_module&) = delete;
    perf_module& operator=(const perf_module&) = delete;
    CUfunction function(const char* name) const;
    void close();
};

struct perf_library {
    CUlibrary value{};
    explicit perf_library(const void* image);
    ~perf_library();
    perf_library(const perf_library&) = delete;
    perf_library& operator=(const perf_library&) = delete;
    CUfunction function(const char* name) const;
    void close();
};
#endif

#ifdef CUDAVM_PERF_NVRTC
void perf_check_nvrtc(nvrtcResult result, const char* operation);
struct perf_source_program {
    nvrtcProgram value{};
    perf_source_program(const char* source, const char* name);
    ~perf_source_program();
    perf_source_program(const perf_source_program&) = delete;
    perf_source_program& operator=(const perf_source_program&) = delete;
    void compile(std::span<const char* const> options) const;
    std::vector<char> ptx() const;
    void close();
};
#endif

// The slice owns setup and correctness checks outside this call. Async work
// requires a completion callback: it runs before and after every batch, with
// the trailing completion included in the measured interval.
perf_result measure_perf(const perf_options& options, const std::function<void()>& operation,
                         const std::function<void()>& synchronize = {});

// A slice returns only after checking its result. The runner then writes one
// JSON record; exceptions produce failure (1) or explicit unavailability (77).
int run_perf(int argc, char** argv, std::string_view slice,
             perf_result (*execute)(const perf_options&));

}
