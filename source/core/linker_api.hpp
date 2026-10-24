#pragma once
#include "result_api.hpp"

// File/archive device linking, used by nvJitLink build extensions and nvcc. Request storage is borrowed.
namespace cudavm {
struct compiler_link_request {
    uint32_t abi_version;
    size_t input_count;
    const char* const* inputs;
    size_t library_input_count;
    const char* const* library_inputs;
    size_t host_input_count;
    const char* const* host_inputs;
    const char *portable, *ir, *manifest, *binary, *target, *module, *architecture;
};
}
extern "C" {
CUDAVM_CORE_API int cudavm_core_link_files_v1(
    const cudavm::compiler_link_request*, cudavm::compiler_result**) noexcept;
}
