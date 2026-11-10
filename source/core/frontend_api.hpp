#pragma once
#include "result_api.hpp"

// AOT CUDA translation units and SDK resource discovery, used by nvcc. Request storage is borrowed.
namespace cudavm {
struct compiler_translation_unit_request {
    uint32_t abi_version;
    const char *input, *host, *bitcode, *manifest, *depfile;
    const char *standard, *architecture, *resources_root;
    size_t include_count;
    const char* const* includes;
    size_t define_count;
    const char* const* defines;
};
struct compiler_translation_unit_request_v2 {
    compiler_translation_unit_request base;
    uint32_t optimization_level;
    const char* dependency_target;
    size_t undefine_count;
    const char* const* undefines;
};
}
extern "C" {
CUDAVM_CORE_API int cudavm_core_compile_translation_unit_v1(
    const cudavm::compiler_translation_unit_request*, cudavm::compiler_result**) noexcept;
CUDAVM_CORE_API int cudavm_core_describe_resources_v1(const char*, cudavm::compiler_result**) noexcept;
CUDAVM_CORE_API int cudavm_core_compile_translation_unit_v2(
    const cudavm::compiler_translation_unit_request_v2*, cudavm::compiler_result**) noexcept;
CUDAVM_CORE_API int cudavm_core_preprocess_translation_unit_v2(
    const cudavm::compiler_translation_unit_request_v2*, cudavm::compiler_result**) noexcept;
}
