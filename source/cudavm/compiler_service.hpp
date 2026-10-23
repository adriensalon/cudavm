#pragma once
#include <cstddef>
#include <cstdint>

#include <cudavm/export.h>
// No LLVM types, C++ ownership or exceptions cross the private ABI.
namespace cudavm {
enum class compiler_status : int { success, invalid, unsupported, out_of_memory };
enum class compiler_input_kind : uint32_t { ptx = 1, nvvm, cubin, nvltoir };
struct compiler_image_request {
    uint32_t abi_version;
    compiler_input_kind kind;
    const char* input;
    size_t length;
    unsigned backend;
};
// Request strings and arrays are borrowed for the duration of each call.
struct compiler_source_header { const char* name; const char* contents; };
struct compiler_source_request {
    uint32_t abi_version;
    const char *source, *name, *standard, *architecture;
    uint32_t relocatable;
    size_t header_count;
    const compiler_source_header* headers;
    size_t expression_count;
    const char* const* expressions;
    size_t include_count;
    const char* const* includes;
    size_t define_count;
    const char* const* defines;
    size_t undefine_count;
    const char* const* undefines;
};
struct compiler_translation_unit_request {
    uint32_t abi_version;
    const char *input, *host, *bitcode, *manifest, *depfile;
    const char *standard, *architecture, *resources_root;
    size_t include_count;
    const char* const* includes;
    size_t define_count;
    const char* const* defines;
};
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
struct compiler_result;
using compiler_output = bool (*)(const char*, size_t, void*) noexcept;
using compiler_compile_image = compiler_status (*)(const compiler_image_request*,
    compiler_output, void*, char*, size_t) noexcept;
}
extern "C" {
CUDAVM_COMPILER_API cudavm::compiler_status cudavm_compiler_compile_image_v1(
    const cudavm::compiler_image_request*, cudavm::compiler_output, void*, char*, size_t) noexcept;
CUDAVM_COMPILER_API cudavm::compiler_status cudavm_compiler_compile_source_v1(
    const cudavm::compiler_source_request*, cudavm::compiler_result**) noexcept;
CUDAVM_COMPILER_API int cudavm_compiler_compile_translation_unit_v1(
    const cudavm::compiler_translation_unit_request*, cudavm::compiler_result**) noexcept;
CUDAVM_COMPILER_API int cudavm_compiler_compile_get_resources_v1(const char*, cudavm::compiler_result**) noexcept;
CUDAVM_COMPILER_API int cudavm_compiler_link_device_v1(
    const cudavm::compiler_link_request*, cudavm::compiler_result**) noexcept;
// Returned strings belong to the result and remain valid until its release.
CUDAVM_COMPILER_API const char* cudavm_compiler_result_output_v1(const cudavm::compiler_result*) noexcept;
CUDAVM_COMPILER_API const char* cudavm_compiler_result_diagnostics_v1(const cudavm::compiler_result*) noexcept;
CUDAVM_COMPILER_API const char* cudavm_compiler_result_lowered_name_v1(const cudavm::compiler_result*, const char*) noexcept;
CUDAVM_COMPILER_API void cudavm_compiler_result_release_v1(cudavm::compiler_result*) noexcept;
}
