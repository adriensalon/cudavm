#pragma once
#include <cstddef>
#include <cstdint>

#include "export.h"
// Module-owned compiler interface. No LLVM types, C++ ownership or exceptions
// cross this versioned ABI. All request storage is borrowed during a call.
namespace cudavm {
enum class compiler_status : int { success, invalid, unsupported, out_of_memory };
enum class compiler_input_kind : uint32_t { ptx = 1, nvvm, cubin, nvltoir, source, cudavm_ir };
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
struct compiler_program;
struct compiler_input_request {
    uint32_t abi_version = 1;
    compiler_input_kind kind = compiler_input_kind::source;
    const char* data = nullptr;
    size_t size = 0;
    const char* name = nullptr;
    // Source inputs use the normalized source options, headers and expressions.
    const compiler_source_request* source = nullptr;
};
struct compiler_pipeline_request {
    uint32_t abi_version = 1;
    uint32_t numerical_helpers = 0;
    uint32_t lower_semantics = 0;
    uint32_t optimization_level = 0; // 0 preserves IR; 1 through 3 select LLVM O1/O2/O3.
};
enum class compiler_output_kind : uint32_t { ptx, spirv, dxil, program_image, bitcode, llvm_ir };
// Values intentionally match cudavmBackend without depending on the Driver.
enum class compiler_environment : uint32_t { cuda, opencl, opengl, vulkan, directx, all };
struct compiler_emit_request {
    uint32_t abi_version = 1;
    compiler_output_kind kind = compiler_output_kind::ptx;
    compiler_environment environment = compiler_environment::cuda;
    const char* architecture = nullptr; // Defaults to the source architecture.
    const char* kernel = nullptr; // Required for raw shader output with multiple kernels.
    uint32_t block[3] = {1, 1, 1};
};
using compiler_output = bool (*)(const char*, size_t, void*) noexcept;
using compiler_compile_image = compiler_status (*)(const compiler_image_request*,
    compiler_output, void*, char*, size_t) noexcept;
}
extern "C" {
// Programs own their IR/context inside the compiler. Import, link and prepare
// publish a new program only on success and never mutate an input program.
// Diagnostics/results are returned on failure when allocation permits.
CUDAVM_CORE_API cudavm::compiler_status cudavm_core_program_import_v1(
    const cudavm::compiler_input_request*, cudavm::compiler_program**, cudavm::compiler_result**) noexcept;
CUDAVM_CORE_API cudavm::compiler_status cudavm_core_program_link_v1(
    size_t, const cudavm::compiler_program* const*, cudavm::compiler_program**, cudavm::compiler_result**) noexcept;
CUDAVM_CORE_API cudavm::compiler_status cudavm_core_program_prepare_v1(
    const cudavm::compiler_program*, const cudavm::compiler_pipeline_request*,
    cudavm::compiler_program**, cudavm::compiler_result**) noexcept;
CUDAVM_CORE_API cudavm::compiler_status cudavm_core_program_emit_v1(
    const cudavm::compiler_program*, const cudavm::compiler_emit_request*, cudavm::compiler_result**) noexcept;
CUDAVM_CORE_API void cudavm_core_program_release_v1(cudavm::compiler_program*) noexcept;
// Size excludes the trailing convenience NUL and is required for binary results.
CUDAVM_CORE_API size_t cudavm_core_result_size_v1(const cudavm::compiler_result*) noexcept;
// File/AOT orchestration and the lazy Driver callback adapter use the same stages.
CUDAVM_CORE_API cudavm::compiler_status cudavm_core_compile_image_v1(
    const cudavm::compiler_image_request*, cudavm::compiler_output, void*, char*, size_t) noexcept;
CUDAVM_CORE_API int cudavm_core_compile_translation_unit_v1(
    const cudavm::compiler_translation_unit_request*, cudavm::compiler_result**) noexcept;
CUDAVM_CORE_API int cudavm_core_describe_resources_v1(const char*, cudavm::compiler_result**) noexcept;
CUDAVM_CORE_API int cudavm_core_link_files_v1(
    const cudavm::compiler_link_request*, cudavm::compiler_result**) noexcept;
// Returned strings belong to the result and remain valid until its release.
CUDAVM_CORE_API const char* cudavm_core_result_output_v1(const cudavm::compiler_result*) noexcept;
CUDAVM_CORE_API const char* cudavm_core_result_diagnostics_v1(const cudavm::compiler_result*) noexcept;
CUDAVM_CORE_API const char* cudavm_core_result_lowered_name_v1(const cudavm::compiler_result*, const char*) noexcept;
CUDAVM_CORE_API void cudavm_core_result_release_v1(cudavm::compiler_result*) noexcept;
}
