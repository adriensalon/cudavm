#pragma once
#include "result_api.hpp"

// In-memory compiler pipeline used by NVRTC and compiler/linker facades.
// Requests borrow their storage. Import, link, and preparation return new
// compiler-owned programs without mutating inputs. No LLVM types cross the ABI.
namespace cudavm {
enum class compiler_input_kind : uint32_t { ptx = 1, nvvm, cubin, nvltoir, source, cudavm_ir, program_image, automatic, fatbinary };
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
// V2 adds a PTX compilation target. The original .target remains the input's
// minimum architecture; the selected target must satisfy that requirement.
struct compiler_input_request_v2 {
    compiler_input_request base;
    const char* architecture = nullptr;
};
// V3 permits unresolved PTX declarations until device linking. For persisted
// programs the requested architecture validates the existing source profile.
struct compiler_input_request_v3 {
    compiler_input_request_v2 base;
    uint32_t relocatable = 0;
};
struct compiler_pipeline_request {
    uint32_t abi_version = 1;
    uint32_t numerical_helpers = 0;
    uint32_t lower_semantics = 0;
    uint32_t optimization_level = 0; // 0 preserves IR; 1 through 3 select LLVM O1/O2/O3.
};
enum class compiler_output_kind : uint32_t { ptx, spirv, dxil, program_image, bitcode, llvm_ir, native_image };
// program_image retains IR and metadata. none omits optional native artifacts;
// native_image retains the existing PTX/descriptor transport for text consumers.
// Values intentionally match cudavmBackend without depending on the Driver.
enum class compiler_environment : uint32_t { cuda, opencl, opengl, vulkan, directx, all, none };
struct compiler_emit_request {
    uint32_t abi_version = 1;
    compiler_output_kind kind = compiler_output_kind::ptx;
    compiler_environment environment = compiler_environment::cuda;
    const char* architecture = nullptr; // Defaults to the source architecture.
    const char* kernel = nullptr; // Required for raw shader output with multiple kernels.
    uint32_t block[3] = {1, 1, 1};
};
}
extern "C" {
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
}

// Local ownership and convenience helpers for this boundary.
namespace cudavm {
struct compiler_program_deleter {
    void operator()(compiler_program* value) const noexcept { cudavm_core_program_release_v1(value); }
};
using compiler_program_ptr = std::unique_ptr<compiler_program, compiler_program_deleter>;
}
