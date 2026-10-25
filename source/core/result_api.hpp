#pragma once
#include "export.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

// Shared compiler status, owned result bytes, diagnostics, and name mappings.
// Result storage stays in core until release. Sizes exclude the convenience NUL;
// a program image's own framing NUL is part of its binary output extent.
namespace cudavm {
enum class compiler_status : int { success, invalid, unsupported, out_of_memory, unsupported_ptx_version, architecture_mismatch, input_type_mismatch, nvvm_version_mismatch };
struct compiler_result;
}
extern "C" {
CUDAVM_CORE_API size_t cudavm_core_result_size_v1(const cudavm::compiler_result*) noexcept;
CUDAVM_CORE_API const char* cudavm_core_result_output_v1(const cudavm::compiler_result*) noexcept;
CUDAVM_CORE_API const char* cudavm_core_result_diagnostics_v1(const cudavm::compiler_result*) noexcept;
CUDAVM_CORE_API const char* cudavm_core_result_lowered_name_v1(const cudavm::compiler_result*, const char*) noexcept;
CUDAVM_CORE_API void cudavm_core_result_release_v1(cudavm::compiler_result*) noexcept;
}

// Local ownership and convenience helpers for this boundary.
namespace cudavm {
struct compiler_result_deleter {
    void operator()(compiler_result* value) const noexcept { cudavm_core_result_release_v1(value); }
};
using compiler_result_ptr = std::unique_ptr<compiler_result, compiler_result_deleter>;
inline std::string compiler_client_text(const char* value) { return value ? value : ""; }
}
