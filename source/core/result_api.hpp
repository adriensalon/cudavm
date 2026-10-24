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
enum class compiler_status : int { success, invalid, unsupported, out_of_memory };
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
inline void validate_compiler_strings(size_t count, const char* const* values) {
    if (count && !values) throw std::runtime_error("null compiler option array");
    for (size_t i = 0; i < count; ++i)
        if (!values[i]) throw std::runtime_error("null compiler option string");
}
template<class Result> int copy_compiler_result(Result& output, int status, compiler_result* result) {
    compiler_result_ptr owned(result);
    output.output = cudavm_core_result_output_v1(result);
    output.diagnostics = cudavm_core_result_diagnostics_v1(result);
    return result ? status : 1;
}
template<class Result, class Function> int run_compiler_facade(Result** output, Function function) noexcept {
    if (!output) return 1;
    *output = nullptr;
    try {
        auto result = std::make_unique<Result>();
        int status = 1;
        try { status = function(*result); }
        catch (const std::exception& error) { result->diagnostics += std::string("cudavm compiler: ") + error.what() + '\n'; }
        catch (...) { result->diagnostics += "cudavm compiler: unknown failure\n"; }
        *output = result.release();
        return status;
    } catch (...) { return 1; }
}
}
