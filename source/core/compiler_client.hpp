#pragma once
#include "compiler_service.hpp"
#include <memory>
#include <stdexcept>
#include <string>
namespace cudavm {
struct compiler_program_deleter {
    void operator()(compiler_program* value) const noexcept { cudavm_core_program_release_v1(value); }
};
using compiler_program_ptr = std::unique_ptr<compiler_program, compiler_program_deleter>;
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
