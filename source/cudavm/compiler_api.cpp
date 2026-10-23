#include "compiler_api.hpp"
using namespace cudavm;
extern "C" CUDAVM_COMPILER_API const char* cudavm_compiler_result_output_v1(const compiler_result* result) noexcept {
    return result ? result->output.c_str() : "";
}
extern "C" CUDAVM_COMPILER_API const char* cudavm_compiler_result_diagnostics_v1(const compiler_result* result) noexcept {
    return result ? result->diagnostics.c_str() : "compiler result is unavailable\n";
}
extern "C" CUDAVM_COMPILER_API const char* cudavm_compiler_result_lowered_name_v1(
        const compiler_result* result, const char* expression) noexcept {
    if (!result || !expression) return nullptr;
    auto found = result->lowered_names.find(expression);
    return found == result->lowered_names.end() ? nullptr : found->second.c_str();
}
extern "C" CUDAVM_COMPILER_API void cudavm_compiler_result_release_v1(compiler_result* result) noexcept { delete result; }
