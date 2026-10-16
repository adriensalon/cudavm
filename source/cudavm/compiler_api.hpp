#pragma once
#include "compiler.hpp"
#include <new>
#include <stdexcept>
namespace cudavm {
inline std::string compiler_option_text(const char* value) { return value ? value : ""; }
inline void append_compiler_option_strings(size_t count, const char* const* input, std::vector<std::string>& output) {
    if (count && !input) throw std::runtime_error("null compiler option array");
    for (size_t i = 0; i < count; ++i) {
        if (!input[i]) throw std::runtime_error("null compiler option string");
        output.emplace_back(input[i]);
    }
}
template<class Result, class F> int run_compiler_api(Result** output, F function) noexcept {
    if (!output) return 1;
    *output = nullptr;
    try {
        auto result = std::make_unique<Result>();
        compiler_options values;
        int status = 0;
        try { function(values, *result); }
        catch (const std::exception& error) {
            status = 1;
            values.diagnostics += std::string("cudavm compiler: ") + error.what() + '\n';
        }
        catch (...) { status = 1; values.diagnostics += "cudavm compiler: unknown failure\n"; }
        result->diagnostics = std::move(values.diagnostics);
        *output = result.release();
        return status;
    } catch (...) { return 1; }
}
}
