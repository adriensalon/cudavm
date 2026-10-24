#include "../core/program_api.hpp"
#include "../core/compiler_profiles.hpp"
#include <nvptxcompiler/export.h>
#include <nvPTXCompiler.h>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wdll-attribute-on-redeclaration"
#endif

namespace cudavm {
namespace {
struct ptx_compiler_program {
    std::mutex mutex;
    std::string source, binary, error, info;
    bool compiled = false;
    int (*callback)(void*, void*) = nullptr;
    void* payload = nullptr;
};
struct ptx_compiler_registry {
    std::mutex mutex;
    uintptr_t next = 0;
    std::unordered_map<nvPTXCompilerHandle, std::shared_ptr<ptx_compiler_program>> programs;
};
ptx_compiler_registry& ptx_programs() { static ptx_compiler_registry value; return value; }
struct ptx_compiler_error { nvPTXCompileResult result; };
void ptx_require(bool condition, nvPTXCompileResult result) { if (!condition) throw ptx_compiler_error{result}; }
template<class Function> nvPTXCompileResult ptx_api(Function function) noexcept {
    try { return function(); }
    catch (const ptx_compiler_error& error) { return error.result; }
    catch (const std::bad_alloc&) { return NVPTXCOMPILE_ERROR_OUT_OF_MEMORY; }
    catch (...) { return NVPTXCOMPILE_ERROR_INTERNAL; }
}
template<class Function> nvPTXCompileResult ptx_with_program(nvPTXCompilerHandle handle, Function function) noexcept {
    return ptx_api([&] {
        std::shared_ptr<ptx_compiler_program> program;
        { auto& registry = ptx_programs(); std::lock_guard lock(registry.mutex);
          auto found = registry.programs.find(handle);
          ptx_require(found != registry.programs.end(), NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE); program = found->second; }
        std::lock_guard lock(program->mutex);
        return function(*program);
    });
}
struct ptx_compiler_options {
    std::string architecture;
    unsigned optimization = 3;
    bool parse_only = false, relocatable = false, verbose = false;
};
ptx_compiler_options ptx_options(int count, const char* const* options, std::string& error) {
    ptx_compiler_options result;
    auto invalid = [&](const std::string& option) {
        error = "cudavm PTX Compiler: unsupported or invalid option: " + option + '\n';
        throw ptx_compiler_error{NVPTXCOMPILE_ERROR_COMPILATION_FAILURE};
    };
    for (int i = 0; i < count; ++i) {
        std::string option(options[i]);
        auto value = [&](const char* long_name, const char* short_name, bool joined = false) -> std::optional<std::string> {
            for (auto* name : {long_name, short_name}) {
                std::string prefix(name);
                if (option == prefix) { if (++i == count) invalid(option); return options[i]; }
                if (option.starts_with(prefix + '=')) return option.substr(prefix.size() + 1);
                if (joined && prefix == short_name && option.starts_with(prefix) && option.size() > prefix.size()) return option.substr(prefix.size());
            }
            return std::nullopt;
        };
        if (auto architecture = value("--gpu-name", "-arch")) {
            bool physical = architecture->starts_with("sm_");
            bool virtual_target = architecture->starts_with("compute_");
            if (!physical && !virtual_target) invalid(option);
            auto number = architecture->substr(physical ? 3 : 8);
            bool supported = false;
            for (auto target : ptx_compile_architectures) supported |= number == std::to_string(target);
            if (!supported) invalid(option);
            result.architecture = "sm_" + number; result.parse_only = virtual_target;
        } else if (auto optimization = value("--opt-level", "-O", true)) {
            if (optimization->size() != 1 || (*optimization)[0] < '0' || (*optimization)[0] > '3') invalid(option);
            result.optimization = unsigned((*optimization)[0] - '0');
        } else if (option == "--compile-only" || option == "-c") result.relocatable = true;
        else if (option == "--verbose" || option == "-v") result.verbose = true;
        else invalid(option);
    }
    if (result.architecture.empty()) invalid("--gpu-name is required");
    return result;
}
void ptx_status(compiler_status status, const compiler_result* result, ptx_compiler_program& program) {
    if (status == compiler_status::success) return;
    program.error += cudavm_core_result_diagnostics_v1(result);
    if (status == compiler_status::out_of_memory) throw std::bad_alloc();
    throw ptx_compiler_error{status == compiler_status::unsupported_ptx_version
        ? NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION : NVPTXCOMPILE_ERROR_COMPILATION_FAILURE};
}
void ptx_checkpoint(ptx_compiler_program& program) {
    if (program.callback && program.callback(program.payload, nullptr) == 1) {
        program.error = "cudavm PTX Compiler: compilation cancelled\n";
        throw ptx_compiler_error{NVPTXCOMPILE_ERROR_CANCELLED};
    }
}
}
}
using namespace cudavm;
extern "C" CUDAVM_NVPTXCOMPILER_API nvPTXCompileResult nvPTXCompilerGetVersion(unsigned* major, unsigned* minor) {
    return ptx_api([&] { ptx_require(major && minor, NVPTXCOMPILE_ERROR_INVALID_INPUT); *major = 13; *minor = 4; return NVPTXCOMPILE_SUCCESS; });
}
extern "C" CUDAVM_NVPTXCOMPILER_API nvPTXCompileResult nvPTXCompilerGetNumSupportedArchs(int* count) {
    return ptx_api([&] { ptx_require(count, NVPTXCOMPILE_ERROR_INVALID_INPUT); *count = int(ptx_compile_architectures.size()); return NVPTXCOMPILE_SUCCESS; });
}
extern "C" CUDAVM_NVPTXCOMPILER_API nvPTXCompileResult nvPTXCompilerGetSupportedArchs(int* architectures) {
    return ptx_api([&] { ptx_require(architectures, NVPTXCOMPILE_ERROR_INVALID_INPUT); std::copy(ptx_compile_architectures.begin(), ptx_compile_architectures.end(), architectures); return NVPTXCOMPILE_SUCCESS; });
}
extern "C" CUDAVM_NVPTXCOMPILER_API nvPTXCompileResult nvPTXCompilerGetNumSupportedVersions(int* count) {
    return ptx_api([&] { ptx_require(count, NVPTXCOMPILE_ERROR_INVALID_INPUT); *count = int(ptx_language_versions.size()); return NVPTXCOMPILE_SUCCESS; });
}
extern "C" CUDAVM_NVPTXCOMPILER_API nvPTXCompileResult nvPTXCompilerGetSupportedVersions(int (*versions)[2]) {
    return ptx_api([&] { ptx_require(versions, NVPTXCOMPILE_ERROR_INVALID_INPUT);
        for (size_t i = 0; i < ptx_language_versions.size(); ++i) std::copy(ptx_language_versions[i].begin(), ptx_language_versions[i].end(), versions[i]);
        return NVPTXCOMPILE_SUCCESS; });
}
extern "C" CUDAVM_NVPTXCOMPILER_API nvPTXCompileResult nvPTXCompilerCreate(nvPTXCompilerHandle* output, size_t length, const char* source) {
    return ptx_api([&] {
        ptx_require(output && source && length, NVPTXCOMPILE_ERROR_INVALID_INPUT);
        auto program = std::make_shared<ptx_compiler_program>();
        if (length && source[length - 1] == 0) --length;
        ptx_require(length != 0, NVPTXCOMPILE_ERROR_INVALID_INPUT);
        if (length) program->source.assign(source, length);
        auto& registry = ptx_programs(); std::lock_guard lock(registry.mutex);
        ptx_require(registry.next != UINTPTR_MAX, NVPTXCOMPILE_ERROR_OUT_OF_MEMORY);
        auto handle = reinterpret_cast<nvPTXCompilerHandle>(++registry.next);
        registry.programs.emplace(handle, std::move(program)); *output = handle; return NVPTXCOMPILE_SUCCESS;
    });
}
extern "C" CUDAVM_NVPTXCOMPILER_API nvPTXCompileResult nvPTXCompilerDestroy(nvPTXCompilerHandle* handle) {
    return ptx_api([&] { ptx_require(handle, NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE);
        auto& registry = ptx_programs(); std::lock_guard lock(registry.mutex);
        ptx_require(registry.programs.erase(*handle) != 0, NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE);
        *handle = nullptr; return NVPTXCOMPILE_SUCCESS; });
}
extern "C" CUDAVM_NVPTXCOMPILER_API nvPTXCompileResult nvPTXCompilerCompile(nvPTXCompilerHandle handle, int count, const char* const* options) {
    return ptx_with_program(handle, [&](ptx_compiler_program& program) {
        ptx_require(count >= 0 && (!count || options), NVPTXCOMPILE_ERROR_INVALID_INPUT);
        for (int i = 0; i < count; ++i) ptx_require(options[i], NVPTXCOMPILE_ERROR_INVALID_INPUT);
        program.compiled = false; program.binary.clear(); program.error.clear(); program.info.clear();
        auto settings = ptx_options(count, options, program.error);
        ptx_checkpoint(program);
        compiler_input_request_v3 request;
        request.base.base = {3, compiler_input_kind::ptx, program.source.data(), program.source.size(), "nvPTXCompiler.ptx", nullptr};
        request.base.architecture = settings.architecture.c_str(); request.relocatable = settings.relocatable;
        compiler_program* imported = nullptr; compiler_result* result = nullptr;
        auto status = cudavm_core_program_import_v1(&request.base.base, &imported, &result);
        compiler_program_ptr input(imported); compiler_result_ptr diagnostic(result);
        ptx_status(status, result, program); ptx_checkpoint(program);
        if (settings.parse_only) { program.compiled = true; return NVPTXCOMPILE_PARSE_ONLY_SUCCESS; }
        compiler_pipeline_request pipeline{1, 1, 0, settings.optimization};
        compiler_program* prepared = nullptr;
        status = cudavm_core_program_prepare_v1(imported, &pipeline, &prepared, &result);
        compiler_program_ptr ready(prepared); compiler_result_ptr prepare_result(result);
        ptx_status(status, result, program); ptx_checkpoint(program);
        compiler_emit_request emission;
        emission.kind = compiler_output_kind::program_image;
        emission.environment = settings.relocatable ? compiler_environment::none : compiler_environment::all;
        status = cudavm_core_program_emit_v1(prepared, &emission, &result);
        compiler_result_ptr output(result); ptx_status(status, result, program); ptx_checkpoint(program);
        std::string binary(cudavm_core_result_output_v1(result), cudavm_core_result_size_v1(result));
        if (settings.verbose) program.info = "cudavm PTX Compiler: target " + settings.architecture + ", optimization "
            + std::to_string(settings.optimization) + ", portable image bytes " + std::to_string(binary.size()) + '\n';
        program.binary = std::move(binary); program.compiled = true; return NVPTXCOMPILE_SUCCESS;
    });
}
extern "C" CUDAVM_NVPTXCOMPILER_API nvPTXCompileResult nvPTXCompilerGetCompiledProgramSize(nvPTXCompilerHandle handle, size_t* size) {
    return ptx_with_program(handle, [&](ptx_compiler_program& program) { ptx_require(size, NVPTXCOMPILE_ERROR_INVALID_INPUT);
        ptx_require(program.compiled, NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE); *size = program.binary.size(); return NVPTXCOMPILE_SUCCESS; });
}
extern "C" CUDAVM_NVPTXCOMPILER_API nvPTXCompileResult nvPTXCompilerGetCompiledProgram(nvPTXCompilerHandle handle, void* output) {
    return ptx_with_program(handle, [&](ptx_compiler_program& program) { ptx_require(output, NVPTXCOMPILE_ERROR_INVALID_INPUT);
        ptx_require(program.compiled, NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE);
        if (!program.binary.empty()) std::memcpy(output, program.binary.data(), program.binary.size()); return NVPTXCOMPILE_SUCCESS; });
}
extern "C" CUDAVM_NVPTXCOMPILER_API nvPTXCompileResult nvPTXCompilerGetErrorLogSize(nvPTXCompilerHandle handle, size_t* size) {
    return ptx_with_program(handle, [&](ptx_compiler_program& program) { ptx_require(size, NVPTXCOMPILE_ERROR_INVALID_INPUT); *size = program.error.empty() ? 0 : program.error.size() + 1; return NVPTXCOMPILE_SUCCESS; });
}
extern "C" CUDAVM_NVPTXCOMPILER_API nvPTXCompileResult nvPTXCompilerGetErrorLog(nvPTXCompilerHandle handle, char* output) {
    return ptx_with_program(handle, [&](ptx_compiler_program& program) { ptx_require(output, NVPTXCOMPILE_ERROR_INVALID_INPUT); std::memcpy(output, program.error.c_str(), program.error.size() + 1); return NVPTXCOMPILE_SUCCESS; });
}
extern "C" CUDAVM_NVPTXCOMPILER_API nvPTXCompileResult nvPTXCompilerGetInfoLogSize(nvPTXCompilerHandle handle, size_t* size) {
    return ptx_with_program(handle, [&](ptx_compiler_program& program) { ptx_require(size, NVPTXCOMPILE_ERROR_INVALID_INPUT); *size = program.info.empty() ? 0 : program.info.size() + 1; return NVPTXCOMPILE_SUCCESS; });
}
extern "C" CUDAVM_NVPTXCOMPILER_API nvPTXCompileResult nvPTXCompilerGetInfoLog(nvPTXCompilerHandle handle, char* output) {
    return ptx_with_program(handle, [&](ptx_compiler_program& program) { ptx_require(output, NVPTXCOMPILE_ERROR_INVALID_INPUT); std::memcpy(output, program.info.c_str(), program.info.size() + 1); return NVPTXCOMPILE_SUCCESS; });
}
extern "C" CUDAVM_NVPTXCOMPILER_API nvPTXCompileResult nvPTXCompilerSetFlowCallback(nvPTXCompilerHandle handle, int (*callback)(void*, void*), void* payload) {
    return ptx_with_program(handle, [&](ptx_compiler_program& program) { ptx_require(callback, NVPTXCOMPILE_ERROR_INVALID_INPUT);
        program.callback = callback; program.payload = payload; return NVPTXCOMPILE_SUCCESS; });
}
