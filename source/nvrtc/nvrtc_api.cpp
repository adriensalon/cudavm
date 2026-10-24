#include "../core/program_api.hpp"
#include <map>
#include <vector>
#include <nvrtc/export.h>
#include <nvrtc.h>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <optional>
#include <list>
#include <unordered_map>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wdll-attribute-on-redeclaration"
#endif
namespace cudavm {
namespace {
struct nvrtc_options {
    std::string standard = "c++17", architecture = "sm_75";
    bool relocatable = false;
    std::vector<std::string> includes, defines, undefines;
};
struct program {
    std::mutex mutex;
    std::string source, name, ptx, log;
    std::vector<std::pair<std::string, std::string>> headers;
    std::vector<std::string> expressions;
    // NVIDIA keeps returned name pointers valid until program destruction,
    // including across subsequent compilation attempts.
    std::list<std::map<std::string, std::string>> lowered_names;
    bool compilation_attempted = false, compiled = false;
};
struct registry {
    std::mutex mutex;
    uintptr_t next = 0;
    std::unordered_map<nvrtcProgram, std::shared_ptr<program>> programs;
};
registry& programs() { static registry value; return value; }
std::shared_ptr<program> lookup(nvrtcProgram handle) {
    auto& registry = programs();
    std::lock_guard lock(registry.mutex);
    auto found = registry.programs.find(handle);
    return found == registry.programs.end() ? nullptr : found->second;
}
struct api_error { nvrtcResult result; };
void require(bool condition, nvrtcResult error) { if (!condition) throw api_error{error}; }
template<class F> nvrtcResult api(F&& function) noexcept {
    try { function(); return NVRTC_SUCCESS; }
    catch (const api_error& error) { return error.result; }
    catch (const std::bad_alloc&) { return NVRTC_ERROR_OUT_OF_MEMORY; }
    catch (...) { return NVRTC_ERROR_INTERNAL_ERROR; }
}
template<class F> nvrtcResult with_program(nvrtcProgram handle, F&& function) noexcept {
    return api([&] {
        auto value = lookup(handle);
        require(bool(value), NVRTC_ERROR_INVALID_PROGRAM);
        std::lock_guard lock(value->mutex);
        function(*value);
    });
}
cudavm::nvrtc_options parse_options(int count, const char* const* input, std::string& log) {
    cudavm::nvrtc_options result;
    auto invalid = [&](const std::string& option) {
        log = "cudavm NVRTC: unsupported or invalid option: " + option + '\n';
        throw api_error{NVRTC_ERROR_INVALID_OPTION};
    };
    for (int i = 0; i < count; ++i) {
        std::string option(input[i]);
        auto value = [&](const char* long_name, const char* short_name, bool joined = false) -> std::optional<std::string> {
            for (const auto* name : {long_name, short_name}) {
                std::string prefix(name);
                if (option == prefix) {
                    if (++i == count) invalid(option);
                    return std::string(input[i]);
                }
                if (option.starts_with(prefix + "=")) return option.substr(prefix.size() + 1);
                if (joined && prefix == short_name && option.starts_with(prefix) && option.size() > prefix.size())
                    return option.substr(prefix.size());
            }
            return std::nullopt;
        };
        if (auto architecture = value("--gpu-architecture", "-arch")) {
            // Physical SM targets also require CUBIN, which is not implemented.
            if (*architecture != "compute_75" && *architecture != "compute_80") invalid(option);
            result.architecture = *architecture == "compute_80" ? "sm_80" : "sm_75";
        } else if (auto standard = value("--std", "-std")) {
            if (*standard != "c++11" && *standard != "c++14" && *standard != "c++17"
                && *standard != "c++20" && *standard != "c++23") invalid(option);
            result.standard = *standard;
        } else if (auto include = value("--include-path", "-I", true)) {
            if (include->empty()) invalid(option);
            result.includes.push_back(*include);
        } else if (auto define = value("--define-macro", "-D", true)) {
            if (define->empty()) invalid(option);
            result.defines.push_back(*define);
        } else if (auto undefine = value("--undefine-macro", "-U", true)) {
            if (undefine->empty()) invalid(option);
            result.undefines.push_back(*undefine);
        } else if (auto relocatable = value("--relocatable-device-code", "-rdc")) {
            if (*relocatable != "true" && *relocatable != "false") invalid(option);
            result.relocatable = *relocatable == "true";
        } else if (option == "--device-c" || option == "-dc") result.relocatable = true;
        else if (option == "--device-w" || option == "-dw") result.relocatable = false;
        else invalid(option);
    }
    return result;
}
}
} // namespace cudavm
using namespace cudavm;
extern "C" CUDAVM_NVRTC_API const char* nvrtcGetErrorString(nvrtcResult result) {
    switch (result) {
#define ERROR_NAME(name) case name: return #name
    ERROR_NAME(NVRTC_SUCCESS); ERROR_NAME(NVRTC_ERROR_OUT_OF_MEMORY);
    ERROR_NAME(NVRTC_ERROR_PROGRAM_CREATION_FAILURE); ERROR_NAME(NVRTC_ERROR_INVALID_INPUT);
    ERROR_NAME(NVRTC_ERROR_INVALID_PROGRAM); ERROR_NAME(NVRTC_ERROR_INVALID_OPTION);
    ERROR_NAME(NVRTC_ERROR_COMPILATION); ERROR_NAME(NVRTC_ERROR_BUILTIN_OPERATION_FAILURE);
    ERROR_NAME(NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION);
    ERROR_NAME(NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION);
    ERROR_NAME(NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID); ERROR_NAME(NVRTC_ERROR_INTERNAL_ERROR);
    ERROR_NAME(NVRTC_ERROR_TIME_FILE_WRITE_FAILED); ERROR_NAME(NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED);
    ERROR_NAME(NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED); ERROR_NAME(NVRTC_ERROR_PCH_CREATE);
    ERROR_NAME(NVRTC_ERROR_CANCELLED); ERROR_NAME(NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED);
    ERROR_NAME(NVRTC_ERROR_BUSY);
#undef ERROR_NAME
    default: return "NVRTC_ERROR unknown";
    }
}
extern "C" CUDAVM_NVRTC_API nvrtcResult nvrtcVersion(int* major, int* minor) {
    return api([&] { require(major && minor, NVRTC_ERROR_INVALID_INPUT); *major = 13; *minor = 4; });
}
extern "C" CUDAVM_NVRTC_API nvrtcResult nvrtcGetNumSupportedArchs(int* count) {
    return api([&] { require(count, NVRTC_ERROR_INVALID_INPUT); *count = 2; });
}
extern "C" CUDAVM_NVRTC_API nvrtcResult nvrtcGetSupportedArchs(int* architectures) {
    return api([&] { require(architectures, NVRTC_ERROR_INVALID_INPUT); architectures[0] = 75; architectures[1] = 80; });
}
extern "C" CUDAVM_NVRTC_API nvrtcResult nvrtcCreateProgram(nvrtcProgram* output, const char* source,
    const char* name, int count, const char* const* headers, const char* const* names) {
    return api([&] {
        require(output && source && count >= 0 && (!count || (headers && names)), NVRTC_ERROR_INVALID_INPUT);
        auto value = std::make_shared<program>();
        value->source = source;
        value->name = name && *name ? name : "default_program";
        for (int i = 0; i < count; ++i) {
            require(headers[i] && names[i] && *names[i], NVRTC_ERROR_INVALID_INPUT);
            value->headers.emplace_back(names[i], headers[i]);
        }
        auto& registry = programs();
        std::lock_guard lock(registry.mutex);
        require(registry.next != UINTPTR_MAX, NVRTC_ERROR_OUT_OF_MEMORY);
        auto handle = reinterpret_cast<nvrtcProgram>(++registry.next);
        registry.programs.emplace(handle, std::move(value));
        *output = handle;
    });
}
extern "C" CUDAVM_NVRTC_API nvrtcResult nvrtcDestroyProgram(nvrtcProgram* handle) {
    return api([&] {
        require(handle, NVRTC_ERROR_INVALID_PROGRAM);
        auto& registry = programs();
        std::lock_guard lock(registry.mutex);
        require(registry.programs.erase(*handle) != 0, NVRTC_ERROR_INVALID_PROGRAM);
        *handle = nullptr;
    });
}
extern "C" CUDAVM_NVRTC_API nvrtcResult nvrtcCompileProgram(nvrtcProgram handle, int count, const char* const* options) {
    return with_program(handle, [&](program& value) {
        require(count >= 0 && (!count || options), NVRTC_ERROR_INVALID_INPUT);
        for (int i = 0; i < count; ++i) require(options[i], NVRTC_ERROR_INVALID_INPUT);
        value.compilation_attempted = true; value.compiled = false;
        value.ptx.clear(); value.log.clear();
        auto settings = parse_options(count, options, value.log);
        try {
            auto pointers = [](const std::vector<std::string>& strings) {
                std::vector<const char*> output;
                for (const auto& text : strings) output.push_back(text.c_str());
                return output;
            };
            std::vector<compiler_source_header> headers;
            for (const auto& [name, contents] : value.headers)
                headers.push_back({name.c_str(), contents.c_str()});
            auto expressions = pointers(value.expressions);
            auto includes = pointers(settings.includes), defines = pointers(settings.defines), undefines = pointers(settings.undefines);
            compiler_source_request request{1, value.source.c_str(), value.name.c_str(),
                settings.standard.c_str(), settings.architecture.c_str(), unsigned(settings.relocatable),
                headers.size(), headers.data(), expressions.size(), expressions.data(),
                includes.size(), includes.data(), defines.size(), defines.data(), undefines.size(), undefines.data()};
            compiler_input_request input;
            input.kind = compiler_input_kind::source; input.source = &request;
            compiler_program* program = nullptr;
            compiler_result* result = nullptr;
            auto status = cudavm_core_program_import_v1(&input, &program, &result);
            compiler_program_ptr imported(program);
            compiler_result_ptr import_result(result);
            value.log = cudavm_core_result_diagnostics_v1(result);
            if (status == compiler_status::out_of_memory) throw std::bad_alloc();
            require(status == compiler_status::success && program, NVRTC_ERROR_COMPILATION);
            compiler_emit_request emission;
            emission.kind = compiler_output_kind::program_image;
            emission.environment = compiler_environment::all;
            status = cudavm_core_program_emit_v1(program, &emission, &result);
            compiler_result_ptr owned(result);
            value.log += cudavm_core_result_diagnostics_v1(result);
            if (status == compiler_status::out_of_memory) throw std::bad_alloc();
            require(status == compiler_status::success && result, NVRTC_ERROR_COMPILATION);
            std::map<std::string, std::string> lowered_names;
            for (const auto& expression : value.expressions) {
                auto* name = cudavm_core_result_lowered_name_v1(result, expression.c_str());
                require(name, NVRTC_ERROR_INTERNAL_ERROR);
                lowered_names.emplace(expression, name);
            }
            std::string ptx = cudavm_core_result_output_v1(result);
            value.lowered_names.push_back(std::move(lowered_names));
            value.ptx = std::move(ptx);
            value.compiled = true;
        } catch (const std::bad_alloc&) { throw; }
        catch (const std::exception& error) {
            value.log += std::string("cudavm NVRTC: ") + error.what() + '\n';
            throw api_error{NVRTC_ERROR_COMPILATION};
        }
    });
}
extern "C" CUDAVM_NVRTC_API nvrtcResult nvrtcGetPTXSize(nvrtcProgram handle, size_t* size) {
    return with_program(handle, [&](program& value) { require(size, NVRTC_ERROR_INVALID_INPUT); *size = value.ptx.size() + 1; });
}
extern "C" CUDAVM_NVRTC_API nvrtcResult nvrtcGetPTX(nvrtcProgram handle, char* output) {
    return with_program(handle, [&](program& value) {
        require(output, NVRTC_ERROR_INVALID_INPUT); std::memcpy(output, value.ptx.c_str(), value.ptx.size() + 1);
    });
}
extern "C" CUDAVM_NVRTC_API nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram handle, size_t* size) {
    return with_program(handle, [&](program& value) { require(size, NVRTC_ERROR_INVALID_INPUT); *size = value.log.size() + 1; });
}
extern "C" CUDAVM_NVRTC_API nvrtcResult nvrtcGetProgramLog(nvrtcProgram handle, char* output) {
    return with_program(handle, [&](program& value) {
        require(output, NVRTC_ERROR_INVALID_INPUT); std::memcpy(output, value.log.c_str(), value.log.size() + 1);
    });
}
extern "C" CUDAVM_NVRTC_API nvrtcResult nvrtcAddNameExpression(nvrtcProgram handle, const char* expression) {
    return with_program(handle, [&](program& value) {
        require(expression && *expression, NVRTC_ERROR_INVALID_INPUT);
        require(!value.compilation_attempted, NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION);
        if (std::find(value.expressions.begin(), value.expressions.end(), expression) == value.expressions.end())
            value.expressions.emplace_back(expression);
    });
}
extern "C" CUDAVM_NVRTC_API nvrtcResult nvrtcGetLoweredName(nvrtcProgram handle, const char* expression, const char** output) {
    return with_program(handle, [&](program& value) {
        require(expression && output, NVRTC_ERROR_INVALID_INPUT);
        require(value.compiled, NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION);
        auto& names = value.lowered_names.back();
        auto found = names.find(expression);
        require(found != names.end(), NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID);
        *output = found->second.c_str();
    });
}
