#include <nvvm.h>
#include <nvvm/export.h>
#include "../core/nvvm_api.hpp"
#include "../core/compiler_profiles.hpp"
#include <array>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <vector>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wdll-attribute-on-redeclaration"
#endif

namespace cudavm {
namespace {
constexpr size_t nvvm_input_limit = size_t(1) << 30;
struct nvvm_module { std::string bytes, name; bool lazy; };
struct nvvm_program {
    std::mutex mutex;
    std::vector<nvvm_module> modules;
    size_t input_size = 0;
    std::string output, log;
};
struct nvvm_registry {
    std::mutex mutex;
    uintptr_t next = 0;
    std::unordered_map<nvvmProgram, std::shared_ptr<nvvm_program>> programs;
};
nvvm_registry& nvvm_programs() { static nvvm_registry value; return value; }
struct nvvm_error { nvvmResult result; };
void nvvm_require(bool condition, nvvmResult error = NVVM_ERROR_INVALID_INPUT) { if (!condition) throw nvvm_error{error}; }
template<class Function> nvvmResult nvvm_api(Function function) noexcept {
    try { function(); return NVVM_SUCCESS; }
    catch (const nvvm_error& error) { return error.result; }
    catch (const std::bad_alloc&) { return NVVM_ERROR_OUT_OF_MEMORY; }
    catch (...) { return NVVM_ERROR_COMPILATION; }
}
template<class Function> nvvmResult nvvm_with_program(nvvmProgram handle, Function function) noexcept {
    return nvvm_api([&] {
        std::shared_ptr<nvvm_program> program;
        { auto& registry = nvvm_programs(); std::lock_guard lock(registry.mutex);
          auto found = registry.programs.find(handle); nvvm_require(found != registry.programs.end(), NVVM_ERROR_INVALID_PROGRAM); program = found->second; }
        std::lock_guard lock(program->mutex); function(*program);
    });
}
[[noreturn]] void nvvm_invalid_option(nvvm_program& program, const std::string& option) {
    program.log = "cudavm libNVVM: unsupported or invalid option: " + option + '\n';
    throw nvvm_error{NVVM_ERROR_INVALID_OPTION};
}
void nvvm_compile(nvvm_program& program, int count, const char** options, bool verify) {
    if (!verify) program.output.clear();
    program.log.clear();
    nvvm_require(count >= 0 && (!count || options));
    compiler_nvvm_request request;
    request.verify_only = verify;
    std::string architecture = "sm_75";
    for (int i = 0; i < count; ++i) {
        nvvm_require(options[i]); std::string option(options[i]);
        if (option.starts_with("-arch=")) {
            auto value = option.substr(6);
            if (std::find(nvvm_compile_architectures.begin(), nvvm_compile_architectures.end(), value) == nvvm_compile_architectures.end()) nvvm_invalid_option(program, option);
            architecture = "sm_" + value.substr(8);
        } else if (option == "-opt=0" || option == "-opt=3") request.optimization = option.back() - '0';
        else if (option == "-gen-lto") request.lto = 1;
        else if (option == "-fma=0" || option == "-fma=1") request.contract_fma = option.back() - '0';
        else if (option == "-ftz=0" || option == "-prec-div=1" || option == "-prec-sqrt=1" || option == "-jump-table-density=101") { /* Strict arithmetic. */ }
        else nvvm_invalid_option(program, option);
    }
    nvvm_require(!program.modules.empty(), NVVM_ERROR_NO_MODULE_IN_PROGRAM);
    std::vector<compiler_nvvm_module> modules;
    for (const auto& input : program.modules) modules.push_back({input.bytes.data(), input.bytes.size(), input.name.c_str(), uint32_t(input.lazy)});
    request.count = modules.size(); request.modules = modules.data(); request.architecture = architecture.c_str();
    compiler_result* result = nullptr;
    auto status = cudavm_core_nvvm_compile_v1(&request, &result);
    compiler_result_ptr owned(result);
    program.log = compiler_client_text(cudavm_core_result_diagnostics_v1(result));
    if (status != compiler_status::success) {
        if (status == compiler_status::out_of_memory) throw nvvm_error{NVVM_ERROR_OUT_OF_MEMORY};
        if (status == compiler_status::nvvm_version_mismatch) throw nvvm_error{NVVM_ERROR_IR_VERSION_MISMATCH};
        throw nvvm_error{verify ? NVVM_ERROR_INVALID_IR : NVVM_ERROR_COMPILATION};
    }
    if (!verify) program.output.assign(cudavm_core_result_output_v1(result), cudavm_core_result_size_v1(result));
}
nvvmResult nvvm_add(nvvmProgram handle, const char* buffer, size_t size, const char* name, bool lazy) {
    return nvvm_with_program(handle, [&](nvvm_program& program) {
        nvvm_require(buffer && size && size <= nvvm_input_limit - program.input_size && program.modules.size() < 4096);
        nvvm_module input{{buffer, size}, name ? name : "nvvm.input." + std::to_string(program.modules.size()), lazy};
        program.modules.push_back(std::move(input)); program.input_size += size;
    });
}
}
}
using namespace cudavm;
extern "C" {
CUDAVM_NVVM_API const char* nvvmGetErrorString(nvvmResult result) {
    constexpr const char* names[]{"NVVM_SUCCESS", "NVVM_ERROR_OUT_OF_MEMORY", "NVVM_ERROR_PROGRAM_CREATION_FAILURE", "NVVM_ERROR_IR_VERSION_MISMATCH",
        "NVVM_ERROR_INVALID_INPUT", "NVVM_ERROR_INVALID_PROGRAM", "NVVM_ERROR_INVALID_IR", "NVVM_ERROR_INVALID_OPTION", "NVVM_ERROR_NO_MODULE_IN_PROGRAM", "NVVM_ERROR_COMPILATION", "NVVM_ERROR_CANCELLED"};
    return unsigned(result) < std::size(names) ? names[unsigned(result)] : "NVVM_ERROR not found";
}
CUDAVM_NVVM_API nvvmResult nvvmVersion(int* major, int* minor) {
    if (!major || !minor) return NVVM_ERROR_INVALID_INPUT;
    *major = 2; *minor = 0; return NVVM_SUCCESS;
}
CUDAVM_NVVM_API nvvmResult nvvmIRVersion(int* major, int* minor, int* debug_major, int* debug_minor) {
    if (!major || !minor || !debug_major || !debug_minor) return NVVM_ERROR_INVALID_INPUT;
    *major = 2; *minor = 0; *debug_major = 3; *debug_minor = 1; return NVVM_SUCCESS;
}
CUDAVM_NVVM_API nvvmResult nvvmLLVMVersion(const char* architecture, int* major) {
    if (!architecture || !major || std::find(nvvm_compile_architectures.begin(), nvvm_compile_architectures.end(), std::string_view(architecture)) == nvvm_compile_architectures.end()) return NVVM_ERROR_INVALID_INPUT;
    *major = 7; return NVVM_SUCCESS;
}
CUDAVM_NVVM_API nvvmResult nvvmCreateProgram(nvvmProgram* handle) {
    return nvvm_api([&] {
        nvvm_require(handle, NVVM_ERROR_INVALID_PROGRAM);
        auto program = std::make_shared<nvvm_program>();
        auto& registry = nvvm_programs(); std::lock_guard lock(registry.mutex);
        nvvm_require(registry.next != UINTPTR_MAX, NVVM_ERROR_PROGRAM_CREATION_FAILURE);
        auto key = reinterpret_cast<nvvmProgram>(++registry.next);
        registry.programs.emplace(key, std::move(program)); *handle = key;
    });
}
CUDAVM_NVVM_API nvvmResult nvvmDestroyProgram(nvvmProgram* handle) {
    return nvvm_api([&] {
        nvvm_require(handle, NVVM_ERROR_INVALID_PROGRAM);
        auto& registry = nvvm_programs(); std::lock_guard lock(registry.mutex);
        nvvm_require(registry.programs.erase(*handle) != 0, NVVM_ERROR_INVALID_PROGRAM); *handle = nullptr;
    });
}
CUDAVM_NVVM_API nvvmResult nvvmAddModuleToProgram(nvvmProgram handle, const char* data, size_t size, const char* name) { return nvvm_add(handle, data, size, name, false); }
CUDAVM_NVVM_API nvvmResult nvvmLazyAddModuleToProgram(nvvmProgram handle, const char* data, size_t size, const char* name) { return nvvm_add(handle, data, size, name, true); }
CUDAVM_NVVM_API nvvmResult nvvmCompileProgram(nvvmProgram handle, int count, const char** options) { return nvvm_with_program(handle, [&](nvvm_program& program) { nvvm_compile(program, count, options, false); }); }
CUDAVM_NVVM_API nvvmResult nvvmVerifyProgram(nvvmProgram handle, int count, const char** options) { return nvvm_with_program(handle, [&](nvvm_program& program) { nvvm_compile(program, count, options, true); }); }
CUDAVM_NVVM_API nvvmResult nvvmGetCompiledResultSize(nvvmProgram handle, size_t* size) {
    return nvvm_with_program(handle, [&](nvvm_program& program) { nvvm_require(size); *size = program.output.size() + (program.output.empty() || program.output.back() != '\0'); });
}
CUDAVM_NVVM_API nvvmResult nvvmGetCompiledResult(nvvmProgram handle, char* buffer) {
    return nvvm_with_program(handle, [&](nvvm_program& program) { nvvm_require(buffer); std::memcpy(buffer, program.output.c_str(), program.output.size()); if (program.output.empty() || program.output.back() != '\0') buffer[program.output.size()] = 0; });
}
CUDAVM_NVVM_API nvvmResult nvvmGetProgramLogSize(nvvmProgram handle, size_t* size) {
    return nvvm_with_program(handle, [&](nvvm_program& program) { nvvm_require(size); *size = program.log.size() + 1; });
}
CUDAVM_NVVM_API nvvmResult nvvmGetProgramLog(nvvmProgram handle, char* buffer) {
    return nvvm_with_program(handle, [&](nvvm_program& program) { nvvm_require(buffer); std::memcpy(buffer, program.log.c_str(), program.log.size() + 1); });
}
}
