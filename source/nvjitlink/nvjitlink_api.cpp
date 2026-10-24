#define NVJITLINK_NO_INLINE
#include <nvJitLink.h>
#include <nvjitlink/export.h>
#include "../core/program_api.hpp"
#include "../core/compiler_profiles.hpp"
#include "../nvfatbin/container_api.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wdll-attribute-on-redeclaration"
#endif

namespace cudavm {
namespace {
// Containers include multiple independently bounded program envelopes.
constexpr size_t nvjitlink_input_limit = fatbin_limit;
struct nvjitlink_options {
    std::string architecture;
    unsigned optimization = 3;
    bool lto = false, ptx = false, relocatable = false, verbose = false, timing = false;
};
struct nvjitlink_program {
    std::mutex mutex;
    nvjitlink_options options;
    nvJitLinkResult configuration = NVJITLINK_SUCCESS;
    std::vector<compiler_program_ptr> inputs;
    std::string cubin, ptx, lto, error, info;
    bool completed = false;
};
struct nvjitlink_registry {
    std::mutex mutex;
    uintptr_t next = 0;
    std::unordered_map<nvJitLinkHandle, std::shared_ptr<nvjitlink_program>> programs;
};
nvjitlink_registry& nvjitlink_programs() { static nvjitlink_registry registry; return registry; }
struct nvjitlink_error { nvJitLinkResult result; };
void nvjitlink_require(bool condition, nvJitLinkResult result = NVJITLINK_ERROR_INVALID_INPUT) {
    if (!condition) throw nvjitlink_error{result};
}
template<class Function> nvJitLinkResult nvjitlink_api(Function function) noexcept {
    try { return function(); }
    catch (const nvjitlink_error& error) { return error.result; }
    catch (...) { return NVJITLINK_ERROR_INTERNAL; }
}
template<class Function> nvJitLinkResult nvjitlink_with_program(nvJitLinkHandle handle, Function function) noexcept {
    return nvjitlink_api([&] {
        std::shared_ptr<nvjitlink_program> program;
        { auto& registry = nvjitlink_programs(); std::lock_guard lock(registry.mutex);
          auto found = registry.programs.find(handle); nvjitlink_require(found != registry.programs.end()); program = found->second; }
        std::lock_guard lock(program->mutex);
        try { return function(*program); }
        catch (const std::exception& error) { program->error += std::string("cudavm nvJitLink: ") + error.what() + '\n'; throw; }
    });
}
[[noreturn]] void nvjitlink_fail(nvjitlink_program& program, nvJitLinkResult result, const std::string& message) {
    program.error += "cudavm nvJitLink: " + message + '\n'; throw nvjitlink_error{result};
}
void nvjitlink_parse_options(nvjitlink_program& program, uint32_t count, const char* const* options) {
    auto& output = program.options;
    bool virtual_target = false;
    for (uint32_t i = 0; i < count; ++i) {
        nvjitlink_require(options[i]); std::string option(options[i]);
        if (option.starts_with("-arch=")) {
            auto target = option.substr(6);
            virtual_target = target.starts_with("compute_");
            if (!virtual_target && !target.starts_with("sm_"))
                nvjitlink_fail(program, NVJITLINK_ERROR_UNRECOGNIZED_ARCH, "unrecognized architecture: " + target);
            auto number = target.substr(virtual_target ? 8 : 3);
            if (std::none_of(ptx_compile_architectures.begin(), ptx_compile_architectures.end(), [&](int value) { return number == std::to_string(value); }))
                nvjitlink_fail(program, NVJITLINK_ERROR_UNSUPPORTED_ARCH, "unsupported architecture: " + target);
            output.architecture = "sm_" + number;
        } else if (option == "-lto") output.lto = true;
        else if (option == "-ptx") output.ptx = true;
        else if (option == "-r") output.relocatable = true;
        else if (option == "-verbose") output.verbose = true;
        else if (option == "-time") output.timing = true;
        else if (option == "-O0" || option == "-O3") output.optimization = option.back() - '0';
        else if (option == "-no-cache" || option == "-split-compile=1") { /* Requests already run without an intermediate cache, serially. */ }
        else nvjitlink_fail(program, NVJITLINK_ERROR_UNRECOGNIZED_OPTION, "unsupported or invalid option: " + option);
    }
    if (output.architecture.empty()) nvjitlink_fail(program, NVJITLINK_ERROR_MISSING_ARCH, "-arch is required");
    if (output.ptx && !output.lto) nvjitlink_fail(program, NVJITLINK_ERROR_LTO_NOT_ENABLED, "-ptx requires -lto");
    if ((virtual_target && !output.ptx) || (output.ptx && output.relocatable))
        nvjitlink_fail(program, NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS, "compute targets require -ptx; -ptx and -r cannot be combined");
}
void nvjitlink_status(nvjitlink_program& program, compiler_status status, const compiler_result* result, nvJitLinkResult failure) {
    auto* diagnostic = cudavm_core_result_diagnostics_v1(result);
    if (diagnostic && *diagnostic) (status == compiler_status::success ? program.info : program.error) += std::string(diagnostic) + '\n';
    if (status == compiler_status::success) return;
    if (status == compiler_status::out_of_memory) failure = NVJITLINK_ERROR_INTERNAL;
    else if (status == compiler_status::architecture_mismatch) failure = NVJITLINK_ERROR_ARCH_MISMATCH;
    else if (status == compiler_status::input_type_mismatch) failure = NVJITLINK_ERROR_INCORRECT_INPUT_TYPE;
    else if (status == compiler_status::unsupported) failure = NVJITLINK_ERROR_UNRECOGNIZED_INPUT;
    throw nvjitlink_error{failure};
}
void nvjitlink_add(nvjitlink_program& program, nvJitLinkInputType type, const void* data, size_t size, const char* name) {
    nvjitlink_require(program.configuration == NVJITLINK_SUCCESS && !program.completed);
    if (!data) nvjitlink_fail(program, NVJITLINK_ERROR_NULL_INPUT, "null input data");
    nvjitlink_require(size && size <= nvjitlink_input_limit);
    compiler_input_kind kind;
    switch (type) {
    case NVJITLINK_INPUT_PTX: kind = compiler_input_kind::ptx; break;
    case NVJITLINK_INPUT_CUBIN: kind = compiler_input_kind::cubin; break;
    case NVJITLINK_INPUT_LTOIR:
        if (!program.options.lto) nvjitlink_fail(program, NVJITLINK_ERROR_LTO_NOT_ENABLED, "LTOIR inputs require -lto");
        kind = compiler_input_kind::nvltoir; break;
    case NVJITLINK_INPUT_ANY: kind = compiler_input_kind::automatic; break;
    case NVJITLINK_INPUT_FATBIN: kind = compiler_input_kind::fatbinary; break;
    case NVJITLINK_INPUT_NONE: nvjitlink_fail(program, NVJITLINK_ERROR_INVALID_INPUT, "input type NONE is invalid");
    default: nvjitlink_fail(program, NVJITLINK_ERROR_UNRECOGNIZED_INPUT, "this container input type is not implemented");
    }
    if ((type == NVJITLINK_INPUT_ANY || type == NVJITLINK_INPUT_FATBIN) && has_fatbin(std::string_view(static_cast<const char*>(data), size))) {
        std::vector<fatbin_record> records;
        try { records = decode_fatbin({static_cast<const char*>(data), size}); }
        catch (const std::exception& error) { nvjitlink_fail(program, NVJITLINK_ERROR_INVALID_INPUT, error.what()); }
        auto* selected = select_fatbin(records, unsigned(std::stoul(program.options.architecture.substr(3))), true);
        if (selected && selected->kind == fatbin_kind::ltoir && !program.options.lto)
            nvjitlink_fail(program, NVJITLINK_ERROR_LTO_NOT_ENABLED, "fatbinary LTOIR inputs require -lto");
    }
    std::string label = name && *name ? name : "nvJitLink.input." + std::to_string(program.inputs.size());
    // The core owns a copy of the input IR before this call returns. Labels are
    // copied as well; neither caller buffers nor open files back the link state.
    compiler_input_request_v3 request;
    request.base.base = {3, kind, static_cast<const char*>(data), size, label.c_str(), nullptr};
    request.base.architecture = program.options.architecture.c_str(); request.relocatable = 1;
    compiler_program* input = nullptr; compiler_result* result = nullptr;
    auto status = cudavm_core_program_import_v1(&request.base.base, &input, &result);
    compiler_program_ptr owned(input); compiler_result_ptr diagnostics(result);
    if (status != compiler_status::success) program.error += label + ": ";
    nvjitlink_status(program, status, result, type == NVJITLINK_INPUT_PTX ? NVJITLINK_ERROR_PTX_COMPILE : NVJITLINK_ERROR_INVALID_INPUT);
    program.inputs.push_back(std::move(owned));
}
std::string nvjitlink_read_file(nvjitlink_program& program, const char* path) {
    std::FILE* file = nullptr;
#if defined(_WIN32)
    fopen_s(&file, path, "rb");
#else
    file = std::fopen(path, "rb");
#endif
    std::unique_ptr<std::FILE, decltype(&std::fclose)> input(file, &std::fclose);
    if (!input) nvjitlink_fail(program, NVJITLINK_ERROR_INVALID_INPUT, std::string("cannot read input file: ") + path);
    nvjitlink_require(std::fseek(input.get(), 0, SEEK_END) == 0);
    auto length = std::ftell(input.get()); nvjitlink_require(length > 0 && static_cast<size_t>(length) <= nvjitlink_input_limit);
    std::string bytes(static_cast<size_t>(length), 0); std::rewind(input.get());
    nvjitlink_require(std::fread(bytes.data(), 1, bytes.size(), input.get()) == bytes.size());
    return bytes;
}
std::string nvjitlink_emit(nvjitlink_program& program, const compiler_program* input, compiler_output_kind kind, compiler_environment environment) {
    compiler_emit_request request; request.kind = kind; request.environment = environment;
    compiler_result* result = nullptr;
    auto status = cudavm_core_program_emit_v1(input, &request, &result);
    compiler_result_ptr output(result); nvjitlink_status(program, status, result, NVJITLINK_ERROR_FINALIZE);
    return {cudavm_core_result_output_v1(result), cudavm_core_result_size_v1(result)};
}
nvJitLinkResult nvjitlink_complete(nvjitlink_program& program) {
    nvjitlink_require(program.configuration == NVJITLINK_SUCCESS);
    nvjitlink_require(!program.completed, NVJITLINK_ERROR_INTERNAL);
    auto start = std::chrono::steady_clock::now();
    if (program.inputs.empty()) {
        const char* empty = ".version 6.3\n.target sm_75\n.address_size 64\n";
        nvjitlink_add(program, NVJITLINK_INPUT_PTX, empty, std::strlen(empty), "nvJitLink.empty");
    }
    std::vector<const compiler_program*> inputs;
    for (const auto& input : program.inputs) inputs.push_back(input.get());
    compiler_program* linked = nullptr; compiler_result* result = nullptr;
    auto status = cudavm_core_program_link_v1(inputs.size(), inputs.data(), &linked, &result);
    compiler_program_ptr graph(linked); compiler_result_ptr diagnostics(result);
    nvjitlink_status(program, status, result, NVJITLINK_ERROR_FINALIZE);
    std::string lto, cubin, ptx;
    if (program.options.lto || program.options.relocatable) {
        auto image = nvjitlink_emit(program, linked, compiler_output_kind::program_image, compiler_environment::none);
        if (program.options.lto) lto = image;
        if (program.options.relocatable) cubin = std::move(image);
    }
    if (!program.options.relocatable) {
        compiler_pipeline_request pipeline{1, 0, 1, program.options.optimization};
        compiler_program* prepared = nullptr;
        status = cudavm_core_program_prepare_v1(linked, &pipeline, &prepared, &result);
        compiler_program_ptr ready(prepared); compiler_result_ptr preparation(result);
        nvjitlink_status(program, status, result, NVJITLINK_ERROR_FINALIZE);
        if (program.options.ptx) ptx = nvjitlink_emit(program, prepared, compiler_output_kind::native_image, compiler_environment::all);
        else cubin = nvjitlink_emit(program, prepared, compiler_output_kind::program_image, compiler_environment::all);
    }
    if (program.options.verbose) program.info += "cudavm nvJitLink: linked " + std::to_string(inputs.size()) + " inputs for "
        + program.options.architecture + ", optimization " + std::to_string(program.options.optimization) + '\n';
    if (program.options.timing) {
        auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        program.info += "cudavm nvJitLink: link time " + std::to_string(elapsed) + " ms\n";
    }
    program.cubin = std::move(cubin); program.ptx = std::move(ptx); program.lto = std::move(lto); program.completed = true;
    return NVJITLINK_SUCCESS;
}
const std::string& nvjitlink_output(nvjitlink_program& program, unsigned kind) {
    nvjitlink_require(kind == 0 ? !program.options.ptx : kind == 1 ? program.options.ptx : program.options.lto);
    nvjitlink_require(program.completed, NVJITLINK_ERROR_INTERNAL);
    return kind == 0 ? program.cubin : kind == 1 ? program.ptx : program.lto;
}
}
}
using namespace cudavm;

extern "C" CUDAVM_NVJITLINK_API nvJitLinkResult __nvJitLinkCreate_13_4(nvJitLinkHandle* output, uint32_t count, const char* const* options) {
    return nvjitlink_api([&] {
        nvjitlink_require(output && (!count || options));
        auto program = std::make_shared<nvjitlink_program>();
        program->configuration = nvjitlink_api([&] { nvjitlink_parse_options(*program, count, options); return NVJITLINK_SUCCESS; });
        auto& registry = nvjitlink_programs(); std::lock_guard lock(registry.mutex);
        nvjitlink_require(registry.next != UINTPTR_MAX, NVJITLINK_ERROR_INTERNAL);
        auto handle = reinterpret_cast<nvJitLinkHandle>(++registry.next);
        registry.programs.emplace(handle, program); *output = handle;
        return program->configuration;
    });
}
extern "C" CUDAVM_NVJITLINK_API nvJitLinkResult nvJitLinkCreate(nvJitLinkHandle* output, uint32_t count, const char** options) {
    return __nvJitLinkCreate_13_4(output, count, options);
}
#define CUDAVM_JITLINK_ENTRY(name, parameters, arguments, body) \
extern "C" CUDAVM_NVJITLINK_API nvJitLinkResult __##name##_13_4 parameters body \
extern "C" CUDAVM_NVJITLINK_API nvJitLinkResult name parameters { return __##name##_13_4 arguments; }

CUDAVM_JITLINK_ENTRY(nvJitLinkDestroy, (nvJitLinkHandle* handle), (handle), {
    return nvjitlink_api([&] { nvjitlink_require(handle); auto& registry = nvjitlink_programs(); std::lock_guard lock(registry.mutex);
        nvjitlink_require(registry.programs.erase(*handle) != 0); *handle = nullptr; return NVJITLINK_SUCCESS; });
})
CUDAVM_JITLINK_ENTRY(nvJitLinkAddData, (nvJitLinkHandle handle, nvJitLinkInputType type, const void* data, size_t size, const char* name), (handle, type, data, size, name), {
    return nvjitlink_with_program(handle, [&](nvjitlink_program& program) { nvjitlink_add(program, type, data, size, name); return NVJITLINK_SUCCESS; });
})
CUDAVM_JITLINK_ENTRY(nvJitLinkAddFile, (nvJitLinkHandle handle, nvJitLinkInputType type, const char* path), (handle, type, path), {
    return nvjitlink_with_program(handle, [&](nvjitlink_program& program) {
        nvjitlink_require(program.configuration == NVJITLINK_SUCCESS && !program.completed && path && *path);
        auto bytes = nvjitlink_read_file(program, path);
        nvjitlink_add(program, type, bytes.data(), bytes.size(), path); return NVJITLINK_SUCCESS;
    });
})
CUDAVM_JITLINK_ENTRY(nvJitLinkComplete, (nvJitLinkHandle handle), (handle), {
    return nvjitlink_with_program(handle, [](nvjitlink_program& program) { return nvjitlink_complete(program); });
})
#define CUDAVM_JITLINK_RESULT(name, kind, pointer_type) \
CUDAVM_JITLINK_ENTRY(nvJitLinkGetLinked##name##Size, (nvJitLinkHandle handle, size_t* size), (handle, size), { \
    return nvjitlink_with_program(handle, [&](nvjitlink_program& program) { nvjitlink_require(size); \
        auto& output = nvjitlink_output(program, kind); *size = output.size() + (kind == 1 ? 1 : 0); return NVJITLINK_SUCCESS; }); \
}) \
CUDAVM_JITLINK_ENTRY(nvJitLinkGetLinked##name, (nvJitLinkHandle handle, pointer_type output), (handle, output), { \
    return nvjitlink_with_program(handle, [&](nvjitlink_program& program) { nvjitlink_require(output); \
        auto& bytes = nvjitlink_output(program, kind); std::memcpy(output, bytes.c_str(), bytes.size() + (kind == 1 ? 1 : 0)); return NVJITLINK_SUCCESS; }); \
})
CUDAVM_JITLINK_RESULT(Cubin, 0, void*)
CUDAVM_JITLINK_RESULT(Ptx, 1, char*)
CUDAVM_JITLINK_RESULT(LTOIR, 2, void*)
#define CUDAVM_JITLINK_LOG(name, member) \
CUDAVM_JITLINK_ENTRY(nvJitLinkGet##name##LogSize, (nvJitLinkHandle handle, size_t* size), (handle, size), { \
    return nvjitlink_with_program(handle, [&](nvjitlink_program& program) { nvjitlink_require(size); \
        *size = program.member.empty() ? 0 : program.member.size() + 1; return NVJITLINK_SUCCESS; }); \
}) \
CUDAVM_JITLINK_ENTRY(nvJitLinkGet##name##Log, (nvJitLinkHandle handle, char* output), (handle, output), { \
    return nvjitlink_with_program(handle, [&](nvjitlink_program& program) { nvjitlink_require(output); \
        std::memcpy(output, program.member.c_str(), program.member.size() + 1); return NVJITLINK_SUCCESS; }); \
})
CUDAVM_JITLINK_LOG(Error, error)
CUDAVM_JITLINK_LOG(Info, info)
#undef CUDAVM_JITLINK_LOG
#undef CUDAVM_JITLINK_RESULT
#undef CUDAVM_JITLINK_ENTRY
extern "C" CUDAVM_NVJITLINK_API nvJitLinkResult nvJitLinkVersion(unsigned* major, unsigned* minor) {
    return nvjitlink_api([&] { nvjitlink_require(major && minor); *major = 13; *minor = 4; return NVJITLINK_SUCCESS; });
}
