#pragma once
#include "program_image_fixture.hpp"
#include "nvvm_image_fixture.hpp"
#include <nvPTXCompiler.h>
#include <nvvm.h>

namespace cudavm {
inline constexpr std::array<std::string_view, 10> program_image_producers{
    "nvrtc_cubin", "nvrtc_rdc", "nvrtc_lto", "ptxcompiler_cubin", "ptxcompiler_rdc", "nvvm_lto",
    "nvjitlink_cubin", "nvjitlink_lto", "nvjitlink_rdc", "nvjitlink_rdc_lto"};
inline void program_image_ptx_status(nvPTXCompileResult status) {
    require(status == NVPTXCOMPILE_SUCCESS, "program PTX Compiler status " + std::to_string(status));
}
inline void program_image_nvvm_status(nvvmResult status) {
    require(status == NVVM_SUCCESS, std::string("program libNVVM status ") + nvvmGetErrorString(status));
}
inline program_image_bytes program_image_produce(std::string_view producer) {
    require(std::find(program_image_producers.begin(), program_image_producers.end(), producer) != program_image_producers.end(), "known public program producer");
    if (producer.starts_with("nvrtc_")) return program_image_nvrtc_produce(producer);
    if (producer.starts_with("ptxcompiler_")) {
        struct owner { nvPTXCompilerHandle value{}; ~owner() { if (value) nvPTXCompilerDestroy(&value); } } program;
        program_image_ptx_status(nvPTXCompilerCreate(&program.value, module_loading_ptx.size(), module_loading_ptx.data()));
        const char* options[]{"-arch=sm_75", "-O0", "-c"};
        program_image_ptx_status(nvPTXCompilerCompile(program.value, producer == "ptxcompiler_rdc" ? 3 : 2, options));
        return program_image_result([&](size_t* size) { program_image_ptx_status(nvPTXCompilerGetCompiledProgramSize(program.value, size)); },
            [&](unsigned char* bytes) { program_image_ptx_status(nvPTXCompilerGetCompiledProgram(program.value, bytes)); });
    }
    if (producer == "nvvm_lto") {
        struct owner { nvvmProgram value{}; ~owner() { if (value) nvvmDestroyProgram(&value); } } program;
        program_image_nvvm_status(nvvmCreateProgram(&program.value));
        program_image_nvvm_status(nvvmAddModuleToProgram(program.value, program_image_nvvm.data(), program_image_nvvm.size(), "program_image.ll"));
        const char* options[]{"-arch=compute_75", "-opt=0", "-gen-lto"};
        const auto status = nvvmCompileProgram(program.value, 3, options);
        if (status != NVVM_SUCCESS) {
            size_t count{}; nvvmGetProgramLogSize(program.value, &count); std::vector<char> text(count ? count : 1);
            nvvmGetProgramLog(program.value, text.data()); require(false, std::string("program libNVVM compile: ") + text.data());
        }
        return program_image_result([&](size_t* size) { program_image_nvvm_status(nvvmGetCompiledResultSize(program.value, size)); },
            [&](unsigned char* bytes) { program_image_nvvm_status(nvvmGetCompiledResult(program.value, reinterpret_cast<char*>(bytes))); });
    }
    auto input = program_image_nvrtc_produce("nvrtc_lto");
    std::vector<std::string> options{"-arch=sm_75", "-lto", "-O0"};
    if (producer == "nvjitlink_rdc" || producer == "nvjitlink_rdc_lto") options.push_back("-r");
    linker_fixture_program linker(options); linker.add(input, "copied_source_input", NVJITLINK_INPUT_LTOIR);
    std::fill(input.begin(), input.end(), 0x57); linker.complete();
    return program_image_linked_result(linker, producer == "nvjitlink_lto" || producer == "nvjitlink_rdc_lto");
}
}
