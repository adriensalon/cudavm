#include "../core/linker_api.hpp"
#include <nvjitlink/cudavm_nvjitlink.h>
struct cudavmLinkResult { std::string output, diagnostics; };
using namespace cudavm;
extern "C" CUDAVM_NVJITLINK_API int cudavmLinkDevice(const cudavmLinkOptions* request, cudavmLinkResult** output) {
    return run_compiler_facade(output, [&](cudavmLinkResult& result) {
        if (!request || (request->abi_version != 1 && request->abi_version != 2))
            throw std::runtime_error("link options require abi_version 1 or 2");
        validate_compiler_strings(request->input_count, request->inputs);
        compiler_link_request normalized{};
        normalized.abi_version = 1;
        normalized.input_count = request->input_count; normalized.inputs = request->inputs;
        if (request->abi_version == 2) {
            const auto& extended = *reinterpret_cast<const cudavmLinkOptionsV2*>(request);
            validate_compiler_strings(extended.library_input_count, extended.library_inputs);
            validate_compiler_strings(extended.host_input_count, extended.host_inputs);
            normalized.library_input_count = extended.library_input_count; normalized.library_inputs = extended.library_inputs;
            normalized.host_input_count = extended.host_input_count; normalized.host_inputs = extended.host_inputs;
        }
        auto portable = compiler_client_text(request->portable), binary = compiler_client_text(request->binary);
        auto target = compiler_client_text(request->target), module = compiler_client_text(request->module);
        if ((!normalized.input_count && !normalized.library_input_count) || portable.empty())
            throw std::runtime_error("link-device requires input --bitcode arguments and --portable output");
        if (!target.empty() && target != "cuda" && target != "all" && target != "opencl" && target != "opengl" && target != "vulkan" && target != "directx")
            throw std::runtime_error("unknown emission target: " + target);
        if (binary.empty() != target.empty()) throw std::runtime_error("--target and --binary must be supplied together");
        if (!module.empty() && target.empty()) throw std::runtime_error("--module requires an emission target and binary output");
        normalized.portable = request->portable; normalized.ir = request->ir; normalized.manifest = request->manifest;
        normalized.binary = request->binary; normalized.target = request->target; normalized.module = request->module;
        normalized.architecture = request->architecture ? request->architecture : "sm_75";
        compiler_result* linked = nullptr;
        auto status = cudavm_core_link_files_v1(&normalized, &linked);
        return copy_compiler_result(result, status, linked);
    });
}
extern "C" CUDAVM_NVJITLINK_API const char* cudavmLinkDiagnostics(const cudavmLinkResult* result) { return result ? result->diagnostics.c_str() : "link result is unavailable\n"; }
extern "C" CUDAVM_NVJITLINK_API void cudavmLinkRelease(cudavmLinkResult* result) { delete result; }
