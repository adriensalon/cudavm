#include "../cudavm/compiler_client.hpp"
#include <nvrtc/cudavm_nvrtc.h>
struct cudavmCompileResult { std::string output, diagnostics; };
using namespace cudavm;
extern "C" CUDAVM_NVRTC_API int cudavmCompileTranslationUnit(const cudavmCompileOptions* request, cudavmCompileResult** output) {
    return run_compiler_facade(output, [&](cudavmCompileResult& result) {
        if (!request || request->abi_version != 1) throw std::runtime_error("compile options require abi_version 1");
        validate_compiler_strings(request->include_count, request->includes);
        validate_compiler_strings(request->define_count, request->defines);
        if (compiler_client_text(request->input).empty() || compiler_client_text(request->host).empty() || compiler_client_text(request->bitcode).empty())
            throw std::runtime_error("compile-tu requires --input, --host and --bitcode");
        auto standard = request->standard ? request->standard : "c++20";
        if (std::string(standard) != "c++17" && std::string(standard) != "c++20" && std::string(standard) != "c++23")
            throw std::runtime_error("--standard must be c++17, c++20 or c++23");
        compiler_translation_unit_request normalized{1, request->input, request->host, request->bitcode, request->manifest, request->depfile,
            standard, request->architecture ? request->architecture : "sm_75", request->resources_root,
            request->include_count, request->includes, request->define_count, request->defines};
        compiler_result* compiled = nullptr;
        auto status = cudavm_compiler_compile_translation_unit_v1(&normalized, &compiled);
        return copy_compiler_result(result, status, compiled);
    });
}
extern "C" CUDAVM_NVRTC_API int cudavmCompileGetResources(const char* root, cudavmCompileResult** output) {
    return run_compiler_facade(output, [&](cudavmCompileResult& result) {
        compiler_result* resources = nullptr;
        auto status = cudavm_compiler_compile_get_resources_v1(root, &resources);
        return copy_compiler_result(result, status, resources);
    });
}
extern "C" CUDAVM_NVRTC_API const char* cudavmCompileOutput(const cudavmCompileResult* result) { return result ? result->output.c_str() : ""; }
extern "C" CUDAVM_NVRTC_API const char* cudavmCompileDiagnostics(const cudavmCompileResult* result) { return result ? result->diagnostics.c_str() : "compiler result is unavailable\n"; }
extern "C" CUDAVM_NVRTC_API void cudavmCompileRelease(cudavmCompileResult* result) { delete result; }
