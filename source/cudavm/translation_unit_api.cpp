#include "compiler_service.hpp"
#include "compiler_api.hpp"
#include <mutex>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FormatVariadic.h>
using namespace cudavm;
extern "C" CUDAVM_COMPILER_API int cudavm_compiler_compile_translation_unit_v1(const compiler_translation_unit_request* request, compiler_result** output) noexcept {
    return run_compiler_api(output, [&](compiler_options& values, compiler_result&) {
        // Frontend serialization is independent of GPU contexts/registry locks.
        std::lock_guard lock(compiler_mutex());
        if (!request || request->abi_version != 1) throw std::runtime_error("compile options require abi_version 1");
        values.input = compiler_option_text(request->input); values.host = compiler_option_text(request->host);
        values.bitcode = compiler_option_text(request->bitcode); values.manifest = compiler_option_text(request->manifest);
        values.depfile = compiler_option_text(request->depfile);
        if (request->standard) values.standard = request->standard;
        if (request->architecture) values.architecture = request->architecture;
        append_compiler_option_strings(request->include_count, request->includes, values.includes);
        append_compiler_option_strings(request->define_count, request->defines, values.defines);
        values.resources = resolve_resources(compiler_option_text(request->resources_root));
        llvm::LLVMContext context;
        auto host = extract_host(values);
        auto module = compile_device(context, values, host.dependencies, &host.symbols);
        mark_translation_unit(*module, values, host);
        auto metadata = describe(*module);
        metadata["host_kernels"] = std::move(host.kernels);
        metadata["host_launches"] = std::move(host.launches);
        metadata["host_registrations"] = std::move(host.registrations);
        metadata["host_symbols"] = std::move(host.symbols);
        llvm::json::Array headers;
        for (const auto& header : host.headers)
            headers.push_back(llvm::json::Object{{"source", header.source}, {"output", header.output}});
        metadata["host_headers"] = std::move(headers);
        metadata["host_source_profile"] = "clang-expanded-user-tokens-v1";
        write_bitcode(values.bitcode, *module);
        write_file(values.host, host.host);
        write_file(values.manifest, llvm::formatv("{0:2}\n", llvm::json::Value(std::move(metadata))).str());
    });
}
extern "C" CUDAVM_COMPILER_API int cudavm_compiler_compile_get_resources_v1(const char* root, compiler_result** output) noexcept {
    return run_compiler_api(output, [&](compiler_options&, compiler_result& result) {
        result.output = llvm::formatv("{0:2}\n", llvm::json::Value(describe_resources(resolve_resources(compiler_option_text(root))))).str();
    });
}
