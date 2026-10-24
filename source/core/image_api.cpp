#include "compiler_client.hpp"
#include <algorithm>
#include <cstring>

// Compatibility adapter for the Driver's lazily discovered callback ABI.
extern "C" CUDAVM_CORE_API cudavm::compiler_status cudavm_core_compile_image_v1(
        const cudavm::compiler_image_request* request, cudavm::compiler_output output,
        void* user, char* diagnostic, size_t capacity) noexcept {
    using namespace cudavm;
    auto report = [&](const char* text) {
        if (diagnostic && capacity) {
            auto size = std::min(std::strlen(text), capacity - 1);
            std::memcpy(diagnostic, text, size); diagnostic[size] = 0;
        }
    };
    if (diagnostic && capacity) diagnostic[0] = 0;
    try {
        if (!request || request->abi_version != 1 || !request->input || !output) {
            report("invalid compiler image request"); return compiler_status::invalid;
        }
        if (request->backend < unsigned(compiler_environment::opencl) || request->backend > unsigned(compiler_environment::directx)) {
            report("invalid PTX compiler request"); return compiler_status::invalid;
        }
        compiler_input_request input{1, request->kind, request->input, request->length, nullptr, nullptr};
        compiler_program* program = nullptr;
        compiler_result* result = nullptr;
        auto status = cudavm_core_program_import_v1(&input, &program, &result);
        compiler_program_ptr imported(program);
        compiler_result_ptr import_result(result);
        if (status != compiler_status::success) { report(cudavm_core_result_diagnostics_v1(result)); return status; }
        compiler_pipeline_request pipeline;
        pipeline.numerical_helpers = 1;
        status = cudavm_core_program_prepare_v1(program, &pipeline, &program, &result);
        compiler_program_ptr prepared(program);
        compiler_result_ptr prepare_result(result);
        if (status != compiler_status::success) { report(cudavm_core_result_diagnostics_v1(result)); return status; }
        compiler_emit_request emission;
        emission.kind = compiler_output_kind::program_image;
        emission.environment = compiler_environment(request->backend);
        status = cudavm_core_program_emit_v1(program, &emission, &result);
        compiler_result_ptr emitted(result);
        if (status != compiler_status::success) {
            report(cudavm_core_result_diagnostics_v1(result));
            return status == compiler_status::invalid ? compiler_status::unsupported : status;
        }
        std::string image(request->input, request->length);
        image += '\n';
        image.append(cudavm_core_result_output_v1(result), cudavm_core_result_size_v1(result));
        if (!output(image.data(), image.size(), user)) { report("PTX output allocation failed"); return compiler_status::out_of_memory; }
        return compiler_status::success;
    } catch (const std::bad_alloc&) { report("PTX compilation allocation failed"); return compiler_status::out_of_memory; }
    catch (...) { report("unknown PTX compiler failure"); return compiler_status::invalid; }
}
