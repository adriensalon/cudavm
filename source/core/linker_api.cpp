#include "compiler_service.hpp"
#include "compiler_api.hpp"
#include "native_emission.hpp"
#include "program.hpp"
#include <algorithm>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
using namespace cudavm;
extern "C" CUDAVM_CORE_API int cudavm_core_link_files_v1(const compiler_link_request* request, compiler_result** output) noexcept {
    return run_compiler_api(output, [&](compiler_options& values, compiler_result&) {
        std::lock_guard lock(compiler_mutex());
        if (!request || request->abi_version != 1) throw std::runtime_error("invalid compiler link request");
        append_compiler_option_strings(request->input_count, request->inputs, values.inputs);
        append_compiler_option_strings(request->library_input_count, request->library_inputs, values.library_inputs);
        append_compiler_option_strings(request->host_input_count, request->host_inputs, values.host_inputs);
        values.portable = compiler_option_text(request->portable); values.ir = compiler_option_text(request->ir);
        values.manifest = compiler_option_text(request->manifest); values.binary = compiler_option_text(request->binary);
        values.target = compiler_option_text(request->target); values.module = compiler_option_text(request->module);
        if (request->architecture) values.architecture = request->architecture;
        if ((values.inputs.empty() && values.library_inputs.empty()) || values.portable.empty())
            throw std::runtime_error("link-device requires input --bitcode arguments and --portable output");
        if (!values.target.empty() && values.target != "cuda" && values.target != "all" && values.target != "opencl"
            && values.target != "opengl" && values.target != "vulkan" && values.target != "directx")
            throw std::runtime_error("unknown emission target: " + values.target);
        if (values.binary.empty() != values.target.empty()) throw std::runtime_error("--target and --binary must be supplied together");
        if (!values.module.empty() && values.target.empty()) throw std::runtime_error("--module requires an emission target and binary output");
        compiler_program linked;
        linked.architecture = values.architecture;
        linked.module = link_device(linked.context, values);
        compiler_pipeline_request pipeline;
        pipeline.lower_semantics = 1;
        auto prepared = prepare_compiler_program(linked, pipeline);
        auto* module = prepared->module.get();
        auto metadata = describe(*module);
        std::string code, embedded;
        if (!values.target.empty()) {
            auto artifact = emit_native(*module, {native_target::nvptx, values.architecture, true, {}});
            code = std::move(artifact.bytes);
            auto available = [&](const char* inventory, const char* field) {
                auto* kernels = artifact.contract.getArray(inventory);
                return kernels && std::all_of(kernels->begin(), kernels->end(), [&](const auto& item) {
                    auto* kernel = item.getAsObject(); return kernel && kernel->getBoolean(field) == true;
                });
            };
            bool spirv = available("native_spirv_kernels", "available");
            bool opencl = available("native_spirv_kernels", "opencl_available");
            bool dxil = available("native_dxil_kernels", "available");
            bool shader_ready = values.target == "opencl" ? opencl : values.target == "directx" ? dxil
                : values.target == "all" || values.target == "cuda" ? spirv && opencl && dxil : spirv;
            metadata["native_emission"] = std::move(artifact.contract);
            metadata["native_module_descriptor"]=true;
            metadata["shader_launch_ready"]=shader_ready;
            metadata["emission_target"]=values.target;
            metadata["image_format"]="ptx-with-native-artifacts";
        }
        if (!values.module.empty()) {
            embedded = embed_cuda(*module, code);
            metadata["runtime_launch_ready"] = true;
            metadata["runtime_launch_scope"] = "blocking-default-stream";
            metadata["host_kernel_registration"] = "cpu-startup-before-ordinary-host-initializers";
        }
        write_bitcode(values.portable, *module);
        if (!values.ir.empty()) {
            std::string ir;
            llvm::raw_string_ostream stream(ir);
            module->print(stream, nullptr);
            write_file(values.ir, ir);
        }
        write_file(values.manifest, llvm::formatv("{0:2}\n", llvm::json::Value(std::move(metadata))).str());
        write_file(values.binary, code);
        write_file(values.module, embedded);
    });
}
