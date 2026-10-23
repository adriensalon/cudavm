#include "ptx_import.hpp"
#include "compiler_service.hpp"
#include "compiler.hpp"
#include "device_numeric_ir.hpp"
#include "spirv_emitter.hpp"
#include "dxil_emitter.hpp"
#include "spirv_artifact.hpp"
#include "dxil_artifact.hpp"
#include <cuda/cudavm_cuda.h>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
// The service owns LLVM and serializes its emission state. The Driver keeps
// only the resulting descriptors and native code, exactly as for AOT modules.
extern "C" CUDAVM_COMPILER_API cudavm::compiler_status cudavm_compiler_compile_image_v1(
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
        switch (request->kind) {
        case compiler_input_kind::ptx: break;
        case compiler_input_kind::nvvm:
            report("NVVM import is not implemented"); return compiler_status::unsupported;
        case compiler_input_kind::cubin:
            report("cubin import is not implemented"); return compiler_status::unsupported;
        case compiler_input_kind::nvltoir:
            report("NVIDIA LTO-IR import is not implemented"); return compiler_status::unsupported;
        default:
            report("unknown compiler input kind"); return compiler_status::invalid;
        }
        auto* input = request->input;
        auto length = request->length;
        auto backend = request->backend;
        if (backend != cudavmBackendVulkan && backend != cudavmBackendOpenGL &&
                backend != cudavmBackendOpenCL && backend != cudavmBackendDirectX) {
            report("invalid PTX compiler request"); return compiler_status::invalid;
        }
        std::lock_guard lock(compiler_mutex());
        llvm::LLVMContext context; auto module = import_ptx(context, {input, length});
        link_device_numeric_ir(*module);
        std::string image(input, length); image += '\n'; image += emit_native_module(*module);
        std::vector<shader_artifact> artifacts;
        for (const auto& function : *module) if (function.hasFnAttribute("cudavm.kernel")) {
            shader_artifact artifact; artifact.symbol = function.getName().str();
            if (backend == cudavmBackendDirectX) {
                auto lowered = lower_native_dxil(*module, artifact.symbol);
                std::vector<dxil_variant> variants;
                for (const auto& shape : dxil_block_shapes) {
                    lowered->getFunction("main")->addFnAttr("hlsl.numthreads",
                        std::to_string(shape[0]) + "," + std::to_string(shape[1]) + "," + std::to_string(shape[2]));
                    variants.push_back({shape, emit_native_dxil(*lowered)});
                }
                artifact.code = encode_dxil_variants(variants);
            } else if (backend == cudavmBackendOpenCL)
                artifact.opencl_code = emit_native_spirv(*lower_native_spirv(*module, artifact.symbol, true));
            else artifact.code = emit_native_spirv(*lower_native_spirv(*module, artifact.symbol));
            artifacts.push_back(std::move(artifact));
        }
        image += backend == cudavmBackendDirectX ? encode_dxil_artifacts(artifacts) : encode_spirv_artifacts(artifacts);
        if (!output(image.data(), image.size(), user)) { report("PTX output allocation failed"); return compiler_status::out_of_memory; }
        return compiler_status::success;
    } catch (const ptx_import_error& error) {
        report(error.what()); return error.unsupported ? compiler_status::unsupported : compiler_status::invalid;
    } catch (const std::bad_alloc&) { report("PTX compilation allocation failed"); return compiler_status::out_of_memory; }
      catch (const std::exception& error) { report(error.what()); return compiler_status::unsupported; }
      catch (...) { report("unknown PTX compiler failure"); return compiler_status::invalid; }
}
