#pragma once
#include "compiler.hpp"
#include <array>

namespace cudavm {
// Artifact targets are independent of Runtime adapter selection. A Vulkan,
// OpenCL or OpenGL session may eventually consume SPIR-V with different ABIs.
enum class native_target { nvptx, spirv, dxil };
struct native_request {
    native_target target = native_target::nvptx;
    std::string architecture = "sm_75";
    bool shader_artifacts = true;
    std::string kernel;
    std::array<unsigned,3> block = {1,1,1};
};
struct native_program {
    std::unique_ptr<llvm::Module> module;
    llvm::json::Object contract;
};
struct native_artifact {
    std::string bytes;
    llvm::json::Object contract;
};
const char* native_target_name(native_target);
// Inspection is side-effect free and reports requirements, not device support.
llvm::json::Object inspect_native(const llvm::Module&, native_target);
// Used after a target changes its DataLayout, before LLVM code generation.
void validate_native_kernel_abi(const llvm::Module&);
// Always owns a clone. Each target passes the same ABI boundary before emission.
native_program prepare_native(const llvm::Module&, const native_request&);
native_artifact emit_native(const llvm::Module&, const native_request&);
}
