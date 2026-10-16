#pragma once
#include "compiler.hpp"
#include <array>
namespace cudavm {
std::unique_ptr<llvm::Module> prepare_native_shader_module(const llvm::Module&);
enum class shader_environment { vulkan, opencl, dxil };
std::unique_ptr<llvm::Module> lower_native_shader(const llvm::Module&, const std::string& kernel,
    shader_environment, const std::array<unsigned,3>& block = {1,1,1});
}
