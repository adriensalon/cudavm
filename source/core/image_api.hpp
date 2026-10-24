#pragma once
#include "program_api.hpp"

// Driver-only lazy JIT adapter. Output is copied through the callback,
// allowing the Driver to discover core without a link-time dependency.
namespace cudavm {
struct compiler_image_request {
    uint32_t abi_version;
    compiler_input_kind kind;
    const char* input;
    size_t length;
    unsigned backend;
};
using compiler_output = bool (*)(const char*, size_t, void*) noexcept;
using compiler_compile_image = compiler_status (*)(const compiler_image_request*,
    compiler_output, void*, char*, size_t) noexcept;
}
extern "C" {
CUDAVM_CORE_API cudavm::compiler_status cudavm_core_compile_image_v1(
    const cudavm::compiler_image_request*, cudavm::compiler_output, void*, char*, size_t) noexcept;
}
