#pragma once
#include "program_api.hpp"

namespace cudavm {
struct compiler_nvvm_module {
    const char* data;
    size_t size;
    const char* name;
    uint32_t lazy;
};
struct compiler_nvvm_request {
    uint32_t abi_version = 1;
    size_t count = 0;
    const compiler_nvvm_module* modules = nullptr;
    const char* architecture = "sm_75";
    uint32_t optimization = 3;
    uint32_t verify_only = 0;
    uint32_t lto = 0;
    uint32_t contract_fma = 1;
};
}
extern "C" CUDAVM_CORE_API cudavm::compiler_status cudavm_core_nvvm_compile_v1(
    const cudavm::compiler_nvvm_request*, cudavm::compiler_result**) noexcept;
