#pragma once
#include "result_api.hpp"

namespace cudavm {
// Ordinary objects carry framed raw TU bitcode in a .cudavm section. Completed
// device-link objects carry a completion record. module is written only when a
// link is needed and owns registration over a self-contained portable image.
struct compiler_object_link_request {
    uint32_t abi_version;
    size_t input_count;
    const char* const* inputs;
    const char* module;
    const char* architecture;
    uint32_t optimization_level;
    uint32_t skip_completed;
};
}
extern "C" {
CUDAVM_CORE_API int cudavm_core_link_objects_v1(
    const cudavm::compiler_object_link_request*, cudavm::compiler_result**) noexcept;
CUDAVM_CORE_API int cudavm_core_device_runtime_v1(cudavm::compiler_result**) noexcept;
}
