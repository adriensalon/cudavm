#pragma once
#include <cstdint>
#include <string_view>
namespace llvm { class Module; class Function; }
namespace cudavm {
inline constexpr uint32_t device_service_heap = 1, device_service_output = 2, device_service_assert = 4;
inline constexpr uint64_t device_heap_address = 0xfffffffd00000000ull;
inline constexpr uint64_t device_output_address = 0xfffffffc00000000ull;
inline constexpr uint32_t device_output_record_bytes = 4096;
inline constexpr uint32_t device_output_header_bytes = 16;
inline constexpr uint64_t device_output_max_bytes = uint64_t(1) << 31;
inline constexpr uint32_t device_assert_fault_word = 5;
inline uint32_t device_service(std::string_view name) {
    if (name == "malloc" || name == "free" || name == "__nv_aligned_device_malloc" || name == "__nv_aligned_device_malloc_impl") return device_service_heap;
    if (name == "vprintf") return device_service_output;
    if (name == "__assertfail") return device_service_output | device_service_assert;
    return 0;
}
void link_shader_services(llvm::Module&);
void validate_device_services(const llvm::Module&);
uint32_t shader_service_requirements(const llvm::Function&);
}
