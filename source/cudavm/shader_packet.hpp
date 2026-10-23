#pragma once
#include <cstdint>
#include <algorithm>
#include <optional>
#include <span>
#include <type_traits>

namespace cudavm {
// Native packets transport integers and pointers as two little-endian 32-bit words.
// Global pointers retain the CUDA virtual address, even in untyped buffers.
struct packet_word64 { uint32_t lo, hi; };
constexpr packet_word64 split_packet_word64(uint64_t n) { return {uint32_t(n), uint32_t(n >> 32)}; }
constexpr uint64_t join_packet_word64(packet_word64 n) { return uint64_t(n.lo) | (uint64_t(n.hi) << 32); }

inline constexpr uint32_t packet_private_high_word = 0xffffffffu;
inline constexpr uint32_t packet_shared_tag = 0x40000000u;
inline constexpr uint32_t packet_local_tag = 0x80000000u;
inline constexpr uint32_t packet_parameter_tag = 0xc0000000u;
inline constexpr uint32_t packet_private_stack_bytes = 16384;
inline constexpr uint32_t packet_arena_header_bytes = 32;
inline constexpr uint32_t packet_stack_fault_word = 3;
inline constexpr uint32_t packet_stack_base_word = 4;
constexpr packet_word64 make_packet_private_pointer(uint32_t space, uint32_t offset) { return {space | offset, packet_private_high_word}; }

// Packet words 0..5: grid and block dimensions. Word 6: byte offset of this
// table, word 7: record count. Parameter bytes still begin at byte 32.
// Sorted by virtual base. Binding zero selects the arena; other bindings select
// persistent native backing storage. Translation never changes CUDA pointer bits.
inline constexpr uint32_t packet_memory_abi = 2;
inline constexpr uint32_t packet_direct_bindings = 2;
inline constexpr uint32_t packet_resource_count = 2 + packet_direct_bindings;
inline constexpr unsigned packet_binding_shift = 61;
inline constexpr uint64_t packet_offset_mask = (uint64_t(1) << packet_binding_shift) - 1;
inline constexpr uint64_t packet_arena_capacity = uint64_t(1) << 32;
struct packet_allocation {
    packet_word64 address;
    uint64_t size, offset;
    uint32_t binding = 0, reserved = 0;
};
inline constexpr uint32_t packet_parameter_header_bytes = 32;
inline constexpr uint32_t packet_allocation_words = 8;
static_assert(sizeof(packet_word64) == 8 && sizeof(packet_allocation) == 32);
static_assert(std::is_trivially_copyable_v<packet_allocation>);

// Host contract for the shader's address resolver. Bounds describe accesses,
// not pointer values: one-past pointers can be stored/compared but not resolved.
inline std::optional<uint64_t> resolve_packet_address(packet_word64 pointer, std::span<const packet_allocation> table) {
    if (pointer.hi == packet_private_high_word) return pointer.lo;
    auto address = join_packet_word64(pointer);
    if (!address) return {};
    auto at = std::upper_bound(table.begin(), table.end(), address,
        [](uint64_t value, const auto& record) { return value < join_packet_word64(record.address); });
    if (at == table.begin()) return {};
    const auto& record = *--at;
    auto delta = address - join_packet_word64(record.address);
    if (delta < record.size && record.offset <= packet_offset_mask && delta <= packet_offset_mask - record.offset &&
        record.binding <= packet_direct_bindings)
        return (uint64_t(record.binding) << packet_binding_shift) | (record.offset + delta);
    return {};
}
}
