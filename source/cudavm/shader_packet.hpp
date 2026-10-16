#pragma once
#include <cstdint>
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
constexpr packet_word64 make_packet_private_pointer(uint32_t space, uint32_t offset) { return {space | offset, packet_private_high_word}; }

// Packet words 0..5: grid and block dimensions. Word 6: byte offset of this
// table, word 7: record count. Parameter bytes still begin at byte 32.
// Allocation extents and arena offsets are bounded below 1 GiB by the runtime.
struct packet_allocation { packet_word64 address; uint32_t size, offset; };
inline constexpr uint32_t packet_parameter_header_bytes = 32;
inline constexpr uint32_t packet_allocation_words = 4;
static_assert(sizeof(packet_word64) == 8 && sizeof(packet_allocation) == 16);
static_assert(std::is_trivially_copyable_v<packet_allocation>);

// Host contract for the shader's address resolver. Bounds describe accesses,
// not pointer values: one-past pointers can be stored/compared but not resolved.
inline std::optional<uint32_t> resolve_packet_address(packet_word64 pointer, std::span<const packet_allocation> table) {
    if (pointer.hi == packet_private_high_word) return pointer.lo;
    auto address = join_packet_word64(pointer);
    if (!address) return {};
    for (auto record : table) {
        auto base = join_packet_word64(record.address);
        if (address >= base && address - base < record.size)
            return record.offset + uint32_t(address - base);
    }
    return {};
}
}
