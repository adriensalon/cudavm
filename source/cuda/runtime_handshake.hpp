#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace cudavm {
// MD2's RFC 1319 substitution permutation. The CUDA 12.4 host handshake uses
// 16-byte HMAC pads (rather than a cryptographic service exposed by our SDK).
inline constexpr std::array<uint8_t, 256> runtime_md2_substitution{
    41,46,67,201,162,216,124,1,61,54,84,161,236,240,6,19,
    98,167,5,243,192,199,115,140,152,147,43,217,188,76,130,202,
    30,155,87,60,253,212,224,22,103,66,111,24,138,23,229,18,
    190,78,196,214,218,158,222,73,160,251,245,142,187,47,238,122,
    169,104,121,145,21,178,7,63,148,194,16,137,11,34,95,33,
    128,127,93,154,90,144,50,39,53,62,204,231,191,247,151,3,
    255,25,48,179,72,165,181,209,215,94,146,42,172,86,170,198,
    79,184,56,210,150,164,125,182,118,252,107,226,156,116,4,241,
    69,157,112,89,100,113,135,32,134,91,207,101,230,45,168,2,
    27,96,37,173,174,176,185,246,28,70,97,105,52,64,126,15,
    85,71,163,35,221,81,175,58,195,92,249,206,186,197,234,38,
    44,83,13,110,133,40,132,9,211,223,205,244,65,129,77,82,
    106,220,55,200,108,193,171,250,36,225,123,8,12,189,177,74,
    120,136,149,139,227,99,232,109,233,203,213,254,59,0,29,57,
    242,239,183,14,102,88,208,228,166,119,114,248,235,117,75,10,
    49,68,80,180,143,237,31,26,219,153,141,51,159,17,131,20};

struct runtime_md2 {
    std::array<uint8_t, 48> state{};
    std::array<uint8_t, 16> checksum{};
    uint8_t position = 0, last = 0;
    void append(uint8_t byte) {
        state[16 + position] = byte;
        state[32 + position] = byte ^ state[position];
        last = checksum[position] ^= runtime_md2_substitution[byte ^ last];
        position = (position + 1) & 15;
        if (!position) {
            uint8_t accumulator = 0;
            for (unsigned round = 0; round < 18; ++round) {
                for (auto& value : state) accumulator = value ^= runtime_md2_substitution[accumulator];
                accumulator = uint8_t(accumulator + round);
            }
        }
    }
    void append(std::span<const uint8_t> bytes) { for (auto byte : bytes) append(byte); }
    std::array<uint8_t, 16> finish() {
        const auto padding = uint8_t(16 - position);
        for (unsigned i = 0; i < padding; ++i) append(padding);
        const auto final_checksum = checksum;
        append(final_checksum);
        std::array<uint8_t, 16> result{};
        for (size_t i = 0; i < result.size(); ++i) result[i] = state[i];
        return result;
    }
};

inline runtime_md2 runtime_handshake_begin() {
    // Versioned protocol material recovered from the pinned producer's
    // calc116Hash, not a key used for application or network authentication.
    constexpr std::array<uint8_t, 16> key{0x14,0x6a,0xdd,0xae,0x53,0xa9,0xa7,0x52,0xaa,0x08,0x41,0x36,0x0b,0xf5,0x5a,0x9f};
    runtime_md2 result;
    for (auto byte : key) result.append(uint8_t(byte ^ 0x36));
    return result;
}
inline std::array<uint8_t, 16> runtime_handshake_finish(runtime_md2 inner) {
    constexpr std::array<uint8_t, 16> key{0x14,0x6a,0xdd,0xae,0x53,0xa9,0xa7,0x52,0xaa,0x08,0x41,0x36,0x0b,0xf5,0x5a,0x9f};
    runtime_md2 outer;
    for (auto byte : key) outer.append(uint8_t(byte ^ 0x5c));
    outer.append(inner.finish());
    return outer.finish();
}
template<class Integer> void runtime_handshake_append(runtime_md2& hash, Integer value) {
    for (size_t i = 0; i < sizeof(Integer); ++i) {
        hash.append(uint8_t(value)); value >>= 8;
    }
}
}
