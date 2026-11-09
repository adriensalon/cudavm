#include "nvidia_lto_import.hpp"
#include "program.hpp"
#include "compiler_profiles.hpp"
#include <algorithm>
#include <array>
#include <cstdint>

namespace cudavm {
// Byte substitution for the CUDA 12.4 NVVM LTO wrapper codec. This encoding
// belongs to the NVIDIA input format, not cudavm's portable program envelope.
constexpr std::array<unsigned char, 256> nvidia_lto_substitution{
    0x64, 0x80, 0x0d, 0x1e, 0xe2, 0xf6, 0xef, 0x16, 0x98, 0xff, 0x29, 0x3a, 0x5e, 0xcf, 0x28, 0xad,
    0x84, 0x45, 0xfc, 0xce, 0x65, 0xab, 0xe3, 0xac, 0xeb, 0x97, 0x49, 0x92, 0xc3, 0x13, 0xa8, 0xf7,
    0xdd, 0x7c, 0xd0, 0xde, 0xf8, 0x78, 0x72, 0xec, 0xa5, 0x12, 0x94, 0x2b, 0x39, 0x53, 0x9a, 0x90,
    0x04, 0x3f, 0xae, 0x99, 0x32, 0x75, 0xbf, 0xc1, 0x02, 0x91, 0xdb, 0xfe, 0x7d, 0xd6, 0xb5, 0xf0,
    0x3d, 0x3c, 0xb3, 0xb6, 0x00, 0xbc, 0x76, 0xa2, 0x9c, 0xe1, 0x50, 0x85, 0x09, 0x4a, 0x66, 0x2d,
    0xb7, 0x6a, 0xd5, 0x79, 0xea, 0x3e, 0x1b, 0xd1, 0xa3, 0x7b, 0x71, 0x95, 0x59, 0xb2, 0x87, 0x8b,
    0x5c, 0x9f, 0x81, 0x18, 0x27, 0xcb, 0x4b, 0x93, 0x11, 0x2a, 0x6b, 0xbd, 0xa0, 0xc8, 0xf5, 0x19,
    0xf2, 0x1f, 0xc2, 0x40, 0x6c, 0xdf, 0x03, 0xda, 0x21, 0x55, 0xb1, 0x82, 0x63, 0x89, 0x77, 0xe6,
    0xa9, 0xc9, 0x73, 0x24, 0xc6, 0xd9, 0xe4, 0x37, 0x17, 0x4f, 0xf4, 0x68, 0xc7, 0x07, 0x86, 0xaf,
    0xfa, 0xa6, 0xb8, 0xdc, 0x88, 0x0a, 0x47, 0x8e, 0xe7, 0x14, 0x1a, 0x83, 0x74, 0x9e, 0x35, 0x7f,
    0xfd, 0x52, 0xe0, 0xa1, 0x41, 0x4c, 0x0b, 0x33, 0x44, 0x8f, 0xf3, 0xbb, 0x08, 0x58, 0x7a, 0x0c,
    0xf1, 0x3b, 0xe5, 0x9b, 0x26, 0xd4, 0x5d, 0xc0, 0xf9, 0x7e, 0xa7, 0x60, 0xe8, 0x51, 0x25, 0x34,
    0xd7, 0x0f, 0x4d, 0x5a, 0xd3, 0x36, 0x1d, 0xc4, 0x8a, 0x2f, 0x20, 0x62, 0xb0, 0xaa, 0x61, 0x8d,
    0x8c, 0x69, 0x01, 0x48, 0x38, 0xed, 0xd8, 0x6e, 0x42, 0xcc, 0x05, 0x2e, 0x22, 0x10, 0x5b, 0x06,
    0x4e, 0x23, 0x46, 0x0e, 0xba, 0xe9, 0x5f, 0xee, 0xb9, 0xb4, 0x30, 0xa4, 0x54, 0xbe, 0x1c, 0x57,
    0xcd, 0xfb, 0x96, 0x9d, 0x31, 0x67, 0xc5, 0x70, 0x15, 0x43, 0x6d, 0x56, 0x2c, 0xd2, 0x6f, 0xca,
};
static void require_nvidia_lto(bool valid, const char* reason, bool unsupported = false) {
    if (!valid) throw compiler_error(unsupported ? compiler_status::unsupported : compiler_status::invalid,
        (std::string("NVIDIA LTO: ") + reason).c_str());
}
nvidia_lto_program decode_nvidia_lto(std::string_view bytes) {
    require_nvidia_lto(bytes.size() >= 24 && bytes.size() <= (size_t(1) << 26), "truncated or oversized wrapper");
    require_nvidia_lto(has_nvidia_lto_ir(bytes), "invalid wrapper magic");
    require_nvidia_lto(bytes.substr(4, 8) == std::string_view("\x01\x3d\x02\x47\x03\x01\x07\x00", 8),
        "unsupported wrapper revision", true);
    auto word = [&](size_t offset, unsigned width) {
        require_nvidia_lto(offset <= bytes.size() && width <= bytes.size() - offset, "truncated wrapper field");
        uint32_t value = 0;
        for (unsigned i = 0; i < width; ++i) value |= uint32_t(static_cast<unsigned char>(bytes[offset + i])) << (i * 8);
        return value;
    };
    require_nvidia_lto(word(12, 2) == 24 && word(14, 2) == 1, "unsupported wrapper layout", true);
    const size_t auxiliary = word(16, 4), payload = word(20, 4);
    require_nvidia_lto(auxiliary >= 28 && auxiliary <= payload && payload <= bytes.size()
        && !(auxiliary % 4) && !(payload % 4), "invalid wrapper offsets");
    require_nvidia_lto(payload - auxiliary >= 4 && bytes.size() - payload >= 4 && !((bytes.size() - payload) % 4),
        "truncated wrapper payload");
    uint32_t architecture = 0, level = 0, seed_offset = 0;
    unsigned seen = 0; bool ended = false;
    for (size_t cursor = 24; cursor < auxiliary;) {
        require_nvidia_lto(auxiliary - cursor >= 4, "truncated metadata record");
        const auto tag = word(cursor, 2); auto value = word(cursor + 2, 2); cursor += 4;
        if (!tag) {
            require_nvidia_lto(!value && cursor == auxiliary, "invalid metadata terminator"); ended = true; break;
        }
        if (value == 0xffff) {
            require_nvidia_lto(auxiliary - cursor >= 4, "truncated extended metadata value"); value = word(cursor, 4); cursor += 4;
        }
        const unsigned bit = tag == 1 ? 1 : tag == 2 ? 2 : tag == 99 ? 4 : 0;
        require_nvidia_lto(bit != 0, "unsupported metadata record", true);
        require_nvidia_lto(!(seen & bit), "duplicate metadata record"); seen |= bit;
        if (tag == 1) architecture = value; else if (tag == 2) level = value; else seed_offset = value;
    }
    require_nvidia_lto(ended && seen == 7, "missing metadata record or terminator");
    require_nvidia_lto(level == 2, "unsupported IR level", true);
    require_nvidia_lto(architecture && architecture % 10 == 0, "invalid source architecture");
    const auto target = "compute_" + std::to_string(architecture / 10);
    require_nvidia_lto(std::find(nvvm_compile_architectures.begin(), nvvm_compile_architectures.end(), target)
        != nvvm_compile_architectures.end(), "unsupported source architecture", true);
    require_nvidia_lto(!(seed_offset % 4) && seed_offset <= payload - auxiliary - 4, "invalid codec seed offset");
    nvidia_lto_program result{std::string(bytes.substr(payload)), "sm_" + std::to_string(architecture / 10)};
    uint32_t seed = word(auxiliary + seed_offset, 4);
    if (seed) {
        unsigned char previous = static_cast<unsigned char>(~seed);
        unsigned remaining = 1; uint32_t key = 0;
        for (auto& byte : result.bitcode) {
            if (!--remaining) { seed = seed * 0x41c64e6du + 0x3039u; key = seed; remaining = 4; }
            else key >>= 8;
            const auto cipher = static_cast<unsigned char>(byte);
            byte = static_cast<char>(nvidia_lto_substitution[cipher ^ previous] ^ static_cast<unsigned char>(key)); previous = cipher;
        }
    }
    require_nvidia_lto(result.bitcode.starts_with(std::string_view("BC\xc0\xde", 4)), "invalid decoded bitcode signature");
    return result;
}
}
