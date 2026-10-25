#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cudavm {
// LLVM/Driver-independent reader for NVIDIA v1 containers, entry version 0x101.
// The member images are alternatives, not separate translation units.
inline constexpr uint32_t nvidia_fatbin_magic = 0xba55ed50;
inline constexpr size_t nvidia_image_limit = size_t(1) << 30;
struct nvidia_container_error : std::runtime_error {
    bool unsupported;
    explicit nvidia_container_error(const char* message, bool unsupported = false)
        : std::runtime_error(message), unsupported(unsupported) {}
};
struct nvidia_container_reader {
    std::string_view bytes;
    uint64_t number(size_t offset, unsigned width = 4) const {
        if (offset > bytes.size() || width > bytes.size() - offset)
            throw nvidia_container_error("truncated NVIDIA fatbinary field");
        uint64_t value = 0;
        for (unsigned i = 0; i < width; ++i) value |= uint64_t(static_cast<unsigned char>(bytes[offset + i])) << (8 * i);
        return value;
    }
};
inline bool has_nvidia_fatbin(std::string_view bytes) {
    return bytes.size() >= 4 && nvidia_container_reader{bytes}.number(0) == nvidia_fatbin_magic;
}
inline bool has_nvidia_fatbin(const void* image) {
    if (!image) return false;
    uint32_t magic; std::memcpy(&magic, image, sizeof(magic)); return magic == nvidia_fatbin_magic;
}
inline bool has_nvidia_cubin(std::string_view bytes) { return bytes.starts_with("\x7f" "ELF"); }
inline bool has_nvidia_cubin(const void* image) {
    return image && std::memcmp(image, "\x7f" "ELF", 4) == 0;
}
inline size_t nvidia_fatbin_extent(std::string_view bytes) {
    nvidia_container_reader reader{bytes};
    if (!has_nvidia_fatbin(bytes)) throw nvidia_container_error("invalid NVIDIA fatbinary magic");
    if (reader.number(4, 2) != 1) throw nvidia_container_error("unsupported NVIDIA fatbinary version", true);
    auto header = reader.number(6, 2), payload = reader.number(8, 8);
    if (header < 16 || header > 4096 || payload > nvidia_image_limit - header)
        throw nvidia_container_error("invalid or oversized NVIDIA fatbinary extent");
    return size_t(header + payload);
}
inline std::string_view nvidia_fatbin_memory(const void* image) {
    if (!image) throw nvidia_container_error("null NVIDIA fatbinary");
    auto* bytes = static_cast<const char*>(image);
    return {bytes, nvidia_fatbin_extent({bytes, 16})};
}
inline std::string decompress_nvidia_fatbin(std::string_view input, size_t size) {
    if (size > nvidia_image_limit) throw nvidia_container_error("oversized compressed NVIDIA image");
    std::string output; output.reserve(size); size_t position = 0;
    auto byte = [&] {
        if (position == input.size()) throw nvidia_container_error("truncated NVIDIA compression token");
        return static_cast<unsigned char>(input[position++]);
    };
    auto length = [&](size_t value) {
        if (value == 15) {
            unsigned next;
            do {
                next = byte();
                if (value > size || next > size - value) throw nvidia_container_error("NVIDIA compression length overflow");
                value += next;
            } while (next == 255);
        }
        return value;
    };
    while (position < input.size()) {
        auto token = byte(); auto literals = length(token >> 4);
        if (literals > input.size() - position || literals > size - output.size())
            throw nvidia_container_error("invalid NVIDIA compression literals");
        output.append(input.substr(position, literals)); position += literals;
        if (position == input.size()) break;
        unsigned offset = byte(); offset |= unsigned(byte()) << 8;
        if (!offset || offset > output.size()) throw nvidia_container_error("invalid NVIDIA compression back reference");
        auto count = length(token & 15);
        if (count > size - output.size() || 4 > size - output.size() - count)
            throw nvidia_container_error("NVIDIA compression output overflow");
        for (size_t i = 0; i < count + 4; ++i) output += output[output.size() - offset];
    }
    if (output.size() != size) throw nvidia_container_error("NVIDIA decompression size mismatch");
    return output;
}
struct nvidia_fatbin_entry {
    unsigned kind, architecture, ptx_version;
    uint64_t flags;
    std::string code;
};
inline std::vector<nvidia_fatbin_entry> decode_nvidia_container(std::string_view bytes) {
    auto extent = nvidia_fatbin_extent(bytes);
    if (extent != bytes.size()) throw nvidia_container_error("NVIDIA fatbinary size mismatch");
    nvidia_container_reader reader{bytes};
    size_t position = size_t(reader.number(6, 2)), decoded = 0;
    std::vector<nvidia_fatbin_entry> entries;
    while (position < extent) {
        if (entries.size() == 4096 || extent - position < 64)
            throw nvidia_container_error("invalid NVIDIA fatbinary entry extent/count");
        auto kind = reader.number(position, 2), version = reader.number(position + 2, 2);
        auto header = reader.number(position + 4), packed = reader.number(position + 8, 8);
        if (version != 0x101) throw nvidia_container_error("unsupported NVIDIA fatbinary entry version", true);
        if (header < 64 || header > extent - position || packed > extent - position - header)
            throw nvidia_container_error("invalid NVIDIA fatbinary entry extent");
        auto arch = reader.number(position + 28), flags = reader.number(position + 40, 8);
        if (!arch || arch > 9999) throw nvidia_container_error("invalid NVIDIA fatbinary architecture");
        auto payload = bytes.substr(position + size_t(header), size_t(packed));
        auto size = flags & 0x2000 ? reader.number(position + 56, 8) : packed;
        if (!size || size > nvidia_image_limit - decoded) throw nvidia_container_error("oversized NVIDIA fatbinary payloads");
        std::string code;
        if (flags & 0x2000) {
            auto compressed = reader.number(position + 16);
            if (!compressed || compressed > packed) throw nvidia_container_error("invalid compressed NVIDIA fatbinary extent");
            code = decompress_nvidia_fatbin(payload.substr(0, size_t(compressed)), size_t(size));
        } else code.assign(payload);
        decoded += size_t(size);
        entries.push_back({unsigned(kind), unsigned(arch), unsigned(reader.number(position + 24)), flags, std::move(code)});
        position += size_t(header + packed);
    }
    return entries;
}
}
