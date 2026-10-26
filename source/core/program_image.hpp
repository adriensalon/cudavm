#pragma once
#include "shader_packet.hpp"
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cudavm {
// No LLVM dependency: compiler-disabled runtimes can consume cached artifacts.
// Bump the IR/native profiles when the producer or semantic ABI becomes incompatible.
inline constexpr std::string_view program_image_magic = "CUDAVM-PROGRAM-IMAGE\n";
inline constexpr std::string_view program_image_ir_profile = "cudavm-llvm-22.1.4-ir-v1";
inline constexpr uint32_t program_image_native_profile = 2;
inline constexpr size_t program_image_limit = 256 * 1024 * 1024;
struct program_image_artifact {
    uint32_t backend = 0, profile = program_image_native_profile, memory_abi = packet_memory_abi;
    uint64_t program_hash = 0;
    std::string code;
};
struct program_image {
    std::string producer = std::string(program_image_ir_profile);
    std::string architecture, module_name, ir, metadata;
    std::map<std::string, std::string, std::less<>> lowered_names;
    std::vector<program_image_artifact> artifacts;
};
inline bool has_program_image(std::string_view bytes) { return bytes.starts_with(program_image_magic); }
inline bool has_program_image(const void* bytes) {
    return bytes && *static_cast<const char*>(bytes) == 'C' && has_program_image(std::string_view(static_cast<const char*>(bytes)));
}
// Corruption detection, not an authenticity signature. Include lengths to avoid
// ambiguous concatenations when binding cached code to its portable program.
inline uint64_t program_image_hash(std::string_view bytes, uint64_t value = 14695981039346656037ull) {
    for (unsigned char byte : bytes) { value ^= byte; value *= 1099511628211ull; }
    return value;
}
inline void append_image_number(std::string& bytes, uint64_t value, unsigned width = 4) {
    if (bytes.size() > program_image_limit - width) throw std::runtime_error("program image exceeds limit");
    for (unsigned i = 0; i < width; ++i) bytes += char(value >> (i * 8));
}
inline void append_image_field(std::string& bytes, std::string_view value) {
    if (value.size() > program_image_limit - 4 || bytes.size() > program_image_limit - 4 - value.size())
        throw std::runtime_error("program image field exceeds limit");
    append_image_number(bytes, value.size()); bytes += value;
}
inline std::string program_image_identity(const program_image& image) {
    std::string bytes;
    for (const auto* field : {&image.producer, &image.architecture, &image.module_name, &image.ir, &image.metadata})
        append_image_field(bytes, *field);
    append_image_number(bytes, image.lowered_names.size());
    for (const auto& [expression, name] : image.lowered_names) { append_image_field(bytes, expression); append_image_field(bytes, name); }
    return bytes;
}
inline std::string encode_program_image(const program_image& image) {
    if (image.lowered_names.size() > 4096 || image.artifacts.size() > 5) throw std::runtime_error("program image inventory exceeds limit");
    std::string bytes;
    append_image_number(bytes, 1); // Format version.
    bytes += program_image_identity(image);
    append_image_number(bytes, image.artifacts.size());
    for (const auto& artifact : image.artifacts) {
        append_image_number(bytes, artifact.backend); append_image_number(bytes, artifact.profile);
        append_image_number(bytes, artifact.memory_abi); append_image_number(bytes, artifact.program_hash, 8);
        append_image_field(bytes, artifact.code);
    }
    append_image_number(bytes, program_image_hash(bytes), 8);
    if (bytes.size() > program_image_limit) throw std::runtime_error("program image exceeds limit");
    std::string result(program_image_magic);
    constexpr char hex[] = "0123456789abcdef";
    result.reserve(result.size() + bytes.size() * 2 + 2);
    for (unsigned char byte : bytes) { result += hex[byte >> 4]; result += hex[byte & 15]; }
    result += '\n'; result += '\0'; // Part of the image extent, including caller copies.
    return result;
}
inline program_image decode_program_image(std::string_view encoded) {
    if (!encoded.empty() && encoded.back() == '\0') encoded.remove_suffix(1);
    if (!has_program_image(encoded)) throw std::runtime_error("invalid program image magic");
    encoded.remove_prefix(program_image_magic.size());
    if (encoded.empty() || encoded.back() != '\n') throw std::runtime_error("truncated program image");
    encoded.remove_suffix(1);
    if (encoded.size() % 2 || encoded.size() / 2 > program_image_limit) throw std::runtime_error("invalid program image extent");
    std::string bytes;
    bytes.reserve(encoded.size() / 2);
    auto digit = [](char c) -> unsigned {
        if (c >= '0' && c <= '9') return unsigned(c - '0');
        if (c >= 'a' && c <= 'f') return unsigned(c - 'a' + 10);
        throw std::runtime_error("invalid program image encoding");
    };
    for (size_t i = 0; i < encoded.size(); i += 2) bytes += char((digit(encoded[i]) << 4) | digit(encoded[i + 1]));
    size_t cursor = 0;
    auto number = [&](unsigned width = 4) {
        if (bytes.size() - cursor < width) throw std::runtime_error("truncated program image field");
        uint64_t value = 0;
        for (unsigned i = 0; i < width; ++i) value |= uint64_t(static_cast<unsigned char>(bytes[cursor++])) << (i * 8);
        return value;
    };
    auto field = [&] {
        auto size = number();
        if (size > bytes.size() - cursor) throw std::runtime_error("truncated program image string");
        auto value = bytes.substr(cursor, size_t(size)); cursor += size_t(size); return value;
    };
    if (bytes.size() < 12) throw std::runtime_error("truncated program image header");
    cursor = bytes.size() - 8;
    auto checksum = number(8);
    if (checksum != program_image_hash(std::string_view(bytes).substr(0, bytes.size() - 8))) throw std::runtime_error("program image checksum mismatch");
    cursor = 0;
    if (number() != 1) throw std::runtime_error("unsupported program image version");
    program_image image;
    image.producer = field(); image.architecture = field(); image.module_name = field(); image.ir = field(); image.metadata = field();
    auto identity = [](const std::string& value) { return !value.empty() && value.find('\0') == value.npos; };
    if (!identity(image.producer) || !identity(image.architecture) || image.ir.empty() || image.metadata.empty())
        throw std::runtime_error("incomplete program image");
    auto count = number();
    if (count > 4096) throw std::runtime_error("too many program image name expressions");
    for (uint64_t i = 0; i < count; ++i) {
        auto expression = field(), name = field();
        if (!identity(expression) || !identity(name) || !image.lowered_names.emplace(expression, name).second)
            throw std::runtime_error("invalid program image name expression");
    }
    auto hash = program_image_hash(program_image_identity(image));
    count = number();
    if (count > 5) throw std::runtime_error("too many program image artifacts");
    std::set<uint32_t> backends;
    for (uint64_t i = 0; i < count; ++i) {
        program_image_artifact artifact;
        artifact.backend = uint32_t(number()); artifact.profile = uint32_t(number()); artifact.memory_abi = uint32_t(number());
        artifact.program_hash = number(8); artifact.code = field();
        if (artifact.backend > 4 || !backends.insert(artifact.backend).second || artifact.program_hash != hash
            || !identity(artifact.code)) throw std::runtime_error("invalid program image artifact binding");
        image.artifacts.push_back(std::move(artifact));
    }
    if (cursor != bytes.size() - 8) throw std::runtime_error("extra program image data");
    return image;
}
inline const program_image_artifact* find_program_image_artifact(const program_image& image, unsigned backend) {
    if (image.producer != program_image_ir_profile) return nullptr;
    for (const auto& artifact : image.artifacts)
        if (artifact.backend == backend && artifact.profile == program_image_native_profile && artifact.memory_abi == packet_memory_abi) return &artifact;
    return nullptr;
}
}
