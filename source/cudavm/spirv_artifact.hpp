#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <set>
#include <map>
#include <algorithm>
namespace cudavm {
struct shader_artifact { std::string symbol, code, diagnostic, opencl_code, opencl_diagnostic; };
inline constexpr std::string_view spirv_artifact_begin = "// CUDAVM-NATIVE-SPIRV-V1-BEGIN\n";
inline constexpr std::string_view spirv_artifact_end = "// CUDAVM-NATIVE-SPIRV-V1-END";
inline constexpr std::string_view spirv_artifact_begin_v2 = "// CUDAVM-NATIVE-SPIRV-V2-BEGIN\n";
inline constexpr std::string_view spirv_artifact_end_v2 = "// CUDAVM-NATIVE-SPIRV-V2-END";
inline std::string encode_spirv_artifacts(const std::vector<shader_artifact>& entries) {
    std::string bytes;
    auto number = [&](uint32_t value) { for (unsigned i = 0; i < 4; ++i) bytes += char(value >> (i * 8)); };
    auto field = [&](const std::string& value) { if (value.size() > 64 * 1024 * 1024) throw std::runtime_error("native shader field exceeds transport limit"); number(uint32_t(value.size())); bytes += value; };
    if (entries.size() > 4096) throw std::runtime_error("too many native shader entries");
    number(uint32_t(entries.size()));
    for (const auto& item : entries) { field(item.symbol); field(item.code); field(item.diagnostic); field(item.opencl_code); field(item.opencl_diagnostic); }
    std::string result(spirv_artifact_begin_v2); constexpr char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (i % 96 == 0) result += "// ";
        auto value = static_cast<unsigned char>(bytes[i]); result += hex[value >> 4]; result += hex[value & 15];
        if (i % 96 == 95 || i + 1 == bytes.size()) result += '\n';
    }
    return result + std::string(spirv_artifact_end_v2) + '\n';
}
inline std::vector<shader_artifact> decode_spirv_artifacts(std::string_view image) {
    bool v2 = image.find(spirv_artifact_begin_v2) != image.npos;
    auto marker = v2 ? spirv_artifact_begin_v2 : spirv_artifact_begin;
    auto start = image.find(marker); if (start == image.npos) return {};
    start += marker.size(); auto stop = image.find(v2 ? spirv_artifact_end_v2 : spirv_artifact_end, start);
    if (stop == image.npos || stop - start > 136 * 1024 * 1024) throw std::runtime_error("invalid native shader transport size");
    std::string bytes; int high = -1;
    for (auto c : image.substr(start, stop - start)) {
        if (c == '/' || c == ' ' || c == '\r' || c == '\n') continue;
        int digit = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
        if (digit < 0) throw std::runtime_error("invalid native shader encoding");
        if (high < 0) high = digit; else { bytes += char((high << 4) | digit); high = -1; }
    }
    if (high >= 0) throw std::runtime_error("truncated native shader byte");
    size_t cursor = 0;
    auto number = [&] { if (bytes.size() - cursor < 4) throw std::runtime_error("truncated native shader field"); uint32_t value = 0; for (unsigned i = 0; i < 4; ++i) value |= uint32_t(static_cast<unsigned char>(bytes[cursor++])) << (i * 8); return value; };
    auto field = [&] { auto size = number(); if (size > bytes.size() - cursor) throw std::runtime_error("truncated native shader string"); auto value = bytes.substr(cursor, size); cursor += size; return value; };
    auto count = number(); if (count > 4096) throw std::runtime_error("too many native shader entries");
    std::vector<shader_artifact> entries;
    for (uint32_t i = 0; i < count; ++i) {
        shader_artifact item{field(), field(), field(), {}, {}};
        if (v2) { item.opencl_code = field(); item.opencl_diagnostic = field(); }
        if (item.symbol.empty() || item.symbol.find('\0') != item.symbol.npos || (item.code.empty() && item.diagnostic.empty() && item.opencl_code.empty() && item.opencl_diagnostic.empty())) throw std::runtime_error("invalid native shader identity");
        entries.push_back(std::move(item));
    }
    if (cursor != bytes.size()) throw std::runtime_error("extra native shader transport data");
    return entries;
}
inline void validate_spirv_words(std::span<const uint32_t> code) {
    if (code.size() < 5 || code.size() > 16 * 1024 * 1024 || code[0] != 0x07230203u || !code[3] || code[4])
        throw std::runtime_error("invalid native SPIR-V header");
    for (size_t i = 5; i < code.size();) {
        auto count = code[i] >> 16, opcode = code[i] & 0xffffu;
        unsigned minimum = 1;
        switch (opcode) {
        case 5: case 6: case 15: minimum = opcode == 15 ? 4 : opcode == 6 ? 4 : 3; break;
        case 16: case 71: case 23: case 33: case 43: case 44: case 46: case 50: case 55: case 62: case 63: case 247: minimum = 3; break;
        case 21: case 28: case 32: case 59: case 61: case 65: case 66: case 83: case 224: minimum = 4; break;
        case 72: case 64: case 81: case 250: case 251: minimum = 4; break;
        case 12: case 54: case 79: case 82: minimum = 5; break;
        case 17: case 19: case 20: case 22: case 30: case 41: case 42: case 248: case 249: case 254: minimum = opcode == 22 || opcode == 41 || opcode == 42 ? 3 : 2; break;
        case 246: minimum = 4; break;
        case 225: minimum = 3; break;
        default: break;
        }
        if (count < minimum || count > code.size() - i) throw std::runtime_error("invalid native SPIR-V instruction");
        if (opcode == 16 && code[i + 2] == 17 && count != 6) throw std::runtime_error("invalid native SPIR-V workgroup size");
        i += count;
    }
}
inline std::vector<uint32_t> decode_spirv_words(std::string_view code, const unsigned* block = nullptr, bool require_local_size = true) {
    if (code.size() < 20 || code.size() % 4 || code.size() > 64 * 1024 * 1024) throw std::runtime_error("invalid native SPIR-V size");
    std::vector<uint32_t> result(code.size() / 4); std::memcpy(result.data(), code.data(), code.size());
    validate_spirv_words(result);
    bool local_size = false;
    for (size_t i = 5; i < result.size();) {
        auto count = result[i] >> 16, opcode = result[i] & 0xffffu;
        if (opcode == 16 && count == 6 && result[i + 2] == 17) {
            if (local_size) throw std::runtime_error("ambiguous native SPIR-V workgroup size");
            local_size = true;
            if (block) for (unsigned axis = 0; axis < 3; ++axis) result[i + 3 + axis] = block[axis];
        }
        i += count;
    }
    if (require_local_size && !local_size) throw std::runtime_error("missing native SPIR-V workgroup size");
    return result;
}
// LLVM's resource-name strings survive as unused Function variables. Remove
// only their i8 definition closure; refuse conversion if any live operation
// actually consumes it. This avoids requiring shaderInt8 for string metadata.
inline std::vector<uint32_t> trim_spirv_resource_names(std::span<const uint32_t> input) {
    validate_spirv_words(input);
    std::set<uint32_t> dead; std::set<size_t> dropped;
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 5; i < input.size(); i += input[i] >> 16) {
            auto count = input[i] >> 16, op = input[i] & 0xffffu;
            bool remove = op == 21 && count == 4 && input[i + 2] == 8;
            if ((op == 23 || op == 28) && count >= 3) remove = dead.contains(input[i + 2]);
            if (op == 32 && count == 4) remove = dead.contains(input[i + 3]);
            if ((op == 30 || op == 33) && count >= 3) for (unsigned n = 2; n < count; ++n) remove |= dead.contains(input[i + n]);
            if (op >= 19 && op <= 33 && remove) {
                changed |= dead.insert(input[i + 1]).second; dropped.insert(i);
            }
            if ((op == 43 || op == 44 || op == 46 || op == 59) && count >= 3 && dead.contains(input[i + 1])) {
                changed |= dead.insert(input[i + 2]).second; dropped.insert(i);
            }
        }
    }
    if (dead.empty()) return {input.begin(), input.end()};
    auto check = [&](uint32_t id) { if (dead.contains(id)) throw std::runtime_error("native shader uses live i8 values; this profile requires fallback"); };
    std::vector<uint32_t> result(input.begin(), input.begin() + 5);
    for (size_t i = 5; i < input.size(); i += input[i] >> 16) {
        auto count = input[i] >> 16, op = input[i] & 0xffffu;
        if (dropped.contains(i) || (op == 17 && count == 2 && input[i + 1] == 39)) continue;
        if ((op == 5 || op == 71 || op == 72) && count > 1 && dead.contains(input[i + 1])) continue;
        switch (op) {
        case 0: case 3: case 4: case 5: case 6: case 7: case 8: case 11: case 14: case 17: case 252: case 255:
        case 19: case 20: case 21: case 22: case 248: case 56: case 253: break;
        case 16: case 71: case 72: check(input[i + 1]); break;
        case 15: {
            check(input[i + 2]); unsigned n = 3;
            for (; n < count; ++n) { auto word = input[i + n]; if (!(word & 255) || !(word & 0xff00) || !(word & 0xff0000) || !(word & 0xff000000)) { ++n; break; } }
            for (; n < count; ++n) check(input[i + n]); break;
        }
        case 23: case 28: check(input[i + 2]); if (op == 28) check(input[i + 3]); break;
        case 32: check(input[i + 3]); break;
        case 30: case 33: for (unsigned n = 2; n < count; ++n) check(input[i + n]); break;
        case 43: case 46: case 50: case 41: case 42: check(input[i + 1]); break;
        case 44: check(input[i + 1]); for (unsigned n = 3; n < count; ++n) check(input[i + n]); break;
        case 54: check(input[i + 1]); check(input[i + 4]); break;
        case 55: check(input[i + 1]); break;
        case 59: check(input[i + 1]); if (count == 5) check(input[i + 4]); break;
        case 61: check(input[i + 1]); check(input[i + 3]); break;
        case 62: check(input[i + 1]); check(input[i + 2]); break;
        case 63: case 64: check(input[i + 1]); check(input[i + 2]); if (op == 64) check(input[i + 3]); break;
        case 79: check(input[i + 1]); check(input[i + 3]); check(input[i + 4]); break;
        case 81: check(input[i + 1]); check(input[i + 3]); break;
        case 82: check(input[i + 1]); check(input[i + 3]); check(input[i + 4]); break;
        case 12: check(input[i + 1]); check(input[i + 3]); for (unsigned n = 5; n < count; ++n) check(input[i + n]); break;
        case 246: check(input[i + 1]); check(input[i + 2]); break;
        case 247: case 249: case 254: check(input[i + 1]); break;
        case 250: for (unsigned n = 1; n < std::min(4u, count); ++n) check(input[i + n]); break;
        case 251: check(input[i + 1]); check(input[i + 2]); for (unsigned n = 4; n < count; n += 2) check(input[i + n]); break;
        case 224: case 225: for (unsigned n = 1; n < count; ++n) check(input[i + n]); break;
        default: if (count > 1) check(input[i + 1]); for (unsigned n = 3; n < count; ++n) check(input[i + n]); break;
        }
        result.insert(result.end(), input.begin() + i, input.begin() + i + count);
    }
    return result;
}
inline std::vector<uint32_t> adapt_spirv_for_opengl(std::span<const uint32_t> input) {
    auto code = trim_spirv_resource_names(input);
    std::set<uint32_t> buffers, resource_pointers, input_output;
    std::map<uint32_t, uint32_t> pointer_spaces;
    for (size_t i = 5; i < code.size(); i += code[i] >> 16) {
        auto count = code[i] >> 16, op = code[i] & 0xffffu;
        if (op == 32 && count == 4) {
            pointer_spaces[code[i + 1]] = code[i + 2];
            if (code[i + 2] == 12) { buffers.insert(code[i + 3]); resource_pointers.insert(code[i + 1]); code[i + 2] = 2; }
        }
        if (op == 59 && count >= 4 && (code[i + 3] == 1 || code[i + 3] == 3)) input_output.insert(code[i + 2]);
    }
    std::map<uint32_t, uint32_t> aliases;
    for (size_t i = 5; i < code.size(); i += code[i] >> 16)
        if ((code[i] & 0xffffu) == 83 && (code[i] >> 16) == 4 && resource_pointers.contains(code[i + 1])) {
            auto original = code[i + 3]; while (aliases.contains(original)) original = aliases.at(original);
            aliases[code[i + 2]] = original;
        }
    std::vector<uint32_t> result(code.begin(), code.begin() + 5); result[1] = 0x00010000u;
    for (size_t i = 5; i < code.size(); i += code[i] >> 16) {
        auto count = code[i] >> 16, op = code[i] & 0xffffu;
        if (op == 83 && count == 4 && resource_pointers.contains(code[i + 1])) {
            continue;
        }
        if (op == 5 && aliases.contains(code[i + 1])) continue;
        std::vector<uint32_t> instruction(code.begin() + i, code.begin() + i + count);
        if (op == 71 && count >= 3 && buffers.contains(instruction[1]) && instruction[2] == 2) instruction[2] = 3;
        if (op == 59 && count >= 4 && instruction[3] == 12) instruction[3] = 2;
        if (op == 15) {
            unsigned name_end = 3;
            for (; name_end < count; ++name_end) { auto word = instruction[name_end]; if (!(word & 255) || !(word & 0xff00) || !(word & 0xff0000) || !(word & 0xff000000)) { ++name_end; break; } }
            instruction.erase(std::remove_if(instruction.begin() + name_end, instruction.end(), [&](uint32_t id) { return !input_output.contains(id); }), instruction.end());
            instruction[0] = (uint32_t(instruction.size()) << 16) | op;
        }
        // Resource aliases occur only as the base of an access chain or the
        // input of another resource copy; descriptor variables retain identity.
        if ((op == 65 || op == 66 || op == 83) && count >= 4 && aliases.contains(instruction[3])) instruction[3] = aliases.at(instruction[3]);
        result.insert(result.end(), instruction.begin(), instruction.end());
    }
    return result;
}
}
