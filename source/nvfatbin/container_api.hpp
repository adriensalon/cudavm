#pragma once
#include "../core/program_image.hpp"
#include "nvidia_container_api.hpp"
#include <algorithm>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace cudavm {
// Shared, LLVM-independent container contract for packagers, core and Driver.
// Payloads are alternative images of one program, not separate link inputs.
inline constexpr std::string_view fatbin_magic = "CVMFATB1";
inline constexpr size_t fatbin_limit = size_t(1) << 30;
enum class fatbin_kind : uint32_t { ptx = 1, cubin = 2, ltoir = 3 };
struct fatbin_record {
    fatbin_kind kind;
    uint32_t architecture;
    std::string identifier, code;
};
struct fatbin_compression { bool enabled = true, force = false; unsigned threshold = 4; };
inline std::vector<std::string_view> fatbin_ptx_header(std::string_view text) {
    std::vector<std::string_view> tokens;
    for (size_t i = 0; i < text.size() && tokens.size() < 6;) {
        if (std::isspace(static_cast<unsigned char>(text[i]))) { ++i; continue; }
        if (text.substr(i).starts_with("//")) { auto end = text.find('\n', i + 2); i = end == text.npos ? text.size() : end + 1; continue; }
        if (text.substr(i).starts_with("/*")) { auto end = text.find("*/", i + 2); if (end == text.npos) break; i = end + 2; continue; }
        size_t begin = i;
        while (i < text.size() && (std::isalnum(static_cast<unsigned char>(text[i])) || text[i] == '.' || text[i] == '_')) ++i;
        if (i == begin) break;
        tokens.push_back(text.substr(begin, i - begin));
    }
    return tokens;
}
inline unsigned fatbin_elf_architecture(std::string_view);
inline bool has_portable_fatbin(std::string_view bytes) { return bytes.starts_with(fatbin_magic); }
inline bool has_portable_fatbin(const void* bytes) {
    if (!bytes) return false;
    auto* text = static_cast<const char*>(bytes);
    for (size_t i = 0; i < fatbin_magic.size(); ++i) if (text[i] != fatbin_magic[i]) return false;
    return true;
}
inline bool has_fatbin(std::string_view bytes) { return has_portable_fatbin(bytes) || has_nvidia_fatbin(bytes); }
inline bool has_fatbin(const void* bytes) { return has_nvidia_fatbin(bytes) || has_portable_fatbin(bytes); }
struct fatbin_reader {
    std::string_view bytes;
    uint64_t number(size_t offset, unsigned width = 4) const {
        if (offset > bytes.size() || width > bytes.size() - offset) throw std::runtime_error("truncated fatbinary field");
        uint64_t value = 0;
        for (unsigned i = 0; i < width; ++i) value |= uint64_t(static_cast<unsigned char>(bytes[offset + i])) << (8 * i);
        return value;
    }
};
inline std::string_view fatbin_memory(const void* image) {
    if (has_nvidia_fatbin(image)) return nvidia_fatbin_memory(image);
    if (!has_fatbin(image)) throw std::runtime_error("invalid fatbinary magic");
    fatbin_reader header{{static_cast<const char*>(image), 24}};
    auto size = header.number(16, 8);
    if (size < 41 || size > fatbin_limit) throw std::runtime_error("invalid fatbinary extent");
    return {static_cast<const char*>(image), size_t(size)};
}
inline void append_fatbin_number(std::string& bytes, uint64_t number, unsigned width = 4) {
    if (width > fatbin_limit - bytes.size()) throw std::runtime_error("fatbinary exceeds size limit");
    for (unsigned i = 0; i < width; ++i) bytes += char(number >> (8 * i));
}
inline void append_fatbin_bytes(std::string& bytes, std::string_view value) {
    if (value.size() > fatbin_limit - bytes.size()) throw std::runtime_error("fatbinary exceeds size limit");
    bytes += value;
}
inline std::string compress_fatbin(std::string_view input, unsigned threshold) {
    std::string output;
    size_t position = 0;
    auto run = [&](size_t start) {
        size_t end = start + 1;
        while (end < input.size() && end - start < 130 && input[end] == input[start]) ++end;
        return end - start;
    };
    while (position < input.size()) {
        auto count = run(position);
        if (count >= threshold) { output += char(0x80 | (count - 3)); output += input[position]; position += count; }
        else {
            auto begin = position++;
            while (position < input.size() && position - begin < 128 && run(position) < threshold) ++position;
            output += char(position - begin - 1); output.append(input.substr(begin, position - begin));
        }
    }
    return output;
}
inline std::string decompress_fatbin(std::string_view input, size_t size) {
    if (size > fatbin_limit) throw std::runtime_error("oversized fatbinary decompression");
    std::string output; output.reserve(size);
    for (size_t position = 0; position < input.size();) {
        auto token = static_cast<unsigned char>(input[position++]);
        size_t count = (token & 0x80) ? (token & 0x7f) + 3 : token + 1;
        if (count > size - output.size()) throw std::runtime_error("fatbinary decompression overflow");
        if (token & 0x80) {
            if (position == input.size()) throw std::runtime_error("truncated fatbinary run");
            output.append(count, input[position++]);
        } else {
            if (count > input.size() - position) throw std::runtime_error("truncated fatbinary literal");
            output.append(input.substr(position, count)); position += count;
        }
    }
    if (output.size() != size) throw std::runtime_error("fatbinary decompression size mismatch");
    return output;
}
inline std::string encode_fatbin(const std::vector<fatbin_record>& records, fatbin_compression compression = {}) {
    if (records.size() > 4096) throw std::runtime_error("fatbinary record limit exceeded");
    std::string bytes(fatbin_magic);
    append_fatbin_number(bytes, 1); append_fatbin_number(bytes, 64);
    append_fatbin_number(bytes, 0, 8); append_fatbin_number(bytes, records.size()); append_fatbin_number(bytes, 0);
    size_t decoded = 0;
    for (const auto& record : records) {
        if (record.code.empty() || record.identifier.size() > 4096 || record.identifier.find('\0') != record.identifier.npos
            || record.code.size() > fatbin_limit - decoded) throw std::runtime_error("invalid fatbinary record");
        decoded += record.code.size();
        auto packed = compression.enabled ? compress_fatbin(record.code, compression.threshold) : std::string();
        bool compressed = compression.enabled && (compression.force || packed.size() < record.code.size());
        auto& code = compressed ? packed : record.code;
        append_fatbin_number(bytes, uint32_t(record.kind)); append_fatbin_number(bytes, record.architecture);
        append_fatbin_number(bytes, compressed); append_fatbin_number(bytes, record.code.size());
        append_fatbin_number(bytes, record.identifier.size()); append_fatbin_number(bytes, code.size());
        append_fatbin_bytes(bytes, record.identifier); append_fatbin_bytes(bytes, code);
    }
    auto size = bytes.size() + 9;
    if (size > fatbin_limit) throw std::runtime_error("fatbinary exceeds size limit");
    for (unsigned i = 0; i < 8; ++i) bytes[16 + i] = char(uint64_t(size) >> (8 * i));
    append_fatbin_number(bytes, program_image_hash(bytes), 8); bytes += '\0';
    return bytes;
}
inline std::vector<fatbin_record> decode_nvidia_fatbin(std::string_view bytes) {
    std::vector<fatbin_record> records;
    for (auto& entry : decode_nvidia_container(bytes)) {
        if (entry.kind != 1 && entry.kind != 2 && entry.kind != 4)
            throw nvidia_container_error("unsupported NVIDIA fatbinary member kind", true);
        if (!(entry.flags & 1)) throw nvidia_container_error("only 64-bit NVIDIA fatbinary members are supported", true);
        if (entry.kind == 1) {
            // NVIDIA pads PTX members with NULs, including compressed images.
            auto end = entry.code.find('\0');
            if (end != entry.code.npos) {
                if (entry.code.find_first_not_of('\0', end) != entry.code.npos)
                    throw nvidia_container_error("embedded NUL in NVIDIA fatbinary PTX");
                entry.code.resize(end);
            }
            auto major = entry.ptx_version >> 16, minor = entry.ptx_version & 0xffff;
            if (!major || major > 9999 || minor > 9) throw nvidia_container_error("invalid NVIDIA fatbinary PTX version metadata");
            auto version = std::to_string(major) + "." + std::to_string(minor);
            auto header = fatbin_ptx_header(entry.code);
            // NVIDIA's producer removes the .version directive from member
            // text and retains its major/minor in the entry header instead.
            if (!header.empty() && header[0] == ".target") {
                entry.code.insert(0, ".version " + version + "\n");
                header = fatbin_ptx_header(entry.code);
            }
            if (header.size() != 6 || header[0] != ".version" || header[2] != ".target"
                || header[1] != version
                || header[3] != "sm_" + std::to_string(entry.architecture) || header[4] != ".address_size" || header[5] != "64")
                throw nvidia_container_error("NVIDIA fatbinary PTX metadata mismatch");
            entry.code += '\0';
        } else if (entry.kind == 2) {
            // Container payload extents include alignment padding after ELF.
            bool valid = false;
            for (unsigned padding = 0; padding < 8 && padding < entry.code.size(); ++padding) {
                if (padding && entry.code[entry.code.size() - padding] != 0) break;
                try {
                    if (fatbin_elf_architecture(std::string_view(entry.code).substr(0, entry.code.size() - padding)) == entry.architecture) {
                        entry.code.resize(entry.code.size() - padding); valid = true; break;
                    }
                } catch (const std::runtime_error&) {}
            }
            if (!valid) throw nvidia_container_error("invalid NVIDIA fatbinary cubin payload");
        }
        records.push_back({entry.kind == 1 ? fatbin_kind::ptx : entry.kind == 2 ? fatbin_kind::cubin : fatbin_kind::ltoir,
            entry.architecture, {}, std::move(entry.code)});
    }
    return records;
}
inline std::vector<fatbin_record> decode_fatbin(std::string_view bytes) {
    if (has_nvidia_fatbin(bytes)) return decode_nvidia_fatbin(bytes);
    if (!has_fatbin(bytes) || bytes.size() < 40 || bytes.size() > fatbin_limit) throw std::runtime_error("invalid fatbinary header");
    fatbin_reader reader{bytes};
    auto size = reader.number(16, 8);
    if (size == bytes.size() + 1) {
        // std::string views may omit the framing NUL retained by c_str().
        if (size < 41 || size > fatbin_limit) throw std::runtime_error("invalid fatbinary extent");
    } else if (size != bytes.size() || bytes.back() != 0) throw std::runtime_error("invalid fatbinary extent");
    if (reader.number(8) != 1 || reader.number(12) != 64 || reader.number(28)) throw std::runtime_error("unsupported fatbinary format");
    size_t end = size_t(size) - 9;
    if (end < 32 || reader.number(end, 8) != program_image_hash(bytes.substr(0, end))) throw std::runtime_error("fatbinary checksum mismatch");
    auto count = reader.number(24);
    if (count > 4096) throw std::runtime_error("fatbinary record limit exceeded");
    std::vector<fatbin_record> records; size_t position = 32, decoded = 0;
    for (uint64_t i = 0; i < count; ++i) {
        if (position > end || end - position < 24) throw std::runtime_error("truncated fatbinary record");
        auto kind = reader.number(position), arch = reader.number(position + 4), codec = reader.number(position + 8);
        auto unpacked = reader.number(position + 12), name = reader.number(position + 16), packed = reader.number(position + 20); position += 24;
        if (kind < 1 || kind > 3 || !arch || arch > 9999 || codec > 1 || !unpacked || name > 4096
            || name > end - position || packed > end - position - name || unpacked > fatbin_limit - decoded)
            throw std::runtime_error("invalid fatbinary record");
        fatbin_record record{fatbin_kind(kind), uint32_t(arch), std::string(bytes.substr(position, size_t(name))), {}}; position += size_t(name);
        if (record.identifier.find('\0') != record.identifier.npos) throw std::runtime_error("invalid fatbinary identifier");
        auto code = bytes.substr(position, size_t(packed)); position += size_t(packed); decoded += size_t(unpacked);
        if (!codec && packed != unpacked) throw std::runtime_error("fatbinary payload size mismatch");
        record.code = codec ? decompress_fatbin(code, size_t(unpacked)) : std::string(code);
        if (record.kind == fatbin_kind::ptx) {
            auto header = fatbin_ptx_header(record.code);
            if (header.size() != 6 || header[0] != ".version" || header[2] != ".target"
                || header[3] != "sm_" + std::to_string(record.architecture) || header[4] != ".address_size" || header[5] != "64"
                || record.code.back() != 0 || record.code.find('\0') != record.code.size() - 1)
                throw std::runtime_error("fatbinary PTX metadata mismatch");
        } else if (has_program_image(std::string_view(record.code))) {
            auto program = decode_program_image(record.code);
            if (program.architecture != "sm_" + std::to_string(record.architecture)) throw std::runtime_error("fatbinary program architecture mismatch");
        } else if (record.kind != fatbin_kind::cubin || fatbin_elf_architecture(record.code) != record.architecture)
            throw std::runtime_error("invalid fatbinary binary payload");
        records.push_back(std::move(record));
    }
    if (position != end) throw std::runtime_error("extra fatbinary data");
    return records;
}
// A source-image cubin retains IR/PTX and can target later CUDA SMs. A genuine
// ELF cubin is native only within its SM family. Shader/core import prefers PTX
// and retained IR, with bounded SM75 SASS import as the last alternative.
inline const fatbin_record* select_fatbin(const std::vector<fatbin_record>& records, unsigned architecture = 0,
                                        bool require_ir = false, bool cuda = false) {
    const fatbin_record* selected = nullptr; unsigned rank = 0;
    for (const auto& record : records) {
        bool program = has_program_image(std::string_view(record.code));
        bool elf = record.code.starts_with("\x7f" "ELF");
        if (architecture && record.architecture > architecture) continue;
        if (record.kind == fatbin_kind::cubin && !program && (!elf || (cuda
            ? (architecture && record.architecture / 10 != architecture / 10) : record.architecture != 75))) continue;
        if (record.kind == fatbin_kind::ltoir && !program) continue;
        if (require_ir && program && architecture && record.architecture != architecture) continue;
        unsigned candidate = record.kind == fatbin_kind::cubin ? 3 : record.kind == fatbin_kind::ptx ? 2 : 1;
        if (elf && !cuda) candidate = 0;
        if (!selected || candidate > rank || (candidate == rank && record.architecture > selected->architecture)) { selected = &record; rank = candidate; }
    }
    return selected;
}
// Validate an ELF payload's extents before packaging it for native CUDA. No
// machine instructions are decoded here; shared-IR cubin import is separate.
inline unsigned fatbin_elf_architecture(std::string_view bytes) {
    fatbin_reader reader{bytes};
    if (bytes.size() < 64 || !bytes.starts_with("\x7f" "ELF") || reader.number(4, 1) != 2
        || reader.number(5, 1) != 1 || reader.number(18, 2) != 190) throw std::runtime_error("invalid CUDA ELF64 image");
    size_t extent = 64;
    auto table = [&](size_t offset, unsigned width_offset, unsigned count_offset, unsigned minimum, bool sections) {
        auto start = reader.number(offset, 8), width = reader.number(width_offset, 2), count = reader.number(count_offset, 2);
        if (!count) { if (sections) throw std::runtime_error("missing cubin section table"); return; }
        if (width < minimum || start > bytes.size() || count * width > bytes.size() - start) throw std::runtime_error("invalid cubin table extent");
        extent = std::max(extent, size_t(start + count * width));
        for (uint64_t i = 0; i < count; ++i) {
            auto position = size_t(start + width * i);
            if (sections && reader.number(position + 4) == 8) continue;
            auto begin = reader.number(position + (sections ? 24 : 8), 8), size = reader.number(position + 32, 8);
            if (begin > bytes.size() || size > bytes.size() - begin) throw std::runtime_error("invalid cubin payload extent");
            extent = std::max(extent, size_t(begin + size));
        }
    };
    table(40, 58, 60, 64, true); table(32, 54, 56, 56, false);
    if (extent != bytes.size()) throw std::runtime_error("cubin size mismatch");
    return unsigned(reader.number(48) & 0xff);
}
}
