#pragma once
#include "ptx_parameters.hpp"
#include "../nvfatbin/nvidia_container_api.hpp"
#include <cstring>
#include <set>

namespace cudavm {
// CUDA fatbinary v1 containers and little-endian ELF64 cubins. Parsing is only
// for ownership and host ABI metadata; NVIDIA's loader executes the original
// image and selects the machine code for the actual device.
struct nvidia_image_reader {
    std::span<const std::byte> bytes;
    void bounds(size_t offset, size_t count) const {
        require(offset <= bytes.size() && count <= bytes.size() - offset, CUDA_ERROR_INVALID_IMAGE, "truncated NVIDIA image");
    }
    template<class T> T read(size_t offset) const {
        bounds(offset, sizeof(T)); T value; std::memcpy(&value, bytes.data() + offset, sizeof(T)); return value;
    }
    std::span<const std::byte> slice(size_t offset, size_t count) const { bounds(offset, count); return bytes.subspan(offset, count); }
    std::string string(size_t offset, size_t count) const {
        auto data = slice(offset, count); auto end = std::find(data.begin(), data.end(), std::byte{});
        require(end != data.end(), CUDA_ERROR_INVALID_IMAGE, "unterminated NVIDIA image string");
        return std::string(reinterpret_cast<const char*>(data.data()), size_t(end - data.begin()));
    }
};
inline size_t nvidia_image_length(const void* image) {
    require(image, CUDA_ERROR_INVALID_IMAGE, "null NVIDIA image");
    uint32_t magic = 0; std::memcpy(&magic, image, sizeof(magic));
    if (magic == nvidia_fatbin_magic) {
        try { return nvidia_fatbin_memory(image).size(); }
        catch (const nvidia_container_error& error) { throw failure(CUDA_ERROR_INVALID_IMAGE, error.what()); }
    }
    if (magic == 0x464c457f) {
        auto data = static_cast<const std::byte*>(image);
        auto word = [&]<class T>(size_t offset) { require(offset <= nvidia_image_limit - sizeof(T), CUDA_ERROR_INVALID_IMAGE, "oversized cubin extent"); T value; std::memcpy(&value, data + offset, sizeof(T)); return value; };
        require(word.template operator()<uint8_t>(4) == 2 && word.template operator()<uint8_t>(5) == 1, CUDA_ERROR_INVALID_IMAGE, "unsupported cubin encoding");
        auto offset = word.template operator()<uint64_t>(40); auto width = word.template operator()<uint16_t>(58), count = word.template operator()<uint16_t>(60);
        auto table_bytes = size_t(width) * count;
        require(width >= 64 && count && table_bytes <= nvidia_image_limit && offset <= nvidia_image_limit - table_bytes, CUDA_ERROR_INVALID_IMAGE, "invalid cubin extent");
        size_t length = size_t(offset) + size_t(width) * count;
        for (size_t i = 0; i < count; ++i) {
            auto pos = size_t(offset) + i * width;
            if (word.template operator()<uint32_t>(pos + 4) == 8) continue; // SHT_NOBITS has no file storage.
            auto start = word.template operator()<uint64_t>(pos + 24), bytes = word.template operator()<uint64_t>(pos + 32);
            require(start <= nvidia_image_limit && bytes <= nvidia_image_limit - start, CUDA_ERROR_INVALID_IMAGE, "invalid cubin data extent");
            length = std::max(length, size_t(start + bytes));
        }
        auto program_offset = word.template operator()<uint64_t>(32); auto program_width = word.template operator()<uint16_t>(54), program_count = word.template operator()<uint16_t>(56);
        auto program_bytes = size_t(program_width) * program_count;
        require(!program_count || (program_width >= 56 && program_bytes <= nvidia_image_limit && program_offset <= nvidia_image_limit - program_bytes), CUDA_ERROR_INVALID_IMAGE, "invalid cubin program extent");
        if (program_count) length = std::max(length, size_t(program_offset) + program_bytes);
        for (size_t i = 0; i < program_count; ++i) {
            auto pos = size_t(program_offset) + i * program_width;
            auto start = word.template operator()<uint64_t>(pos + 8), bytes = word.template operator()<uint64_t>(pos + 32);
            require(start <= nvidia_image_limit && bytes <= nvidia_image_limit - start, CUDA_ERROR_INVALID_IMAGE, "invalid cubin segment extent");
            length = std::max(length, size_t(start + bytes));
        }
        return length;
    }
    return 0;
}
inline std::vector<std::byte> nvidia_decompress(std::span<const std::byte> input, size_t size) {
    try {
        auto decoded = decompress_nvidia_fatbin({reinterpret_cast<const char*>(input.data()), input.size()}, size);
        auto bytes = std::as_bytes(std::span(decoded.data(), decoded.size()));
        return {bytes.begin(), bytes.end()};
    } catch (const nvidia_container_error& error) { throw failure(CUDA_ERROR_INVALID_IMAGE, error.what()); }
}
struct nvidia_module_metadata {
    std::map<std::string, std::vector<kernel_parameter>> kernels;
    std::set<std::string> unavailable;
    void merge(const nvidia_module_metadata& other) {
        unavailable.insert(other.unavailable.begin(), other.unavailable.end());
        for (const auto& [name, layout] : other.kernels) {
            auto [it, added] = kernels.emplace(name, layout);
            if (!added && (it->second.size() != layout.size() || !std::equal(layout.begin(), layout.end(), it->second.begin(), [](auto a, auto b) { return a.offset == b.offset && a.size == b.size; }))) unavailable.insert(name);
        }
    }
};
inline nvidia_module_metadata nvidia_elf_metadata(std::span<const std::byte> image) {
    nvidia_image_reader in{image};
    require(in.read<uint32_t>(0) == 0x464c457f && in.read<uint8_t>(4) == 2 && in.read<uint8_t>(5) == 1 && in.read<uint16_t>(18) == 190,
        CUDA_ERROR_INVALID_IMAGE, "NVIDIA metadata requires a little-endian ELF64 cubin");
    auto offset = in.read<uint64_t>(40); auto width = in.read<uint16_t>(58), count = in.read<uint16_t>(60), strings = in.read<uint16_t>(62);
    require(width >= 64 && count && strings < count, CUDA_ERROR_INVALID_IMAGE, "invalid cubin section table");
    in.bounds(size_t(offset), size_t(width) * count);
    auto program_offset = in.read<uint64_t>(32); auto program_width = in.read<uint16_t>(54), program_count = in.read<uint16_t>(56);
    if (program_count) {
        require(program_width >= 56, CUDA_ERROR_INVALID_IMAGE, "invalid cubin program table");
        in.bounds(size_t(program_offset), size_t(program_width) * program_count);
        for (size_t i = 0; i < program_count; ++i) {
            auto pos = size_t(program_offset) + i * program_width;
            in.bounds(size_t(in.read<uint64_t>(pos + 8)), size_t(in.read<uint64_t>(pos + 32)));
        }
    }
    auto section = [&](size_t index) { require(index < count, CUDA_ERROR_INVALID_IMAGE, "invalid cubin section index"); return size_t(offset) + index * width; };
    auto data = [&](size_t index) { auto header = section(index); return in.slice(size_t(in.read<uint64_t>(header + 24)), size_t(in.read<uint64_t>(header + 32))); };
    nvidia_image_reader names{data(strings)}; nvidia_module_metadata result;
    // STO_CUDA_ENTRY distinguishes kernels from device functions.
    for (size_t index = 0; index < count; ++index) {
        auto header = section(index); if (in.read<uint32_t>(header + 4) != 2) continue;
        nvidia_image_reader symbols{data(index)}, text{data(in.read<uint32_t>(header + 40))};
        auto stride = in.read<uint64_t>(header + 56); require(stride >= 24 && symbols.bytes.size() % stride == 0, CUDA_ERROR_INVALID_IMAGE, "invalid cubin symbol table");
        for (size_t pos = 0; pos < symbols.bytes.size(); pos += size_t(stride))
            if ((symbols.read<uint8_t>(pos + 4) & 15) == 2 && (symbols.read<uint8_t>(pos + 5) & 16)) {
                auto start = symbols.read<uint32_t>(pos); auto name = text.string(start, text.bytes.size() - std::min<size_t>(start, text.bytes.size())); result.kernels.emplace(std::move(name), std::vector<kernel_parameter>{});
            }
    }
    std::set<std::string> described;
    for (size_t index = 0; index < count; ++index) {
        auto start = in.read<uint32_t>(section(index)); auto name = names.string(start, names.bytes.size() - std::min<size_t>(start, names.bytes.size()));
        constexpr std::string_view prefix = ".nv.info.";
        if (!name.starts_with(prefix) || !result.kernels.contains(name.substr(prefix.size()))) continue;
        auto kernel_name = name.substr(prefix.size()); nvidia_image_reader info{data(index)};
        std::map<unsigned, kernel_parameter> layout; bool supported = true;
        for (size_t pos = 0; pos < info.bytes.size();) {
            auto format = info.read<uint8_t>(pos), attribute = info.read<uint8_t>(pos + 1); auto size = info.read<uint16_t>(pos + 2);
            require(format >= 1 && format <= 4, CUDA_ERROR_INVALID_IMAGE, "invalid cubin information record");
            auto payload = format == 4 ? size_t(size) : 0; info.bounds(pos + 4, payload);
            if (attribute == 0x17) {
                require(format == 4 && size == 12, CUDA_ERROR_INVALID_IMAGE, "invalid cubin parameter information");
                auto ordinal = info.read<uint16_t>(pos + 8), parameter_offset = info.read<uint16_t>(pos + 10);
                size_t bytes = info.read<uint32_t>(pos + 12) >> 18;
                if (!bytes || ordinal >= 4096 || bytes > 65536 - size_t(parameter_offset) || !layout.emplace(ordinal, kernel_parameter{parameter_offset, bytes}).second) supported = false;
            }
            pos += 4 + payload;
        }
        auto& target = result.kernels.at(kernel_name); size_t end = 0;
        for (auto [ordinal, parameter] : layout) {
            if (ordinal != target.size() || parameter.offset < end) supported = false;
            target.push_back(parameter); end = parameter.offset + parameter.size;
        }
        if (!supported) result.unavailable.insert(kernel_name); described.insert(kernel_name);
    }
    for (auto& [name, layout] : result.kernels) if (!described.contains(name)) result.unavailable.insert(name);
    return result;
}
inline nvidia_module_metadata nvidia_image_metadata(std::span<const std::byte> image) {
    require(image.size() <= nvidia_image_limit, CUDA_ERROR_INVALID_IMAGE, "oversized NVIDIA image");
    nvidia_image_reader in{image}; auto magic = in.read<uint32_t>(0);
    if (magic == 0x464c457f) return nvidia_elf_metadata(image);
    if (magic != nvidia_fatbin_magic) {
        std::string_view text(reinterpret_cast<const char*>(image.data()), image.size()); nvidia_module_metadata result;
        auto layouts = ptx_parameter_layouts(text);
        for (auto& name : ptx_entry_names(text)) {
            if (auto it = layouts.find(name); it != layouts.end()) result.kernels.emplace(name, it->second);
            else { result.kernels.emplace(name, std::vector<kernel_parameter>{}); result.unavailable.insert(name); }
        }
        return result;
    }
    nvidia_module_metadata result;
    std::vector<nvidia_fatbin_entry> entries;
    try { entries = decode_nvidia_container({reinterpret_cast<const char*>(image.data()), image.size()}); }
    catch (const nvidia_container_error& error) { throw failure(CUDA_ERROR_INVALID_IMAGE, error.what()); }
    for (const auto& record : entries) {
        if (record.kind == 1 || record.kind == 2) {
            auto data = std::as_bytes(std::span(record.code.data(), record.code.size()));
            // Sections contain PTX or cubin, never another fatbinary. Avoid
            // recursive parsing of untrusted container records.
            nvidia_module_metadata entry;
            if (record.kind == 2) entry = nvidia_elf_metadata(data);
            else {
                nvidia_image_reader text{data};
                auto entry_magic = text.read<uint32_t>(0);
                require(entry_magic != nvidia_fatbin_magic && entry_magic != 0x464c457f, CUDA_ERROR_INVALID_IMAGE, "invalid PTX fatbinary entry");
                entry = nvidia_image_metadata(data);
            }
            result.merge(entry);
        }
    }
    return result;
}
}
