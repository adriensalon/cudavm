#pragma once
#include "nvidia_image.hpp"

namespace cudavm {
struct nvidia_binding_image { std::vector<std::byte> bytes; bool ptx = false; };

// Select native code for an independently owned registration binding. Leave
// compilation of PTX to NVIDIA's linker and execution of cubins to its loader.
inline nvidia_binding_image nvidia_select_binding_image(std::span<const std::byte> image, unsigned architecture) {
    (void)nvidia_image_metadata(image);
    nvidia_image_reader in{image};
    if (in.read<uint32_t>(0) != nvidia_fatbin_magic)
        return {{image.begin(), image.end()}, in.read<uint32_t>(0) != 0x464c457f};
    nvidia_binding_image cubin, ptx; unsigned cubin_arch = 0, ptx_arch = 0;
    size_t position = in.read<uint16_t>(6), end = position + size_t(in.read<uint64_t>(8));
    while (position < end) {
        auto kind = in.read<uint16_t>(position); auto arch = in.read<uint32_t>(position + 28);
        auto header = in.read<uint32_t>(position + 4); auto bytes = size_t(in.read<uint64_t>(position + 8));
        bool compatible = arch <= architecture && (kind == 1 || arch / 10 == architecture / 10);
        if (compatible && ((kind == 2 && arch >= cubin_arch) || (kind == 1 && arch >= ptx_arch))) {
            auto data = in.slice(position + header, bytes); std::vector<std::byte> decoded;
            if (in.read<uint64_t>(position + 40) & 0x2000)
                decoded = nvidia_decompress(data.first(in.read<uint32_t>(position + 16)), size_t(in.read<uint64_t>(position + 56)));
            else decoded.assign(data.begin(), data.end());
            if (kind == 2) { cubin = {std::move(decoded), false}; cubin_arch = arch; }
            else { ptx = {std::move(decoded), true}; ptx_arch = arch; }
        }
        position += header + bytes;
    }
    if (!cubin.bytes.empty()) return cubin;
    require(!ptx.bytes.empty(), CUDA_ERROR_NO_BINARY_FOR_GPU, "no compatible managed registration binding image");
    return ptx;
}

// Managed cubin references use native address relocations. Resolve these to
// registration-owned storage before loading a disposable per-context module.
// Only address fields change; unrelated relocations remain unchanged. Reject
// unfamiliar relocation encodings rather than guessing their bit fields.
inline std::vector<std::byte> nvidia_bind_managed_symbols(std::span<const std::byte> image,
        const std::map<std::string, std::pair<CUdeviceptr, size_t>>& bindings) {
    (void)nvidia_elf_metadata(image); nvidia_image_reader in{image};
    auto output = std::vector(image.begin(), image.end());
    auto write = [&]<class T>(size_t offset, T value) { in.bounds(offset, sizeof(value)); std::memcpy(output.data() + offset, &value, sizeof(value)); };
    auto table = size_t(in.read<uint64_t>(40)); auto width = in.read<uint16_t>(58), count = in.read<uint16_t>(60);
    auto section = [&](size_t index) { require(index < count, CUDA_ERROR_INVALID_IMAGE, "invalid managed relocation section"); return table + index * width; };
    std::map<std::pair<size_t, size_t>, CUdeviceptr> symbols; std::set<std::string> found;
    std::vector<size_t> symbol_tables;
    for (size_t index = 0; index < count; ++index) {
        auto sh = section(index); if (in.read<uint32_t>(sh + 4) != 2) continue;
        symbol_tables.push_back(index);
        auto start = size_t(in.read<uint64_t>(sh + 24)), bytes = size_t(in.read<uint64_t>(sh + 32)), stride = size_t(in.read<uint64_t>(sh + 56));
        auto strings = section(in.read<uint32_t>(sh + 40));
        nvidia_image_reader names{in.slice(size_t(in.read<uint64_t>(strings + 24)), size_t(in.read<uint64_t>(strings + 32)))};
        for (size_t pos = 0; pos < bytes; pos += stride) {
            auto name_offset = in.read<uint32_t>(start + pos);
            auto name = names.string(name_offset, names.bytes.size() - std::min<size_t>(name_offset, names.bytes.size()));
            auto item = bindings.find(name); if (item == bindings.end()) continue;
            require((in.read<uint8_t>(start + pos + 5) & 4) && in.read<uint64_t>(start + pos + 16) == item->second.second,
                CUDA_ERROR_INVALID_IMAGE, "managed binding disagrees with cubin symbol");
            symbols.emplace(std::pair(index, pos / stride), item->second.first); found.insert(name);
            // This disposable module must not register a second managed
            // allocation or let the native loader rebind its managed references.
            write(start + pos + 5, uint8_t(in.read<uint8_t>(start + pos + 5) & ~4u));
        }
    }
    require(found.size() == bindings.size(), CUDA_ERROR_INVALID_IMAGE, "managed binding symbol is absent from cubin");
    for (size_t index = 0; index < count; ++index) {
        auto sh = section(index); auto kind = in.read<uint32_t>(sh + 4); if (kind != 9 && kind != 4) continue;
        auto start = size_t(in.read<uint64_t>(sh + 24)), bytes = size_t(in.read<uint64_t>(sh + 32)), stride = size_t(in.read<uint64_t>(sh + 56));
        require(stride >= (kind == 9 ? 16 : 24) && bytes % stride == 0, CUDA_ERROR_INVALID_IMAGE, "invalid managed relocation table");
        in.bounds(start, bytes); auto target = section(in.read<uint32_t>(sh + 44));
        size_t symbol_table = in.read<uint32_t>(sh + 40);
        if (in.read<uint32_t>(section(symbol_table) + 4) != 2) {
            // Some native cuLink outputs retain a link to CUDA_INFO after
            // removing an intermediate symbol table. NVIDIA's loader uses
            // their sole remaining symbol table for these relocation ordinals.
            require(in.read<uint32_t>(section(symbol_table) + 4) == 0x70000000 && symbol_tables.size() == 1,
                CUDA_ERROR_INVALID_IMAGE, "invalid managed relocation symbol table");
            symbol_table = symbol_tables.front();
        }
        auto target_start = size_t(in.read<uint64_t>(target + 24)), target_bytes = size_t(in.read<uint64_t>(target + 32));
        size_t retained = 0;
        for (size_t pos = 0; pos < bytes; pos += stride) {
            auto info = in.read<uint64_t>(start + pos + 8); auto item = symbols.find({symbol_table, size_t(info >> 32)});
            if (item == symbols.end()) { std::memcpy(output.data() + start + retained, image.data() + start + pos, stride); retained += stride; continue; }
            auto relocation = uint32_t(info);
            require(relocation == 2 || relocation == 56 || relocation == 57, CUDA_ERROR_NOT_SUPPORTED, "unsupported managed address relocation");
            auto offset = in.read<uint64_t>(start + pos);
            require(in.read<uint32_t>(target + 4) == 1 && offset <= target_bytes && 8 <= target_bytes - offset,
                CUDA_ERROR_INVALID_IMAGE, "managed relocation exceeds target storage");
            if (relocation == 2) { // R_CUDA_64, including pointers in initializers.
                auto addend = kind == 9 ? in.read<uint64_t>(target_start + size_t(offset)) : in.read<uint64_t>(start + pos + 16);
                write(target_start + size_t(offset), item->second + addend);
            } else { // R_CUDA_ABS32_LO_32 / HI_32: native instruction immediate.
                auto immediate = in.read<uint32_t>(target_start + size_t(offset) + 4);
                require(kind == 4 || immediate == 0, CUDA_ERROR_NOT_SUPPORTED, "implicit managed instruction addend is unavailable");
                auto address = item->second + (kind == 4 ? in.read<uint64_t>(start + pos + 16) : 0);
                write(target_start + size_t(offset) + 4, uint32_t(relocation == 56 ? address : address >> 32));
            }
        }
        write(sh + 32, uint64_t(retained));
    }
    return output;
}
}
