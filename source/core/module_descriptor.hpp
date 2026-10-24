#pragma once
#include "shader_packet.hpp"
#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <stdexcept>

// Compiler/Driver transport, carried in comments in otherwise ordinary PTX.
// NVIDIA consumes the same PTX; native module adapters consume this versioned payload.
namespace cudavm {
struct kernel_parameter_descriptor { uint32_t size, alignment, offset, pointer; };
struct kernel_descriptor {
    std::string symbol;
    uint32_t shared_bytes = 0, local_bytes = 0;
    std::vector<kernel_parameter_descriptor> parameters;
    uint32_t stack_bytes = 0;
    uint32_t memory_abi = packet_memory_abi;
    bool call_frames = false;
    uint32_t services = 0;
};
struct global_relocation { uint32_t offset, target; uint64_t addend; };
struct global_descriptor { std::string symbol; uint32_t alignment; std::string initializer; std::vector<global_relocation> relocations; };
struct module_descriptor { std::vector<kernel_descriptor> entries; std::vector<global_descriptor> globals; };
inline constexpr std::string_view module_descriptor_begin = "// CUDAVM-NATIVE-MODULE-V1-BEGIN\n";
inline constexpr std::string_view module_descriptor_end = "// CUDAVM-NATIVE-MODULE-V1-END";
inline bool has_module_descriptor(std::string_view text) { return text.find(module_descriptor_begin) != text.npos; }
inline std::string encode_module_descriptor(const std::vector<kernel_descriptor>& entries, const std::vector<global_descriptor>& globals = {}) {
    if (entries.size() > 4096 || globals.size() > 4096) throw std::runtime_error("native module inventory exceeds transport limit");
    std::string bytes;
    auto number = [&](uint32_t n) { for (unsigned i=0;i<4;++i) bytes += char(n >> (8*i)); };
    auto string = [&](const std::string& s) { if (s.size() > 64 * 1024 * 1024) throw std::runtime_error("native module field exceeds transport limit"); number(uint32_t(s.size())); bytes += s; };
    number(uint32_t(entries.size()));
    for (const auto& e : entries) {
        string(e.symbol); number(e.shared_bytes); number(e.local_bytes); number(uint32_t(e.parameters.size()));
        for (auto p : e.parameters) { number(p.size); number(p.alignment); number(p.offset); number(p.pointer); }
    }
    {
        number(uint32_t(globals.size()));
        for(const auto& g:globals) {
            string(g.symbol); number(g.alignment); string(g.initializer); number(uint32_t(g.relocations.size()));
            for(auto r:g.relocations) { number(r.offset); number(r.target); number(uint32_t(r.addend)); number(uint32_t(r.addend>>32)); }
        }
    }
    if (std::any_of(entries.begin(), entries.end(), [](const auto& entry) { return entry.stack_bytes != 0; })) {
        // Optional STK1 trailer. Older loaders reject the extra payload instead
        // of executing shaders with a missing stack base or fault observation.
        number(0x314b5453); number(uint32_t(entries.size()));
        for (const auto& entry : entries) number(entry.stack_bytes);
    }
    if (std::any_of(entries.begin(), entries.end(), [](const auto& entry) { return entry.call_frames; })) {
        number(0x314c4143); number(uint32_t(entries.size()));
        for (const auto& entry : entries) number(entry.call_frames ? 1u : 0u);
    }
    if (std::any_of(entries.begin(), entries.end(), [](const auto& entry) { return entry.services != 0; })) {
        number(0x31565253); number(uint32_t(entries.size()));
        for (const auto& entry : entries) number(entry.services);
    }
    // ADR2 makes incompatible loaders reject widened records and native slots.
    number(0x32524441); number(uint32_t(entries.size()));
    for (const auto& entry : entries) number(entry.memory_abi);
    std::string out(module_descriptor_begin);
    constexpr char hex[] = "0123456789abcdef";
    for (size_t i=0;i<bytes.size();++i) {
        if (i%96 == 0) out += "// ";
        auto b = static_cast<unsigned char>(bytes[i]); out += hex[b>>4]; out += hex[b&15];
        if (i%96 == 95 || i+1 == bytes.size()) out += '\n';
    }
    return out + std::string(module_descriptor_end) + '\n';
}
inline module_descriptor decode_module_descriptor(std::string_view ptx) {
    auto marker = module_descriptor_begin;
    auto start = ptx.find(marker);
    if (start == ptx.npos) throw std::runtime_error("PTX has no cudavm native module payload; compile source with cudavm AOT or NVRTC");
    if (ptx.find(marker, start + marker.size()) != ptx.npos) throw std::runtime_error("duplicate native module descriptor");
    start += marker.size(); auto stop = ptx.find(module_descriptor_end, start);
    if (stop == ptx.npos || stop - start > 136 * 1024 * 1024) throw std::runtime_error("truncated native module payload");
    std::string bytes; int high = -1;
    for (size_t i=start;i<stop;++i) {
        char c=ptx[i]; if (c=='/' || c==' ' || c=='\r' || c=='\n') continue;
        int n = c>='0' && c<='9' ? c-'0' : c>='a' && c<='f' ? c-'a'+10 : -1;
        if (n<0) throw std::runtime_error("invalid native module payload encoding");
        if (high<0) high=n; else { bytes += char((high<<4)|n); high=-1; }
    }
    if (high>=0) throw std::runtime_error("truncated native module payload byte");
    size_t pos=0;
    auto number = [&]() { if (bytes.size()-pos<4) throw std::runtime_error("truncated native module field"); uint32_t n=0; for(unsigned i=0;i<4;++i) n|=uint32_t(static_cast<unsigned char>(bytes[pos++]))<<(8*i); return n; };
    auto string = [&]() { auto size=number(); if (size>bytes.size()-pos) throw std::runtime_error("truncated native module string"); auto s=bytes.substr(pos,size); pos+=size; return s; };
    auto count=number(); if(count>4096) throw std::runtime_error("too many native module entries");
    module_descriptor result;
    for(uint32_t i=0;i<count;++i) {
        kernel_descriptor e; e.memory_abi=1; e.symbol=string(); e.shared_bytes=number(); e.local_bytes=number(); auto params=number();
        if(params>4096) throw std::runtime_error("too many native module parameters");
        for(uint32_t j=0;j<params;++j) e.parameters.push_back({number(),number(),number(),number()});
        result.entries.push_back(std::move(e));
    }
    {
        auto globals=number(); if(globals>4096) throw std::runtime_error("too many native module globals");
        for(uint32_t i=0;i<globals;++i) {
            global_descriptor g; g.symbol=string(); g.alignment=number(); g.initializer=string(); auto relocations=number();
            if(relocations>bytes.size()/16) throw std::runtime_error("too many native module relocations");
            for(uint32_t j=0;j<relocations;++j) { auto offset=number(),target=number(),low=number(),high=number(); g.relocations.push_back({offset,target,uint64_t(low)|(uint64_t(high)<<32)}); }
            result.globals.push_back(std::move(g));
        }
    }
    if (pos != bytes.size()) {
        auto trailer = number();
        if (trailer == 0x314b5453) {
            if (number() != result.entries.size()) throw std::runtime_error("invalid native stack descriptor");
            for (auto& entry : result.entries) {
                entry.stack_bytes = number();
                if (entry.stack_bytes && (entry.stack_bytes != packet_private_stack_bytes || entry.local_bytes != entry.stack_bytes))
                    throw std::runtime_error("invalid native stack budget");
            }
            if (pos == bytes.size()) return result;
            trailer = number();
        }
        if (trailer == 0x314c4143) {
            if (number() != result.entries.size()) throw std::runtime_error("invalid native call descriptor");
            for (auto& entry : result.entries) {
                auto flags = number();
                if (flags > 1 || (flags && !entry.stack_bytes)) throw std::runtime_error("invalid native call stack");
                entry.call_frames = flags != 0;
            }
            trailer = number();
        }
        if (trailer == 0x31565253) {
            if (number() != result.entries.size()) throw std::runtime_error("invalid native service descriptor");
            for (auto& entry : result.entries) {
                entry.services = number();
                if (entry.services & ~7u || ((entry.services & 4u) && !(entry.services & 2u))) throw std::runtime_error("invalid native service flags");
            }
            trailer = number();
        }
        if (trailer != 0x32524441 || number() != result.entries.size()) throw std::runtime_error("invalid native memory descriptor");
        for (auto& entry : result.entries) {
            entry.memory_abi = number();
            if (entry.memory_abi != packet_memory_abi) throw std::runtime_error("unsupported native memory ABI");
        }
    }
    if(pos!=bytes.size()) throw std::runtime_error("extra native module payload data");
    return result;
}
inline std::vector<kernel_descriptor> decode_kernel_descriptors(std::string_view ptx) { return decode_module_descriptor(ptx).entries; }
}
