#pragma once
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
        kernel_descriptor e; e.symbol=string(); e.shared_bytes=number(); e.local_bytes=number(); auto params=number();
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
    if(pos!=bytes.size()) throw std::runtime_error("extra native module payload data");
    return result;
}
inline std::vector<kernel_descriptor> decode_kernel_descriptors(std::string_view ptx) { return decode_module_descriptor(ptx).entries; }
}
