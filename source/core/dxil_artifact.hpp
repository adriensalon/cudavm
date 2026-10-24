#pragma once
#include "spirv_artifact.hpp"
#include <array>
namespace cudavm {
inline constexpr std::string_view dxil_artifact_begin = "// CUDAVM-NATIVE-DXIL-V1-BEGIN\n", dxil_artifact_end = "// CUDAVM-NATIVE-DXIL-V1-END";
struct dxil_variant { std::array<unsigned,3> block; std::string code; };
inline constexpr std::array<std::array<unsigned,3>,33> dxil_block_shapes{{{1,1,1},{2,1,1},{4,1,1},{8,1,1},{16,1,1},{32,1,1},
    {64,1,1},{128,1,1},{256,1,1},{512,1,1},{1024,1,1},{4,2,1},{4,2,2},
    {13,1,1},{45,1,1},{96,1,1},{17,1,1},{31,1,1},{33,1,1},{63,1,1},{8,4,3},{7,5,3},{8,4,4},{8,3,3},
    {8,4,1},{8,4,2},{4,4,2},{2,2,8},{2,2,1},{1,2,2},{2,1,2},{8,8,1},{2,2,2}}};
inline uint32_t read_dxil_word(std::string_view bytes, size_t& cursor) {
    if (cursor > bytes.size() || bytes.size() - cursor < 4) throw std::runtime_error("truncated native DXIL field");
    uint32_t value = 0; for (unsigned i=0;i<4;++i) value |= uint32_t(static_cast<unsigned char>(bytes[cursor++])) << (i*8); return value;
}
inline void append_dxil_word(std::string& bytes, uint32_t value) { for (unsigned i=0;i<4;++i) bytes += char(value >> (i*8)); }
inline void validate_dxil_container(std::string_view code) {
    if (code.size() < 32 || code.size() > 64*1024*1024 || code.substr(0,4) != "DXBC") throw std::runtime_error("invalid native DXIL container");
    size_t at=20; auto version=read_dxil_word(code,at), size=read_dxil_word(code,at), count=read_dxil_word(code,at);
    if (version!=1 || size!=code.size() || count>4096 || count>(code.size()-32)/4) throw std::runtime_error("invalid native DXIL part table");
    bool dxil=false; std::vector<std::pair<size_t,size_t>> ranges;
    for (uint32_t i=0;i<count;++i) {
        auto offset=read_dxil_word(code,at); size_t part=offset;
        if (offset<32+count*4 || offset>code.size() || code.size()-offset<8) throw std::runtime_error("invalid native DXIL part offset");
        auto kind=code.substr(offset,4); part+=4; auto bytes=read_dxil_word(code,part);
        if (bytes>code.size()-part) throw std::runtime_error("truncated native DXIL part");
        for (auto [start,stop]:ranges) if (offset<stop && part+bytes>start) throw std::runtime_error("overlapping native DXIL parts");
        ranges.emplace_back(offset,part+bytes);
        if (kind=="DXIL") { if(dxil || bytes<24) throw std::runtime_error("invalid native DXIL program"); dxil=true; }
    }
    if (!dxil) throw std::runtime_error("missing native DXIL program");
}
struct dxil_program_info { std::array<unsigned,3> block{}; uint64_t features=0; };
inline dxil_program_info inspect_dxil_program(std::string_view code) {
    validate_dxil_container(code); dxil_program_info result; bool psv=false, flags=false;
    size_t at=28; auto count=read_dxil_word(code,at);
    for(uint32_t i=0;i<count;++i) {
        auto offset=read_dxil_word(code,at); auto kind=code.substr(offset,4); size_t part=offset+4; auto bytes=read_dxil_word(code,part);
        if(kind=="DXIL") {
            if(static_cast<unsigned char>(code[part])!=0x60 || code[part+2]!=5 || code[part+3]!=0) throw std::runtime_error("native DXIL requires a compute shader model 6.0 program");
        }
        if(kind=="PSV0") {
            if(psv || bytes<52) throw std::runtime_error("invalid native DXIL pipeline validation metadata");
            psv=true; size_t cursor=part; auto size=read_dxil_word(code,cursor);
            if(size<48 || size>bytes-4 || static_cast<unsigned char>(code[part+28])!=5) throw std::runtime_error("unsupported native DXIL pipeline validation metadata");
            cursor=part+40; for(auto& dimension:result.block) dimension=read_dxil_word(code,cursor);
        }
        if(kind=="SFI0") {
            if(flags || bytes!=8) throw std::runtime_error("invalid native DXIL shader feature metadata");
            flags=true; size_t cursor=part; auto low=read_dxil_word(code,cursor), high=read_dxil_word(code,cursor); result.features=uint64_t(low)|(uint64_t(high)<<32);
        }
    }
    if(!psv || !flags) throw std::runtime_error("missing native DXIL dispatch or feature metadata"); return result;
}
inline std::string encode_dxil_variants(const std::vector<dxil_variant>& variants) {
    if (variants.empty() || variants.size()>64) throw std::runtime_error("invalid native DXIL variant count");
    std::string bytes("CDXL"); append_dxil_word(bytes,1); append_dxil_word(bytes,uint32_t(variants.size()));
    for (const auto& v:variants) { validate_dxil_container(v.code); for(auto dimension:v.block) append_dxil_word(bytes,dimension); append_dxil_word(bytes,uint32_t(v.code.size())); bytes+=v.code; }
    return bytes;
}
inline std::vector<dxil_variant> decode_dxil_variants(std::string_view bytes) {
    if (bytes.size()>64*1024*1024 || bytes.substr(0,4)!="CDXL") throw std::runtime_error("invalid native DXIL bundle");
    size_t at=4; if(read_dxil_word(bytes,at)!=1) throw std::runtime_error("unsupported native DXIL bundle version");
    auto count=read_dxil_word(bytes,at); if(!count || count>64) throw std::runtime_error("invalid native DXIL variant count");
    std::vector<dxil_variant> result; std::set<std::array<unsigned,3>> selected;
    for(uint32_t i=0;i<count;++i) {
        dxil_variant v; uint64_t threads=1;
        for(auto& dimension:v.block) { dimension=read_dxil_word(bytes,at); if(!dimension || dimension>1024) throw std::runtime_error("invalid native DXIL block shape"); threads*=dimension; }
        if(threads>1024 || v.block[2]>64 || !selected.insert(v.block).second) throw std::runtime_error("invalid or duplicate native DXIL block shape");
        auto size=read_dxil_word(bytes,at); if(size>bytes.size()-at) throw std::runtime_error("truncated native DXIL variant");
        v.code=bytes.substr(at,size); at+=size;
        if(inspect_dxil_program(v.code).block!=v.block) throw std::runtime_error("native DXIL block metadata differs from variant");
        result.push_back(std::move(v));
    }
    if(at!=bytes.size()) throw std::runtime_error("extra native DXIL bundle data"); return result;
}
inline std::string encode_dxil_artifacts(const std::vector<shader_artifact>& entries) {
    auto bytes=cudavm::encode_spirv_artifacts(entries);
    bytes.replace(bytes.find(cudavm::spirv_artifact_end_v2),cudavm::spirv_artifact_end_v2.size(),dxil_artifact_end);
    bytes.replace(0,cudavm::spirv_artifact_begin_v2.size(),dxil_artifact_begin); return bytes;
}
inline std::vector<shader_artifact> decode_dxil_artifacts(std::string_view image) {
    auto start=image.find(dxil_artifact_begin); if(start==image.npos) return {};
    start+=dxil_artifact_begin.size(); auto stop=image.find(dxil_artifact_end,start);
    if(stop==image.npos || stop-start>136*1024*1024) throw std::runtime_error("invalid native DXIL transport size");
    return cudavm::decode_spirv_artifacts(std::string(cudavm::spirv_artifact_begin_v2)+std::string(image.substr(start,stop-start))+std::string(cudavm::spirv_artifact_end_v2));
}
}
