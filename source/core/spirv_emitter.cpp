#include "spirv_emitter.hpp"
#include "shader_lowering.hpp"
#include "spirv_artifact.hpp"
#include <map>
#include <cstdio>
#include <cstdlib>
#include <llvm/IR/Module.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Transforms/Utils/Cloning.h>
namespace llvm {
extern "C" bool SPIRVTranslate(Module*, std::string&, std::string&, const std::vector<std::string>&, CodeGenOptLevel, Triple);
}
namespace cudavm {
std::unique_ptr<llvm::Module> lower_native_spirv(const llvm::Module& input, const std::string& symbol, bool opencl) {
    return lower_native_shader(input, symbol, opencl ? shader_environment::opencl : shader_environment::vulkan);
}
std::string emit_native_spirv(const llvm::Module& input) {
    // The owned target caches named scope IDs across contexts. Give every
    // emission the same scope registry before parsing IR, so a prior kernel's
    // device/workgroup registration order cannot change this kernel's scope.
    llvm::LLVMContext context;
    for(auto name:{"subgroup","workgroup","device"})context.getOrInsertSyncScopeID(name);
    std::string assembly;llvm::raw_string_ostream stream(assembly);input.print(stream,nullptr);
    llvm::SMDiagnostic parse_diagnostic;
    auto module=llvm::parseAssemblyString(assembly,parse_diagnostic,context);
    if(!module)throw std::runtime_error("native SPIR-V emission could not clone target IR");
    std::string bytes, diagnostic;
    if(std::getenv("CUDAVM_TRACE_NATIVE")) std::fprintf(stderr,"native SPIR-V emit module=%s\n",input.getModuleIdentifier().c_str());
    if (!llvm::SPIRVTranslate(module.get(), bytes, diagnostic, {}, llvm::CodeGenOptLevel::None, module->getTargetTriple()))
        throw std::runtime_error("native SPIR-V emission failed: " + diagnostic);
    // CUDA block barriers order global and shared memory. The HLSL group
    // intrinsic supplies WorkgroupMemory; add UniformMemory for our SSBO ABI.
    // Allocate fresh constants so a user's equal-valued integer is untouched.
    bool opencl = module->getTargetTriple().getOS() != llvm::Triple::Vulkan;
    auto code = cudavm::decode_spirv_words(bytes, nullptr, !opencl);
    // OpenCL requires Int8 support. In particular, unused opaque transport
    // pointers may be emitted as byte pointers in a no-op kernel signature.
    // Only shader environments need the restricted i8 metadata cleanup.
    if (!opencl) code = cudavm::trim_spirv_resource_names(code);
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> constants;
    std::vector<uint32_t> added;
    size_t first_function = code.size(),first_type=code.size();
    std::vector<uint32_t> coherent;
    if(!opencl) for(size_t i=5;i<code.size();i+=code[i]>>16) {
        auto opcode=code[i]&0xffffu;
        if(opcode>=19 && opcode<=39 && first_type==code.size())first_type=i;
        if(opcode==59 && (code[i+3]==2 || code[i+3]==12))coherent.insert(coherent.end(),{(3u<<16)|71u,code[i+2],23u});
    }
    for (size_t i = 5; i < code.size(); i += code[i] >> 16) {
        auto count = code[i] >> 16, opcode = code[i] & 0xffffu;
        if (opcode == 43 && count == 4) constants[code[i + 2]] = {code[i + 1], code[i + 3]};
        if (opcode == 54 && first_function == code.size()) first_function = i;
        if ((opcode == 224 && count == 4) || (opcode == 225 && count == 3)) {
            auto operand=i+(opcode==224?3:2);
            auto original = constants.find(code[operand]);
            if (original == constants.end()) throw std::runtime_error("native SPIR-V barrier requires constant memory semantics");
            auto id = code[3]++;
            auto semantics=original->second.second;
            // LLVM's HLSL group barrier defaults to SeqCst. CUDA block/warp
            // rendezvous need acquire/release; explicit fences retain their
            // own ordering (SC shader fences use the common arena mutex).
            if(opcode==224)semantics=(semantics&~0x1eu)|0x8u;
            semantics|=0x100u|(opencl?0x200u:0x40u);
            added.insert(added.end(), {(4u << 16) | 43u, original->second.first, id, semantics});
            code[operand] = id;
        }
    }
    code.insert(code.begin() + first_function, added.begin(), added.end());
    // GLSL450 resource accesses participating in the device mutex/fence
    // protocol must bypass non-coherent per-invocation caches.
    if(!coherent.empty())code.insert(code.begin()+first_type,coherent.begin(),coherent.end());
    bytes.assign(reinterpret_cast<const char*>(code.data()), code.size() * 4);
    return bytes;
}
std::string emit_native_spirv_payload(const llvm::Module& input, llvm::json::Array* inventory) {
    std::vector<cudavm::shader_artifact> entries;
    for (const auto& kernel : input) if (!kernel.isDeclaration() && (kernel.hasFnAttribute("cudavm.kernel") || kernel.getCallingConv() == llvm::CallingConv::PTX_Kernel)) {
        cudavm::shader_artifact entry; entry.symbol = kernel.getName().str();
        try { entry.code = emit_native_spirv(*lower_native_spirv(input, entry.symbol)); }
        catch (const std::exception& error) { entry.diagnostic = error.what(); if(std::getenv("CUDAVM_TRACE_NATIVE")) std::fprintf(stderr,"native SPIR-V rejected %s: %s\n",entry.symbol.c_str(),error.what()); }
        try { entry.opencl_code = emit_native_spirv(*lower_native_spirv(input, entry.symbol, true)); }
        catch (const std::exception& error) { entry.opencl_diagnostic = error.what(); if(std::getenv("CUDAVM_TRACE_NATIVE")) std::fprintf(stderr,"native OpenCL rejected %s: %s\n",entry.symbol.c_str(),error.what()); }
        if (inventory) inventory->push_back(llvm::json::Object{{"symbol", entry.symbol}, {"available", !entry.code.empty()},
            {"bytes", int64_t(entry.code.size())}, {"diagnostic", entry.diagnostic},
            {"opencl_available", !entry.opencl_code.empty()}, {"opencl_bytes", int64_t(entry.opencl_code.size())}, {"opencl_diagnostic", entry.opencl_diagnostic}});
        entries.push_back(std::move(entry));
    }
    return cudavm::encode_spirv_artifacts(entries);
}
}
