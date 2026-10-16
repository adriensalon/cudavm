#include "dxil_emitter.hpp"
#include "dxil_artifact.hpp"
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <llvm/AsmParser/Parser.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Analysis/DXILResource.h>
#include <llvm/PassInfo.h>
#include <llvm/PassRegistry.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/Utils/Cloning.h>
extern "C" void LLVMInitializeDirectXTargetInfo();
extern "C" void LLVMInitializeDirectXTarget();
extern "C" void LLVMInitializeDirectXTargetMC();
extern "C" void LLVMInitializeDirectXAsmPrinter();
namespace cudavm {
struct directx_coherent_resources : llvm::ModulePass {
    static char ID;
    directx_coherent_resources():llvm::ModulePass(ID) {}
    void getAnalysisUsage(llvm::AnalysisUsage& usage) const override {
        usage.addRequired<llvm::DXILResourceWrapperPass>();usage.setPreservesAll();
    }
    bool runOnModule(llvm::Module&) override {
        // The CUDA arena and its representation/SC mutexes communicate across
        // workgroups. DeviceMemoryBarrier requires a globally coherent UAV for
        // this visibility. Set both the resource metadata and annotate props
        // through LLVM's shared resource analysis before either is serialized.
        for(auto& resource:getAnalysis<llvm::DXILResourceWrapperPass>().getResourceMap().uavs()) resource.GloballyCoherent=true;
        return false;
    }
};
char directx_coherent_resources::ID;
struct directx_pass_manager : llvm::legacy::PassManager {
    bool coherent_resources=false;
    void add(llvm::Pass* pass) override {
        auto* info=llvm::PassRegistry::getPassRegistry()->getPassInfo(pass->getPassID());
        if(info && llvm::StringRef(info->getPassArgument())=="dxil-translate-metadata") {
            llvm::legacy::PassManager::add(new directx_coherent_resources());coherent_resources=true;
        }
        llvm::legacy::PassManager::add(pass);
    }
};
std::unique_ptr<llvm::Module> lower_native_dxil(const llvm::Module& input,const std::string& symbol,const std::array<unsigned,3>& block) {
    uint64_t threads=1; for(auto dimension:block) { if(!dimension || dimension>1024) throw std::runtime_error("invalid native DXIL block shape"); threads*=dimension; }
    if(threads>1024 || block[2]>64) throw std::runtime_error("native DXIL block exceeds compute shader limits");
    return lower_native_shader(input,symbol,shader_environment::dxil,block);
}
std::string emit_native_dxil(const llvm::Module& input) {
    if(std::getenv("CUDAVM_TRACE_NATIVE")) std::fprintf(stderr,"native DXIL emit module=%s\n",input.getModuleIdentifier().c_str());
    // The DXIL writer serializes the context's metadata-kind registry, including
    // kinds absent from this shader. Isolate emission from earlier CUDA/SPIR-V
    // compilations so equivalent shaders produce identical containers.
    std::string assembly; llvm::raw_string_ostream assembly_out(assembly);
    input.print(assembly_out,nullptr);
    llvm::LLVMContext context; llvm::SMDiagnostic diagnostic;
    auto module=llvm::parseAssemblyString(assembly,diagnostic,context);
    if(!module) throw std::runtime_error("native DXIL emission could not clone target IR");
    static std::once_flag initialized;
    std::call_once(initialized,[]{ LLVMInitializeDirectXTargetInfo(); LLVMInitializeDirectXTarget(); LLVMInitializeDirectXTargetMC(); LLVMInitializeDirectXAsmPrinter(); });
    std::string error; auto* target=llvm::TargetRegistry::lookupTarget(module->getTargetTriple(),error);
    if(!target) throw std::runtime_error(error);
    llvm::TargetOptions settings;
    std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(module->getTargetTriple(),"","",settings,std::nullopt));
    if(!machine) throw std::runtime_error("native DXIL target machine creation failed");
    module->setDataLayout(machine->createDataLayout());
    llvm::SmallVector<char,0> storage; llvm::raw_svector_ostream out(storage); directx_pass_manager passes;
    if(machine->addPassesToEmitFile(passes,out,nullptr,llvm::CodeGenFileType::ObjectFile)) throw std::runtime_error("native DXIL container emission is unavailable");
    if(!passes.coherent_resources) throw std::runtime_error("native DXIL pipeline lacks the CUDA resource-coherence hook");
    passes.run(*module);
    std::string bytes(storage.begin(),storage.end()); cudavm::validate_dxil_container(bytes); return bytes;
}
std::string emit_native_dxil_payload(const llvm::Module& input, llvm::json::Array* inventory) {
    std::vector<cudavm::shader_artifact> entries;
    for(const auto& kernel:input) if(!kernel.isDeclaration() && (kernel.hasFnAttribute("cudavm.kernel") || kernel.getCallingConv()==llvm::CallingConv::PTX_Kernel)) {
        cudavm::shader_artifact entry; entry.symbol=kernel.getName().str();
        try {
            auto module=lower_native_dxil(input,entry.symbol); std::vector<cudavm::dxil_variant> variants;
            for(const auto& shape:cudavm::dxil_block_shapes) {
                module->getFunction("main")->addFnAttr("hlsl.numthreads",std::to_string(shape[0])+","+std::to_string(shape[1])+","+std::to_string(shape[2]));
                variants.push_back({shape,emit_native_dxil(*module)});
            }
            entry.code=cudavm::encode_dxil_variants(variants);
        } catch(const std::exception& error) { entry.diagnostic=error.what(); }
        if(inventory) inventory->push_back(llvm::json::Object{{"symbol",entry.symbol},{"available",!entry.code.empty()},
            {"bytes",int64_t(entry.code.size())},{"diagnostic",entry.diagnostic},{"shader_model","6.0"},{"validation_required",true},
            {"block_variants",int64_t(entry.code.empty()?0:cudavm::dxil_block_shapes.size())}});
        entries.push_back(std::move(entry));
    }
    return cudavm::encode_dxil_artifacts(entries);
}
}
