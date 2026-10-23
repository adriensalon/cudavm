#include "compiler.hpp"
#include "shader_lowering.hpp"
#include "shader_collectives.hpp"
#include "shader_local.hpp"
#include "module_descriptor.hpp"
#include <algorithm>
#include <map>
#include <set>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/Support/raw_ostream.h>
namespace cudavm {
namespace {
using global_indices = std::map<const llvm::GlobalVariable*, uint32_t>;
// Shared variables are per-kernel resources. After helper inlining, walk the
// kernel's constant operands (including GEP/cast expressions), rather than
// charging every shared global in the linked multi-kernel module.
void collect_shared(const llvm::Value* value,std::set<const llvm::Constant*>& visited,
    std::set<const llvm::GlobalVariable*>& globals) {
    auto* constant=llvm::dyn_cast<llvm::Constant>(value);
    if(!constant || !visited.insert(constant).second)return;
    if(auto* global=llvm::dyn_cast<llvm::GlobalVariable>(constant)) {
        if(global->getAddressSpace()==3)globals.insert(global);
        return;
    }
    for(const auto& operand:constant->operands())collect_shared(operand.get(),visited,globals);
}
// Initializers use the device DataLayout, including aggregate padding. Pointer
// bytes are relocated after the Driver assigns module-local CUDA addresses.
struct global_initializer {
    const llvm::DataLayout& dl;
    const global_indices& indices;
    cudavm::global_descriptor& out;
    [[noreturn]] void fail(const llvm::Constant& c) {
        std::string text; llvm::raw_string_ostream stream(text); c.print(stream);
        throw std::runtime_error("shader global initializer for " + out.symbol + " does not support: " + text);
    }
    const llvm::GlobalVariable* pointer(const llvm::Constant* c, uint64_t& addend) {
        if(auto* g=llvm::dyn_cast<llvm::GlobalVariable>(c)) return g;
        if(auto* e=llvm::dyn_cast<llvm::ConstantExpr>(c)) {
            if(e->getOpcode()==llvm::Instruction::BitCast || e->getOpcode()==llvm::Instruction::AddrSpaceCast) return pointer(e->getOperand(0),addend);
            if(auto* gep=llvm::dyn_cast<llvm::GEPOperator>(e)) {
                llvm::APInt offset(64,0);
                if(!gep->accumulateConstantOffset(dl,offset))fail(*c);
                addend+=offset.getZExtValue(); return pointer(llvm::cast<llvm::Constant>(gep->getPointerOperand()),addend);
            }
        }
        fail(*c);
    }
    void scalar(uint64_t bits, unsigned bytes, size_t offset) {
        for(unsigned i=0;i<bytes;++i)out.initializer[offset+i]=char(bits>>(i*8));
    }
    void relocate(const llvm::Constant* address, size_t offset) {
        uint64_t addend=0; auto* target=pointer(address,addend); auto index=indices.find(target);
        if(index==indices.end() || addend>dl.getTypeAllocSize(target->getValueType()).getFixedValue())fail(*address);
        out.relocations.push_back({uint32_t(offset),index->second,addend});
    }
    void write(const llvm::Constant* c, llvm::Type* type, size_t offset) {
        if(llvm::isa<llvm::ConstantAggregateZero>(c) || llvm::isa<llvm::ConstantPointerNull>(c) || llvm::isa<llvm::UndefValue>(c) || llvm::isa<llvm::PoisonValue>(c))return;
        if(auto* n=llvm::dyn_cast<llvm::ConstantInt>(c)) { if(n->getBitWidth()>64)fail(*c);scalar(n->getZExtValue(),unsigned(dl.getTypeStoreSize(type).getFixedValue()),offset);return; }
        if(auto* n=llvm::dyn_cast<llvm::ConstantFP>(c)) { auto bits=n->getValueAPF().bitcastToAPInt();if(bits.getBitWidth()>64)fail(*c);scalar(bits.getZExtValue(),unsigned(dl.getTypeStoreSize(type).getFixedValue()),offset);return; }
        if(auto* expression=llvm::dyn_cast<llvm::ConstantExpr>(c); expression && expression->getOpcode()==llvm::Instruction::PtrToInt) {
            // V3 relocations store a complete 64-bit module address. Narrow or
            // widened address representations need a different storage contract.
            auto* address=expression->getOperand(0);
            if(!type->isIntegerTy(64) || dl.getPointerTypeSizeInBits(address->getType())!=64
                || dl.isNonIntegralPointerType(address->getType()))fail(*c);
            relocate(address,offset);return;
        }
        if(type->isPointerTy()) {
            relocate(c,offset);return;
        }
        if(auto* structure=llvm::dyn_cast<llvm::StructType>(type)) {
            auto* layout=dl.getStructLayout(structure);
            for(unsigned i=0;i<structure->getNumElements();++i) { auto* element=c->getAggregateElement(i);if(!element)fail(*c);write(element,structure->getElementType(i),offset+layout->getElementOffset(i)); }
            return;
        }
        llvm::Type* element=nullptr; uint64_t count=0;
        if(auto* array=llvm::dyn_cast<llvm::ArrayType>(type)) {element=array->getElementType();count=array->getNumElements();}
        if(element) {
            auto stride=dl.getTypeAllocSize(element).getFixedValue();
            for(uint64_t i=0;i<count;++i) {auto* value=c->getAggregateElement(unsigned(i));if(!value)fail(*c);write(value,element,offset+i*stride);}return;
        }
        fail(*c);
    }
};
}
std::string emit_native_module(const llvm::Module& input) {
    auto m = prepare_native_shader_module(input);
    global_indices indices; std::vector<cudavm::global_descriptor> globals;
    const auto& dl=m->getDataLayout();
    for(auto& g:m->globals()) if(g.getAddressSpace()==1 || g.getAddressSpace()==4) {
        // Declaration-only globals have no module storage. CUDA builtin helper
        // bodies can still reference threadIdx/blockIdx after kernel inlining.
        // Actual unresolved kernel/initializer data is rejected by value()
        // or relocation lookup because it has no defined global index.
        if(g.isDeclaration())continue;
        if(globals.size()>=4096)throw std::runtime_error("too many shader globals");
        auto bytes=dl.getTypeAllocSize(g.getValueType()).getFixedValue();
        auto alignment=g.getAlign().valueOrOne().value();
        if(!g.hasInitializer() || !bytes || bytes>=(1u<<30) || alignment>256 || !dl.isLittleEndian() || dl.getPointerSize()!=8)
            throw std::runtime_error("shader module requires defined, nonempty globals with alignment at most 256: "+g.getName().str());
        indices[&g]=uint32_t(globals.size()); globals.push_back({g.getName().str(),uint32_t(alignment),std::string(size_t(bytes),'\0'),{}});
    }
    for(auto [g,index]:indices) global_initializer{dl,indices,globals[index]}.write(g->getInitializer(),g->getValueType(),0);
    std::vector<cudavm::kernel_descriptor> entries;
    for (auto& f : *m) if (!f.isDeclaration() && (f.hasFnAttribute("cudavm.kernel") || f.getCallingConv() == llvm::CallingConv::PTX_Kernel)) {
        cudavm::kernel_descriptor entry; entry.symbol = f.getName().str();
        uint64_t offset = 0;
        for (auto& arg : f.args()) {
            auto* byval = f.getParamByValType(arg.getArgNo()); auto* t = byval ? byval : arg.getType();
            auto align = byval && f.getParamAlign(arg.getArgNo()) ? f.getParamAlign(arg.getArgNo())->value() : dl.getABITypeAlign(t).value();
            offset = llvm::alignTo(offset, align);
            auto bytes = dl.getTypeAllocSize(t).getFixedValue();
            if (!bytes || bytes > 65536 || offset + bytes > 65536) throw std::runtime_error("native kernel parameter packet exceeds 64 KiB");
            entry.parameters.push_back({uint32_t(bytes), uint32_t(align), uint32_t(offset), !byval && t->isPointerTy() ? 1u : 0u});
            offset += bytes;
        }
        std::set<const llvm::Constant*> visited;
        std::set<const llvm::GlobalVariable*> used;
        for (auto& i : llvm::instructions(f)) for (const auto& operand : i.operands()) collect_shared(operand.get(), visited, used);
        uint64_t shared = 0, dynamic_alignment = 16;
        for (const auto& g : m->globals()) if (used.contains(&g)) {
            auto alignment = g.getAlign().value_or(dl.getABITypeAlign(g.getValueType())).value();
            auto bytes = dl.getTypeAllocSize(g.getValueType()).getFixedValue();
            if (!bytes) { dynamic_alignment = std::max(dynamic_alignment, alignment); continue; }
            shared = llvm::alignTo(shared, alignment);
            shared += bytes;
        }
        if (shared > 32768) throw std::runtime_error("native static shared memory exceeds 32 KiB");
        shared=llvm::alignTo(shared,dynamic_alignment);
        bool collectives=native_atomic_polling(f),named_barriers=false;
        for(const auto& bb:f) for(const auto& input:bb) if(auto* call=llvm::dyn_cast<llvm::CallInst>(&input);call && call->getCalledFunction()) {
            auto* function=call->getCalledFunction(); auto name=function->hasFnAttribute("cudavm.cuda.origin")?function->getFnAttribute("cudavm.cuda.origin").getValueAsString():function->getName();
            auto operation=native_collective(name); collectives|=operation>=0; named_barriers|=operation>=36;
        }
        if(collectives) shared+=native_collective_shared_bytes;
        if(named_barriers) shared+=native_named_barrier_shared_bytes;
        if(shared>32768) throw std::runtime_error("native shared storage and collective scratch exceed 32 KiB");
        entry.shared_bytes = uint32_t(shared);
        entry.local_bytes=shader_local_layout(f).bytes();
        if (entry.local_bytes == shader_local_capacity) entry.stack_bytes = shader_local_capacity;
        entries.push_back(std::move(entry));
    }
    return cudavm::encode_module_descriptor(entries, globals);
}
}
