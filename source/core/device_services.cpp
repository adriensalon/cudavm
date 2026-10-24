#include "device_services.hpp"
#include "device_sources.hpp"
#include <stdexcept>
#include <set>
#include <map>
#include <optional>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SHA256.h>
namespace cudavm {
struct shader_output_plan {
    std::string symbol;
    std::vector<std::pair<unsigned, uint64_t>> fields;
    std::optional<uint64_t> format;
    std::optional<std::string> literal;
};
namespace {
uint64_t scan_output_format(llvm::StringRef format) {
    unsigned count = 0, strings = 0; bool conversion = false;
    for (unsigned char type : format) {
        if (!conversion) { conversion = type == '%'; continue; }
        if (type == '%') { conversion = false; continue; }
        if (type == 'h' || type == 'l' || type == '-' || type == '+' || type == ' ' || type == '#' || type == '.' || (type >= '0' && type <= '9')) continue;
        bool star = type == '*';
        if (!star) conversion = false;
        if ((star || std::string_view("cdiouxXpeEfFgGaAs").find(char(type)) != std::string_view::npos) && count < 32) {
            if (type == 's') strings |= 1u << count;
            ++count;
        }
    }
    return (uint64_t(strings) << 32) | count;
}
std::optional<shader_output_plan> plan_output(const llvm::CallInst& call, const llvm::DataLayout& layout) {
    shader_output_plan plan;
    if (llvm::isa<llvm::ConstantPointerNull>(call.getArgOperand(0))) return {};
    auto* arguments = call.getArgOperand(1);
    if (!llvm::isa<llvm::ConstantPointerNull>(arguments)) {
        llvm::APInt offset(layout.getIndexTypeSizeInBits(arguments->getType()), 0);
        auto* base = arguments->stripAndAccumulateConstantOffsets(layout, offset, true);
        auto* allocation = llvm::dyn_cast<llvm::AllocaInst>(base);
        if (!offset.isZero() || !allocation || !llvm::isa<llvm::ConstantInt>(allocation->getArraySize()) ||
            !llvm::cast<llvm::ConstantInt>(allocation->getArraySize())->isOne()) return {};
        auto* structure = llvm::dyn_cast<llvm::StructType>(allocation->getAllocatedType());
        if (!structure) return {};
        auto* fields = layout.getStructLayout(structure);
        for (unsigned index = 0; index < std::min(32u, structure->getNumElements()); ++index) {
            auto* type = structure->getElementType(index);
            unsigned code = type->isPointerTy() ? 7 : type->isDoubleTy() ? 5 : type->isIntegerTy(64) ? 4 : type->isIntegerTy(32) ? 2 : 0;
            if (!code) return {};
            plan.fields.emplace_back(code, fields->getElementOffset(index));
        }
    }
    llvm::StringRef format;
    auto* constant = llvm::dyn_cast<llvm::GlobalVariable>(llvm::getUnderlyingObject(call.getArgOperand(0)));
    if (constant && constant->hasLocalLinkage() && constant->isConstant() && llvm::getConstantStringInfo(call.getArgOperand(0), format)) {
        plan.format = scan_output_format(format);
        if (format.size() <= 256) plan.literal = format.str();
    }
    plan.symbol = "__cudavm_service_printf";
    for (auto [code, offset] : plan.fields) plan.symbol += "_" + std::to_string(code) + "x" + std::to_string(offset);
    plan.symbol += plan.format ? "_c" + std::to_string(*plan.format) : "_dynamic";
    if (plan.literal) {
        auto digest = llvm::SHA256::hash(llvm::ArrayRef(reinterpret_cast<const uint8_t*>(plan.literal->data()), plan.literal->size()));
        constexpr char hex[] = "0123456789abcdef"; plan.symbol += "_t";
        for (auto byte : digest) { plan.symbol += hex[byte >> 4]; plan.symbol += hex[byte & 15]; }
    }
    return plan;
}
std::string output_plan_source(const shader_output_plan& plan) {
    std::string format;
    if (plan.literal) {
        auto bytes = *plan.literal + '\0';
        std::string words;
        for (size_t at = 0; at < bytes.size(); at += 4) {
            uint32_t word = 0;
            for (unsigned byte = 0; byte < 4 && at + byte < bytes.size(); ++byte)
                word |= uint32_t(uint8_t(bytes[at + byte])) << (byte * 8);
            words += render_device_source("printf_word.cu.in", {
                {"index", std::to_string(144 + at / 4)}, {"value", std::to_string(word)}});
        }
        format = render_device_source("printf_literal.cu.in", {
            {"end", std::to_string(576 + bytes.size())}, {"words", words}});
    } else format = device_source("printf_text.cu");
    std::string fields;
    for (size_t index = 0; index < plan.fields.size(); ++index) {
        auto [code, offset] = plan.fields[index];
        auto expression = code == 7 ? render_device_source("printf_pointer_code.cu.in",
            {{"mask", std::to_string(1u << index)}}) : std::to_string(code);
        fields += render_device_source("printf_field.cu.in", {
            {"index", std::to_string(index)}, {"code", expression},
            {"load_type", code >= 3 ? "u64" : "unsigned"}, {"offset", std::to_string(offset)},
            {"slot", std::to_string(16 + index * 4)},
            {"string", code == 7 ? device_source("printf_string.cu") : std::string_view{}}});
    }
    return render_device_source("printf_plan.cu.in", {
        {"symbol", plan.symbol}, {"parsed", plan.format ? std::to_string(*plan.format) + "ull" : "__cudavm_service_scan(format)"},
        {"field_count", std::to_string(plan.fields.size())}, {"format", format}, {"fields", fields}});
}
}
uint32_t shader_service_requirements(const llvm::Function& entry) {
    uint32_t flags = 0;
    std::set<const llvm::Function*> visited;
    std::vector<const llvm::Function*> pending{&entry};
    while (!pending.empty()) {
        auto* function = pending.back(); pending.pop_back();
        if (!visited.insert(function).second) continue;
        for (const auto& block : *function) for (const auto& instruction : block)
            if (auto* call = llvm::dyn_cast<llvm::CallInst>(&instruction); call && call->getCalledFunction()) {
                auto* target = call->getCalledFunction(); auto name = target->getName();
                flags |= device_service(name.str());
                if (name == "__cudavm_service_buffer") {
                    auto* index = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(0));
                    if (index) flags |= index->isZero() ? device_service_heap : device_service_output;
                }
                if (name == "__cudavm_service_assert") flags |= device_service_output | device_service_assert;
                if (!target->isDeclaration()) pending.push_back(target);
            }
    }
    return flags;
}
void validate_device_services(const llvm::Module& module) {
    for (const auto& function : module) if (!function.use_empty() && device_service(function.getName().str())) {
        auto name = function.getName(); auto* signature = function.getFunctionType();
        auto pointer = [&](unsigned index) { return index < signature->getNumParams() && signature->getParamType(index)->isPointerTy(); };
        auto integer = [&](unsigned index, unsigned bits) { return index < signature->getNumParams() && signature->getParamType(index)->isIntegerTy(bits); };
        bool valid = !signature->isVarArg();
        if (name == "malloc" || name == "__nv_aligned_device_malloc" || name == "__nv_aligned_device_malloc_impl")
            valid &= signature->getReturnType()->isPointerTy() && signature->getNumParams() == (name == "malloc" ? 1u : 2u) && integer(0, 64) && (name == "malloc" || integer(1, 64));
        else if (name == "free") valid &= signature->getReturnType()->isVoidTy() && signature->getNumParams() == 1 && pointer(0);
        else if (name == "vprintf") valid &= signature->getReturnType()->isIntegerTy(32) && signature->getNumParams() == 2 && pointer(0) && pointer(1);
        else valid &= signature->getReturnType()->isVoidTy() && signature->getNumParams() == 5 && pointer(0) && pointer(1) && integer(2, 32) && pointer(3) && integer(4, 64);
        if (!valid) throw std::runtime_error("invalid CUDA device service signature: " + name.str());
    }
}
void link_shader_services(llvm::Module& module) {
    validate_device_services(module);
    bool needed = false;
    for (auto& function : module) needed |= function.isDeclaration() && !function.use_empty() && device_service(function.getName().str());
    if (!needed) return;
    std::map<std::string, shader_output_plan> plans;
    std::vector<std::pair<llvm::CallInst*, std::string>> planned_calls;
    if (auto* output = module.getFunction("vprintf")) for (auto* user : output->users())
        if (auto* call = llvm::dyn_cast<llvm::CallInst>(user)) if (auto plan = plan_output(*call, module.getDataLayout())) {
            planned_calls.emplace_back(call, plan->symbol); plans.emplace(plan->symbol, std::move(*plan));
        }
    std::string source(device_source("device_services.cu"));
    for (const auto& [name, plan] : plans) source += output_plan_source(plan);
    auto files = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
    files->addFile("/cudavm_services.cu", 0, llvm::MemoryBuffer::getMemBufferCopy(source));
    const char* args[]{"-triple", "nvptx64-nvidia-cuda", "-aux-triple", LLVM_HOST_TRIPLE, "-fcuda-is-device", "-x", "cuda", "-std=c++17",
        "-target-cpu", "sm_75", "-emit-llvm", "-disable-llvm-passes", "-relaxed-aliasing", "-O1", "-nostdsysteminc", "-nobuiltininc", "/cudavm_services.cu"};
    std::string log; llvm::raw_string_ostream stream(log);
    auto invocation = std::make_shared<clang::CompilerInvocation>();
    clang::CompilerInstance compiler(invocation);
    compiler.createDiagnostics(new clang::TextDiagnosticPrinter(stream, invocation->getDiagnosticOpts()));
    if (!clang::CompilerInvocation::CreateFromArgs(*invocation, args, compiler.getDiagnostics())) throw std::runtime_error(log);
    compiler.setVirtualFileSystem(files); compiler.createFileManager(); compiler.createSourceManager();
    clang::EmitLLVMOnlyAction action(&module.getContext());
    if (!compiler.ExecuteAction(action)) throw std::runtime_error("device services compilation failed: " + log);
    auto services = action.takeModule();
    if (!services) throw std::runtime_error("device services frontend produced no IR");
    services->setDataLayout(module.getDataLayout());
    if (auto* flags = services->getModuleFlagsMetadata()) services->eraseNamedMetadata(flags);
    // These bounded claims never wait for an invocation holding a lock. Keep
    // them native even when an ordinary user polling loop needs continuations.
    for (auto& function : *services) for (auto& block : function) for (auto& instruction : block)
        if (instruction.isAtomic()) instruction.setMetadata("cudavm.nonblocking", llvm::MDNode::get(module.getContext(), {}));
    for (auto& function : *services) if (!function.isDeclaration() &&
        (function.getName() == "free" || function.getName() == "vprintf" || function.getName().starts_with("__cudavm_service_printf")
            || function.getName().starts_with("_ZN6cudavm8allocate") || function.getName().starts_with("_ZN6cudavm6record")))
        function.addFnAttr("cudavm.service.helper");
    if (auto* output = services->getFunction("vprintf")) output->addFnAttr("cudavm.service.storage");
    for (auto& function : *services) if (function.getName().starts_with("__cudavm_service_printf")) function.addFnAttr("cudavm.service.storage");
    if (llvm::Linker::linkModules(module, std::move(services))) throw std::runtime_error("device services linking failed");
    for (auto [call, name] : planned_calls) {
        llvm::IRBuilder<> builder(call);
        auto* replacement = builder.CreateCall(module.getFunction(name), {call->getArgOperand(0), call->getArgOperand(1)});
        call->replaceAllUsesWith(replacement); call->eraseFromParent();
    }
    if (auto* output = module.getFunction("vprintf")) {
        std::vector<llvm::CallInst*> calls;
        for (auto* user : output->users()) if (auto* call = llvm::dyn_cast<llvm::CallInst>(user)) calls.push_back(call);
        for (auto* call : calls) {
            llvm::IRBuilder<> builder(call); llvm::Value* replacement = nullptr;
            if (llvm::isa<llvm::ConstantPointerNull>(call->getArgOperand(0))) replacement = builder.getInt32(-1);
            else if (llvm::isa<llvm::ConstantPointerNull>(call->getArgOperand(1))) {
                llvm::StringRef format;
                if (llvm::getConstantStringInfo(call->getArgOperand(0), format)) {
                    bool literal = true;
                    for (size_t at = 0; at < format.size(); ++at) if (format[at] == '%')
                        if (++at == format.size() || format[at] != '%') { literal = false; break; }
                    if (literal) replacement = builder.CreateCall(module.getFunction("__cudavm_service_literal"), {call->getArgOperand(0)});
                }
            }
            if (replacement) { call->replaceAllUsesWith(replacement); call->eraseFromParent(); }
        }
    }
    for (auto& function : module) if (device_service(function.getName().str()) && !function.isDeclaration()) {
        if (function.getName() == "__assertfail") function.addFnAttr(llvm::Attribute::NoReturn);
        if (!function.hasFnAttribute("cudavm.service.helper")) {
            function.removeFnAttr(llvm::Attribute::NoInline); function.addFnAttr(llvm::Attribute::AlwaysInline);
        }
    }
}
}
