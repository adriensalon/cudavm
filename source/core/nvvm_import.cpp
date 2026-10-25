#include "nvvm_import.hpp"
#include "program.hpp"
#include "compiler_profiles.hpp"
#include "device_conversions.hpp"
#include "device_integers.hpp"
#include "device_math.hpp"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <set>
#include <llvm/AsmParser/Parser.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DiagnosticInfo.h>
#include <llvm/IR/DiagnosticPrinter.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace cudavm {
namespace {
constexpr auto annotations_key = "cudavm.nvvm.input.annotations";
constexpr auto flags_key = "cudavm.nvvm.input.flags";
constexpr auto debug_key = "cudavm.nvvm.input.debug";
bool supported_nvvm_intrinsic(const llvm::Function& function) {
    auto name = function.getName();
    if (!name.starts_with("llvm.nvvm.")) {
        // Keep target-specific and host introspection intrinsics away from the
        // NVPTX selector, which can terminate on an unlegalizable operation.
        constexpr std::string_view generic[]{"llvm.memcpy", "llvm.memmove", "llvm.memset", "llvm.assume",
            "llvm.lifetime.start", "llvm.lifetime.end", "llvm.stacksave", "llvm.stackrestore", "llvm.trap",
            "llvm.sqrt", "llvm.fma", "llvm.fmuladd", "llvm.fabs", "llvm.copysign", "llvm.minnum", "llvm.maxnum",
            "llvm.minimum", "llvm.maximum", "llvm.ctpop", "llvm.ctlz", "llvm.cttz", "llvm.bswap", "llvm.bitreverse",
            "llvm.fshl", "llvm.fshr", "llvm.sadd.with.overflow", "llvm.uadd.with.overflow", "llvm.ssub.with.overflow",
            "llvm.usub.with.overflow", "llvm.smul.with.overflow", "llvm.umul.with.overflow"};
        auto base = llvm::Intrinsic::getBaseName(function.getIntrinsicID());
        return std::find(std::begin(generic), std::end(generic), std::string_view(base)) != std::end(generic);
    }
    for (auto prefix : {"llvm.nvvm.read.ptx.sreg.tid.", "llvm.nvvm.read.ptx.sreg.ctaid.",
            "llvm.nvvm.read.ptx.sreg.ntid.", "llvm.nvvm.read.ptx.sreg.nctaid.", "llvm.nvvm.barrier.cta.",
            "llvm.nvvm.shfl.sync.", "llvm.nvvm.vote.", "llvm.nvvm.match.", "llvm.nvvm.redux.sync."})
        if (name.starts_with(prefix)) return true;
    for (auto exact : {"llvm.nvvm.read.ptx.sreg.warpsize", "llvm.nvvm.read.ptx.sreg.laneid", "llvm.nvvm.barrier0",
            "llvm.nvvm.barrier0.popc", "llvm.nvvm.barrier0.and", "llvm.nvvm.barrier0.or", "llvm.nvvm.activemask",
            "llvm.nvvm.bar.warp.sync", "llvm.nvvm.membar.cta", "llvm.nvvm.membar.gl", "llvm.nvvm.membar.sys",
            "llvm.nvvm.isspacep.global", "llvm.nvvm.isspacep.const", "llvm.nvvm.isspacep.shared", "llvm.nvvm.isspacep.local"})
        if (name == exact) return true;
    for (auto exact : {"llvm.nvvm.prmt", "llvm.nvvm.prmt.f4e", "llvm.nvvm.prmt.b4e", "llvm.nvvm.prmt.rc8",
            "llvm.nvvm.prmt.ecl", "llvm.nvvm.prmt.ecr", "llvm.nvvm.prmt.rc16"})
        if (name == exact) return true;
    for (const auto& conversion : device_conversions) if (name == conversion.intrinsic) return true;
    for (const auto& conversion : double_conversions) if (name == conversion.intrinsic) return true;
    for (const auto& operation : device_integer_operations) if (name == operation.intrinsic) return true;
    for (const auto& operation : device_dot_operations) if (name == operation.intrinsic) return true;
    for (const auto& operation : device_math_operations) if (operation.intrinsic && name == operation.intrinsic) return true;
    return false;
}
std::string protect_nvvm_metadata(std::string_view input) {
    std::string output;
    for (size_t i = 0; i < input.size();) {
        if (input[i] == ';') {
            auto end = input.find('\n', i); if (end == input.npos) end = input.size();
            output.append(input.substr(i, end - i)); i = end;
        } else if (input[i] == '"') {
            output += input[i++];
            while (i < input.size()) {
                char c = input[i++]; output += c;
                if (c == '\\' && i < input.size()) output += input[i++];
                else if (c == '"') break;
            }
        } else if (input[i] == '!') {
            auto first = i++;
            std::string name;
            while (i < input.size()) {
                unsigned char c = input[i];
                if (c == '\\') {
                    if (i + 2 >= input.size() || !std::isxdigit(static_cast<unsigned char>(input[i+1])) || !std::isxdigit(static_cast<unsigned char>(input[i+2])))
                        throw compiler_error(compiler_status::invalid, "NVVM: malformed metadata name escape");
                    auto hex = [](unsigned char value) { return std::isdigit(value) ? value - '0' : std::tolower(value) - 'a' + 10; };
                    name += char(hex(input[i+1]) * 16 + hex(input[i+2])); i += 3;
                } else if (std::isalnum(c) || c == '_' || c == '.' || c == '$' || c == '-') { name += char(c); ++i; }
                else break;
            }
            if (name.starts_with("cudavm.")) throw compiler_error(compiler_status::invalid, "NVVM: reserved cudavm metadata");
            if (name == "nvvm.annotations") output += "!cudavm.nvvm.input.annotations";
            else if (name == "llvm.module.flags") output += "!cudavm.nvvm.input.flags";
            else if (name == "llvm.dbg.cu") output += "!cudavm.nvvm.input.debug";
            else output.append(input.substr(first, i - first));
        } else output += input[i++];
    }
    return output;
}
void nvvm_require(bool condition, const std::string& message) {
    if (!condition) throw compiler_error(compiler_status::invalid, ("NVVM: " + message).c_str());
}
void rename_metadata(llvm::Module& module, llvm::StringRef from, llvm::StringRef to) {
    if (auto* original = module.getNamedMetadata(from)) {
        auto* replacement = module.getOrInsertNamedMetadata(to);
        for (auto* row : original->operands()) replacement->addOperand(row);
        module.eraseNamedMetadata(original);
    }
}
void verify_nvvm(llvm::Module& module) {
    std::string diagnostic;
    llvm::raw_string_ostream stream(diagnostic);
    nvvm_require(!llvm::verifyModule(module, &stream), diagnostic);
}
std::set<std::string> unresolved(const llvm::Module& module) {
    std::set<std::string> names;
    for (const auto& value : module.global_values())
        if (value.isDeclaration() && !value.use_empty() && !value.getName().starts_with("llvm.")) names.insert(value.getName().str());
    return names;
}
void compatible_symbols(const llvm::Module& target, const llvm::Module& source) {
    nvvm_require(target.getDataLayoutStr() == source.getDataLayoutStr(), "incompatible module data layouts");
    for (const auto& value : source.global_values()) {
        if (value.hasLocalLinkage()) continue;
        auto* existing = target.getNamedValue(value.getName());
        if (!existing || existing->hasLocalLinkage()) continue;
        nvvm_require(existing->getValueType() == value.getValueType() && existing->getAddressSpace() == value.getAddressSpace(),
            "incompatible declaration: " + value.getName().str());
        auto* left = llvm::dyn_cast<llvm::Function>(existing);
        auto* right = llvm::dyn_cast<llvm::Function>(&value);
        if (left && right) nvvm_require(left->getAttributes().getRetAttrs() == right->getAttributes().getRetAttrs()
                && left->getCallingConv() == right->getCallingConv(), "incompatible function ABI: " + value.getName().str());
        if (left && right) for (unsigned i = 0; i < left->arg_size(); ++i)
            nvvm_require(left->getParamByValType(i) == right->getParamByValType(i)
                && left->getParamStructRetType(i) == right->getParamStructRetType(i)
                && left->getParamAlign(i) == right->getParamAlign(i), "incompatible parameter ABI: " + value.getName().str());
    }
}
}

std::unique_ptr<llvm::Module> import_nvvm(llvm::LLVMContext& context, std::string_view bytes, const char* label, const char* architecture) {
    nvvm_require(!bytes.empty() && bytes.size() <= (size_t(1) << 30), "empty or oversized module");
    nvvm_require(architecture && std::string_view(architecture).starts_with("sm_"), "invalid target architecture");
    auto target = "compute_" + std::string(architecture + 3);
    nvvm_require(std::find(nvvm_compile_architectures.begin(), nvvm_compile_architectures.end(), target) != nvvm_compile_architectures.end(), "unsupported target architecture");
    auto name = label ? label : "nvvm.input";
    llvm::StringRef input(bytes.data(), bytes.size());
    std::unique_ptr<llvm::Module> module;
    // Delay LLVM's NVVM annotation upgrade until the external metadata has been
    // checked. Its upgrader assumes well-formed nodes and may assert otherwise.
    if (llvm::isBitcode(reinterpret_cast<const unsigned char*>(bytes.data()), reinterpret_cast<const unsigned char*>(bytes.data() + bytes.size()))) {
        auto parsed = llvm::getLazyBitcodeModule(llvm::MemoryBufferRef(input, name), context, false);
        if (!parsed) throw compiler_error(compiler_status::invalid, ("NVVM: " + llvm::toString(parsed.takeError())).c_str());
        module = std::move(*parsed);
        for (const auto& metadata : module->named_metadata()) nvvm_require(!metadata.getName().starts_with("cudavm."), "reserved cudavm input metadata");
        rename_metadata(*module, "nvvm.annotations", annotations_key);
        rename_metadata(*module, "llvm.module.flags", flags_key);
        nvvm_require(!module->getNamedMetadata("llvm.dbg.cu"), "debug information is outside the supported import profile");
        if (auto error = module->materializeAll())
            throw compiler_error(compiler_status::invalid, ("NVVM: " + llvm::toString(std::move(error))).c_str());
    } else {
        if (input.ends_with('\0')) input = input.drop_back();
        nvvm_require(!input.contains('\0'), "embedded NUL in text input");
        auto text = protect_nvvm_metadata(input.str());
        llvm::SMDiagnostic diagnostic;
        module = llvm::parseAssemblyString(text, diagnostic, context);
        if (!module) {
            std::string message; llvm::raw_string_ostream stream(message); diagnostic.print(name, stream);
            throw compiler_error(compiler_status::invalid, message.c_str());
        }
    }
    for (const auto& metadata : module->named_metadata())
        nvvm_require(!metadata.getName().starts_with("cudavm.") || metadata.getName() == annotations_key || metadata.getName() == flags_key || metadata.getName() == debug_key,
            "reserved cudavm input metadata");
    auto* version = module->getNamedMetadata("nvvmir.version");
    bool supported = version && version->getNumOperands();
    if (version) for (auto* row : version->operands()) {
        nvvm_require(row->getNumOperands() == 2 || row->getNumOperands() == 4, "invalid nvvmir.version tuple");
        unsigned values[4]{2,0,3,1};
        for (unsigned i = 0; i < row->getNumOperands(); ++i) {
            auto* value = llvm::mdconst::dyn_extract_or_null<llvm::ConstantInt>(row->getOperand(i));
            nvvm_require(value && value->getBitWidth() == 32, "invalid nvvmir.version field");
            values[i] = unsigned(value->getZExtValue());
        }
        supported &= values[0] == 2 && values[1] == 0 && values[2] == 3 && values[3] == 1;
    }
    if (!supported) throw compiler_error(compiler_status::nvvm_version_mismatch, "NVVM: expected IR 2.0 and debug metadata 3.1");
    nvvm_require(module->getTargetTriple().getArch() == llvm::Triple::nvptx64, "expected nvptx64 target");
    const auto& layout = module->getDataLayout();
    nvvm_require(!module->getDataLayoutStr().empty() && layout.isLittleEndian(), "expected an explicit little-endian CUDA data layout");
    for (unsigned space : {0u,1u,4u}) nvvm_require(layout.getPointerSizeInBits(space) == 64 && !layout.isNonIntegralAddressSpace(space), "expected integral 64-bit CUDA addresses");
    for (unsigned space : {3u,5u}) nvvm_require((layout.getPointerSizeInBits(space) == 32 || layout.getPointerSizeInBits(space) == 64)
        && !layout.isNonIntegralAddressSpace(space), "unsupported shared/local pointer layout");
    for (unsigned width : {1u,8u,16u,32u,64u,128u}) {
        auto* type = llvm::IntegerType::get(context, width);
        nvvm_require(layout.getABITypeAlign(type).value() == std::max(1u, width / 8), "unsupported integer data layout");
    }
    nvvm_require(layout.getABITypeAlign(llvm::Type::getFloatTy(context)).value() == 4
        && layout.getABITypeAlign(llvm::Type::getDoubleTy(context)).value() == 8, "unsupported floating-point data layout");
    nvvm_require(module->getModuleInlineAsm().empty() && module->ifunc_empty(), "module assembly and indirect resolvers are unsupported");
    nvvm_require(!module->getNamedMetadata("llvm.dbg.cu") && !module->getNamedMetadata(debug_key), "debug information is outside the supported import profile");
    if (auto* annotations = module->getNamedMetadata(annotations_key)) {
        for (auto* row : annotations->operands()) {
            nvvm_require(row->getNumOperands() >= 3 && row->getNumOperands() % 2, "invalid nvvm.annotations tuple");
            auto* function = llvm::mdconst::dyn_extract_or_null<llvm::Function>(row->getOperand(0));
            nvvm_require(function, "kernel annotation requires a function");
            for (unsigned i = 1; i < row->getNumOperands(); i += 2) {
                auto* key = llvm::dyn_cast_or_null<llvm::MDString>(row->getOperand(i));
                auto* value = llvm::mdconst::dyn_extract_or_null<llvm::ConstantInt>(row->getOperand(i + 1));
                nvvm_require(key && key->getString() == "kernel" && value && value->getBitWidth() == 32 && value->getZExtValue() <= 1,
                    "only boolean kernel annotations are supported");
                if (!value->isZero()) function->setCallingConv(llvm::CallingConv::PTX_Kernel);
            }
        }
        module->eraseNamedMetadata(annotations);
    }
    // Restore ordinary module flags only after LLVM verification can safely
    // inspect them; debug metadata translation is deliberately a separate profile.
    if (auto* flags = module->getNamedMetadata(flags_key)) {
        for (auto* row : flags->operands()) {
            nvvm_require(row->getNumOperands() == 3, "invalid module flag");
            auto* key = llvm::dyn_cast_or_null<llvm::MDString>(row->getOperand(1));
            nvvm_require(key && key->getString() != "Debug Info Version", "debug module flags are unsupported");
        }
        rename_metadata(*module, flags_key, "llvm.module.flags");
    }
    for (auto& variable : module->globals()) {
        nvvm_require(!variable.isThreadLocal() && !variable.hasAvailableExternallyLinkage(), "unsupported global storage linkage");
        nvvm_require(variable.getAddressSpace() == 0 || variable.getAddressSpace() == 1 || variable.getAddressSpace() == 3 || variable.getAddressSpace() == 4,
            "unsupported global address space");
        if (variable.getAddressSpace() == 3 && variable.hasInitializer())
            nvvm_require(llvm::isa<llvm::UndefValue>(variable.getInitializer()), "NVVM 2.0 shared storage requires undef initialization");
        nvvm_require(variable.getName() != "llvm.global_ctors" && variable.getName() != "llvm.global_dtors", "device initialization requires runtime support");
    }
    for (auto& function : *module) {
        nvvm_require(!function.isVarArg() && !function.hasPersonalityFn() && !function.hasPrefixData() && !function.hasPrologueData()
            && !function.hasAvailableExternallyLinkage(), "unsupported function ABI");
        nvvm_require(!function.getName().starts_with("llvm.") || function.isIntrinsic(), "unrecognized LLVM/NVVM intrinsic: " + function.getName().str());
        if (function.isIntrinsic()) nvvm_require(supported_nvvm_intrinsic(function), "unsupported LLVM/NVVM intrinsic: " + function.getName().str());
        for (auto& block : function) for (auto& instruction : block) {
            nvvm_require(!instruction.isEHPad() && !llvm::isa<llvm::InvokeInst, llvm::CallBrInst, llvm::IndirectBrInst, llvm::VAArgInst, llvm::ResumeInst>(instruction), "unsupported control flow");
            if (auto* call = llvm::dyn_cast<llvm::CallBase>(&instruction)) nvvm_require(!call->isInlineAsm(), "inline assembly requires a separate import profile");
        }
        if (function.getCallingConv() == llvm::CallingConv::PTX_Kernel)
            nvvm_require(!function.isDeclaration() && function.getReturnType()->isVoidTy(), "invalid kernel signature");
    }
    verify_nvvm(*module);
    compiler_options settings;
    settings.input = name; settings.architecture = architecture; settings.bitcode = program_input_identity(*module);
    mark_translation_unit(*module, settings, {});
    return module;
}

void compile_nvvm(const compiler_nvvm_request& request, compiler_result& result) {
    require_compiler_request(request.abi_version == 1 && request.count && request.modules && request.architecture
        && (request.optimization == 0 || request.optimization == 3) && request.verify_only <= 1 && request.lto <= 1 && request.contract_fma <= 1, "invalid NVVM request");
    compiler_program program;
    program.architecture = request.architecture;
    program.context.setDiagnosticHandlerCallBack([](const llvm::DiagnosticInfo* diagnostic, void* opaque) {
        llvm::raw_string_ostream stream(*static_cast<std::string*>(opaque));
        llvm::DiagnosticPrinterRawOStream printer(stream); diagnostic->print(printer); stream << '\n';
    }, &result.diagnostics);
    std::vector<std::unique_ptr<llvm::Module>> libraries;
    for (size_t i = 0; i < request.count; ++i) {
        const auto& input = request.modules[i];
        nvvm_require(input.data && input.size && input.lazy <= 1, "invalid module input");
        if (input.lazy) nvvm_require(llvm::isBitcode(reinterpret_cast<const unsigned char*>(input.data), reinterpret_cast<const unsigned char*>(input.data + input.size)), "lazy modules require bitcode");
        auto module = import_nvvm(program.context, {input.data, input.size}, input.name, request.architecture);
        // Composite linking owns one translation unit and one source ABI tuple.
        for (const char* key : {"cudavm.source.abi", "cudavm.translation.units", "cudavm.ir.stage"})
            if (auto* metadata = module->getNamedMetadata(key)) module->eraseNamedMetadata(metadata);
        if (input.lazy) { libraries.push_back(std::move(module)); continue; }
        if (!program.module) program.module = std::move(module);
        else { compatible_symbols(*program.module, *module); nvvm_require(!llvm::Linker::linkModules(*program.module, std::move(module)), "module link failed"); }
    }
    nvvm_require(bool(program.module), "at least one eagerly added module is required");
    // Repeat lazy-library selection to close dependencies across library order.
    std::set<std::string> imported;
    for (;;) {
        auto needed = unresolved(*program.module);
        bool changed = false;
        for (const auto& library : libraries) {
            bool selected = false;
            for (const auto& value : library->global_values())
                if (!value.isDeclaration() && !value.hasLocalLinkage() && needed.contains(value.getName().str()) && !imported.contains(value.getName().str())) selected = true;
            if (!selected) continue;
            compatible_symbols(*program.module, *library);
            auto clone = llvm::CloneModule(*library);
            auto previous_needed = needed;
            nvvm_require(!llvm::Linker::linkModules(*program.module, std::move(clone), llvm::Linker::LinkOnlyNeeded,
                [&](llvm::Module&, const llvm::StringSet<>& names) { for (const auto& name : names) imported.insert(name.getKey().str()); }), "lazy module link failed");
            needed = unresolved(*program.module);
            nvvm_require(needed != previous_needed, "lazy module selection made no progress"); changed = true;
        }
        if (!changed) break;
    }
    for (const auto& name : imported) if (auto* value = program.module->getNamedValue(name); value && !value->isDeclaration()) value->setLinkage(llvm::GlobalValue::InternalLinkage);
    verify_nvvm(*program.module);
    if (request.verify_only) return;
    if (!request.contract_fma) {
        std::vector<llvm::CallInst*> contractions;
        for (auto& function : *program.module) for (auto& block : function) for (auto& instruction : block) {
            if (llvm::isa<llvm::FPMathOperator>(instruction)) instruction.setHasAllowContract(false);
            if (auto* call = llvm::dyn_cast<llvm::CallInst>(&instruction); call && call->getIntrinsicID() == llvm::Intrinsic::fmuladd) contractions.push_back(call);
        }
        for (auto* call : contractions) {
            llvm::IRBuilder<> builder(call);
            auto* value = builder.CreateFAdd(builder.CreateFMul(call->getArgOperand(0), call->getArgOperand(1)), call->getArgOperand(2));
            call->replaceAllUsesWith(value); call->eraseFromParent();
        }
    }
    compiler_options settings;
    settings.input = "nvvm.program"; settings.architecture = request.architecture; settings.bitcode = program_input_identity(*program.module);
    mark_translation_unit(*program.module, settings, {});
    compiler_emit_request emission;
    emission.kind = request.lto ? compiler_output_kind::program_image : compiler_output_kind::ptx;
    emission.environment = request.lto ? compiler_environment::none : compiler_environment::cuda;
    // LTO remains relocatable, including unresolved helpers supplied by a later
    // public link. Text PTX compilation closes and lowers the device program.
    if (request.lto) { emit_compiler_program(program, emission, result); return; }
    auto prepared = prepare_compiler_program(program, {1, 0, 1, request.optimization});
    emit_compiler_program(*prepared, emission, result);
}
}
