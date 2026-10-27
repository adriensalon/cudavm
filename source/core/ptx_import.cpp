#include "ptx_import.hpp"
#include "compiler.hpp"
#include "compiler_profiles.hpp"
#include "device_conversions.hpp"
#include "device_services.hpp"
#include "device_math.hpp"
#include "device_integers.hpp"
#include "integer_ir.hpp"
#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <set>
#include <vector>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/GlobalAlias.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Linker/Linker.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Error.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace cudavm {
namespace {
struct ptx_token { std::string text; unsigned line, column; };
[[noreturn]] void ptx_fail(const ptx_token& token, const std::string& reason, bool unsupported = false, bool version = false, bool architecture = false) {
    throw ptx_import_error(unsupported, "PTX " + std::to_string(token.line) + ":" +
        std::to_string(token.column) + ": " + reason + " (" + token.text + ")", version, architecture);
}
std::vector<ptx_token> ptx_tokens(std::string_view input) {
    // Public binary/container APIs include the PTX text terminator in their
    // byte extent. Embedded NULs still reach the lexer and fail validation.
    if (!input.empty() && input.back() == '\0') input.remove_suffix(1);
    std::vector<ptx_token> result;
    size_t at = 0; unsigned line = 1, column = 1;
    auto advance = [&] { if (input[at++] == '\n') { ++line; column = 1; } else ++column; };
    while (at < input.size()) {
        if (std::isspace(static_cast<unsigned char>(input[at]))) { advance(); continue; }
        if (input.substr(at, 2) == "//") { while (at < input.size() && input[at] != '\n') advance(); continue; }
        if (input.substr(at, 2) == "/*") {
            ptx_token start{"/*", line, column}; advance(); advance();
            while (at < input.size() && input.substr(at, 2) != "*/") advance();
            if (at == input.size()) ptx_fail(start, "unterminated comment");
            advance(); advance(); continue;
        }
        ptx_token token{{}, line, column};
        if (input[at] == '"') {
            token.text += input[at]; advance();
            while (at < input.size() && input[at] != '"' && input[at] != '\n') {
                if (input[at] == '\\') {
                    token.text += input[at]; advance();
                    if (at == input.size() || input[at] == '\n') ptx_fail(token,"unterminated string escape");
                }
                token.text += input[at]; advance();
            }
            if (at == input.size() || input[at] != '"') ptx_fail(token, "unsupported or unterminated string literal", true);
            token.text += input[at]; advance();
        } else if (std::string_view("{}()[]<>,;:@!+-|=").find(input[at]) != std::string_view::npos) {
            token.text += input[at]; advance();
        } else {
            while (at < input.size()) {
                auto c = static_cast<unsigned char>(input[at]);
                bool exponent_sign=(c=='+' || c=='-') && !token.text.empty() && std::isdigit(static_cast<unsigned char>(token.text.front())) &&
                    (token.text.back()=='e' || token.text.back()=='E');
                if (!std::isalnum(c) && c != '_' && c != '$' && c != '%' && c != '.' && !exponent_sign) break;
                token.text += char(c); advance();
            }
            if (token.text.empty()) ptx_fail(token, "unrecognized character");
        }
        result.push_back(std::move(token));
        if (result.size() > 1000000) ptx_fail(result.back(), "token resource limit exceeded", true);
    }
    result.push_back({"<eof>", line, column}); return result;
}
uint64_t ptx_unsigned(const ptx_token& token) {
    auto text = llvm::StringRef(token.text);
    if (text.ends_with("U")) text = text.drop_back();
    uint64_t value = 0;
    if (text.empty() || text.getAsInteger(0, value)) ptx_fail(token, "invalid integer literal");
    return value;
}
uint64_t ptx_floating_bits(const ptx_token& token,unsigned bits,bool negative) {
    llvm::APFloat value(llvm::APFloat::IEEEdouble());
    auto converted=value.convertFromString(token.text,llvm::APFloat::rmNearestTiesToEven);
    if(!converted){llvm::consumeError(converted.takeError());ptx_fail(token,"invalid floating literal");}
    if(negative)value.changeSign();
    bool loses=false;
    if(bits!=64)value.convert(bits==16?llvm::APFloat::IEEEhalf():llvm::APFloat::IEEEsingle(),llvm::APFloat::rmNearestTiesToEven,&loses);
    return value.bitcastToAPInt().getZExtValue();
}
bool ptx_identifier(const std::string& text) {
    if (text.empty() || (!std::isalpha(static_cast<unsigned char>(text[0])) && text[0] != '_' && text[0] != '$')) return false;
    return std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isalnum(c) || c == '_' || c == '$'; });
}
struct ptx_type {
    unsigned bits; char kind;
    bool signed_integer() const { return kind == 's'; }
    bool accepts_register(ptx_type other) const {
        if (kind == 'p' || other.kind == 'p') return kind == other.kind;
        if (kind == 'b' || other.kind == 'b') return true;
        return (kind == 'f') == (other.kind == 'f');
    }
    llvm::IntegerType* storage(llvm::LLVMContext& context) const { return llvm::IntegerType::get(context, bits); }
    llvm::Type* value(llvm::LLVMContext& context) const { return kind == 'f' ? (bits==16?llvm::Type::getHalfTy(context):bits==64?llvm::Type::getDoubleTy(context):llvm::Type::getFloatTy(context)) : storage(context); }
};
ptx_type ptx_scalar(const ptx_token& token, bool memory = false) {
    auto name = token.text; if (name.starts_with('.')) name.erase(0, 1);
    if (name == "f16x2" && !memory) return {32, 'h'};
    if (name == "pred" && !memory) return {1, 'p'};
    for (auto prefix : {'b', 'u', 's', 'f'}) for (unsigned bits : {8u, 16u, 32u, 64u})
        if (name == prefix + std::to_string(bits)) {
            if ((prefix == 'f' && bits == 8) || (!memory && bits == 8)) break;
            return {bits, prefix};
        }
    ptx_fail(token, "unsupported scalar type", true);
}
struct ptx_instruction {
    ptx_token opcode;
    std::string predicate;
    bool negate = false, label = false;
    std::vector<std::vector<ptx_token>> operands;
};
struct ptx_register { ptx_type type; llvm::AllocaInst* slot; };
struct ptx_parameter {
    ptx_token name;
    ptx_type type;
    uint64_t count = 1;
    unsigned alignment = 1;
    bool array = false;
    bool register_parameter = false;
    uint64_t bytes() const { return count * (type.bits / 8); }
    llvm::Type* storage(llvm::LLVMContext& context) const {
        return array ? llvm::ArrayType::get(type.storage(context), count) : static_cast<llvm::Type*>(type.storage(context));
    }
};
using ptx_signature = std::pair<std::vector<ptx_parameter>, std::vector<ptx_parameter>>;
std::string ptx_signature_key(const ptx_signature& signature) {
    std::string key;
    for (const auto* list : {&signature.first, &signature.second}) {
        key += "(";
        for (const auto& p : *list) key += std::string(p.register_parameter ? "r" : "p") +
            (p.type.kind == 'f' ? "f" : "i") + std::to_string(p.type.bits) + ":" +
            std::to_string(p.count) + ":" + std::to_string(p.alignment) + (p.array ? "a;" : ";");
        key += ")";
    }
    return key;
}
llvm::FunctionType* ptx_function_type(llvm::LLVMContext& context, const ptx_signature& signature) {
    std::vector<llvm::Type*> inputs;
    for (const auto& p : signature.first) inputs.push_back(p.storage(context));
    auto* output = signature.second.empty() ? llvm::Type::getVoidTy(context) : signature.second[0].storage(context);
    return llvm::FunctionType::get(output, inputs, false);
}
struct ptx_call_prototype {
    ptx_signature signature;
    std::vector<ptx_token> targets;
};
struct ptx_body {
    ptx_token name;
    llvm::Function* function;
    std::vector<ptx_parameter> parameters, results;
    std::map<std::string, ptx_register> registers;
    std::map<std::string, std::pair<ptx_parameter, llvm::Value*>> slots;
    std::map<std::string, llvm::Value*> locals;
    std::vector<ptx_instruction> instructions;
    std::map<std::string, ptx_call_prototype> prototypes;
    std::map<std::string,std::vector<std::string>> vectors;
};
struct ptx_global {
    ptx_parameter declaration;
    llvm::GlobalVariable* variable;
    std::vector<std::vector<ptx_token>> initializer;
};

class ptx_parser {
    std::vector<ptx_token> tokens;
    size_t cursor = 0;
    llvm::LLVMContext& context;
    std::unique_ptr<llvm::Module> module;
    std::vector<ptx_body> bodies;
    std::map<std::string, ptx_signature> signatures;
    std::map<std::string,llvm::GlobalValue::LinkageTypes> declared_linkages;
    std::vector<ptx_global> globals;
    std::map<std::string, ptx_token> symbols;
    unsigned symbol_counter = 0;
    unsigned source_sm = 0;
    unsigned source_version = 0;
    bool relocatable = false;
    std::vector<std::pair<ptx_token,ptx_token>> aliases;
    const ptx_token& peek() const { return tokens.at(cursor); }
    ptx_token take() { auto token = peek(); if (cursor + 1 < tokens.size()) ++cursor; return token; }
    bool accept(const char* text) { if (peek().text != text) return false; take(); return true; }
    ptx_token expect(const char* text) { auto token = take(); if (token.text != text) ptx_fail(token, std::string("expected '") + text + "'"); return token; }
    ptx_token identifier() { auto token = take(); if (!ptx_identifier(token.text)) ptx_fail(token, "expected identifier"); return token; }

    bool debug_directive() {
        if (peek().text != ".file" && peek().text != ".loc") return false;
        auto directive=take(); auto file=take();
        if (!ptx_unsigned(file)) ptx_fail(file,"invalid debug file index");
        if (directive.text==".file") {
            auto path=take(); if(path.text.size()<2 || path.text.front()!='"' || path.text.back()!='"')ptx_fail(path,"expected debug filename");
            if(accept(",")){ptx_unsigned(take());expect(",");ptx_unsigned(take());}
        } else {
            ptx_unsigned(take());ptx_unsigned(take());
            while(accept(",")) {
                auto property=take();
                if(property.text=="function_name") {
                    auto value=take();if(value.text.empty() || (value.text.front()!='"' && !ptx_identifier(value.text)))ptx_fail(value,"invalid debug function name");
                } else if(property.text=="inlined_at") {ptx_unsigned(take());ptx_unsigned(take());ptx_unsigned(take());}
                else ptx_fail(property,"unsupported location property",true);
            }
        }
        return true;
    }
    bool pragma() {
        if (!accept(".pragma")) return false;
        do {
            auto hint=take();
            if(hint.text!="\"nounroll\"" && hint.text!="\"used\"")ptx_fail(hint,"unsupported pragma",true);
        } while(accept(","));
        expect(";"); return true;
    }
    void function_directives(llvm::Function& function,bool kernel) {
        bool shape=false;
        while(peek().text.starts_with('.')) {
            if(pragma())continue;
            auto directive=take();
            if(directive.text==".noreturn") {
                if(kernel || !function.getReturnType()->isVoidTy())ptx_fail(directive,"invalid noreturn function");
                function.addFnAttr(llvm::Attribute::NoReturn); continue;
            }
            bool required=directive.text==".reqntid", maximum=directive.text==".maxntid";
            bool hint=directive.text==".maxnreg" || directive.text==".minnctapersm" || directive.text==".maxnctapersm";
            if(!kernel || (!required && !maximum && !hint))ptx_fail(directive,"unsupported kernel directive",true);
            std::vector<uint64_t> values;
            do {
                auto token=take();auto value=ptx_unsigned(token);
                if(!value || value>UINT32_MAX || values.size()>=(hint?1u:3u))ptx_fail(token,"invalid kernel directive value");
                values.push_back(value);
            } while(accept(","));
            if(required || maximum) {
                if(shape)ptx_fail(directive,"conflicting block shape directives");shape=true;
                uint64_t threads=1;
                for(auto value:values) {if(value>1024 || threads>1024/value)ptx_fail(directive,"block size exceeds 1024 threads");threads*=value;}
                auto key=required?"reqntid":"maxntid";
                std::string dimensions;
                for(unsigned axis=0;axis<3;++axis) {
                    auto value=axis<values.size()?values[axis]:1;
                    dimensions+=(axis?",":"")+std::to_string(value);
                    function.addFnAttr(std::string("cudavm.ptx.")+key+"."+char('x'+axis),std::to_string(value));
                    auto* data=llvm::MDNode::get(context,{llvm::ValueAsMetadata::get(&function),llvm::MDString::get(context,std::string(key)+char('x'+axis)),
                        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(context),value))});
                    module->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(data);
                }
                // LLVM 22 reads launch bounds from attributes during NVPTX
                // emission; the imported per-axis attributes serve shaders.
                function.addFnAttr(std::string("nvvm.")+key,dimensions);
            } else function.addFnAttr("cudavm.ptx"+directive.text,std::to_string(values[0]));
        }
    }
    std::pair<std::string,llvm::FunctionType*> external_builtin(const std::string& name) {
        auto* pointer=llvm::PointerType::get(context,0);auto* word=llvm::Type::getInt32Ty(context);auto* wide=llvm::Type::getInt64Ty(context);
        auto* real=llvm::Type::getFloatTy(context);auto* empty=llvm::Type::getVoidTy(context);
        auto signature=[&](const std::string& symbol,llvm::Type* result,std::vector<llvm::Type*> args) {
            return std::make_pair(symbol,llvm::FunctionType::get(result,args,false));
        };
        if(name=="malloc")return signature(name,pointer,{wide});
        if(name=="free")return signature(name,empty,{pointer});
        if(name=="__nv_aligned_device_malloc" || name=="__nv_aligned_device_malloc_impl")return signature(name,pointer,{wide,wide});
        if(name=="vprintf")return signature(name,word,{pointer,pointer});
        if(name=="__assertfail")return signature(name,empty,{pointer,pointer,word,pointer,wide});
        for(const auto& operation:device_math_operations) if(name==std::string("__nv_")+operation.name) {
            std::string symbol;
            if(operation.intrinsic)symbol=operation.intrinsic;
            else if(operation.llvm_intrinsic)symbol=operation.llvm_intrinsic;
            else {
                symbol=operation.name;symbol.pop_back();symbol="__cudavm_native_"+symbol;
            }
            return signature(symbol,real,std::vector<llvm::Type*>(operation.arguments,real));
        }
        if(name=="__nv_fmaf_rn")return signature("llvm.fma.f32",real,{real,real,real});
        for(const auto& conversion:device_conversions)if(name==std::string("__nv_")+conversion.name) {
            auto* floating=conversion.real_bits==64?llvm::Type::getDoubleTy(context):real;
            auto* integer=llvm::Type::getIntNTy(context,conversion.integer_bits);
            return signature(conversion.intrinsic,conversion.to_real?static_cast<llvm::Type*>(floating):integer,{conversion.to_real?static_cast<llvm::Type*>(integer):floating});
        }
        for(const auto& conversion:double_conversions)if(name==std::string("__nv_")+conversion.name)
            return signature(conversion.intrinsic,real,{llvm::Type::getDoubleTy(context)});
        for(const auto& operation:device_integer_operations)if(name==std::string("__nv_")+operation.name) {
            auto* type=llvm::Type::getIntNTy(context,operation.bits);
            return signature(operation.intrinsic,type,std::vector<llvm::Type*>(operation.arguments,type));
        }
        return {{},nullptr};
    }
    bool resolved_external(const llvm::Function& function) {
        return external_builtin(function.getName().str()).second || std::any_of(aliases.begin(),aliases.end(),[&](const auto& alias){return alias.first.text==function.getName();});
    }
    void normalize_external_abi() {
        std::vector<llvm::Function*> declarations;
        for(auto& function:*module)if(function.isDeclaration() && !function.use_empty() && external_builtin(function.getName().str()).second)declarations.push_back(&function);
        for(auto* function:declarations) {
            auto name=function->getName().str();auto [symbol,type]=external_builtin(name);
            const auto& signature=signatures.at(name);
            auto fail=[&] {ptx_fail(symbols.at(name),"incompatible external builtin ABI");};
            if(function->arg_size()!=type->getNumParams() || signature.second.size()>1)fail();
            for(const auto& parameter:signature.first)if(parameter.array)fail();
            if(!signature.second.empty() && signature.second[0].array)fail();
            auto compatible=[](llvm::Type* bits,llvm::Type* native) {
                return bits->isIntegerTy(native->isPointerTy()?64:native->getPrimitiveSizeInBits());
            };
            for(unsigned i=0;i<function->arg_size();++i)if(!compatible(function->getArg(i)->getType(),type->getParamType(i)))fail();
            if(type->getReturnType()->isVoidTy()!=function->getReturnType()->isVoidTy() ||
                (!type->getReturnType()->isVoidTy() && !compatible(function->getReturnType(),type->getReturnType())))fail();
            if(symbol==name)function->setName(name+"$ptx_abi");
            auto native=module->getOrInsertFunction(symbol,type);
            llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context,"entry",function));
            std::vector<llvm::Value*> arguments;
            for(unsigned i=0;i<function->arg_size();++i) {
                auto* value=static_cast<llvm::Value*>(function->getArg(i));auto* target=type->getParamType(i);
                arguments.push_back(target->isPointerTy()?builder.CreateIntToPtr(value,target):builder.CreateBitCast(value,target));
            }
            auto* call=builder.CreateCall(native,arguments);
            if(type->getReturnType()->isVoidTy())builder.CreateRetVoid();
            else builder.CreateRet(type->getReturnType()->isPointerTy()?builder.CreatePtrToInt(call,function->getReturnType()):builder.CreateBitCast(call,function->getReturnType()));
            function->setLinkage(llvm::GlobalValue::InternalLinkage);function->addFnAttr(llvm::Attribute::AlwaysInline);
        }
    }


    unsigned alignment() {
        auto token = take(); auto value = ptx_unsigned(token);
        if (!value || value > 65536 || (value & (value - 1))) ptx_fail(token, "invalid storage alignment");
        return unsigned(value);
    }
    ptx_parameter declaration(bool unsized = false, bool parameter = false, bool register_parameter = false) {
        unsigned align = accept(".align") ? alignment() : 0;
        auto type = ptx_scalar(take(), true);
        bool pointer = accept(".ptr");
        if (pointer) {
            if (!parameter || register_parameter || type.bits != 64 || (type.kind != 'u' && type.kind != 'b'))
                ptx_fail(peek(), "pointer annotations require a scalar 64-bit parameter");
            // These constrain the pointee, not the parameter slot. Keep the
            // ordinary 64-bit ABI and do not invent stronger LLVM assumptions.
            if (peek().text == ".global" || peek().text == ".const" || peek().text == ".shared" || peek().text == ".local") take();
            if (accept(".align")) alignment();
        }
        auto name = take();
        if (!ptx_identifier(register_parameter && name.text.starts_with('%') ? name.text.substr(1) : name.text))
            ptx_fail(name, "expected identifier");
        ptx_parameter result{name, type, 1, align ? align : type.bits / 8, false};
        result.register_parameter = register_parameter;
        if (accept("[")) {
            if (pointer) ptx_fail(name, "pointer parameter cannot be an array");
            result.array = true;
            result.count = peek().text == "]" ? 0 : ptx_unsigned(take());
            expect("]");
            if ((!result.count && !unsized) || result.count > (1u << 24)) ptx_fail(name, "invalid or excessive storage extent");
            if (peek().text == "[") ptx_fail(peek(), "multidimensional declarations are unsupported", true);
        }
        if (result.alignment < type.bits / 8) ptx_fail(name, "storage alignment is smaller than element size");
        if (register_parameter && (result.array || align || type.bits < 32))
            ptx_fail(name, "register parameters require scalar 32- or 64-bit types", true);
        return result;
    }
    std::vector<ptx_parameter> parameter_list(bool kernel = false, bool prototype = false) {
        expect("("); std::vector<ptx_parameter> result; std::set<std::string> names;
        uint64_t extent = 0;
        if (!accept(")")) for (;;) {
            bool reg = !kernel && accept(".reg");
            if (!reg) expect(".param");
            auto parameter = declaration(false, true, reg);
            if (!(prototype && parameter.name.text == "_") && !names.insert(parameter.name.text).second) ptx_fail(parameter.name, "duplicate parameter");
            extent = llvm::alignTo(extent, parameter.alignment) + parameter.bytes();
            if (extent > 65536) ptx_fail(parameter.name, "function parameters exceed 64 KiB", true);
            result.push_back(parameter);
            if (accept(")")) break;
            expect(",");
        }
        return result;
    }
    void storage(bool external, const std::string& space, ptx_body* body = nullptr,
            std::map<std::string, std::string>* scope = nullptr,
            llvm::GlobalValue::LinkageTypes linkage = llvm::GlobalValue::ExternalLinkage) {
        auto data = declaration(external && space == ".shared");
        auto original = data.name.text;
        if (scope) {
            if (scope->contains(original)) ptx_fail(data.name, "duplicate storage declaration");
            data.name.text = body->name.text + "$storage$" + std::to_string(symbol_counter++);
            (*scope)[original] = data.name.text;
        }
        unsigned address_space = space == ".shared" ? 3 : space == ".const" ? 4 : 1;
        if (external && space == ".shared" && data.count) ptx_fail(data.name, "extern shared storage must be unsized", true);
        if (!external && !data.count) ptx_fail(data.name, "defined storage must have an extent");
        auto* old = module->getNamedValue(data.name.text);
        symbols[data.name.text] = data.name;
        auto* variable = llvm::dyn_cast_or_null<llvm::GlobalVariable>(old);
        if (old && (!variable || variable->getAddressSpace() != address_space || variable->getValueType() != data.storage(context)))
            ptx_fail(data.name, "conflicting storage declaration");
        if (variable && !external && variable->hasInitializer()) ptx_fail(data.name, "duplicate storage definition");
        // CUDA constant storage is read-only to kernels, but host symbol copies
        // may update it between launches. LLVM must not fold its initial value.
        if (!variable) variable = new llvm::GlobalVariable(*module, data.storage(context), false,
            scope ? llvm::GlobalValue::InternalLinkage : llvm::GlobalValue::ExternalLinkage, nullptr, data.name.text,
            nullptr, llvm::GlobalVariable::NotThreadLocal, address_space);
        if (variable->getAlign() && variable->getAlign()->value() != data.alignment) ptx_fail(data.name, "conflicting storage alignment");
        variable->setAlignment(llvm::Align(data.alignment));
        std::vector<std::vector<ptx_token>> initializer;
        if (accept("=")) {
            if (external || address_space == 3) ptx_fail(data.name, "external/shared storage cannot have an initializer");
            bool list = accept("{");
            if (!list || peek().text != "}") for (;;) {
                initializer.emplace_back();
                while (peek().text != "," && peek().text != ";" && peek().text != "}") {
                    if (peek().text == "<eof>") ptx_fail(peek(), "unterminated initializer");
                    initializer.back().push_back(take());
                }
                if (!list || !accept(",")) break;
                if (peek().text == "}") break;
            }
            if (list) expect("}");
            if (initializer.size() > data.count) ptx_fail(data.name, "too many initializer elements");
        }
        expect(";");
        if (!external || address_space == 3) {
            variable->setLinkage(scope ? llvm::GlobalValue::InternalLinkage : linkage);
            if(linkage==llvm::GlobalValue::CommonLinkage) {
                variable->setMetadata("cudavm.ptx.common",llvm::MDNode::get(context,{}));
                if(!initializer.empty()) {
                    variable->setLinkage(llvm::GlobalValue::ExternalLinkage);
                    variable->setMetadata("cudavm.ptx.common.initialized",llvm::MDNode::get(context,{}));
                }
            }
            variable->setInitializer(address_space == 3 ? llvm::UndefValue::get(data.storage(context)) : llvm::Constant::getNullValue(data.storage(context)));
            globals.push_back({data, variable, std::move(initializer)});
        }
    }
    void callable(bool kernel, bool external, llvm::GlobalValue::LinkageTypes linkage = llvm::GlobalValue::ExternalLinkage) {
        std::vector<ptx_parameter> results;
        if (!kernel && peek().text == "(") results = parameter_list();
        if (results.size() > 1) ptx_fail(results[1].name, "PTX ABI permits a single return parameter");
        auto name = identifier();
        if(name.text.starts_with("__cudavm_native_") || name.text.starts_with("_ZN6cudavm"))ptx_fail(name,"reserved compiler symbol",true);
        auto parameters = peek().text == "(" ? parameter_list(kernel) : std::vector<ptx_parameter>{};
        if (module->size() >= 4096) ptx_fail(name, "function resource limit exceeded", true);
        std::vector<llvm::Type*> inputs, outputs;
        for (const auto& parameter : parameters) inputs.push_back(kernel && (parameter.array || parameter.alignment != parameter.type.bits / 8)
            ? llvm::PointerType::get(context, 0) : parameter.storage(context));
        for (const auto& parameter : results) outputs.push_back(parameter.storage(context));
        auto* output = outputs.empty() ? llvm::Type::getVoidTy(context) : outputs[0];
        auto* type = llvm::FunctionType::get(output, inputs, false);
        auto* function = module->getFunction(name.text);
        if (module->getNamedValue(name.text) && (!function || function->getFunctionType() != type)) ptx_fail(name, "conflicting function declaration");
        if (function && function->hasFnAttribute("cudavm.kernel") != kernel) ptx_fail(name, "conflicting function kind");
        symbols[name.text] = name;
        if (auto found = signatures.find(name.text); found != signatures.end())
            if (ptx_signature_key(found->second) != ptx_signature_key({parameters, results})) ptx_fail(name, "conflicting function parameter layout");
        if (!function) function = llvm::Function::Create(type, llvm::GlobalValue::ExternalLinkage, name.text, *module);
        if (kernel) function->addFnAttr("cudavm.kernel");
        signatures[name.text] = {parameters, results};
        declared_linkages[name.text]=linkage;
        if (!kernel) function->addFnAttr("cudavm.ptx.signature", ptx_signature_key(signatures.at(name.text)));
        for (unsigned i = 0; i < parameters.size(); ++i) {
            if (kernel && inputs[i]->isPointerTy()) {
                function->addParamAttr(i, llvm::Attribute::getWithByValType(context, parameters[i].storage(context)));
                function->addParamAttr(i, llvm::Attribute::getWithAlignment(context, llvm::Align(parameters[i].alignment)));
            }
            function->getArg(i)->setName(parameters[i].name.text);
        }
        function_directives(*function,kernel);
        if (accept(";")) {
            if (kernel) ptx_fail(name, "kernel declarations require a definition", true);
            return;
        }
        if (external) ptx_fail(name, "extern function cannot have a body");
        if (!function->isDeclaration()) ptx_fail(name, "duplicate function definition");
        function->setLinkage(linkage);
        if (peek().text != "{") ptx_fail(peek(), "unsupported kernel directive or declaration", true);
        expect("{");
        ptx_body body{name, function, parameters, results, {}, {}, {}, {}, {}, {}};
        auto* entry = llvm::BasicBlock::Create(context, "entry", function); llvm::IRBuilder<> allocations(entry);
        std::vector<std::map<std::string, std::string>> scopes(1);
        auto add_slot = [&](const ptx_parameter& parameter, llvm::Value* input = nullptr, bool byval = false) {
            if (body.slots.contains(parameter.name.text) || body.registers.contains(parameter.name.text)) ptx_fail(parameter.name, "duplicate parameter");
            llvm::Value* slot = input;
            if (!byval) {
                auto* storage = allocations.CreateAlloca(parameter.storage(context)); storage->setAlignment(llvm::Align(parameter.alignment)); slot = storage;
                if (input) allocations.CreateAlignedStore(input, slot, llvm::Align(parameter.alignment));
            }
            if (parameter.register_parameter) body.registers.emplace(parameter.name.text, ptx_register{parameter.type, llvm::cast<llvm::AllocaInst>(slot)});
            else body.slots.emplace(parameter.name.text, std::make_pair(parameter, slot));
            scopes.back()[parameter.name.text] = parameter.name.text;
        };
        for (unsigned i = 0; i < parameters.size(); ++i) add_slot(parameters[i], function->getArg(i), kernel && inputs[i]->isPointerTy());
        for (const auto& parameter : results) add_slot(parameter);
        auto resolve = [&](ptx_token token) {
            for (auto scope = scopes.rbegin(); scope != scopes.rend(); ++scope)
                if (auto found = scope->find(token.text); found != scope->end()) { token.text = found->second; break; }
            auto dot=token.text.rfind('.');
            if(dot!=std::string::npos && dot+2<token.text.size() && (token.text[dot+1]=='b' || token.text[dot+1]=='h'))
                for(auto scope=scopes.rbegin();scope!=scopes.rend();++scope)
                    if(auto found=scope->find(token.text.substr(0,dot));found!=scope->end()){token.text=found->second+token.text.substr(dot);break;}
            return token;
        };
        while (true) {
            if (accept("}")) { if (scopes.size() == 1) break; scopes.pop_back(); continue; }
            if (accept("{")) { scopes.emplace_back(); continue; }
            if (peek().text == "<eof>") ptx_fail(peek(), "unterminated function body");
            if (debug_directive() || pragma()) continue;
            if (accept(".param")) {
                auto parameter = declaration();
                for(;;) {
                    auto original = parameter.name.text;
                    if (scopes.back().contains(original)) ptx_fail(parameter.name, "duplicate local parameter");
                    auto current=parameter;current.name.text=name.text+"$param$"+std::to_string(symbol_counter++);
                    add_slot(current);scopes.back().erase(current.name.text);scopes.back()[original]=current.name.text;
                    if(accept(";"))break;
                    expect(",");parameter.name=identifier();parameter.count=1;parameter.array=false;
                    if(accept("[")){parameter.array=true;parameter.count=ptx_unsigned(take());expect("]");if(!parameter.count || parameter.count>(1u<<24))ptx_fail(parameter.name,"invalid parameter extent");}
                }
                continue;
            }
            bool external_storage = accept(".extern");
            if (peek().text == ".shared") { take(); storage(external_storage, ".shared", &body, &scopes.back()); continue; }
            if (external_storage) ptx_fail(peek(), "unsupported external body declaration", true);
            if (accept(".local")) {
                auto data = declaration(); auto original = data.name.text;
                if (scopes.back().contains(original)) ptx_fail(data.name, "duplicate local storage declaration");
                if (peek().text == "=") ptx_fail(peek(), "local storage cannot have an initializer");
                expect(";");
                auto unique = name.text + "$local$" + std::to_string(symbol_counter++);
                auto* slot = allocations.CreateAlloca(data.storage(context), nullptr, unique);
                slot->setAlignment(llvm::Align(data.alignment));
                body.locals.emplace(unique, allocations.CreateAddrSpaceCast(slot, llvm::PointerType::get(context, 5)));
                scopes.back()[original] = unique;
                continue;
            }
            if (accept(".reg")) {
                unsigned lanes=accept(".v2")?2:accept(".v4")?4:1;
                auto type = ptx_scalar(take());
                for (;;) {
                    auto reg = take();
                    if (!(reg.text.starts_with('%') ? ptx_identifier(reg.text.substr(1)) : ptx_identifier(reg.text))) ptx_fail(reg, "invalid register name");
                    unsigned count = 0;
                    if (accept("<")) {
                        auto token = take(); auto value = ptx_unsigned(token);
                        if (!value || value > 65536) ptx_fail(token, "register resource limit exceeded", true);
                        count = unsigned(value); expect(">");
                    }
                    for (unsigned i = 0; i < std::max(1u, count); ++i) {
                        auto symbol = reg.text + (count ? std::to_string(i) : "");
                        if (scopes.back().contains(symbol)) ptx_fail(reg, "duplicate register");
                        if (body.registers.size() >= 65536) ptx_fail(reg, "register resource limit exceeded", true);
                        auto unique = symbol + "$" + std::to_string(symbol_counter++);
                        scopes.back()[symbol] = unique;
                        if(lanes==1)body.registers.emplace(unique, ptx_register{type, allocations.CreateAlloca(type.storage(context))});
                        else for(unsigned lane=0;lane<lanes;++lane) {
                            const char components[]{'x','y','z','w'};
                            auto scalar=unique+"."+std::to_string(lane);
                            scopes.back()[symbol+"."+components[lane]]=scalar;body.vectors[unique].push_back(scalar);
                            body.registers.emplace(scalar,ptx_register{type,allocations.CreateAlloca(type.storage(context))});
                        }
                    }
                    if (accept(";")) break;
                    expect(",");
                }
                continue;
            }
            ptx_instruction instruction{};
            if (accept("@")) { instruction.negate = accept("!"); instruction.predicate = resolve(take()).text; }
            instruction.opcode = take();
            if (accept(":")) {
                if (!instruction.predicate.empty() || !ptx_identifier(instruction.opcode.text)) ptx_fail(instruction.opcode, "invalid label");
                if (peek().text == ".callprototype" || peek().text == ".calltargets") {
                    auto label = instruction.opcode;
                    if (scopes.back().contains(label.text)) ptx_fail(label, "duplicate call prototype");
                    auto unique = name.text + "$prototype$" + std::to_string(symbol_counter++);
                    scopes.back()[label.text] = unique;
                    ptx_call_prototype prototype;
                    if (accept(".callprototype")) {
                        if (peek().text == "(") prototype.signature.second = parameter_list(false, true);
                        if (prototype.signature.second.size() > 1) ptx_fail(label, "PTX ABI permits a single return parameter");
                        expect("_");
                        if (peek().text == "(") prototype.signature.first = parameter_list(false, true);
                    } else {
                        expect(".calltargets");
                        do { prototype.targets.push_back(identifier()); } while (accept(","));
                    }
                    expect(";"); body.prototypes.emplace(unique, std::move(prototype)); continue;
                }
                instruction.label = true;
            } else {
                if (instruction.opcode.text.starts_with('.')) ptx_fail(instruction.opcode, "unsupported body directive", true);
                if (!accept(";")) {
                    instruction.operands.emplace_back(); unsigned nesting = 0;
                    while (!accept(";")) {
                        if ((peek().text == "}" && !nesting) || peek().text == "<eof>") ptx_fail(peek(), "expected instruction terminator");
                        auto token = take();
                        if (token.text == "(" || token.text == "[" || token.text == "{") ++nesting;
                        if (token.text == ")" || token.text == "]" || token.text == "}") { if (!nesting) ptx_fail(token, "unbalanced operand"); --nesting; }
                        if (token.text == "," && !nesting) instruction.operands.emplace_back();
                        else instruction.operands.back().push_back(resolve(token));
                    }
                    if (nesting) ptx_fail(instruction.opcode, "unbalanced operand");
                    for (const auto& operand : instruction.operands) if (operand.empty()) ptx_fail(instruction.opcode, "empty operand");
                }
            }
            for(auto& operand:instruction.operands)if(operand.size()==6 && operand.front().text=="[" && operand.back().text=="]" &&
                (operand[2].text=="+" || operand[2].text=="-") && (operand[3].text=="+" || operand[3].text=="-")) {
                operand[2].text=operand[2].text==operand[3].text?"+":"-";operand.erase(operand.begin()+3);
            }
            body.instructions.push_back(std::move(instruction));
            if (body.instructions.size() > 65536) ptx_fail(peek(), "instruction resource limit exceeded", true);
        }
        if (kernel) {
            llvm::SmallVector<llvm::Metadata*> layout; uint64_t offset = 0;
            for (const auto& parameter : parameters) {
                offset = llvm::alignTo(offset, parameter.alignment); llvm::SmallVector<llvm::Metadata*> fields;
                for (uint64_t value : {parameter.bytes(), uint64_t(parameter.alignment), offset})
                    fields.push_back(llvm::ConstantAsMetadata::get(allocations.getInt64(value)));
                layout.push_back(llvm::MDNode::get(context, fields)); offset += parameter.bytes();
            }
            function->setMetadata("cudavm.kernel.layout", llvm::MDNode::get(context, layout));
        }
        bodies.push_back(std::move(body));
    }

    // Reuse the exact, checked image assembly contract of the CUDA headers.
    // NVPTX preserves these operations; shader clones resolve object descriptors.
    template<class Read, class Write> void image_instruction(const std::vector<std::string>& parts,
            const ptx_instruction& instruction, llvm::IRBuilder<>& b, Read read, Write write) {
        const auto& op = parts[0];
        bool texture = op == "tex", writing = op == "sust", fetch = texture && parts.size() > 1 && parts[1] == "1d";
        unsigned components = 4, bits = 32;
        ptx_type data{32, 'b'}, coordinate{32, 's'};
        if (texture) {
            if (parts.size() != 5 || parts[2] != "v4" ||
                (parts[3] != "f32" && parts[3] != "s32" && parts[3] != "u32") ||
                !(fetch ? parts[4] == "s32" : parts[1] == "2d" && parts[4] == "f32"))
                ptx_fail(instruction.opcode, "unsupported texture instruction profile", true);
            data = ptx_scalar({parts[3], instruction.opcode.line, instruction.opcode.column});
            coordinate = {32, fetch ? 's' : 'f'};
        } else {
            unsigned type_index = parts.size() == 6 ? 4 : 3;
            if ((parts.size() != 5 && parts.size() != 6) || parts[1] != "b" || parts[2] != "2d" ||
                (parts.size() == 6 && parts[3] != "v2" && parts[3] != "v4") ||
                (parts.back() != "zero" && parts.back() != "clamp" && parts.back() != "trap"))
                ptx_fail(instruction.opcode, "unsupported surface instruction profile", true);
            data = ptx_scalar({parts[type_index], instruction.opcode.line, instruction.opcode.column}, true);
            components = parts.size() == 6 ? (parts[3] == "v2" ? 2 : 4) : 1;
            bits = data.bits;
            if (data.kind != 'b' || components * bits > 128) ptx_fail(instruction.opcode, "unsupported surface transfer type", true);
        }
        if (instruction.operands.size() != 2) ptx_fail(instruction.opcode, "wrong image operand count");
        auto fields = [&](std::vector<ptx_token> tokens, unsigned count) {
            if (!tokens.empty() && tokens.front().text == "{") {
                if (tokens.back().text != "}") ptx_fail(instruction.opcode, "invalid image vector");
                tokens.erase(tokens.begin()); tokens.pop_back();
            }
            std::vector<std::vector<ptx_token>> result(1);
            for (const auto& token : tokens) {
                if (token.text == ",") result.emplace_back();
                else result.back().push_back(token);
            }
            if (result.size() != count || llvm::any_of(result, [](const auto& field) { return field.empty(); }))
                ptx_fail(instruction.opcode, "invalid image vector size");
            return result;
        };
        const auto& address = instruction.operands[writing ? 0 : 1];
        if (address.size() < 5 || address.front().text != "[" || address.back().text != "]" || address[2].text != ",")
            ptx_fail(instruction.opcode, "invalid image address");
        auto coordinates = fields({address.begin() + 3, address.end() - 1}, fetch ? 1 : 2);
        auto values = fields(instruction.operands[writing ? 1 : 0], components);
        std::vector<llvm::Value*> arguments{read({address[1]}, {64, 'u'})};
        for (const auto& c : coordinates) arguments.push_back(read(c, coordinate));
        auto* component_type = texture ? data.value(context) : static_cast<llvm::Type*>(llvm::IntegerType::get(context, std::max(16u, bits)));
        std::string constraints, text, placeholders;
        auto constraint = texture && data.kind == 'f' ? "f" : bits < 32 ? "h" : bits == 64 ? "l" : "r";
        for (unsigned i = 0; i < components; ++i) {
            if (i) placeholders += ",";
            placeholders += "$" + std::to_string(writing ? i + 3 : i);
            if (!writing) constraints += std::string(i ? ",=" : "=") + constraint;
        }
        auto address_text = "[$" + std::to_string(writing ? 0 : components) + ",{$" +
            std::to_string(writing ? 1 : components + 1) + (fetch ? "" : ",$" + std::to_string(writing ? 2 : components + 2)) + "}]";
        constraints += writing ? "l,r,r" : texture ? (fetch ? ",l,r" : ",l,f,f") : ",l,r,r";
        if (writing) for (const auto& v : values) {
            arguments.push_back(b.CreateZExtOrTrunc(read(v, data, true), component_type));
            constraints += std::string(",") + constraint;
        }
        text = instruction.opcode.text + (writing ? address_text + ",{" + placeholders + "};" : "{" + placeholders + "}," + address_text + ";");
        llvm::Type* output = b.getVoidTy();
        if (!writing) output = components == 1 ? component_type : llvm::StructType::get(context, std::vector<llvm::Type*>(components, component_type));
        std::vector<llvm::Type*> inputs; for (auto* argument : arguments) inputs.push_back(argument->getType());
        auto* assembly = llvm::InlineAsm::get(llvm::FunctionType::get(output, inputs, false), text, constraints, true);
        auto* call = b.CreateCall(assembly, arguments);
        if (!writing) for (unsigned i = 0; i < components; ++i)
            write(values[i], components == 1 ? static_cast<llvm::Value*>(call) : b.CreateExtractValue(call, i),
                texture ? data : ptx_type{std::max(16u, bits), 'b'}, !texture);
    }
    template<class Read,class Write> bool numeric_instruction(const std::vector<std::string>& parts,
            const ptx_instruction& instruction,llvm::IRBuilder<>& b,Read read,Write write) {
        const auto& op=parts[0];const auto& operands=instruction.operands;
        auto type=ptx_scalar({parts.back(),instruction.opcode.line,instruction.opcode.column},op=="ld" || op=="st");
        auto arity=[&](unsigned n){if(operands.size()!=n)ptx_fail(instruction.opcode,"wrong operand count");};
        auto helper=[&](const std::string& name,std::vector<llvm::Value*> args,llvm::Type* result)->llvm::Value* {
            std::vector<llvm::Type*> types;for(auto* arg:args)types.push_back(arg->getType());
            return b.CreateCall(module->getOrInsertFunction("__cudavm_native_"+name,llvm::FunctionType::get(result,types,false)),args);
        };
        auto word=[&](llvm::Value* v)->llvm::Value* {return v->getType()->isFloatingPointTy()?b.CreateBitCast(v,llvm::Type::getIntNTy(context,v->getType()->getPrimitiveSizeInBits())):v;};
        auto real=[&](llvm::Value* v)->llvm::Value* {return b.CreateBitCast(helper("word_f16_to_f32",{b.CreateZExtOrTrunc(word(v),b.getInt32Ty())},b.getInt32Ty()),b.getFloatTy());};
        auto done=[&](llvm::Value* v,ptx_type t){write(operands[0],v,t,false);return true;};
        auto flush_float = [&](llvm::Value* value) -> llvm::Value* {
            auto* bits = word(value);
            return b.CreateBitCast(b.CreateSelect(b.CreateICmpULT(b.CreateAnd(bits, b.getInt32(0x7fffffff)), b.getInt32(0x800000)),
                b.CreateAnd(bits, b.getInt32(0x80000000)), bits), b.getFloatTy());
        };
        if(type.kind=='f' && type.bits==32 && (op=="add" || op=="sub" || op=="mul" || op=="fma") &&
            (std::find(parts.begin(),parts.end(),"sat")!=parts.end() || (parts.size()>2 && (parts[1]=="rz" || parts[1]=="rm" || parts[1]=="rp")))) {
            unsigned at=1,mode=0;const std::vector<std::string> modes{"rn","rz","rm","rp"};
            if(at<parts.size()-1)if(auto found=std::find(modes.begin(),modes.end(),parts[at]);found!=modes.end()){mode=unsigned(found-modes.begin());++at;}
            bool flush=at<parts.size()-1 && parts[at]=="ftz";if(flush)++at;
            bool saturate=at<parts.size()-1 && parts[at]=="sat";if(saturate)++at;
            if(at!=parts.size()-1)ptx_fail(instruction.opcode,"unsupported FP32 arithmetic modifier",true);
            arity(op=="fma"?4:3);auto* a=read(operands[1],type);auto* c=read(operands[2],type);
            if(flush){a=flush_float(a);c=flush_float(c);}
            llvm::Value* result=nullptr;
            if(op=="add" || op=="sub")result=helper("word_f32_add_mode",{word(a),op=="sub"?b.CreateXor(word(c),b.getInt32(0x80000000)):word(c),b.getInt32(mode)},b.getInt32Ty());
            else {
                llvm::Value* addend=op=="fma"?read(operands[3],type):nullptr;
                if(addend && flush)addend=flush_float(addend);
                auto* bits=addend?word(addend):b.CreateAnd(b.CreateXor(word(a),word(c)),b.getInt32(0x80000000));
                result=helper("word_f32_fma_mode",{word(a),word(c),bits,b.getInt32(mode)},b.getInt32Ty());
            }
            if(saturate) {
                auto* zero=b.CreateOr(b.CreateICmpSLT(result,b.getInt32(0)),b.CreateICmpUGT(b.CreateAnd(result,b.getInt32(0x7fffffff)),b.getInt32(0x7f800000)));
                result=b.CreateSelect(zero,b.getInt32(0),b.CreateSelect(b.CreateICmpUGT(result,b.getInt32(0x3f800000)),b.getInt32(0x3f800000),result));
            }
            result=b.CreateBitCast(result,b.getFloatTy());return done(flush?flush_float(result):result,type);
        }
        if (op == "cvt" && parts.size() >= 3) {
            auto destination = ptx_scalar({parts[parts.size()-2], instruction.opcode.line, instruction.opcode.column});
            bool extended = (type.kind == 'f' && type.bits == 64) || (destination.kind == 'f' && destination.bits == 64) ||
                (destination.kind == 'f' && destination.bits == 16 && type.kind != 'f') ||
                (type.kind == 'f' && type.bits == 16 && destination.kind != 'f');
            if (extended) {
                arity(2);
                unsigned mode = 0;
                bool integral = false;
                if (parts.size() == 4) {
                    const std::vector<std::string> modes{"rn","rz","rm","rp","rni","rzi","rmi","rpi"};
                    auto found = std::find(modes.begin(), modes.end(), parts[1]);
                    if (found == modes.end()) ptx_fail(instruction.opcode, "unsupported extended conversion modifier", true);
                    mode = unsigned(found - modes.begin()) % 4; integral = found - modes.begin() >= 4;
                } else if (parts.size() != 3) ptx_fail(instruction.opcode, "unsupported extended conversion modifier", true);
                auto* input = read(operands[1], type, type.kind != 'f');
                llvm::Value* result = nullptr;
                if (type.kind != 'f' && destination.kind == 'f') {
                    if (parts.size() != 4 || integral || (type.kind != 's' && type.kind != 'u'))
                        ptx_fail(instruction.opcode, "integer conversion requires a floating rounding mode");
                    auto* extended_input = type.signed_integer() ? b.CreateSExtOrTrunc(input,b.getInt64Ty()) : b.CreateZExtOrTrunc(input,b.getInt64Ty());
                    if (destination.bits == 16) {
                        auto* negative = type.signed_integer() ? b.CreateICmpSLT(extended_input,b.getInt64(0)) : b.getFalse();
                        auto* magnitude = b.CreateSelect(negative,b.CreateNeg(extended_input),extended_input);
                        result = helper("word_f16_round",{magnitude,b.getInt32(0),b.CreateZExt(negative,b.getInt32Ty()),b.getInt32(mode)},b.getInt32Ty());
                        result = b.CreateBitCast(b.CreateTrunc(result,b.getInt16Ty()),b.getHalfTy());
                    } else {
                        result = helper(type.signed_integer()?"word_i64_to_f64":"word_u64_to_f64",{extended_input,b.getInt32(mode)},b.getInt64Ty());
                        result = b.CreateBitCast(result,b.getDoubleTy());
                    }
                } else if (type.kind == 'f' && destination.kind != 'f') {
                    if (parts.size() != 4 || !integral || (destination.kind != 's' && destination.kind != 'u'))
                        ptx_fail(instruction.opcode, "floating conversion requires an integer destination and rounding mode");
                    if (type.bits == 16) {
                        result = helper("word_f16_to_integer",{b.CreateZExt(word(input),b.getInt32Ty()),b.getInt32(mode),b.getInt32(destination.bits),b.getInt32(!destination.signed_integer())},b.getInt64Ty());
                        result = b.CreateSelect(b.CreateICmpUGT(b.CreateAnd(word(input),b.getInt16(0x7fff)),b.getInt16(0x7c00)),b.getInt64(0),result);
                    } else {
                        auto* bits = word(input);
                        auto* magnitude = b.CreateAnd(bits,b.getInt64(0x7fffffffffffffffull));
                        auto* negative = b.CreateICmpSLT(bits,b.getInt64(0));
                        auto* nan = b.CreateICmpUGT(magnitude,b.getInt64(0x7ff0000000000000ull));
                        auto* overflow = b.CreateICmpUGE(magnitude,b.getInt64(uint64_t(1023 + destination.bits - unsigned(destination.signed_integer())) << 52));
                        result = helper("word_f64_to_i64",{bits,b.getInt32(mode)},b.getInt64Ty());
                        auto low = destination.signed_integer() ? llvm::APInt::getSignedMinValue(destination.bits).sext(64) : llvm::APInt(64,0);
                        auto high = destination.signed_integer() ? llvm::APInt::getSignedMaxValue(destination.bits).zext(64) : llvm::APInt::getMaxValue(destination.bits).zext(64);
                        auto* minimum = llvm::ConstantInt::get(context,low); auto* maximum = llvm::ConstantInt::get(context,high);
                        if (destination.bits < 64) {
                            result = b.CreateSelect(destination.signed_integer()?b.CreateICmpSLT(result,minimum):b.CreateICmpULT(result,minimum),minimum,result);
                            result = b.CreateSelect(destination.signed_integer()?b.CreateICmpSGT(result,maximum):b.CreateICmpUGT(result,maximum),maximum,result);
                        }
                        result = b.CreateSelect(overflow,b.CreateSelect(negative,minimum,maximum),result);
                        if (!destination.signed_integer()) result = b.CreateSelect(negative,b.getInt64(0),result);
                        result = b.CreateSelect(nan,b.getInt64(0),result);
                    }
                    result = b.CreateTruncOrBitCast(result,destination.storage(context));
                } else if (type.kind == 'f' && destination.kind == 'f') {
                    if (type.bits == destination.bits) {
                        if (parts.size() != 4 || !integral) ptx_fail(instruction.opcode,"integral conversion requires an integer rounding mode");
                        result = b.CreateBitCast(helper("word_f64_integral",{word(input),b.getInt32(mode)},b.getInt64Ty()),b.getDoubleTy());
                    } else if (destination.bits == 64) {
                        if (parts.size() != 3 || integral) ptx_fail(instruction.opcode,"widening conversion has no rounding modifier");
                        if (type.bits == 16) input = real(input);
                        result = b.CreateBitCast(helper("word_f32_to_f64",{word(input)},b.getInt64Ty()),b.getDoubleTy());
                    } else {
                        if (parts.size() != 4 || integral) ptx_fail(instruction.opcode,"narrowing conversion requires a floating rounding mode");
                        result = helper(destination.bits == 16?"word_f64_to_f16":"word_f64_to_f32",{word(input),b.getInt32(mode)},b.getInt32Ty());
                        if (destination.bits == 16) result = b.CreateTrunc(result,b.getInt16Ty());
                        result = b.CreateBitCast(result,destination.value(context));
                    }
                }
                if (!result) ptx_fail(instruction.opcode,"unsupported extended conversion",true);
                write(operands[0],result,destination,destination.kind != 'f'); return true;
            }
        }
        if(type.kind=='f' && type.bits==64 && op=="rcp" && parts==std::vector<std::string>{"rcp","rn","f64"}) {
            arity(2);return done(b.CreateBitCast(helper("word_f64_div",{b.getInt64(0x3ff0000000000000ull),word(read(operands[1],type))},b.getInt64Ty()),b.getDoubleTy()),type);
        }
        if(type.kind=='f' && type.bits==64 && (op=="add" || op=="sub" || op=="mul" || op=="fma" || op=="sqrt") && parts.size()==3) {
            const std::vector<std::string> modes{"rn","rz","rm","rp"};auto mode=std::find(modes.begin(),modes.end(),parts[1]);
            if(mode==modes.end())ptx_fail(instruction.opcode,"unsupported FP64 arithmetic modifier",true);
            arity(op=="sqrt"?2:op=="fma"?4:3);
            std::vector<llvm::Value*> values{word(read(operands[1],type))};
            if(op!="sqrt") {auto* next=word(read(operands[2],type));if(op=="sub")next=b.CreateXor(next,b.getInt64(0x8000000000000000ull));values.push_back(next);}
            if(op=="fma")values.push_back(word(read(operands[3],type)));
            values.push_back(b.getInt32(unsigned(mode-modes.begin())));
            return done(b.CreateBitCast(helper("word_f64_"+(op=="sub"?std::string("add"):op)+"_mode",values,b.getInt64Ty()),b.getDoubleTy()),type);
        }
        if (type.kind == 'f' && type.bits == 64 && (op == "add" || op == "sub" || op == "mul" || op == "div")) {
            if (parts.size() != 2 && (parts.size() != 3 || parts[1] != "rn")) ptx_fail(instruction.opcode,"unsupported FP64 arithmetic modifier",true);
            arity(3); auto* a=read(operands[1],type); auto* c=read(operands[2],type);
            auto* result = op=="add"?b.CreateFAdd(a,c):op=="sub"?b.CreateFSub(a,c):op=="mul"?b.CreateFMul(a,c):b.CreateFDiv(a,c);
            return done(result,type);
        }
        if(op=="testp" && parts.size()==3 && type.kind=='f' && type.bits>=32) {
            arity(2);auto* bits=word(read(operands[1],type));
            auto constant=[&](uint64_t value){return llvm::ConstantInt::get(bits->getType(),value);};
            auto* magnitude=b.CreateAnd(bits,constant(type.bits==64?0x7fffffffffffffffull:0x7fffffffull));
            auto* infinity=constant(type.bits==64?0x7ff0000000000000ull:0x7f800000ull);
            auto* normal=constant(type.bits==64?0x0010000000000000ull:0x00800000ull);
            llvm::Value* result=nullptr;
            if(parts[1]=="finite")result=b.CreateICmpULT(magnitude,infinity);
            else if(parts[1]=="infinite")result=b.CreateICmpEQ(magnitude,infinity);
            else if(parts[1]=="number")result=b.CreateICmpULE(magnitude,infinity);
            else if(parts[1]=="notanumber")result=b.CreateICmpUGT(magnitude,infinity);
            else if(parts[1]=="normal")result=b.CreateAnd(b.CreateICmpUGE(magnitude,normal),b.CreateICmpULT(magnitude,infinity));
            else if(parts[1]=="subnormal")result=b.CreateAnd(b.CreateICmpNE(magnitude,constant(0)),b.CreateICmpULT(magnitude,normal));
            else ptx_fail(instruction.opcode,"unsupported floating classification",true);
            return done(result,{1,'p'});
        }
        if(op=="slct" && (parts.size()==3 || (parts.size()==4 && parts[1]=="ftz"))) {
            arity(4);bool flush=parts.size()==4;
            auto destination=ptx_scalar({parts[parts.size()-2],instruction.opcode.line,instruction.opcode.column});
            if((type.kind!='s' && type.kind!='f') || type.bits!=32 || (flush && type.kind!='f'))ptx_fail(instruction.opcode,"unsupported selection type",true);
            auto* condition=read(operands[3],type);if(flush)condition=flush_float(condition);
            auto* selected=type.kind=='f'?b.CreateFCmpOGE(condition,llvm::ConstantFP::get(b.getFloatTy(),0)):b.CreateICmpSGE(condition,b.getInt32(0));
            return done(b.CreateSelect(selected,read(operands[1],destination),read(operands[2],destination)),destination);
        }
        if (op == "cvt" && parts.size() >= 3) {
            auto destination = ptx_scalar({parts[parts.size() - 2], instruction.opcode.line, instruction.opcode.column});
            bool from_float = type.kind == 'f' && type.bits == 32;
            bool to_float = destination.kind == 'f' && destination.bits == 32;
            bool from_integer = (type.kind == 's' || type.kind == 'u') && type.bits >= 16;
            bool to_integer = (destination.kind == 's' || destination.kind == 'u') && destination.bits >= 16;
            if ((from_float && (to_float || to_integer)) || (to_float && from_integer)) {
                unsigned at = 1, mode = from_float && to_integer ? 1 : 0;
                bool rounded = false, integral = false;
                const std::vector<std::string> float_modes{"rn", "rz", "rm", "rp"};
                const std::vector<std::string> integer_modes{"rni", "rzi", "rmi", "rpi"};
                for (unsigned i = 0; i < 4; ++i) {
                    if (parts[at] == float_modes[i]) { mode = i; rounded = true; ++at; break; }
                    if (parts[at] == integer_modes[i]) { mode = i; rounded = integral = true; ++at; break; }
                }
                bool flush = parts[at] == "ftz"; if (flush) ++at;
                bool saturate = parts[at] == "sat"; if (saturate) ++at;
                if (at != parts.size() - 2 || ((from_integer || to_integer) && !rounded) || (rounded && integral != from_float) ||
                        (from_float && to_float && rounded && !integral))
                    ptx_fail(instruction.opcode, "unsupported FP32 conversion modifier", true);
                arity(2);
                auto* input = read(operands[1], type, from_integer);
                if (from_float && flush) input = flush_float(input);
                llvm::Value* result = input;
                if (from_float && to_float && rounded) {
                    // NVPTX uses native rounding instructions; shader lowering
                    // maps these intrinsics to the shared exact helpers.
                    constexpr llvm::Intrinsic::ID modes[]{llvm::Intrinsic::roundeven, llvm::Intrinsic::trunc,
                        llvm::Intrinsic::floor, llvm::Intrinsic::ceil};
                    result = b.CreateIntrinsic(b.getFloatTy(), modes[mode], {input});
                } else if (from_integer) {
                    auto width = std::max(32u, type.bits);
                    input = type.signed_integer() ? b.CreateSExtOrTrunc(input, b.getIntNTy(width)) : b.CreateZExtOrTrunc(input, b.getIntNTy(width));
                    for (const auto& conversion : device_conversions)
                        if (conversion.to_real && conversion.real_bits == 32 && conversion.integer_bits == width &&
                                conversion.signed_value == type.signed_integer() && conversion.mode == mode) {
                            result = b.CreateCall(module->getOrInsertFunction(conversion.intrinsic,
                                llvm::FunctionType::get(b.getFloatTy(), {input->getType()}, false)), {input});
                            break;
                        }
                } else if (to_integer) {
                    auto* bits = word(input);
                    auto* magnitude = b.CreateAnd(bits, b.getInt32(0x7fffffff));
                    auto* negative = b.CreateICmpSLT(bits, b.getInt32(0));
                    auto* nan = b.CreateICmpUGT(magnitude, b.getInt32(0x7f800000));
                    auto limit = (127 + destination.bits - unsigned(destination.signed_integer())) << 23;
                    auto* overflow = b.CreateICmpUGE(magnitude, b.getInt32(limit));
                    auto* invalid = b.CreateOr(nan, overflow);
                    if (!destination.signed_integer()) invalid = b.CreateOr(invalid, negative);
                    auto* safe = b.CreateSelect(invalid, llvm::ConstantFP::get(b.getFloatTy(), 0), input);
                    // PTX cvt clamps and maps NaN to zero. CUDA's conversion
                    // intrinsics have a narrower valid range, so guard their
                    // input before reusing the shared rounding implementation.
                    auto width = std::max(32u, destination.bits);
                    for (const auto& conversion : device_conversions)
                        if (!conversion.to_real && conversion.real_bits == 32 && conversion.integer_bits == width &&
                                conversion.signed_value == destination.signed_integer() && conversion.mode == mode) {
                            result = b.CreateCall(module->getOrInsertFunction(conversion.intrinsic,
                                llvm::FunctionType::get(b.getIntNTy(width), {safe->getType()}, false)), {safe});
                            break;
                        }
                    auto minimum = destination.signed_integer() ? llvm::APInt::getSignedMinValue(destination.bits).sext(width) : llvm::APInt(width, 0);
                    auto maximum = destination.signed_integer() ? llvm::APInt::getSignedMaxValue(destination.bits).zext(width) : llvm::APInt::getMaxValue(destination.bits).zext(width);
                    auto* low = llvm::ConstantInt::get(context, minimum);
                    auto* high = llvm::ConstantInt::get(context, maximum);
                    if (destination.bits < width) {
                        result = b.CreateSelect(destination.signed_integer() ? b.CreateICmpSLT(result, low) : b.CreateICmpULT(result, low), low, result);
                        result = b.CreateSelect(destination.signed_integer() ? b.CreateICmpSGT(result, high) : b.CreateICmpUGT(result, high), high, result);
                    }
                    result = b.CreateSelect(overflow, b.CreateSelect(negative, low, high), result);
                    if (!destination.signed_integer()) result = b.CreateSelect(negative, low, result);
                    result = b.CreateSelect(nan, llvm::ConstantInt::get(b.getIntNTy(width), 0), result);
                    result = b.CreateTruncOrBitCast(result, destination.storage(context));
                }
                if (to_float && saturate) {
                    auto* bits = word(result);
                    auto* zero = b.CreateOr(b.CreateICmpSLT(bits, b.getInt32(0)),
                        b.CreateICmpUGT(b.CreateAnd(bits, b.getInt32(0x7fffffff)), b.getInt32(0x7f800000)));
                    bits = b.CreateSelect(zero, b.getInt32(0), b.CreateSelect(b.CreateICmpUGT(bits, b.getInt32(0x3f800000)), b.getInt32(0x3f800000), bits));
                    result = b.CreateBitCast(bits, b.getFloatTy());
                }
                if (to_float && flush) result = flush_float(result);
                write(operands[0], result, destination, to_integer);
                return true;
            }
        }
        if ((op == "setp" || op == "set") && type.kind == 'f' && type.bits >= 32) {
            if (parts.size() < (op == "setp" ? 3u : 4u)) ptx_fail(instruction.opcode, "unsupported FP32 comparison modifier", true);
            unsigned at = 2;
            auto boolean = parts[at] == "and" || parts[at] == "or" || parts[at] == "xor" ? parts[at++] : std::string{};
            bool flush = parts[at] == "ftz"; if (flush) ++at;
            if (flush && type.bits != 32) ptx_fail(instruction.opcode, "ftz requires FP32", true);
            if (at != parts.size() - (op == "setp" ? 1 : 2)) ptx_fail(instruction.opcode, "unsupported FP32 comparison modifier", true);
            arity(boolean.empty() ? 3 : 4);
            auto* left = read(operands[1], type); auto* right = read(operands[2], type);
            if (flush) {
                left = flush_float(left); right = flush_float(right);
            }
            const std::map<std::string, llvm::CmpInst::Predicate> comparisons{
                {"eq", llvm::CmpInst::FCMP_OEQ}, {"ne", llvm::CmpInst::FCMP_ONE}, {"lt", llvm::CmpInst::FCMP_OLT},
                {"le", llvm::CmpInst::FCMP_OLE}, {"gt", llvm::CmpInst::FCMP_OGT}, {"ge", llvm::CmpInst::FCMP_OGE},
                {"equ", llvm::CmpInst::FCMP_UEQ}, {"neu", llvm::CmpInst::FCMP_UNE}, {"ltu", llvm::CmpInst::FCMP_ULT},
                {"leu", llvm::CmpInst::FCMP_ULE}, {"gtu", llvm::CmpInst::FCMP_UGT}, {"geu", llvm::CmpInst::FCMP_UGE},
                {"nan", llvm::CmpInst::FCMP_UNO}, {"num", llvm::CmpInst::FCMP_ORD}};
            if (!comparisons.contains(parts[1])) ptx_fail(instruction.opcode, "unsupported FP32 comparison", true);
            // The common shader pass preserves subnormals with the same exact
            // comparison helper used by CUDA source and imported NVVM.
            auto* comparison = b.CreateFCmp(comparisons.at(parts[1]), left, right);
            llvm::Value* predicate = nullptr;
            if (!boolean.empty()) {
                auto tokens = operands[3]; bool negate = tokens.front().text == "!";
                if (negate) tokens.erase(tokens.begin());
                predicate = read(tokens, {1, 'p'}); if (negate) predicate = b.CreateNot(predicate);
            }
            auto combine = [&](llvm::Value* value) -> llvm::Value* {
                return boolean == "and" ? b.CreateAnd(value, predicate) : boolean == "or" ? b.CreateOr(value, predicate) :
                    boolean == "xor" ? b.CreateXor(value, predicate) : value;
            };
            if (op == "setp") {
                const auto& outputs = operands[0];
                if (outputs.size() != 1 && (outputs.size() != 3 || outputs[1].text != "|")) ptx_fail(instruction.opcode, "invalid comparison predicate destinations");
                // Both results use the incoming c, even when c aliases p or q.
                auto* positive = combine(comparison); auto* negative = combine(b.CreateNot(comparison));
                if (outputs[0].text != "_") write({outputs[0]}, positive, {1, 'p'}, false);
                if (outputs.size() == 3 && outputs[2].text != "_") write({outputs[2]}, negative, {1, 'p'}, false);
                return true;
            }
            auto destination = ptx_scalar({parts[parts.size() - 2], instruction.opcode.line, instruction.opcode.column});
            if (destination.bits != 32 || (destination.kind != 'f' && destination.kind != 's' && destination.kind != 'u'))
                ptx_fail(instruction.opcode, "unsupported FP32 comparison result type", true);
            auto* result = b.CreateSelect(combine(comparison), b.getInt32(destination.kind == 'f' ? 0x3f800000 : 0xffffffff), b.getInt32(0));
            return done(destination.kind == 'f' ? b.CreateBitCast(result, b.getFloatTy()) : result, destination);
        }
        if(op=="mov" && parts.size()==2 && (operands.size()==2) && (operands[0].size()>1 || operands[1].size()>1)) {
            if(type.kind!='b' || (type.bits!=32 && type.bits!=64))return false;
            bool packing=operands[1].size()>1;
            const auto& fields=operands[packing?1:0];
            unsigned lanes=unsigned(fields.size()-1)/2;
            if((lanes!=2 && lanes!=4) || fields.size()!=lanes*2+1 || fields.front().text!="{" || fields.back().text!="}")ptx_fail(instruction.opcode,"invalid packed register list");
            for(unsigned lane=1;lane<lanes;++lane)if(fields[lane*2].text!=",")ptx_fail(instruction.opcode,"invalid packed register separator");
            auto width=type.bits/lanes;
            // PTX has no scalar byte registers; byte moves use wider registers.
            ptx_type element{width,'b'};
            if(packing) {
                llvm::Value* result=llvm::ConstantInt::get(type.storage(context),0);
                for(unsigned lane=0;lane<lanes;++lane)result=b.CreateOr(result,b.CreateShl(b.CreateZExt(read({fields[lane*2+1]},element,true),type.storage(context)),width*lane));
                return done(result,type);
            }
            auto* value=read(operands[1],type);
            for(unsigned lane=0;lane<lanes;++lane)write({fields[lane*2+1]},b.CreateTrunc(b.CreateLShr(value,width*lane),element.storage(context)),element,true);
            return true;
        }
        if(op=="cvt" && parts.size()>=3) {
            auto destination=ptx_scalar({parts[parts.size()-2],instruction.opcode.line,instruction.opcode.column});
            if(destination.kind=='f' && destination.bits==16 && type.kind=='f' && type.bits==16 && parts.size()==4) {
                const std::map<std::string,unsigned> modes{{"rzi",0},{"rmi",1},{"rpi",2},{"rni",4}};
                if(!modes.contains(parts[1]))return false;
                arity(2);auto* x=helper("integral",{real(read(operands[1],type)),b.getInt32(modes.at(parts[1]))},b.getFloatTy());
                auto* v=helper("word_f32_to_f16",{word(x),b.getInt32(0)},b.getInt32Ty());
                return done(b.CreateBitCast(b.CreateTrunc(v,b.getInt16Ty()),b.getHalfTy()),destination);
            }
            if(destination.kind=='f' && destination.bits==32 && type.kind=='f' && type.bits==16 && parts.size()==3) {
                arity(2);return done(real(read(operands[1],type)),destination);
            }
            const std::vector<std::string> modes{"rn","rz","rm","rp"};
            auto found=parts.size()==4?std::find(modes.begin(),modes.end(),parts[1]):modes.end();
            if(destination.kind=='f' && destination.bits==16 && type.kind=='f' && type.bits==32 && found!=modes.end()) {
                arity(2);auto* v=helper("word_f32_to_f16",{word(read(operands[1],type)),b.getInt32(unsigned(found-modes.begin()))},b.getInt32Ty());
                return done(b.CreateBitCast(b.CreateTrunc(v,b.getInt16Ty()),b.getHalfTy()),destination);
            }
        }
        if((type.kind=='f' || type.kind=='h') && (op=="neg" || op=="abs") && parts.size()==2) {
            arity(2);auto* v=word(read(operands[1],type));uint64_t sign=type.kind=='h'?0x80008000ull:type.bits==16?0x8000ull:type.bits==64?0x8000000000000000ull:0x80000000ull;
            auto* mask=llvm::ConstantInt::get(v->getType(),sign);
            v=op=="neg"?b.CreateXor(v,mask):b.CreateAnd(v,b.CreateNot(mask));
            return done(type.kind=='f'?b.CreateBitCast(v,type.value(context)):v,type);
        }
        if((op=="min" || op=="max") && type.kind=='f' && type.bits==64 && parts.size()==2) {
            arity(3);auto* a=read(operands[1],type);auto* c=read(operands[2],type);
            auto* left=word(a);auto* right=word(c);auto* magnitude=b.getInt64(0x7fffffffffffffffull);
            auto* zeros=b.CreateICmpEQ(b.CreateOr(b.CreateAnd(left,magnitude),b.CreateAnd(right,magnitude)),b.getInt64(0));
            auto* selected=b.CreateSelect(op=="min"?b.CreateFCmpOLT(a,c):b.CreateFCmpOGT(a,c),a,c);
            selected=b.CreateSelect(zeros,b.CreateBitCast(op=="min"?b.CreateOr(left,right):b.CreateAnd(left,right),b.getDoubleTy()),selected);
            selected=b.CreateSelect(b.CreateFCmpUNO(c,c),a,selected);
            return done(b.CreateSelect(b.CreateFCmpUNO(a,a),c,selected),type);
        }
        if((op=="min" || op=="max") && ((type.kind=='f' && type.bits<64) || type.kind=='h') &&
            (parts.size()==2 || (parts.size()==3 && parts[1]=="NaN" && (type.bits==16 || type.kind=='h')))) {
            arity(3);auto* a=read(operands[1],type);auto* c=read(operands[2],type);
            auto compute=[&](llvm::Value* a,llvm::Value* c)->llvm::Value* {
                bool half=type.bits==16 || type.kind=='h';
                if(!half)return helper(op,{a,c},b.getFloatTy());
                return helper("word_f16_minmax",{b.CreateZExtOrTrunc(word(a),b.getInt32Ty()),b.CreateZExtOrTrunc(word(c),b.getInt32Ty()),b.getInt32(op=="max"),b.getInt32(parts.size()==3)},b.getInt32Ty());
            };
            if(type.kind=='h') {
                auto lane=[&](unsigned shift){return compute(b.CreateTrunc(b.CreateLShr(a,shift),b.getInt16Ty()),b.CreateTrunc(b.CreateLShr(c,shift),b.getInt16Ty()));};
                return done(b.CreateOr(lane(0),b.CreateShl(lane(16),16)),type);
            }
            auto* v=compute(a,c);return done(type.bits==16?b.CreateBitCast(b.CreateTrunc(v,b.getInt16Ty()),b.getHalfTy()):v,type);
        }
        if(op=="setp" && parts.size()==3 && type.kind=='f') {
            const std::map<std::string,llvm::CmpInst::Predicate> predicates{
                {"eq",llvm::CmpInst::FCMP_OEQ},{"ne",llvm::CmpInst::FCMP_ONE},{"lt",llvm::CmpInst::FCMP_OLT},{"le",llvm::CmpInst::FCMP_OLE},{"gt",llvm::CmpInst::FCMP_OGT},{"ge",llvm::CmpInst::FCMP_OGE},
                {"equ",llvm::CmpInst::FCMP_UEQ},{"neu",llvm::CmpInst::FCMP_UNE},{"ltu",llvm::CmpInst::FCMP_ULT},{"leu",llvm::CmpInst::FCMP_ULE},{"gtu",llvm::CmpInst::FCMP_UGT},{"geu",llvm::CmpInst::FCMP_UGE},{"nan",llvm::CmpInst::FCMP_UNO},{"num",llvm::CmpInst::FCMP_ORD}};
            if(!predicates.contains(parts[1]))return false;
            arity(3);return done(b.CreateFCmp(predicates.at(parts[1]),read(operands[1],type),read(operands[2],type)),{1,'p'});
        }
        if(op=="set" && parts.size()==4 && (type.kind=='f' || type.kind=='h')) {
            arity(3);auto destination=ptx_scalar({parts[2],instruction.opcode.line,instruction.opcode.column});
            if((type.kind=='f' && (type.bits!=16 || destination.kind!='f' || destination.bits!=16)) ||
                (type.kind=='h' && destination.kind!='h' && (destination.kind!='u' || destination.bits!=32)))return false;
            const std::map<std::string,llvm::CmpInst::Predicate> comparisons{{"eq",llvm::CmpInst::FCMP_OEQ},{"ne",llvm::CmpInst::FCMP_ONE},{"nan",llvm::CmpInst::FCMP_UNO}};
            if(!comparisons.contains(parts[1]))return false;
            auto* a=read(operands[1],type);auto* c=read(operands[2],type);
            auto lane=[&](unsigned shift)->llvm::Value* {
                auto half=[&](llvm::Value* x){return b.CreateBitCast(b.CreateTrunc(b.CreateLShr(x,shift),b.getInt16Ty()),b.getHalfTy());};
                auto* predicate=b.CreateFCmp(comparisons.at(parts[1]),type.kind=='h'?half(a):a,type.kind=='h'?half(c):c);
                return b.CreateSelect(predicate,b.getInt32(destination.kind=='u'?65535:0x3c00),b.getInt32(0));
            };
            auto* v=lane(0);
            if(type.kind=='h')v=b.CreateOr(v,b.CreateShl(lane(16),16));
            else v=b.CreateBitCast(b.CreateTrunc(v,b.getInt16Ty()),b.getHalfTy());
            return done(v,destination);
        }
        if((op=="fma" || op=="add" || op=="sub" || op=="mul") && (type.kind=='f' || type.kind=='h') &&
                (parts.size()==2 || (parts.size()==3 && parts[1]=="rn") ||
                    (type.kind=='f' && type.bits==32 && parts==std::vector<std::string>{op,"rn","ftz","f32"}) ||
                    ((type.bits==16 || type.kind=='h') && ((parts.size()==3 && parts[1]=="sat") || (parts.size()==4 && parts[1]=="rn" && parts[2]=="sat"))) ||
                    (op=="fma" && (type.bits==16 || type.kind=='h') && parts.size()==4 && parts[1]=="rn" && parts[2]=="relu"))) {
            bool fused=op=="fma";arity(fused?4:3);
            bool flush=type.kind=='f' && type.bits==32 && parts.size()==4 && parts[2]=="ftz";
            auto* a=read(operands[1],type);auto* c=read(operands[2],type);
            if(flush){a=flush_float(a);c=flush_float(c);}
            auto compute=[&](llvm::Value* a,llvm::Value* c,llvm::Value* d)->llvm::Value* {
                if(type.kind=='f' && type.bits==32 && !fused) {
                    if(op=="sub")c=b.CreateBitCast(b.CreateXor(word(c),b.getInt32(0x80000000)),b.getFloatTy());
                    return helper(op=="mul"?"mul_f32":"add_f32",{a,c},b.getFloatTy());
                }
                llvm::Value* v=fused?b.CreateIntrinsic(a->getType(),llvm::Intrinsic::fma,{a,c,d}):op=="add"?b.CreateFAdd(a,c):op=="sub"?b.CreateFSub(a,c):b.CreateFMul(a,c);
                bool saturate=std::find(parts.begin(),parts.end(),"sat")!=parts.end();
                bool relu=std::find(parts.begin(),parts.end(),"relu")!=parts.end();
                if(saturate || relu) {
                        auto* bits=word(v);auto* magnitude=b.CreateAnd(bits,b.getInt16(0x7fff));
                        auto* nan=b.CreateICmpUGT(magnitude,b.getInt16(0x7c00));
                        auto* negative=b.CreateICmpNE(b.CreateAnd(bits,b.getInt16(0x8000)),b.getInt16(0));
                        if(saturate)bits=b.CreateSelect(b.CreateOr(nan,negative),b.getInt16(0),b.CreateSelect(b.CreateICmpUGT(bits,b.getInt16(0x3c00)),b.getInt16(0x3c00),bits));
                        else bits=b.CreateSelect(nan,b.getInt16(0x7fff),b.CreateSelect(negative,b.getInt16(0),bits));
                        v=b.CreateBitCast(bits,b.getHalfTy());
                }
                return v;
            };
            auto* d=fused?read(operands[3],type):nullptr;
            if(flush && d)d=flush_float(d);
            if(type.kind=='h') {
                auto lane=[&](unsigned shift){auto half=[&](llvm::Value* x){return b.CreateBitCast(b.CreateTrunc(b.CreateLShr(x,shift),b.getInt16Ty()),b.getHalfTy());};return b.CreateZExt(word(compute(half(a),half(c),d?half(d):nullptr)),b.getInt32Ty());};
                return done(b.CreateOr(lane(0),b.CreateShl(lane(16),16)),type);
            }
            auto* result=compute(a,c,d);
            return done(flush?flush_float(result):result,type);
        }
        if((op=="div" || op=="rcp" || op=="sqrt") && type.kind=='f' && type.bits==32 && parts.size()>=3 &&
            (parts[1]=="rn" || (op=="div" && (parts[1]=="approx" || parts[1]=="full")))) {
            bool flush=parts.size()==4 && parts[2]=="ftz";
            if(parts.size()!=(flush?4u:3u))ptx_fail(instruction.opcode,"unsupported FP32 arithmetic modifier",true);
            arity(op=="div"?3:2);auto* x=read(operands[1],type);if(flush)x=flush_float(x);
            llvm::Value* result=nullptr;
            if(op=="sqrt")result=helper("sqrt",{x},b.getFloatTy());
            else if(op=="rcp")result=b.CreateFDiv(llvm::ConstantFP::get(b.getFloatTy(),1),x);
            else {auto* y=read(operands[2],type);if(flush)y=flush_float(y);result=b.CreateFDiv(x,y);}
            return done(flush?flush_float(result):result,type);
        }
        if((op=="rcp" || op=="sqrt" || op=="rsqrt" || op=="ex2" || op=="lg2" || op=="sin" || op=="cos" || op=="tanh") &&
                ((type.kind=='f' && type.bits==32 && (parts==std::vector<std::string>{op,"approx","ftz","f32"} || parts==std::vector<std::string>{op,"approx","f32"})) ||
                 (op=="tanh" && (type.bits==16 || type.kind=='h') && parts.size()==3 && parts[1]=="approx"))) {
            arity(2);auto* a=read(operands[1],type);
            auto name=op=="ex2"?"exp2":op=="lg2"?"log2":op;
            auto compute=[&](llvm::Value* x)->llvm::Value* {
                bool half=type.bits==16 || type.kind=='h',flush=parts.size()==4;
                if(half)x=real(x);
                if(flush)x=helper("flush",{x},b.getFloatTy());
                x=helper(name,{x},b.getFloatTy());
                if(flush)x=helper("flush",{x},b.getFloatTy());
                if(half)x=b.CreateTrunc(helper("word_f32_to_f16",{word(x),b.getInt32(0)},b.getInt32Ty()),b.getInt16Ty());
                return x;
            };
            if(type.kind=='h') {
                auto lane=[&](unsigned shift){return b.CreateZExt(compute(b.CreateTrunc(b.CreateLShr(a,shift),b.getInt16Ty())),b.getInt32Ty());};
                return done(b.CreateOr(lane(0),b.CreateShl(lane(16),16)),type);
            }
            auto* v=compute(a);return done(type.bits==16?b.CreateBitCast(v,b.getHalfTy()):v,type);
        }
        return false;
    }
    llvm::Value* high_product(llvm::IRBuilder<>& b, llvm::Value* left, llvm::Value* right, ptx_type type) {
        if (type.bits == 16) {
            auto extend = [&](llvm::Value* value) { return type.signed_integer() ? b.CreateSExt(value, b.getInt32Ty()) : b.CreateZExt(value, b.getInt32Ty()); };
            return b.CreateTrunc(b.CreateLShr(b.CreateMul(extend(left), extend(right)), 16), type.storage(context));
        }
        auto name = std::string("llvm.nvvm.mulhi.") + (type.bits == 64 ? (type.signed_integer() ? "ll" : "ull") : (type.signed_integer() ? "i" : "ui"));
        return b.CreateCall(module->getOrInsertFunction(name,
            llvm::FunctionType::get(type.storage(context), {left->getType(), right->getType()}, false)), {left, right});
    }
    template<class Read, class Write> bool integer_instruction(const std::vector<std::string>& parts,
            const ptx_instruction& instruction, llvm::IRBuilder<>& b, Read read, Write write, llvm::AllocaInst* carry) {
        const auto& op = parts[0]; const auto& operands = instruction.operands; const auto& opcode = instruction.opcode;
        if((op=="setp" || op=="set") && parts.size()>=3 && (parts.back().starts_with('s') || parts.back().starts_with('u') || parts.back().starts_with('b'))) {
            auto type=ptx_scalar({parts.back(),opcode.line,opcode.column});
            unsigned at=2;std::string boolean;
            if(at<parts.size() && (parts[at]=="and" || parts[at]=="or" || parts[at]=="xor"))boolean=parts[at++];
            if(parts.size()!=at+(op=="set"?2u:1u) || operands.size()!=(boolean.empty()?3u:4u) || type.bits<16)ptx_fail(opcode,"unsupported integer comparison",true);
            bool unsigned_value=type.kind!='s';
            const std::map<std::string,llvm::CmpInst::Predicate> comparisons{{"eq",llvm::CmpInst::ICMP_EQ},{"ne",llvm::CmpInst::ICMP_NE},
                {"lt",unsigned_value?llvm::CmpInst::ICMP_ULT:llvm::CmpInst::ICMP_SLT},{"le",unsigned_value?llvm::CmpInst::ICMP_ULE:llvm::CmpInst::ICMP_SLE},
                {"gt",unsigned_value?llvm::CmpInst::ICMP_UGT:llvm::CmpInst::ICMP_SGT},{"ge",unsigned_value?llvm::CmpInst::ICMP_UGE:llvm::CmpInst::ICMP_SGE},
                {"lo",llvm::CmpInst::ICMP_ULT},{"ls",llvm::CmpInst::ICMP_ULE},{"hi",llvm::CmpInst::ICMP_UGT},{"hs",llvm::CmpInst::ICMP_UGE}};
            if(!comparisons.contains(parts[1]) || (type.kind=='b' && parts[1]!="eq" && parts[1]!="ne"))ptx_fail(opcode,"unsupported instruction or modifier",true);
            auto* compared=b.CreateICmp(comparisons.at(parts[1]),read(operands[1],type),read(operands[2],type));
            llvm::Value* incoming=nullptr;
            if(!boolean.empty()) {auto tokens=operands[3];bool negate=tokens.front().text=="!";if(negate)tokens.erase(tokens.begin());incoming=read(tokens,{1,'p'});if(negate)incoming=b.CreateNot(incoming);}
            auto combine=[&](llvm::Value* value)->llvm::Value* {return boolean=="and"?b.CreateAnd(value,incoming):boolean=="or"?b.CreateOr(value,incoming):boolean=="xor"?b.CreateXor(value,incoming):value;};
            if(op=="setp") {
                auto positive=combine(compared),negative=combine(b.CreateNot(compared));
                const auto& destination=operands[0];if(destination.size()!=1 && (destination.size()!=3 || destination[1].text!="|"))ptx_fail(opcode,"invalid comparison destinations");
                if(destination[0].text!="_")write({destination[0]},positive,{1,'p'},false);
                if(destination.size()==3 && destination[2].text!="_")write({destination[2]},negative,{1,'p'},false);
            } else {
                auto destination=ptx_scalar({parts[parts.size()-2],opcode.line,opcode.column});
                if(destination.bits!=32 || (destination.kind!='s' && destination.kind!='u' && destination.kind!='f'))ptx_fail(opcode,"invalid comparison result type");
                auto* result=b.CreateSelect(combine(compared),b.getInt32(destination.kind=='f'?0x3f800000u:0xffffffffu),b.getInt32(0));
                write(operands[0],destination.kind=='f'?b.CreateBitCast(result,b.getFloatTy()):result,destination,false);
            }
            return true;
        }
        if (op == "mul24" || op == "mad24" || op == "sad" || op == "abs" || op == "dp4a" || op == "dp2a") {
            if(op=="abs" && parts.back().starts_with('f'))return false;
            auto type=ptx_scalar({parts.back(),opcode.line,opcode.column});
            auto invalid=[&] { ptx_fail(opcode,"unsupported integer instruction",true); };
            if(type.kind!='s' && type.kind!='u')invalid();
            if(op=="abs") {
                if(parts.size()!=2 || type.kind!='s' || type.bits<16)invalid();
                if(operands.size()!=2)ptx_fail(opcode,"wrong operand count");
                auto* a=read(operands[1],type);write(operands[0],b.CreateSelect(b.CreateICmpSLT(a,llvm::ConstantInt::get(a->getType(),0)),b.CreateNeg(a),a),type,false);return true;
            }
            if(type.bits!=32)invalid();
            unsigned count=op=="mul24"?3:4;
            if(operands.size()!=count)ptx_fail(opcode,"wrong operand count");
            auto lane=[&](llvm::Value* value,unsigned shift,unsigned bits,bool sign)->llvm::Value* {
                auto* small=b.CreateTrunc(b.CreateLShr(value,shift),b.getIntNTy(bits));
                return sign?b.CreateSExt(small,b.getInt64Ty()):b.CreateZExt(small,b.getInt64Ty());
            };
            auto* a=read(operands[1],{32,'u'});auto* c=read(operands[2],{32,'u'});
            llvm::Value* result=nullptr;
            if(op=="mul24" || op=="mad24") {
                bool sat=parts.size()==4 && parts[2]=="sat";
                if((parts.size()!=3 && !sat) || (parts[1]!="lo" && parts[1]!="hi") || (sat && (op!="mad24" || parts[1]!="hi" || type.kind!='s')))invalid();
                result=b.CreateMul(lane(a,0,24,type.signed_integer()),lane(c,0,24,type.signed_integer()));
                if(parts[1]=="hi")result=type.signed_integer()?b.CreateAShr(result,16):b.CreateLShr(result,16);
                result=b.CreateTrunc(result,b.getInt32Ty());
                if(op=="mad24") {
                    auto* addend=read(operands[3],{32,'b'});
                    if(sat)result=b.CreateBinaryIntrinsic(llvm::Intrinsic::sadd_sat,result,addend);
                    else result=b.CreateAdd(result,addend);
                }
            } else if(op=="sad") {
                if(parts.size()!=2)invalid();
                auto* difference=b.CreateSub(lane(a,0,32,type.signed_integer()),lane(c,0,32,type.signed_integer()));
                result=b.CreateTrunc(b.CreateSelect(b.CreateICmpSLT(difference,b.getInt64(0)),b.CreateNeg(difference),difference),b.getInt32Ty());
                result=b.CreateAdd(result,read(operands[3],{32,'b'}));
            } else {
                bool four=op=="dp4a";
                if(parts.size()!=(four?3u:4u) || (!four && parts[1]!="lo" && parts[1]!="hi"))invalid();
                auto atype=ptx_scalar({parts[parts.size()-2],opcode.line,opcode.column});
                if(atype.bits!=32 || (atype.kind!='s' && atype.kind!='u') || source_sm<61)invalid();
                result=b.CreateZExt(read(operands[3],{32,'b'}),b.getInt64Ty());
                for(unsigned i=0;i<(four?4u:2u);++i)
                    result=b.CreateAdd(result,b.CreateMul(lane(a,i*(four?8:16),four?8:16,atype.signed_integer()),lane(c,(i+(!four && parts[1]=="hi"?2:0))*8,8,type.signed_integer())));
                result=b.CreateTrunc(result,b.getInt32Ty());
            }
            write(operands[0],result,{32,'u'},false);return true;
        }
        if (op.size()>1 && (op.back()=='2' || op.back()=='4') && op.front()=='v') {
            auto base=op.substr(0,op.size()-1);bool comparison=base=="vset";
            if(base!="vadd" && base!="vsub" && base!="vavrg" && base!="vabsdiff" && base!="vmin" && base!="vmax" && !comparison)return false;
            unsigned bits=op.back()=='2'?16:8;
            if(operands.size()!=4)ptx_fail(opcode,"wrong operand count");
            if(parts.size()<4 || parts.size()>6)ptx_fail(opcode,"unsupported packed instruction modifier",true);
            unsigned modifier=4;bool saturate=modifier<parts.size() && parts[modifier]=="sat";if(saturate)++modifier;
            bool accumulate=modifier<parts.size() && parts[modifier]=="add";if(accumulate)++modifier;
            if(modifier!=parts.size() || (comparison && saturate))ptx_fail(opcode,"unsupported packed instruction modifier",true);
            auto dtype=ptx_scalar({parts[1],opcode.line,opcode.column});
            auto atype=ptx_scalar({parts[comparison?1:2],opcode.line,opcode.column});
            auto btype=ptx_scalar({parts[comparison?2:3],opcode.line,opcode.column});
            for(auto type:{dtype,atype,btype})if(type.bits!=32 || (type.kind!='s' && type.kind!='u'))ptx_fail(opcode,"invalid packed instruction type");
            auto selector=[&](std::vector<ptx_token> tokens,bool source,std::string defaults) {
                if(tokens.size()!=1)ptx_fail(opcode,"packed operands require registers");
                auto dot=tokens[0].text.rfind('.');
                if(dot!=std::string::npos && dot+1<tokens[0].text.size() && (tokens[0].text[dot+1]=='h' || tokens[0].text[dot+1]=='b')) {
                    auto suffix=tokens[0].text.substr(dot+1);tokens[0].text.resize(dot);
                    if(suffix.empty() || suffix[0]!=(bits==16?'h':'b'))ptx_fail(opcode,"invalid packed selector");
                    defaults=suffix.substr(1);
                }
                if(defaults.empty() || defaults.size()>(32/bits) || (source && defaults.size()!=32/bits))ptx_fail(opcode,"invalid packed selector");
                unsigned selected=0;
                for(char digit:defaults) {
                    unsigned index=unsigned(digit-'0');
                    if(index>=(source?64/bits:32/bits) || (!source && (selected&(1u<<index))))ptx_fail(opcode,"invalid packed selector");
                    selected|=1u<<index;
                }
                return std::make_pair(tokens,std::move(defaults));
            };
            auto [a_operand,a_selector]=selector(operands[1],true,bits==16?"10":"3210");
            auto [b_operand,b_selector]=selector(operands[2],true,bits==16?"32":"7654");
            auto [destination,mask]=selector(operands[0],false,bits==16?"10":"3210");
            auto* a=read(a_operand,{32,'u'});auto* c=read(b_operand,{32,'u'});auto* merge=read(operands[3],{32,'u'});
            auto* pair=b.CreateOr(b.CreateZExt(a,b.getInt64Ty()),b.CreateShl(b.CreateZExt(c,b.getInt64Ty()),32));
            llvm::Value* result=accumulate?merge:b.getInt32(0);
            for(unsigned shift=0;shift<32;shift+=bits) {
                auto lane=shift/bits;
                if(mask.find(char('0'+lane))==mask.npos) {
                    if(!accumulate)result=b.CreateOr(result,b.CreateAnd(merge,b.getInt32(((1u<<bits)-1)<<shift)));
                    continue;
                }
                auto extend=[&](const std::string& selected,ptx_type type)->llvm::Value* {
                    auto index=unsigned(selected[selected.size()-1-lane]-'0');
                    auto* small=b.CreateTrunc(b.CreateLShr(pair,index*bits),b.getIntNTy(bits));
                    return type.signed_integer()?b.CreateSExt(small,b.getInt32Ty()):b.CreateZExt(small,b.getInt32Ty());
                };
                auto* x=extend(a_selector,atype);auto* y=extend(b_selector,btype);llvm::Value* value=nullptr;
                if(comparison) {
                    const std::map<std::string,llvm::CmpInst::Predicate> predicates{{"eq",llvm::CmpInst::ICMP_EQ},{"ne",llvm::CmpInst::ICMP_NE},{"lt",llvm::CmpInst::ICMP_SLT},{"le",llvm::CmpInst::ICMP_SLE},{"gt",llvm::CmpInst::ICMP_SGT},{"ge",llvm::CmpInst::ICMP_SGE}};
                    if(!predicates.contains(parts[3]))ptx_fail(opcode,"invalid packed comparison");
                    value=b.CreateZExt(b.CreateICmp(predicates.at(parts[3]),x,y),b.getInt32Ty());
                } else if(base=="vadd" || base=="vavrg") {
                    value=b.CreateAdd(x,y);
                    if(base=="vavrg")value=b.CreateAShr(b.CreateAdd(value,b.CreateZExt(b.CreateICmpSGE(value,b.getInt32(0)),b.getInt32Ty())),1);
                } else if(base=="vsub" || base=="vabsdiff") {
                    value=b.CreateSub(x,y);
                    if(base=="vabsdiff")value=b.CreateSelect(b.CreateICmpSLT(value,b.getInt32(0)),b.CreateNeg(value),value);
                } else value=b.CreateSelect(base=="vmin"?b.CreateICmpSLT(x,y):b.CreateICmpSGT(x,y),x,y);
                if(saturate) {
                    auto* minimum=b.getInt32(dtype.signed_integer()?uint32_t(0)-(1u<<(bits-1)):0);
                    auto* maximum=b.getInt32(dtype.signed_integer()?(1u<<(bits-1))-1:(1u<<bits)-1);
                    value=b.CreateSelect(b.CreateICmpSLT(value,minimum),minimum,b.CreateSelect(b.CreateICmpSGT(value,maximum),maximum,value));
                }
                result=accumulate?b.CreateAdd(result,value):b.CreateOr(result,b.CreateShl(b.CreateAnd(value,b.getInt32((1u<<bits)-1)),shift));
            }
            write(destination,result,{32,'u'},false);return true;
        }
        auto arity = [&](unsigned count) { if (operands.size() != count) ptx_fail(opcode, "wrong operand count"); };
        auto operand = [&](unsigned index, ptx_type type) { return read(operands.at(index), type); };
        auto unsupported = [&] { ptx_fail(opcode, "unsupported integer instruction or modifier", true); };
        bool cc = std::find(parts.begin(), parts.end(), "cc") != parts.end();
        if (op == "addc" || op == "subc" || op == "madc" || (cc && (op == "add" || op == "sub" || op == "mad"))) {
            bool multiply = op == "mad" || op == "madc", input_carry = op == "addc" || op == "subc" || op == "madc";
            if (parts.size() != 2u + unsigned(multiply) + unsigned(cc) ||
                    (cc && parts[parts.size() - 2] != "cc") ||
                    (multiply && parts[1] != "lo" && parts[1] != "hi")) unsupported();
            auto type = ptx_scalar({parts.back(), opcode.line, opcode.column});
            if ((type.kind != 'u' && type.kind != 's') || (type.bits != 32 && type.bits != 64)) unsupported();
            arity(multiply ? 4 : 3);
            auto* left = operand(1, type); auto* right = operand(2, type);
            if (multiply) {
                left = parts[1] == "hi" ? high_product(b, left, right, type) : b.CreateMul(left, right);
                right = operand(3, type);
            }
            bool subtract = op == "sub" || op == "subc";
            auto* value = subtract ? b.CreateSub(left, right) : b.CreateAdd(left, right);
            llvm::Value* overflow = subtract ? b.CreateICmpULT(left, right) : b.CreateICmpULT(value, left);
            if (input_carry) {
                auto* flag = b.CreateLoad(b.getInt1Ty(), carry);
                // NVIDIA uses a carry/no-borrow encoding for CC.CF. Subc
                // consumes its complement; mixed add/sub chains observe the
                // same flag, rather than two independently named states.
                auto* incoming = b.CreateZExt(subtract ? b.CreateNot(flag) : flag, type.storage(context));
                auto* final = subtract ? b.CreateSub(value, incoming) : b.CreateAdd(value, incoming);
                overflow = b.CreateOr(overflow, subtract ? b.CreateICmpULT(value, incoming) : b.CreateICmpULT(final, value));
                value = final;
            }
            if (cc) b.CreateStore(subtract ? b.CreateNot(overflow) : overflow, carry);
            write(operands[0], value, type);
            return true;
        }
        if (op == "prmt") {
            if (parts.size() < 2 || parts[1] != "b32" || parts.size() > 3) unsupported();
            auto mode = parts.size() == 3 ? parts[2] : "";
            if (!byte_permutation_mode(mode)) unsupported();
            arity(4);
            auto* low = operand(1, {32, 'b'}); auto* high = operand(2, {32, 'b'}); auto* control = operand(3, {32, 'b'});
            write(operands[0], byte_permutation_ir(b, low, high, control, mode), {32, 'b'}); return true;
        }
        if (op == "lop3") {
            bool predicate = parts.size() == 3 && (parts[1] == "or" || parts[1] == "and");
            if (parts != std::vector<std::string>{"lop3", "b32"} && !(predicate && parts[2] == "b32")) unsupported();
            if (predicate && (source_version < 82 || source_sm < 70)) ptx_fail(opcode, "predicate lop3 requires PTX 8.2 and SM 70", true);
            arity(predicate ? 6 : 5);
            const auto& literal = operands[4];
            if (literal.size() != 1 || !std::isdigit(static_cast<unsigned char>(literal[0].text[0]))) ptx_fail(opcode, "lop3 requires an immediate lookup table");
            auto table = ptx_unsigned(literal[0]); if (table > 255) ptx_fail(literal[0], "lop3 lookup table exceeds 8 bits");
            llvm::Value* nodes[8];
            for (unsigned i = 0; i < 8; ++i) nodes[i] = b.getInt32(table & (1u << i) ? UINT32_MAX : 0);
            for (unsigned level = 0; level < 3; ++level) {
                auto* mask = operand(3 - level, {32, 'b'});
                for (unsigned i = 0; i < (4u >> level); ++i) nodes[i] = nodes[i * 2] == nodes[i * 2 + 1] ? nodes[i * 2] :
                    b.CreateOr(b.CreateAnd(b.CreateNot(mask), nodes[i * 2]), b.CreateAnd(mask, nodes[i * 2 + 1]));
            }
            if (!predicate) write(operands[0], nodes[0], {32, 'b'});
            else {
                const auto& destination = operands[0];
                if (destination.size() != 3 || destination[1].text != "|") ptx_fail(opcode, "predicate lop3 requires paired destinations");
                if (destination[2].text == "_") ptx_fail(opcode, "predicate lop3 requires a predicate destination");
                auto* test = b.CreateICmpNE(nodes[0], b.getInt32(0)); auto* q = operand(5, {1, 'p'});
                auto* result = parts[1] == "or" ? b.CreateOr(test, q) : b.CreateAnd(test, q);
                if (destination[0].text != "_") write({destination[0]}, nodes[0], {32, 'b'});
                write({destination[2]}, result, {1, 'p'});
            }
            return true;
        }
        if (op == "shf") {
            if (parts.size() != 4 || (parts[1] != "l" && parts[1] != "r") ||
                    (parts[2] != "wrap" && parts[2] != "clamp") || parts[3] != "b32") unsupported();
            arity(4);
            auto* low = operand(1, {32, 'b'}); auto* high = operand(2, {32, 'b'}); auto* shift = operand(3, {32, 'u'});
            auto intrinsic = parts[1] == "l" ? llvm::Intrinsic::fshl : llvm::Intrinsic::fshr;
            auto* native = llvm::Intrinsic::getOrInsertDeclaration(module.get(), intrinsic, {b.getInt32Ty()});
            llvm::Value* value = b.CreateCall(native, {high, low, shift});
            if (parts[2] == "clamp") value = b.CreateSelect(b.CreateICmpULT(shift, b.getInt32(32)), value, parts[1] == "l" ? low : high);
            write(operands[0], value, {32, 'b'}); return true;
        }
        if (op != "bfe" && op != "bfi" && op != "brev" && op != "popc" && op != "clz" && op != "bfind" && op != "cnot") return false;
        bool shift_amount = op == "bfind" && parts.size() == 3 && parts[1] == "shiftamt";
        if (parts.size() != 2u + unsigned(shift_amount)) unsupported();
        auto type = ptx_scalar({parts.back(), opcode.line, opcode.column});
        bool integer = op == "bfe" || op == "bfind";
        if ((integer ? (type.kind != 's' && type.kind != 'u') : type.kind != 'b') ||
                (type.bits != 32 && type.bits != 64 && !(op == "cnot" && type.bits == 16))) unsupported();
        arity(op == "bfe" ? 4 : op == "bfi" ? 5 : 2);
        auto* input = operand(1, type); llvm::Value* value = nullptr;
        if (op == "bfe" || op == "bfi") {
            auto* position = b.CreateAnd(operand(op == "bfe" ? 2 : 3, {32, 'u'}), b.getInt32(255));
            auto* length = b.CreateAnd(operand(op == "bfe" ? 3 : 4, {32, 'u'}), b.getInt32(255));
            auto* bound = b.getInt32(type.bits); auto* zero = llvm::ConstantInt::get(type.storage(context), 0);
            auto* position_valid = b.CreateICmpULT(position, bound);
            auto* safe = b.CreateZExtOrTrunc(b.CreateSelect(position_valid, position, b.getInt32(type.bits - 1)), type.storage(context));
            auto* bounded_length = b.CreateSelect(b.CreateICmpULT(length, bound), length, bound);
            auto* padding = b.CreateZExtOrTrunc(b.CreateAnd(b.CreateSub(bound, bounded_length), b.getInt32(type.bits - 1)), type.storage(context));
            auto* mask = b.CreateSelect(b.CreateICmpEQ(length, b.getInt32(0)), zero,
                b.CreateLShr(llvm::ConstantInt::getAllOnesValue(type.storage(context)), padding));
            if (op == "bfe") {
                auto* shifted = type.signed_integer() ? b.CreateAShr(input, safe) : b.CreateLShr(input, safe);
                if (type.signed_integer()) value = b.CreateSelect(b.CreateICmpEQ(length, b.getInt32(0)), zero,
                    b.CreateAShr(b.CreateShl(shifted, padding), padding));
                else value = b.CreateSelect(position_valid, b.CreateAnd(shifted, mask), zero);
            } else {
                auto* target = operand(2, type);
                mask = b.CreateSelect(position_valid, b.CreateShl(mask, safe), zero);
                value = b.CreateOr(b.CreateAnd(target, b.CreateNot(mask)), b.CreateAnd(b.CreateShl(input, safe), mask));
            }
        } else if (op == "cnot") value = b.CreateZExt(b.CreateICmpEQ(input, llvm::ConstantInt::get(type.storage(context), 0)), type.storage(context));
        else {
            if (op == "bfind" && type.signed_integer()) input = b.CreateSelect(b.CreateICmpSLT(input, llvm::ConstantInt::get(type.storage(context), 0)), b.CreateNot(input), input);
            auto intrinsic = op == "brev" ? llvm::Intrinsic::bitreverse : op == "popc" ? llvm::Intrinsic::ctpop : llvm::Intrinsic::ctlz;
            auto* native = llvm::Intrinsic::getOrInsertDeclaration(module.get(), intrinsic, {type.storage(context)});
            value = intrinsic == llvm::Intrinsic::ctlz ? b.CreateCall(native, {input, b.getFalse()}) : b.CreateCall(native, {input});
            if (op != "brev") {
                value = b.CreateZExtOrTrunc(value, b.getInt32Ty());
                if (op == "bfind") value = b.CreateSelect(b.CreateICmpEQ(input, llvm::ConstantInt::get(type.storage(context), 0)), b.getInt32(UINT32_MAX),
                    shift_amount ? value : b.CreateSub(b.getInt32(type.bits - 1), value));
                type = {32, 'u'};
            }
        }
        write(operands[0], value, type); return true;
    }
    void lower(ptx_body& body) {
        auto& function = *body.function; auto* entry = &function.getEntryBlock();
        const auto& registers = body.registers;
        std::vector<ptx_instruction> instructions;
        // CUDA compilers use vector parameter operations for aggregate fields.
        // Expand them into scalar accesses before SSA construction; each lane
        // retains the original predicate and its byte offset.
        for (auto instruction : body.instructions) {
            for(auto& operand:instruction.operands)if(operand.size()==1 && body.vectors.contains(operand[0].text)) {
                auto token=operand[0];const auto& lanes=body.vectors.at(token.text);operand.clear();
                token.text="{";operand.push_back(token);
                for(const auto& lane:lanes) {if(operand.size()>1){token.text=",";operand.push_back(token);}token.text=lane;operand.push_back(token);}
                token.text="}";operand.push_back(token);
            }
            auto text = instruction.opcode.text; size_t at = text.find(".v2."); unsigned lanes = 2;
            if (at == text.npos) { at = text.find(".v4."); lanes = 4; }
            if (at == text.npos || (!text.starts_with("ld.") && !text.starts_with("st."))) { instructions.push_back(instruction); continue; }
            bool load = text.starts_with("ld.");
            if (instruction.operands.size() != 2) ptx_fail(instruction.opcode, "wrong operand count");
            auto type = ptx_scalar({text.substr(text.rfind('.') + 1), instruction.opcode.line, instruction.opcode.column}, true);
            const auto& fields = instruction.operands[load ? 0 : 1]; const auto& address = instruction.operands[load ? 1 : 0];
            if (fields.size() != lanes * 2 + 1 || fields.front().text != "{" || fields.back().text != "}") ptx_fail(instruction.opcode, "invalid vector register list");
            if ((address.size() != 3 && address.size() != 5) || address.front().text != "[" || address.back().text != "]") ptx_fail(instruction.opcode, "unsupported address expression", true);
            int64_t displacement = 0;
            if (address.size() == 5) {
                if (address[2].text != "+" && address[2].text != "-") ptx_fail(address[2], "invalid address displacement");
                auto amount = ptx_unsigned(address[3]); if (amount > INT32_MAX - 32) ptx_fail(address[3], "address displacement exceeds supported range", true);
                displacement = int64_t(amount) * (address[2].text == "-" ? -1 : 1);
            }
            if (text.starts_with("ld.param.") || text.starts_with("st.param.")) {
                auto found = body.slots.find(address[1].text); auto align = lanes * type.bits / 8;
                if (found == body.slots.end() || found->second.first.alignment < align || displacement % align) ptx_fail(address[1], "vector parameter access is misaligned");
            }
            for (unsigned lane = 0; lane < lanes; ++lane) {
                if (lane && fields[lane * 2].text != ",") ptx_fail(instruction.opcode, "invalid vector register separator");
                auto scalar = instruction; scalar.opcode.text.erase(at, 3);
                scalar.operands[load ? 0 : 1] = {fields[lane * 2 + 1]};
                auto offset = displacement + lane * (type.bits / 8);
                auto token = instruction.opcode;
                auto sign = token; sign.text = offset < 0 ? "-" : "+";
                auto amount = token; amount.text = std::to_string(offset < 0 ? -offset : offset);
                scalar.operands[load ? 1 : 0] = {address.front(), address[1], sign, amount, address.back()};
                instructions.push_back(std::move(scalar));
            }
        }
        llvm::IRBuilder<> builder(entry);
        // The implicit carry flag is local to this PTX function, shared by its
        // lexical scopes and promoted with the ordinary registers into SSA.
        auto* carry = builder.CreateAlloca(builder.getInt1Ty(), nullptr, "ptx.cc.cf");
        auto reg = [&](const ptx_token& token) -> const ptx_register& {
            auto found = registers.find(token.text);
            if (found == registers.end()) ptx_fail(token, "undeclared register");
            return found->second;
        };
        auto scalar = [&](const std::vector<ptx_token>& tokens, ptx_type type, bool truncate_memory = false) -> llvm::Value* {
            if (tokens.empty() || tokens.size() > 2) ptx_fail(tokens.empty() ? ptx_token{"operand", 0, 0} : tokens.front(), "expected scalar operand");
            bool invert=type.kind=='p' && tokens.size()==2 && tokens[0].text=="!";
            bool negative = tokens.size() == 2 && !invert;
            if (negative && tokens[0].text != "-") ptx_fail(tokens[0], "expected negative immediate");
            const auto& token = tokens.back();
            llvm::Value* value = nullptr;
            if (token.text.starts_with('%') || registers.contains(token.text)) {
                if (negative) ptx_fail(token, "register negation requires an instruction", true);
                auto found = registers.find(token.text);
                if (found != registers.end()) {
                    if (!type.accepts_register(found->second.type)) ptx_fail(token, "register type does not match instruction");
                    if (found->second.type.bits != type.bits && (!truncate_memory || found->second.type.bits < type.bits))
                        ptx_fail(token, "register width does not match instruction");
                    value = builder.CreateLoad(found->second.type.storage(context), found->second.slot);
                    value = builder.CreateZExtOrTrunc(value, type.storage(context));
                } else {
                    if (type.bits != 32 || type.kind == 'f') ptx_fail(token, "special register requires a 32-bit integer operand");
                    const std::string families[]{"tid", "ntid", "ctaid", "nctaid"};
                    const std::string operations[]{"thread_id", "block_size", "block_id", "grid_size"};
                    for (unsigned family = 0; family < 4; ++family) for (char axis : {'x', 'y', 'z'})
                        if (token.text == "%" + families[family] + "." + axis) {
                            auto name = "cudavm.op." + operations[family] + "." + axis;
                            auto callee = module->getOrInsertFunction(name, llvm::FunctionType::get(builder.getInt32Ty(), {}, false));
                            auto* builtin = llvm::cast<llvm::Function>(callee.getCallee());
                            builtin->addFnAttr("cudavm.cuda.origin", "llvm.nvvm.read.ptx.sreg." + families[family] + "." + axis);
                            value = builder.CreateCall(callee);
                        }
                    for (auto special : {"laneid", "warpsize", "lanemask.eq", "lanemask.lt", "lanemask.le", "lanemask.gt", "lanemask.ge"}) {
                        auto spelling=std::string("%")+special;
                        if(spelling.starts_with("%lanemask."))spelling[9]='_';
                        if (token.text == spelling || token.text == std::string("%") + special)
                            value = builder.CreateCall(module->getOrInsertFunction(std::string("llvm.nvvm.read.ptx.sreg.") + special,
                                llvm::FunctionType::get(builder.getInt32Ty(), {}, false)));
                    }
                    if (!value) ptx_fail(token, token.text.find('.') == std::string::npos ? "undeclared register" : "unsupported special register",
                        token.text.find('.') != std::string::npos);
                }
            } else if (body.locals.contains(token.text)) {
                if (negative || (type.bits != 32 && type.bits != 64) || type.kind == 'f') ptx_fail(token, "invalid local address operand");
                value = builder.CreatePtrToInt(body.locals.at(token.text), type.storage(context));
            } else if (auto* global = module->getGlobalVariable(token.text, true)) {
                if (negative || (type.bits != 32 && type.bits != 64) || type.kind == 'f') ptx_fail(token, "invalid symbol address operand");
                value = builder.CreatePtrToInt(global, type.storage(context));
            } else if (auto* target = module->getFunction(token.text)) {
                if (negative || type.bits != 64 || type.kind == 'f' || target->hasFnAttribute("cudavm.kernel"))
                    ptx_fail(token, "function address requires a 64-bit device-function operand");
                if (!relocatable && target->isDeclaration() && !resolved_external(*target)) ptx_fail(token, "unresolved device function");
                value = builder.CreatePtrToInt(target, type.storage(context));
            } else if ((token.text.starts_with("0f") && token.text.size() == 10) || (token.text.starts_with("0d") && token.text.size() == 18)) {
                auto width = token.text[1] == 'd' ? 64u : 32u;
                if (type.bits != width || negative) ptx_fail(token, "invalid floating bit literal");
                uint64_t bits = 0;
                if (llvm::StringRef(token.text).drop_front(2).getAsInteger(16, bits)) ptx_fail(token, "invalid binary32 literal");
                value = llvm::ConstantInt::get(type.storage(context), bits);
            } else {
                if(type.kind=='f')value=llvm::ConstantInt::get(type.storage(context),ptx_floating_bits(token,type.bits,negative));
                else {
                    uint64_t integer = token.text == "WARP_SZ" ? 32 : ptx_unsigned(token);
                    value = llvm::ConstantInt::get(type.storage(context),type.kind=='p'?uint64_t(integer!=0):negative ? uint64_t(0) - integer : integer);
                }
            }
            return type.kind == 'f' ? builder.CreateBitCast(value, type.value(context)) : invert?builder.CreateNot(value):value;
        };
        auto write = [&](const std::vector<ptx_token>& destination, llvm::Value* value, ptx_type result_type, bool relaxed = false) {
            if (destination.size() != 1) ptx_fail(destination.front(), "expected register destination");
            const auto& output = reg(destination[0]);
            if (!result_type.accepts_register(output.type)) ptx_fail(destination[0], "destination register type mismatch");
            if (value->getType()->isFloatingPointTy()) value = builder.CreateBitCast(value, llvm::Type::getIntNTy(context,value->getType()->getPrimitiveSizeInBits()));
            auto bits = value->getType()->getIntegerBitWidth();
            if ((output.type.bits != bits && (!relaxed || output.type.bits < bits)) ||
                    (output.type.bits == 1) != (bits == 1)) ptx_fail(destination[0], "destination register width mismatch");
            value = result_type.signed_integer() ? builder.CreateSExtOrTrunc(value, output.type.storage(context)) : builder.CreateZExtOrTrunc(value, output.type.storage(context));
            builder.CreateStore(value, output.slot);
        };
        auto memory_address = [&](const std::vector<ptx_token>& address, unsigned space, unsigned atomic_alignment = 0) {
            if ((address.size() != 3 && address.size() != 5) || address.front().text != "[" || address.back().text != "]")
                ptx_fail(address.front(), "unsupported address expression", true);
            int64_t displacement = 0;
            if (address.size() == 5) {
                if (address[2].text != "+" && address[2].text != "-") ptx_fail(address[2], "invalid address displacement");
                auto amount = ptx_unsigned(address[3]);
                if (amount > INT32_MAX) ptx_fail(address[3], "address displacement exceeds supported range", true);
                displacement = int64_t(amount) * (address[2].text == "-" ? -1 : 1);
            }
            if (atomic_alignment) if (auto* symbol = module->getGlobalVariable(address[1].text, true)) {
                if ((space && symbol->getAddressSpace() != space) ||
                        (symbol->getAddressSpace() != 1 && symbol->getAddressSpace() != 3))
                    ptx_fail(address[1], "atomic address space disagrees with storage");
                if (symbol->getAlign().valueOrOne().value() < atomic_alignment || displacement % atomic_alignment)
                    ptx_fail(address[1], "atomic storage is misaligned");
            }
            unsigned width = 64;
            if (space == 3 && registers.contains(address[1].text)) width = registers.at(address[1].text).type.bits;
            if (width != 32 && width != 64) ptx_fail(address[1], "invalid address register width");
            auto* bits = scalar({address[1]}, {width, 'u'});
            if (displacement) bits = builder.CreateAdd(bits, llvm::ConstantInt::get(bits->getType(), uint64_t(displacement)));
            return builder.CreateIntToPtr(bits, llvm::PointerType::get(context, space));
        };
        std::map<std::string, llvm::BasicBlock*> labels;
        for (const auto& instruction : instructions) if (instruction.label) {
            if (labels.contains(instruction.opcode.text)) ptx_fail(instruction.opcode, "duplicate label");
            labels[instruction.opcode.text] = llvm::BasicBlock::Create(context, instruction.opcode.text, &function);
        }
        for (const auto& instruction : instructions) {
            auto& opcode = instruction.opcode;
            if (instruction.label) {
                auto* block = labels.at(opcode.text);
                if (!builder.GetInsertBlock()->getTerminator()) builder.CreateBr(block);
                builder.SetInsertPoint(block); continue;
            }
            if (builder.GetInsertBlock()->getTerminator()) builder.SetInsertPoint(llvm::BasicBlock::Create(context, "unreachable", &function));
            std::vector<std::string> parts;
            llvm::SmallVector<llvm::StringRef> split; llvm::StringRef(opcode.text).split(split, '.');
            for (auto part : split) parts.push_back(part.str());
            auto arity = [&](size_t size) { if (instruction.operands.size() != size) ptx_fail(opcode, "wrong operand count"); };
            auto operand = [&](size_t index, ptx_type type) { return scalar(instruction.operands.at(index), type); };
            auto branch_target = [&] {
                arity(1); const auto& target = instruction.operands[0];
                if (target.size() != 1 || !labels.contains(target[0].text)) ptx_fail(opcode, "undefined branch label");
                return labels.at(target[0].text);
            };
            llvm::Value* predicate = nullptr;
            if (!instruction.predicate.empty()) {
                ptx_token token{instruction.predicate, opcode.line, opcode.column};
                if (reg(token).type.bits != 1) ptx_fail(token, "instruction predicate requires .pred register");
                predicate = scalar({token}, {1, 'p'});
                if (instruction.negate) predicate = builder.CreateNot(predicate);
            }
            if (parts == std::vector<std::string>{"bra"} || parts == std::vector<std::string>{"bra", "uni"}) {
                auto* target = branch_target();
                if (predicate) {
                    auto* next = llvm::BasicBlock::Create(context, "fallthrough", &function);
                    builder.CreateCondBr(predicate, target, next); builder.SetInsertPoint(next);
                } else builder.CreateBr(target);
                continue;
            }
            llvm::BasicBlock* next = nullptr;
            if (predicate) {
                auto* execute = llvm::BasicBlock::Create(context, "predicated", &function);
                next = llvm::BasicBlock::Create(context, "fallthrough", &function);
                builder.CreateCondBr(predicate, execute, next); builder.SetInsertPoint(execute);
            }
            if(parts.size()==3 && parts[0]=="mov" && (parts[1]=="v2" || parts[1]=="v4")) {
                arity(2);unsigned lanes=parts[1]=="v2"?2:4;
                auto type=ptx_scalar({parts[2],opcode.line,opcode.column});
                for(const auto& fields:instruction.operands)if(fields.size()!=lanes*2+1 || fields.front().text!="{" || fields.back().text!="}")ptx_fail(opcode,"invalid vector register list");
                for(const auto& fields:instruction.operands)for(unsigned lane=1;lane<lanes;++lane)if(fields[lane*2].text!=",")ptx_fail(opcode,"invalid vector register separator");
                std::vector<llvm::Value*> values;
                for(unsigned lane=0;lane<lanes;++lane)values.push_back(scalar({instruction.operands[1][lane*2+1]},type));
                for(unsigned lane=0;lane<lanes;++lane)write({instruction.operands[0][lane*2+1]},values[lane],type);
            } else if (parts.size() == 2 && (parts[0] == "alloca" || parts[0] == "stacksave" || parts[0] == "stackrestore")) {
                if (source_version < 73 || source_sm < 52) ptx_fail(opcode,"stack instructions require PTX 7.3 and sm_52",true);
                auto type = ptx_scalar({parts[1],opcode.line,opcode.column});
                if (type.kind != 'u' || (type.bits != 32 && type.bits != 64)) ptx_fail(opcode,"stack instructions require u32 or u64");
                if (parts[0] == "alloca") {
                    if (instruction.operands.size() != 2 && instruction.operands.size() != 3) ptx_fail(opcode,"wrong operand count");
                    uint64_t align = 8;
                    if (instruction.operands.size() == 3) {
                        if (instruction.operands[2].size() != 1) ptx_fail(opcode,"alloca alignment must be immediate");
                        align = ptx_unsigned(instruction.operands[2][0]);
                        if (!align || (align & (align-1)) || align > (1u<<23)) ptx_fail(opcode,"invalid alloca alignment");
                    }
                    // LLVM's dynamic stack allocation produces a generic
                    // pointer. PTX alloca exposes a local-space address.
                    auto* allocation = builder.CreateAlloca(builder.getInt8Ty(),operand(1,type),"ptx.dynamic");
                    allocation->setAlignment(llvm::Align(align));
                    allocation->setMetadata("cudavm.ptx.dynamic",llvm::MDNode::get(context,{}));
                    auto* local=builder.CreateAddrSpaceCast(allocation,llvm::PointerType::get(context,5));
                    write(instruction.operands[0],builder.CreatePtrToInt(local,type.storage(context)),type);
                } else if (parts[0] == "stacksave") {
                    arity(1);
                    auto* pointer = builder.CreateIntrinsic(llvm::PointerType::get(context,0),llvm::Intrinsic::stacksave,{});
                    write(instruction.operands[0],builder.CreatePtrToInt(pointer,type.storage(context)),type);
                } else {
                    arity(1); auto* pointer=builder.CreateIntToPtr(operand(0,type),llvm::PointerType::get(context,0));
                    builder.CreateIntrinsic(builder.getVoidTy(),llvm::Intrinsic::stackrestore,{pointer});
                }
            } else if (opcode.text == "ret" || opcode.text == "exit") {
                arity(0);
                if (opcode.text == "exit" && !function.hasFnAttribute("cudavm.kernel")) ptx_fail(opcode, "exit in device functions is unsupported", true);
                llvm::Value* result = nullptr;
                for (unsigned i = 0; i < body.results.size(); ++i) {
                    const auto& parameter = body.results[i];
                    auto* slot = parameter.register_parameter ? body.registers.at(parameter.name.text).slot : body.slots.at(parameter.name.text).second;
                    auto* value = builder.CreateAlignedLoad(parameter.storage(context), slot, llvm::Align(parameter.alignment));
                    result = value;
                }
                if (result) builder.CreateRet(result); else builder.CreateRetVoid();
            } else if (opcode.text == "bar.sync" || opcode.text == "bar.arrive" ||
                       opcode.text == "barrier.sync" || opcode.text == "barrier.arrive" ||
                       opcode.text == "barrier.cta.sync" || opcode.text == "barrier.cta.arrive" ||
                       opcode.text == "barrier.cta.sync.aligned" || opcode.text == "barrier.cta.arrive.aligned") {
                if(opcode.text.starts_with("barrier.cta.") && source_version<78)ptx_fail(opcode,"barrier.cta requires PTX 7.8",true);
                bool arrive = opcode.text.find("arrive") != std::string::npos;
                if (instruction.operands.empty() || instruction.operands.size() > 2 || (arrive && instruction.operands.size() != 2))
                    ptx_fail(opcode, "wrong operand count");
                std::vector<llvm::Value*> args;
                for (unsigned i = 0; i < instruction.operands.size(); ++i) {
                    auto* value = operand(i, {32, 'u'});
                    if (auto* constant = llvm::dyn_cast<llvm::ConstantInt>(value)) {
                        auto number = constant->getZExtValue();
                        if ((!i && number >= 16) || (i && (!number || number % 32))) ptx_fail(opcode, "invalid CTA barrier operand");
                    }
                    args.push_back(value);
                }
                auto aligned = opcode.text.starts_with("bar.") || opcode.text.ends_with(".aligned");
                auto name = std::string("llvm.nvvm.barrier.cta.") + (arrive?"arrive":"sync") + (aligned?".aligned.":".") + (args.size()==1?"all":"count");
                auto callee = module->getOrInsertFunction(name,
                    llvm::FunctionType::get(builder.getVoidTy(), std::vector<llvm::Type*>(args.size(), builder.getInt32Ty()), false));
                llvm::cast<llvm::Function>(callee.getCallee())->addFnAttr(llvm::Attribute::Convergent);
                builder.CreateCall(callee, args)->setConvergent();
            } else if (opcode.text.starts_with("bar.red.") || opcode.text.starts_with("barrier.red.") || opcode.text.starts_with("barrier.cta.red.")) {
                if(opcode.text.starts_with("barrier.cta.") && source_version<78)ptx_fail(opcode,"barrier.cta requires PTX 7.8",true);
                std::vector<std::string> fields=parts;
                bool aligned=fields[0]=="bar" || fields.back()=="aligned";
                if (fields.back()=="aligned") fields.pop_back();
                auto red=std::find(fields.begin(),fields.end(),"red");
                if (red==fields.end() || std::distance(red,fields.end())!=3 ||
                    (red[1]!="popc" && red[1]!="and" && red[1]!="or") || red[2]!=(red[1]=="popc"?"u32":"pred"))
                    ptx_fail(opcode,"unsupported barrier reduction",true);
                if (instruction.operands.size()!=3 && instruction.operands.size()!=4) ptx_fail(opcode,"wrong operand count");
                std::vector<llvm::Value*> args{operand(1,{32,'u'})};
                if (auto* id=llvm::dyn_cast<llvm::ConstantInt>(args[0]);id && id->getZExtValue()>=16) ptx_fail(opcode,"invalid CTA barrier operand");
                if (instruction.operands.size()==4) {
                    auto* count=operand(2,{32,'u'});
                    if (auto* n=llvm::dyn_cast<llvm::ConstantInt>(count);n && (!n->getZExtValue() || n->getZExtValue()%32)) ptx_fail(opcode,"invalid CTA barrier operand");
                    args.push_back(count);
                }
                auto predicate=instruction.operands.back();bool negate=predicate.front().text=="!";
                if(negate)predicate.erase(predicate.begin());
                auto* value=scalar(predicate,{1,'p'});args.push_back(negate?builder.CreateNot(value):value);
                std::vector<llvm::Type*> types;for(auto* arg:args)types.push_back(arg->getType());
                auto* result=red[1]=="popc"?builder.getInt32Ty():builder.getInt1Ty();
                auto callee=module->getOrInsertFunction("llvm.nvvm.barrier.cta.red."+red[1]+(aligned?".aligned.":".")+(args.size()==2?"all":"count"),llvm::FunctionType::get(result,types,false));
                auto* call=builder.CreateCall(callee,args);call->setConvergent();
                llvm::cast<llvm::Function>(callee.getCallee())->addFnAttr(llvm::Attribute::Convergent);
                write(instruction.operands[0],call,red[1]=="popc"?ptx_type{32,'u'}:ptx_type{1,'p'});
            } else if (parts[0] == "vote" || parts[0] == "shfl" || parts[0] == "match" || parts[0] == "redux" ||
                    parts[0] == "activemask" || opcode.text == "bar.warp.sync") {
                auto collective = [&](const std::string& name, llvm::Type* result, std::vector<llvm::Value*> args) -> llvm::Value* {
                    std::vector<llvm::Type*> types; for (auto* arg : args) types.push_back(arg->getType());
                    auto callee = module->getOrInsertFunction("llvm.nvvm." + name, llvm::FunctionType::get(result, types, false));
                    llvm::cast<llvm::Function>(callee.getCallee())->addFnAttr(llvm::Attribute::Convergent);
                    auto* call = builder.CreateCall(callee, args); call->setConvergent(); return call;
                };
                if (opcode.text == "bar.warp.sync") {
                    arity(1); collective("bar.warp.sync", builder.getVoidTy(), {operand(0, {32, 'u'})});
                } else if (parts == std::vector<std::string>{"activemask", "b32"}) {
                    arity(1);
                    if (source_version < 62) ptx_fail(opcode, "activemask requires PTX 6.2", true);
                    write(instruction.operands[0], collective("activemask", builder.getInt32Ty(), {}), {32, 'b'});
                } else if (parts.size() == 4 && parts[0] == "vote" && parts[1] == "sync" &&
                        ((parts[2] == "ballot" && parts[3] == "b32") ||
                         ((parts[2] == "all" || parts[2] == "any" || parts[2] == "uni") && parts[3] == "pred"))) {
                    arity(3); auto input = instruction.operands[1];
                    bool negate = input.front().text == "!"; if (negate) input.erase(input.begin());
                    auto* test = scalar(input, {1, 'p'}); if (negate) test = builder.CreateNot(test);
                    auto* mask = operand(2, {32, 'u'}); bool ballot = parts[2] == "ballot";
                    auto* result_type = ballot ? builder.getInt32Ty() : builder.getInt1Ty();
                    auto* result = collective("vote." + (parts[2] == "uni" ? std::string("all") : parts[2]) + ".sync", result_type, {mask, test});
                    if (parts[2] == "uni") result = builder.CreateOr(result,
                        collective("vote.all.sync", result_type, {mask, builder.CreateNot(test)}));
                    write(instruction.operands[0], result, ballot ? ptx_type{32, 'b'} : ptx_type{1, 'p'});
                } else if (parts.size() == 4 && parts[0] == "shfl" && parts[1] == "sync" && parts[3] == "b32" &&
                        (parts[2] == "idx" || parts[2] == "up" || parts[2] == "down" || parts[2] == "bfly")) {
                    arity(5); const auto& destination = instruction.operands[0];
                    if (destination.size() != 1 && (destination.size() != 3 || destination[1].text != "|"))
                        ptx_fail(opcode, "invalid shuffle destination");
                    auto* data = operand(1, {32, 'b'}); auto* offset = operand(2, {32, 'u'});
                    auto* control = operand(3, {32, 'u'}); auto* mask = operand(4, {32, 'u'});
                    bool predicate_result = destination.size() == 3 && destination[2].text != "_";
                    llvm::Type* result_type = predicate_result ? static_cast<llvm::Type*>(llvm::StructType::get(context, {builder.getInt32Ty(), builder.getInt1Ty()})) : builder.getInt32Ty();
                    auto* result = collective("shfl.sync." + parts[2] + (predicate_result ? ".i32p" : ".i32"), result_type, {mask, data, offset, control});
                    if (destination[0].text != "_") write({destination[0]}, predicate_result ? builder.CreateExtractValue(result, 0) : result, {32, 'b'});
                    if (predicate_result) write({destination[2]}, builder.CreateExtractValue(result, 1), {1, 'p'});
                } else if (parts.size() == 4 && parts[0] == "match" && parts[2] == "sync" &&
                        (parts[1] == "any" || parts[1] == "all") && (parts[3] == "b32" || parts[3] == "b64")) {
                    arity(3);
                    if (source_version < 60 || source_sm < 70) ptx_fail(opcode, "match.sync requires PTX 6.0 and sm_70", true);
                    bool all = parts[1] == "all";
                    const auto& destination = instruction.operands[0];
                    if (destination.size() != 1 && (!all || destination.size() != 3 || destination[1].text != "|"))
                        ptx_fail(opcode, "invalid match destination");
                    unsigned width = parts[3] == "b64" ? 64 : 32;
                    auto* data = operand(1, {width, 'b'}); auto* mask = operand(2, {32, 'u'});
                    llvm::Type* result_type = all ? static_cast<llvm::Type*>(llvm::StructType::get(context, {builder.getInt32Ty(), builder.getInt1Ty()})) : builder.getInt32Ty();
                    auto* result = collective("match." + parts[1] + ".sync.i" + std::to_string(width) + (all ? "p" : ""), result_type, {mask, data});
                    if (destination[0].text != "_") write({destination[0]}, all ? builder.CreateExtractValue(result, 0) : result, {32, 'b'});
                    if (destination.size() == 3 && destination[2].text != "_") write({destination[2]}, builder.CreateExtractValue(result, 1), {1, 'p'});
                } else if (parts.size() == 4 && parts[0] == "redux" && parts[1] == "sync") {
                    arity(3);
                    if (source_version < 70 || source_sm < 80) ptx_fail(opcode, "redux.sync requires PTX 7.0 and sm_80", true);
                    bool arithmetic = parts[2] == "add" || parts[2] == "min" || parts[2] == "max";
                    bool bitwise = parts[2] == "and" || parts[2] == "or" || parts[2] == "xor";
                    if ((!arithmetic || (parts[3] != "u32" && parts[3] != "s32")) && (!bitwise || parts[3] != "b32"))
                        ptx_fail(opcode, "unsupported warp reduction operation/type", true);
                    auto type = ptx_scalar({parts[3], opcode.line, opcode.column});
                    auto name = (parts[2] == "min" || parts[2] == "max") && parts[3] == "u32" ? "u" + parts[2] : parts[2];
                    auto* result = collective("redux.sync." + name, builder.getInt32Ty(), {operand(1, type), operand(2, {32, 'u'})});
                    if (instruction.operands[0].size() != 1 || instruction.operands[0][0].text != "_")
                        write(instruction.operands[0], result, type);
                } else ptx_fail(opcode, "unsupported warp instruction or modifier", true);
            } else if (parts[0] == "membar" || parts[0] == "fence") {
                arity(0);
                bool legacy = parts[0] == "membar";
                auto order = legacy ? llvm::AtomicOrdering::SequentiallyConsistent : llvm::AtomicOrdering::AcquireRelease;
                unsigned index = 1;
                if (!legacy && parts.size() > 2) {
                    if (parts[index] == "sc") order = llvm::AtomicOrdering::SequentiallyConsistent;
                    else if (parts[index] != "acq_rel") ptx_fail(opcode, "unsupported fence semantics", true);
                    ++index;
                }
                if (parts.size() != index + 1) ptx_fail(opcode, "unsupported fence modifier", true);
                auto scope = parts[index];
                if (scope != "cta" && scope != (legacy ? "gl" : "gpu")) ptx_fail(opcode, "unsupported fence scope", true);
                if (!legacy && source_sm < 70) ptx_fail(opcode, "fence requires sm_70", true);
                builder.CreateFence(order, context.getOrInsertSyncScopeID(scope == "cta" ? "block" : "device"));
            } else if (parts[0] == "atom" || parts[0] == "red") {
                bool reduction = parts[0] == "red";
                auto type = ptx_scalar({parts.back(), opcode.line, opcode.column});
                bool floating = type.kind == 'f' && (type.bits == 32 || type.bits == 64);
                if (floating && type.bits == 64 && source_sm < 60) ptx_fail(opcode, "FP64 atomic add requires sm_60", true);
                if (!floating && ((type.bits != 32 && type.bits != 64) || (type.kind != 'b' && type.kind != 'u' && type.kind != 's')))
                    ptx_fail(opcode, "unsupported atomic type", true);
                std::string operation, scope, semantics, space;
                for (unsigned i = 1; i + 1 < parts.size(); ++i) {
                    const auto& part = parts[i];
                    auto assign = [&](std::string& field) {
                        if (!field.empty()) ptx_fail(opcode, "duplicate atomic qualifier");
                        field = part;
                    };
                    if (part == "relaxed" || part == "acquire" || part == "release" || part == "acq_rel") assign(semantics);
                    else if (part == "cta" || part == "gpu") assign(scope);
                    else if (part == "global" || part == "shared") assign(space);
                    else if (part == "add" || part == "min" || part == "max" || part == "and" || part == "or" || part == "xor" ||
                            part == "exch" || part == "cas" || part == "inc" || part == "dec") assign(operation);
                    else ptx_fail(opcode, "unsupported atomic qualifier", true);
                }
                if (operation.empty()) ptx_fail(opcode, "missing atomic operation");
                bool bits = operation == "and" || operation == "or" || operation == "xor" || operation == "exch" || operation == "cas";
                bool wrap = operation == "inc" || operation == "dec";
                if ((floating && operation != "add") || (bits && type.kind != 'b') || (!bits && type.kind == 'b') || (wrap && (type.kind != 'u' || type.bits != 32)) ||
                        (!floating && operation == "add" && type.bits == 64 && type.kind != 'u') ||
                        (reduction && (operation == "cas" || operation == "exch")))
                    ptx_fail(opcode, "invalid atomic operation/type combination");
                if (reduction && (semantics == "acquire" || semantics == "acq_rel")) ptx_fail(opcode, "reduction cannot acquire");
                if ((!semantics.empty() && source_sm < 70) || (!scope.empty() && source_sm < 60))
                    ptx_fail(opcode, "atomic qualifier is unavailable for the source target", true);
                auto order = semantics == "acquire" ? llvm::AtomicOrdering::Acquire : semantics == "release" ? llvm::AtomicOrdering::Release :
                    semantics == "acq_rel" ? llvm::AtomicOrdering::AcquireRelease : llvm::AtomicOrdering::Monotonic;
                auto failure = llvm::isAcquireOrStronger(order) ? llvm::AtomicOrdering::Acquire : llvm::AtomicOrdering::Monotonic;
                auto sync = context.getOrInsertSyncScopeID(scope == "cta" ? "block" : "device");
                unsigned address_index = reduction ? 0 : 1;
                arity(address_index + (operation == "cas" ? 3 : 2));
                auto* pointer = memory_address(instruction.operands[address_index], space == "shared" ? 3 : space == "global" ? 1 : 0, type.bits / 8);
                auto* input = operand(address_index + 1, type);
                llvm::Value* previous = nullptr;
                if (operation == "cas") {
                    auto* exchange = builder.CreateAtomicCmpXchg(pointer, input, operand(address_index + 2, type), llvm::Align(type.bits / 8), order, failure, sync);
                    previous = builder.CreateExtractValue(exchange, 0);
                } else {
                    auto operation_id = operation == "add" ? (floating ? llvm::AtomicRMWInst::FAdd : llvm::AtomicRMWInst::Add) : operation == "and" ? llvm::AtomicRMWInst::And :
                        operation == "or" ? llvm::AtomicRMWInst::Or : operation == "xor" ? llvm::AtomicRMWInst::Xor :
                        operation == "inc" ? llvm::AtomicRMWInst::UIncWrap : operation == "dec" ? llvm::AtomicRMWInst::UDecWrap :
                        operation == "min" ? (type.signed_integer() ? llvm::AtomicRMWInst::Min : llvm::AtomicRMWInst::UMin) :
                        operation == "max" ? (type.signed_integer() ? llvm::AtomicRMWInst::Max : llvm::AtomicRMWInst::UMax) : llvm::AtomicRMWInst::Xchg;
                    previous = builder.CreateAtomicRMW(operation_id, pointer, input, llvm::Align(type.bits / 8), order, sync);
                }
                if (!reduction) {
                    const auto& destination = instruction.operands[0];
                    if (destination.size() != 1 || destination[0].text != "_") write(destination, previous, type);
                }
            } else if (opcode.text == "call" || opcode.text == "call.uni") {
                auto list = [&](const std::vector<ptx_token>& tokens) {
                    std::vector<std::vector<ptx_token>> names;
                    if (tokens.size() < 2 || tokens.front().text != "(" || tokens.back().text != ")") ptx_fail(opcode, "expected call parameter list");
                    for (size_t i = 1; i + 1 < tokens.size();) {
                        names.push_back({tokens[i++]});
                        if (names.back()[0].text == "-") {
                            if (i + 1 >= tokens.size()) ptx_fail(opcode, "empty call parameter");
                            names.back().push_back(tokens[i++]);
                        }
                        if (i + 1 < tokens.size() && tokens[i++].text != ",") ptx_fail(tokens[i - 1], "expected call parameter separator");
                    }
                    if (tokens.size() > 2 && tokens[tokens.size() - 2].text == ",") ptx_fail(opcode, "empty call parameter");
                    return names;
                };
                unsigned target_index = !instruction.operands.empty() && instruction.operands[0].front().text == "(" ? 1 : 0;
                if (instruction.operands.size() < target_index + 1 || instruction.operands.size() > target_index + 3) ptx_fail(opcode, "wrong call operand count");
                const auto& target = instruction.operands[target_index];
                if (target.size() != 1) ptx_fail(opcode, "expected device function or address register");
                bool indirect = registers.contains(target[0].text);
                auto* callee = indirect ? nullptr : module->getFunction(target[0].text);
                ptx_signature signature;
                std::vector<llvm::Metadata*> targets;
                if (indirect) {
                    if (instruction.operands.size() != target_index + 3) ptx_fail(opcode, "indirect call requires a validated call prototype");
                    const auto& label = instruction.operands.back();
                    if (label.size() != 1 || !body.prototypes.contains(label[0].text)) ptx_fail(opcode, "undefined call prototype");
                    const auto& prototype = body.prototypes.at(label[0].text);
                    signature = prototype.signature;
                    for (const auto& token : prototype.targets) {
                        auto* function = module->getFunction(token.text);
                        if (!function || (!relocatable && function->isDeclaration() && !resolved_external(*function))) ptx_fail(token, "unresolved device function");
                        if (function->hasFnAttribute("cudavm.kernel")) ptx_fail(token, "cannot call a kernel");
                        if (targets.empty()) signature = signatures.at(token.text);
                        else if (ptx_signature_key(signature) != ptx_signature_key(signatures.at(token.text))) ptx_fail(token, "call target signature mismatch");
                        targets.push_back(llvm::ValueAsMetadata::get(function));
                    }
                } else {
                    if (!callee || (!relocatable && callee->isDeclaration() && !resolved_external(*callee))) ptx_fail(target[0], "unresolved device function");
                    if (callee->hasFnAttribute("cudavm.kernel")) ptx_fail(target[0], "cannot call a kernel");
                    if (instruction.operands.size() > target_index + 2) ptx_fail(opcode, "direct call cannot have a call prototype");
                    signature = signatures.at(target[0].text);
                }
                auto inputs = instruction.operands.size() > target_index + 1 ? list(instruction.operands[target_index + 1]) : std::vector<std::vector<ptx_token>>{};
                auto outputs = target_index ? list(instruction.operands[0]) : std::vector<std::vector<ptx_token>>{};
                if (inputs.size() != signature.first.size() || outputs.size() != signature.second.size()) ptx_fail(opcode, "call parameter count mismatch");
                auto slot = [&](const std::vector<ptx_token>& operand, const ptx_parameter& expected, bool output) -> llvm::Value* {
                    if (operand.size() != 1) ptx_fail(opcode, "call requires declared .param storage");
                    const auto& token = operand[0];
                    auto found = body.slots.find(token.text);
                    if (found == body.slots.end()) ptx_fail(token, "call requires declared .param storage");
                    const auto& actual = found->second.first;
                    if (actual.bytes() != expected.bytes() || actual.alignment < expected.alignment) ptx_fail(token, "call parameter layout mismatch");
                    if (output && std::any_of(body.parameters.begin(), body.parameters.end(), [&](const auto& p) { return p.name.text == token.text; }))
                        ptx_fail(token, "cannot overwrite input parameter");
                    if (!output && std::any_of(body.results.begin(), body.results.end(), [&](const auto& p) { return p.name.text == token.text; }))
                        ptx_fail(token, "cannot read function return parameter");
                    return found->second.second;
                };
                std::vector<llvm::Value*> arguments;
                for (unsigned i = 0; i < inputs.size(); ++i) {
                    const auto& p = signature.first[i];
                    arguments.push_back(p.register_parameter ? builder.CreateBitCast(scalar(inputs[i], p.type), p.storage(context)) :
                        builder.CreateAlignedLoad(p.storage(context), slot(inputs[i], p, false), llvm::Align(p.alignment)));
                }
                auto* type = ptx_function_type(context, signature);
                llvm::Value* address = callee;
                if (indirect) address = builder.CreateIntToPtr(scalar(target, {64, 'u'}), builder.getPtrTy());
                auto* call = builder.CreateCall(type, address, arguments); call->setConvergent();
                if (indirect) {
                    call->addFnAttr(llvm::Attribute::get(context, "cudavm.ptx.signature", ptx_signature_key(signature)));
                    if (!targets.empty()) call->setMetadata("callees", llvm::MDNode::get(context, targets));
                }
                builder.CreateStore(llvm::UndefValue::get(builder.getInt1Ty()), carry);
                for (unsigned i = 0; i < outputs.size(); ++i) {
                    const auto& p = signature.second[i];
                    if (p.register_parameter) write(outputs[i], call, p.type);
                    else builder.CreateAlignedStore(call, slot(outputs[i], p, true), llvm::Align(p.alignment));
                }
            }
            else if (parts[0] == "isspacep") {
                arity(2);
                if (parts.size() != 2 || (parts[1] != "global" && parts[1] != "const" && parts[1] != "shared" && parts[1] != "local"))
                    ptx_fail(opcode, "unsupported address space query", true);
                unsigned width = 64;
                const auto& address = instruction.operands[1];
                if (address.size() == 1 && registers.contains(address[0].text)) width = reg(address[0]).type.bits;
                if (width != 32 && width != 64) ptx_fail(opcode, "address space query requires a 32- or 64-bit address");
                auto* source_address = operand(1, {width, 'u'});
                if (width == 32) ptx_fail(opcode, "32-bit generic address queries require separate lowering", true);
                auto* bits = builder.CreateZExtOrTrunc(source_address, builder.getInt64Ty());
                auto* pointer = builder.CreateIntToPtr(bits, llvm::PointerType::get(context, 0));
                auto callee = module->getOrInsertFunction("llvm.nvvm.isspacep." + parts[1],
                    llvm::FunctionType::get(builder.getInt1Ty(), {pointer->getType()}, false));
                write(instruction.operands[0], builder.CreateCall(callee, {pointer}), {1, 'p'});
            }
            else if (parts[0] == "tex" || parts[0] == "suld" || parts[0] == "sust") image_instruction(parts, instruction, builder, scalar, write);
            else if (integer_instruction(parts, instruction, builder, scalar, write, carry)) {}
            else if (parts.size() >= 2) {
                const auto& op = parts[0];
                auto type = ptx_scalar({parts.back(), opcode.line, opcode.column}, op == "ld" || op == "st");
                llvm::Value* value = nullptr;
                if (type.kind == 'p') {
                    if (parts.size() != 2 || (op != "mov" && op != "and" && op != "or" && op != "xor" && op != "not"))
                        ptx_fail(opcode, "unsupported predicate instruction", true);
                    arity(op == "not" || op == "mov" ? 2 : 3); auto* left = operand(1, type);
                    value = op == "mov" ? left : op == "not" ? builder.CreateNot(left) : op == "and" ? builder.CreateAnd(left, operand(2, type)) :
                        op == "or" ? builder.CreateOr(left, operand(2, type)) : builder.CreateXor(left, operand(2, type));
                    write(instruction.operands[0], value, type);
                }
                else if (numeric_instruction(parts,instruction,builder,scalar,write)) {}
                else if (op == "ld" || op == "st") {
                    arity(2);
                    bool load = op == "ld", parameter = parts.size() == 3 && parts[1] == "param";
                    bool global = (parts.size() == 3 && parts[1] == "global") ||
                        (load && parts.size() == 4 && parts[1] == "global" && parts[2] == "nc");
                    bool shared = parts.size() == 3 && parts[1] == "shared";
                    bool local = parts.size() == 3 && parts[1] == "local";
                    bool constant = parts.size() == 3 && parts[1] == "const";
                    bool generic = parts.size() == 2;
                    llvm::AtomicOrdering order = llvm::AtomicOrdering::NotAtomic;
                    llvm::SyncScope::ID sync = llvm::SyncScope::System;
                    if (parts.size() >= 4 && (parts[1] == "relaxed" || parts[1] == "acquire" || parts[1] == "release")) {
                        if ((load && parts[1] == "release") || (!load && parts[1] == "acquire")) ptx_fail(opcode, "invalid load/store memory ordering");
                        if (parts[2] != "cta" && parts[2] != "gpu") ptx_fail(opcode, "unsupported memory scope", true);
                        if (source_sm < 70) ptx_fail(opcode, "ordered memory access requires sm_70", true);
                        if ((type.bits != 32 && type.bits != 64) || type.kind == 'f') ptx_fail(opcode, "unsupported ordered memory type", true);
                        order = parts[1] == "acquire" ? llvm::AtomicOrdering::Acquire : parts[1] == "release" ? llvm::AtomicOrdering::Release : llvm::AtomicOrdering::Monotonic;
                        sync = context.getOrInsertSyncScopeID(parts[2] == "cta" ? "block" : "device");
                        generic = parts.size() == 4;
                        global = parts.size() == 5 && parts[3] == "global";
                        shared = parts.size() == 5 && parts[3] == "shared";
                    }
                    if ((!parameter && !global && !shared && !local && !constant && !generic) || (!load && constant)) ptx_fail(opcode, "unsupported memory instruction", true);
                    const auto& address = instruction.operands[load ? 1 : 0];
                    if (address.size() < 3 || address.front().text != "[" || address.back().text != "]" ||
                        (address.size() != 3 && address.size() != 5)) ptx_fail(opcode, "unsupported address expression", true);
                    int64_t displacement = 0;
                    if (address.size() == 5) {
                        if (address[2].text != "+" && address[2].text != "-") ptx_fail(address[2], "invalid address displacement");
                        auto amount = ptx_unsigned(address[3]);
                        if (amount > INT32_MAX) ptx_fail(address[3], "address displacement exceeds supported range", true);
                        displacement = int64_t(amount) * (address[2].text == "-" ? -1 : 1);
                    }
                    llvm::Value* pointer = nullptr;
                    if (parameter) {
                        auto found = body.slots.find(address[1].text);
                        if (found == body.slots.end()) ptx_fail(address[1], "unknown function parameter");
                        if (displacement < 0 || uint64_t(displacement) + type.bits / 8 > found->second.first.bytes() || displacement % (type.bits / 8) || found->second.first.alignment < type.bits / 8)
                            ptx_fail(address[1], "parameter load is out of bounds or misaligned");
                        if (!load && std::any_of(body.parameters.begin(), body.parameters.end(), [&](const auto& p) { return p.name.text == address[1].text; }))
                            ptx_fail(address[1], "cannot overwrite input parameter");
                        if (load && std::any_of(body.results.begin(), body.results.end(), [&](const auto& p) { return p.name.text == address[1].text; }))
                            ptx_fail(address[1], "cannot read function return parameter");
                        pointer = builder.CreateGEP(builder.getInt8Ty(), found->second.second, builder.getInt64(uint64_t(displacement)));
                    } else {
                        unsigned width = 64;
                        if ((shared || local) && registers.contains(address[1].text)) width = registers.at(address[1].text).type.bits;
                        if (width != 32 && width != 64) ptx_fail(address[1], "invalid address register width");
                        auto* pointer_bits = scalar({address[1]}, {width, 'u'});
                        if (displacement) pointer_bits = builder.CreateAdd(pointer_bits, llvm::ConstantInt::get(pointer_bits->getType(), uint64_t(displacement)));
                        pointer = builder.CreateIntToPtr(pointer_bits, llvm::PointerType::get(context, shared ? 3 : local ? 5 : constant ? 4 : global ? 1 : 0));
                    }
                    if (load) {
                        auto* memory = builder.CreateAlignedLoad(type.storage(context), pointer, llvm::Align(type.bits / 8));
                        if (order != llvm::AtomicOrdering::NotAtomic) memory->setAtomic(order, sync);
                        value = memory;
                    }
                    else {
                        auto* data = scalar(instruction.operands[1], type, true);
                        if (type.kind == 'f') data = builder.CreateBitCast(data, type.storage(context));
                        auto* memory = builder.CreateAlignedStore(data, pointer, llvm::Align(type.bits / 8));
                        if (order != llvm::AtomicOrdering::NotAtomic) memory->setAtomic(order, sync);
                    }
                    if (load) write(instruction.operands[0], value, type, true);
                } else if (op == "cvta" && (type.bits == 32 || type.bits == 64) && type.kind == 'u' &&
                        (parts.size() == 3 || (parts.size() == 4 && parts[1] == "to"))) {
                    arity(2); bool to = parts.size() == 4; auto space = parts[to ? 2 : 1];
                    if (space != "global" && space != "shared" && space != "const" && space != "local") ptx_fail(opcode, "unsupported address space conversion", true);
                    auto* specific = llvm::PointerType::get(context, space == "shared" ? 3 : space == "const" ? 4 : space == "local" ? 5 : 1);
                    auto* generic = llvm::PointerType::get(context, 0);
                    auto address = instruction.operands[1]; llvm::Value* bits = nullptr;
                    if (!to && address.size() == 3 && (body.locals.contains(address[0].text) || module->getGlobalVariable(address[0].text, true)) &&
                            (address[1].text == "+" || address[1].text == "-")) {
                        auto displacement = ptx_unsigned(address[2]);
                        if (displacement > INT32_MAX) ptx_fail(address[2], "address displacement exceeds supported range", true);
                        bits = builder.CreateAdd(scalar({address[0]}, type), llvm::ConstantInt::get(type.storage(context),
                            address[1].text == "-" ? uint64_t(0) - displacement : displacement));
                    } else bits = operand(1, type);
                    auto* pointer = builder.CreateIntToPtr(bits, to ? generic : specific);
                    value = builder.CreatePtrToInt(builder.CreateAddrSpaceCast(pointer, to ? specific : generic), type.storage(context));
                    write(instruction.operands[0], value, type);
                } else if (op == "cvt" && parts.size() == 3 && type.kind != 'f' && type.kind != 'b' && type.bits >= 16) {
                    arity(2); auto destination = ptx_scalar({parts[1], opcode.line, opcode.column});
                    if (destination.kind != 's' && destination.kind != 'u') ptx_fail(opcode, "unsupported conversion", true);
                    auto* input = scalar(instruction.operands[1], type, true);
                    value = type.signed_integer() ? builder.CreateSExtOrTrunc(input, destination.storage(context)) : builder.CreateZExtOrTrunc(input, destination.storage(context));
                    write(instruction.operands[0], value, destination, true);
                } else if (op == "setp" && parts.size() == 3 && (type.kind == 's' || type.kind == 'u'
                        || (type.kind == 'b' && (parts[1] == "eq" || parts[1] == "ne"))) && type.bits >= 16) {
                    arity(3); auto* left = operand(1, type); auto* right = operand(2, type);
                    const std::map<std::string, llvm::CmpInst::Predicate> comparisons{
                        {"eq", llvm::CmpInst::ICMP_EQ}, {"ne", llvm::CmpInst::ICMP_NE},
                        {"lt", type.signed_integer() ? llvm::CmpInst::ICMP_SLT : llvm::CmpInst::ICMP_ULT},
                        {"le", type.signed_integer() ? llvm::CmpInst::ICMP_SLE : llvm::CmpInst::ICMP_ULE},
                        {"gt", type.signed_integer() ? llvm::CmpInst::ICMP_SGT : llvm::CmpInst::ICMP_UGT},
                        {"ge", type.signed_integer() ? llvm::CmpInst::ICMP_SGE : llvm::CmpInst::ICMP_UGE}};
                    if (!comparisons.contains(parts[1])) ptx_fail(opcode, "unsupported comparison", true);
                    write(instruction.operands[0], builder.CreateICmp(comparisons.at(parts[1]), left, right), {1, 'p'});
                } else if (op == "mov" && parts.size() == 2) {
                    arity(2); write(instruction.operands[0], operand(1, type), type);
                } else if (op == "selp" && parts.size() == 2) {
                    arity(4); write(instruction.operands[0], builder.CreateSelect(operand(3, {1, 'p'}), operand(1, type), operand(2, type)), type);
                } else if ((op == "min" || op == "max" || op == "div" || op == "rem") && parts.size() == 2 &&
                        (type.kind == 's' || type.kind == 'u') && type.bits >= 16) {
                    arity(3); auto* left = operand(1, type); auto* right = operand(2, type);
                    if (op == "min" || op == "max") {
                        auto predicate = type.signed_integer() ? (op == "min" ? llvm::CmpInst::ICMP_SLT : llvm::CmpInst::ICMP_SGT) :
                            (op == "min" ? llvm::CmpInst::ICMP_ULT : llvm::CmpInst::ICMP_UGT);
                        value = builder.CreateSelect(builder.CreateICmp(predicate, left, right), left, right);
                    } else {
                        // PTX zero-divisor results are unspecified. Keep those
                        // values local instead of introducing LLVM poison; also
                        // guard the signed minimum/-1 overflow case.
                        auto* zero = llvm::ConstantInt::get(type.storage(context), 0);
                        auto* invalid = builder.CreateICmpEQ(right, zero);
                        if (type.signed_integer()) invalid = builder.CreateOr(invalid, builder.CreateAnd(
                            builder.CreateICmpEQ(left, llvm::ConstantInt::get(context, llvm::APInt::getSignedMinValue(type.bits))),
                            builder.CreateICmpEQ(right, llvm::ConstantInt::getSigned(type.storage(context), -1))));
                        auto* safe = builder.CreateSelect(invalid, llvm::ConstantInt::get(type.storage(context), 1), right);
                        value = type.signed_integer() ? (op == "div" ? builder.CreateSDiv(left, safe) : builder.CreateSRem(left, safe)) :
                            (op == "div" ? builder.CreateUDiv(left, safe) : builder.CreateURem(left, safe));
                    }
                    write(instruction.operands[0], value, type);
                } else if ((op == "mul" || op == "mad") && parts.size() == 3 && (type.kind == 's' || type.kind == 'u') &&
                        (parts[1] == "lo" || (parts[1] == "hi" && type.bits >= 16) || (parts[1] == "wide" && (type.bits == 16 || type.bits == 32)))) {
                    arity(op == "mad" ? 4 : 3); auto* left = operand(1, type); auto* right = operand(2, type);
                    auto result_type = type;
                    if (parts[1] == "hi") value = high_product(builder, left, right, type);
                    else {
                        if (parts[1] == "wide") {
                            result_type.bits *= 2;
                            left = type.signed_integer() ? builder.CreateSExt(left, result_type.storage(context)) : builder.CreateZExt(left, result_type.storage(context));
                            right = type.signed_integer() ? builder.CreateSExt(right, result_type.storage(context)) : builder.CreateZExt(right, result_type.storage(context));
                        }
                        value = builder.CreateMul(left, right);
                    }
                    if (op == "mad") value = builder.CreateAdd(value, operand(3, result_type));
                    write(instruction.operands[0], value, result_type);
                } else if ((op == "add" || op == "sub" || op == "mul") && type.kind == 'f' &&
                        (parts.size() == 2 || (parts.size() == 3 && parts[1] == "rn"))) {
                    arity(3); auto* left = operand(1, type); auto* right = operand(2, type);
                    value = op == "add" ? builder.CreateFAdd(left, right) : op == "sub" ? builder.CreateFSub(left, right) : builder.CreateFMul(left, right);
                    write(instruction.operands[0], value, type);
                } else if (parts.size() == 2 && type.kind != 'f' && type.bits >= 16 &&
                        ((op == "add" || op == "sub" || op == "neg") ? (type.kind == 'u' || type.kind == 's') :
                        ((op == "and" || op == "or" || op == "xor" || op == "not" || op == "shl") ? type.kind == 'b' :
                        (op == "shr" && (type.kind == 'b' || type.kind == 'u' || type.kind == 's'))))) {
                    bool unary = op == "neg" || op == "not"; arity(unary ? 2 : 3);
                    auto* left = operand(1, type);
                    if (unary) value = op == "neg" ? builder.CreateNeg(left) : builder.CreateNot(left);
                    else if (op == "shl" || op == "shr") {
                        auto* shift = operand(2, {32, 'u'});
                        auto* excessive = builder.CreateICmpUGE(shift, builder.getInt32(type.bits));
                        // PTX clamps shifts; LLVM overshifts would introduce poison.
                        auto* safe = builder.CreateZExtOrTrunc(builder.CreateSelect(excessive, builder.getInt32(type.bits - 1), shift), type.storage(context));
                        value = op == "shl" ? builder.CreateShl(left, safe) : type.signed_integer() ? builder.CreateAShr(left, safe) : builder.CreateLShr(left, safe);
                        if (!type.signed_integer() || op == "shl") value = builder.CreateSelect(excessive, llvm::ConstantInt::get(type.storage(context), 0), value);
                    } else {
                        auto* right = operand(2, type);
                        if (op == "add") value = builder.CreateAdd(left, right);
                        else if (op == "sub") value = builder.CreateSub(left, right);
                        else if (op == "and") value = builder.CreateAnd(left, right);
                        else if (op == "or") value = builder.CreateOr(left, right);
                        else value = builder.CreateXor(left, right);
                    }
                    write(instruction.operands[0], value, type);
                } else ptx_fail(opcode, "unsupported instruction or modifier", true);
            } else ptx_fail(opcode, "unsupported instruction", true);
            if (next) {
                if (!builder.GetInsertBlock()->getTerminator()) builder.CreateBr(next);
                builder.SetInsertPoint(next);
            }
        }
        if (!builder.GetInsertBlock()->getTerminator()) ptx_fail(tokens[cursor - 1], "kernel control flow falls through its closing brace");
        llvm::FunctionAnalysisManager analyses; llvm::PassBuilder passes; passes.registerFunctionAnalyses(analyses);
        llvm::SROAPass(llvm::SROAOptions::ModifyCFG).run(function, analyses);
        llvm::PromotePass().run(function, analyses);
        // Promotion exposes addresses moved into PTX registers. Reject a known
        // incompatible target before the optimizer can devirtualize its call.
        for (auto& block : function) for (auto& instruction : block) {
            auto* call = llvm::dyn_cast<llvm::CallInst>(&instruction);
            if (!call || !call->hasFnAttr("cudavm.ptx.signature")) continue;
            auto* address = call->getCalledOperand()->stripPointerCasts();
            if (auto* cast = llvm::dyn_cast<llvm::Operator>(address); cast && cast->getOpcode() == llvm::Instruction::IntToPtr)
                if (auto* bits = llvm::dyn_cast<llvm::PtrToIntOperator>(cast->getOperand(0)); bits && bits->getType()->isIntegerTy(64))
                    address = bits->getOperand(0)->stripPointerCasts();
            if (auto* target = llvm::dyn_cast<llvm::Function>(address)) {
                auto signature = target->getFnAttribute("cudavm.ptx.signature");
                if (!signature.isValid() || signature.getValueAsString() != call->getFnAttr("cudavm.ptx.signature").getValueAsString())
                    ptx_fail(body.name, "call target signature mismatch");
                if (auto* targets = call->getMetadata("callees"); targets && !llvm::any_of(targets->operands(), [&](const auto& operand) {
                    return llvm::cast<llvm::ValueAsMetadata>(operand.get())->getValue() == target;
                })) ptx_fail(body.name, "call target is absent from target list");
            }
        }
    }
public:
    ptx_parser(llvm::LLVMContext& owner, std::string_view input) : tokens(ptx_tokens(input)), context(owner), module(std::make_unique<llvm::Module>("imported.ptx", owner)) {}
    std::unique_ptr<llvm::Module> parse(std::string_view architecture = {}, bool retain_declarations = false) {
        relocatable = retain_declarations;
        expect(".version"); auto version = take();
        // Bound accepted language versions independently of opcode support.
        // Later versions remain explicit obligations even for old opcodes.
        if (!supported_ptx_version(version.text))
            ptx_fail(version, "unsupported PTX version (expected 6.0-6.5, 7.0-7.8 or 8.0-8.4)", true, true);
        source_version = unsigned((version.text[0] - '0') * 10 + version.text[2] - '0');
        expect(".target"); auto target = take();
        const std::map<std::string, unsigned> targets{{"sm_50", 60}, {"sm_52", 60}, {"sm_53", 60}, {"sm_60", 60},
            {"sm_61", 60}, {"sm_62", 60}, {"sm_70", 60}, {"sm_72", 61}, {"sm_75", 63},
            {"sm_80", 70}, {"sm_86", 71}, {"sm_87", 74}, {"sm_89", 78}, {"sm_90", 78}};
        if (!targets.contains(target.text)) ptx_fail(target, "unsupported PTX target", true);
        source_sm = unsigned(std::stoul(target.text.substr(3)));
        if (unsigned((version.text[0] - '0') * 10 + version.text[2] - '0') < targets.at(target.text))
            ptx_fail(target, "target is unavailable in this PTX version");
        auto selected = architecture.empty() ? target.text : std::string(architecture);
        if (!targets.contains(selected)) ptx_fail(target, "unsupported PTX compilation target", true);
        if (std::stoul(selected.substr(3)) < std::stoul(target.text.substr(3)))
            ptx_fail(target, "PTX compilation target is older than the input target", false, false, true);
        if (peek().text == ",") ptx_fail(peek(), "unsupported target option", true);
        expect(".address_size"); auto address = take();
        if (address.text != "64") ptx_fail(address, "PTX import requires 64-bit addresses", true);
        module->setTargetTriple(llvm::Triple("nvptx64-nvidia-cuda"));
        module->getOrInsertNamedMetadata("cudavm.import.architecture")->addOperand(
            llvm::MDNode::get(context, llvm::MDString::get(context, selected)));
        module->getOrInsertNamedMetadata("cudavm.import.source.architecture")->addOperand(
            llvm::MDNode::get(context, llvm::MDString::get(context, target.text)));
        module->setDataLayout("e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64");
        module->getOrInsertNamedMetadata("cudavm.ir.stage")->addOperand(llvm::MDNode::get(context, llvm::MDString::get(context, "cudavm.portable")));
        while (peek().text != "<eof>") {
            if(debug_directive() || pragma())continue;
            if(accept(".section")) {
                auto section=take(); if(!section.text.starts_with(".debug_"))ptx_fail(section,"unsupported section",true);
                expect("{");unsigned depth=1;
                while(depth) {auto token=take();if(token.text=="<eof>")ptx_fail(token,"unterminated debug section");if(token.text=="{")++depth;if(token.text=="}")--depth;}
                continue;
            }
            if(accept(".alias")) {
                auto alias=identifier();expect(",");auto target=identifier();expect(";");aliases.emplace_back(alias,target);continue;
            }
            bool visible = accept(".visible"), weak = !visible && accept(".weak");
            bool common = !visible && !weak && accept(".common");
            bool external = !visible && !weak && !common && accept(".extern");
            auto linkage = common?llvm::GlobalValue::CommonLinkage:weak ? llvm::GlobalValue::WeakAnyLinkage : visible || external || !relocatable
                ? llvm::GlobalValue::ExternalLinkage : llvm::GlobalValue::InternalLinkage;
            if(common && peek().text!=".global")ptx_fail(peek(),"common linkage requires global storage");
            if (accept(".entry")) callable(true, external, weak ? llvm::GlobalValue::WeakAnyLinkage : llvm::GlobalValue::ExternalLinkage);
            else if (accept(".func")) callable(false, external, linkage);
            else if (peek().text == ".global" || peek().text == ".const" || peek().text == ".shared") {
                auto space = take();
                if (weak && space.text == ".shared") ptx_fail(space, "weak shared storage is unsupported", true);
                storage(external, space.text, nullptr, nullptr, linkage);
            } else ptx_fail(peek(), "unsupported module declaration or directive", true);
        }
        bool has_kernel = false;
        for (const auto& body : bodies) has_kernel |= body.function->hasFnAttribute("cudavm.kernel");
        if (!has_kernel && !relocatable) ptx_fail(peek(), "PTX module contains no kernels");
        for (auto& global : globals) if (!global.initializer.empty()) {
            const auto& data = global.declaration;
            std::vector<llvm::Constant*> elements(data.count, llvm::ConstantInt::get(data.type.storage(context), 0));
            for (unsigned i = 0; i < global.initializer.size(); ++i) {
                auto tokens = global.initializer[i];
                if (tokens.empty()) ptx_fail(data.name, "empty initializer element");
                if (tokens[0].text == "generic" && tokens.size() >= 4 && tokens[1].text == "(" && tokens[3].text == ")") {
                    tokens.erase(tokens.begin(), tokens.begin() + 2); tokens.erase(tokens.begin() + 1);
                }
                auto* target = module->getNamedValue(tokens[0].text);
                if (target) {
                    auto* function = llvm::dyn_cast<llvm::Function>(target);
                    if (data.type.bits != 64 || data.type.kind == 'f' || (function ? function->hasFnAttribute("cudavm.kernel") :
                        target->getAddressSpace() != 1 && target->getAddressSpace() != 4)) ptx_fail(tokens[0], "symbol initializer requires a 64-bit global or device-function address", true);
                    if (function && !relocatable && function->isDeclaration() && !resolved_external(*function)) ptx_fail(tokens[0], "unresolved device function");
                    llvm::Constant* address = target;
                    if (tokens.size() == 3 && (tokens[1].text == "+" || tokens[1].text == "-")) {
                        if (function) ptx_fail(tokens[0], "function address cannot have a displacement");
                        auto offset = ptx_unsigned(tokens[2]);
                        if (tokens[1].text == "-" || offset > module->getDataLayout().getTypeAllocSize(llvm::cast<llvm::GlobalVariable>(target)->getValueType()).getFixedValue()) ptx_fail(tokens[0], "initializer relocation is out of bounds");
                        address = llvm::ConstantExpr::getGetElementPtr(llvm::Type::getInt8Ty(context), target, llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), offset));
                    } else if (tokens.size() != 1) ptx_fail(tokens[0], "unsupported initializer expression", true);
                    elements[i] = llvm::ConstantExpr::getPtrToInt(address, data.type.storage(context));
                } else {
                    bool negative = tokens.size() == 2 && tokens[0].text == "-";
                    if (tokens.size() != (negative ? 2u : 1u)) ptx_fail(tokens[0], "unsupported initializer expression", true);
                    const auto& literal = tokens.back(); uint64_t bits;
                    if (((literal.text.starts_with("0f") && literal.text.size() == 10 && data.type.bits == 32) ||
                         (literal.text.starts_with("0d") && literal.text.size() == 18 && data.type.bits == 64)) && !negative) {
                        if (llvm::StringRef(literal.text).drop_front(2).getAsInteger(16, bits)) ptx_fail(literal, "invalid binary32 literal");
                    } else {
                        if(data.type.kind=='f')bits=ptx_floating_bits(literal,data.type.bits,negative);
                        else {bits = ptx_unsigned(literal); if (negative) bits = uint64_t(0) - bits;}
                    }
                    elements[i] = llvm::ConstantInt::get(data.type.storage(context), bits);
                }
            }
            global.variable->setInitializer(data.array ? llvm::ConstantArray::get(llvm::cast<llvm::ArrayType>(data.storage(context)), elements) : elements[0]);
        }
        for (auto& body : bodies) lower(body);
        for(const auto& [name,target]:aliases) {
            auto* alias=module->getFunction(name.text);auto* aliasee=module->getFunction(target.text);
            if(!alias || !alias->isDeclaration() || !aliasee || aliasee->isDeclaration() || alias->hasFnAttribute("cudavm.kernel") || aliasee->hasFnAttribute("cudavm.kernel") ||
                alias->getFunctionType()!=aliasee->getFunctionType() ||
                ptx_signature_key(signatures.at(name.text))!=ptx_signature_key(signatures.at(target.text)))ptx_fail(name,"invalid function alias ABI");
            auto* value=llvm::GlobalAlias::create(alias->getValueType(),0,declared_linkages.at(name.text),name.text+"$alias",aliasee,module.get());
            value->takeName(alias);alias->replaceAllUsesWith(value);alias->eraseFromParent();
        }
        normalize_external_abi();
        for (auto& global : module->globals())
            if (!relocatable && global.isDeclaration() && global.getAddressSpace() != 3 && !global.use_empty()) ptx_fail(symbols.at(global.getName().str()), "unresolved device storage");
        // PTX enters the same portable CUDA semantic identities as source IR.
        // Keep native intrinsics out of persisted programs and native emission.
        import_device_semantics(*module);
        std::string diagnostic; llvm::raw_string_ostream stream(diagnostic);
        if (llvm::verifyModule(*module, &stream)) throw ptx_import_error(false, "invalid imported PTX IR: " + diagnostic);
        return std::move(module);
    }
};
}
std::unique_ptr<llvm::Module> import_ptx(llvm::LLVMContext& context, std::string_view input, std::string_view architecture, bool relocatable) {
    return ptx_parser(context, input).parse(architecture, relocatable);
}
llvm::Value* import_inline_ptx(llvm::CallInst& call) {
    auto* assembly=llvm::cast<llvm::InlineAsm>(call.getCalledOperand());
    auto fail=[](const std::string& message) { throw std::runtime_error("inline PTX: "+message); };
    if(assembly->isAlignStack() || assembly->getDialect()!=llvm::InlineAsm::AD_ATT)
        fail("unsupported assembly dialect or stack alignment");
    auto constraints=assembly->ParseConstraints();
    if(constraints.empty() && !assembly->getConstraintString().empty())fail("invalid constraints");
    std::vector<llvm::Type*> outputs;
    if(auto* structure=llvm::dyn_cast<llvm::StructType>(call.getType()))
        outputs.assign(structure->element_begin(),structure->element_end());
    else if(!call.getType()->isVoidTy())outputs.push_back(call.getType());
    struct operand { llvm::Type* type; llvm::Value* input; std::string text; unsigned bits; };
    std::vector<operand> operands;
    std::string register_prefix="%__cudavm_operand";
    while(assembly->getAsmString().contains(register_prefix))register_prefix+='_';
    bool memory=false,inputs=false,clobbers=false;
    unsigned argument=0,output=0;
    for(const auto& constraint:constraints) {
        if(constraint.isIndirect || constraint.isMultipleAlternative || constraint.isCommutative || constraint.Codes.size()!=1)
            fail("unsupported indirect, alternative or commutative constraint");
        const auto& code=constraint.Codes[0];
        if(constraint.Type==llvm::InlineAsm::isClobber) {
            if(code!="{memory}")fail("unsupported clobber "+code);
            memory=true;clobbers=true;continue;
        }
        if(clobbers)fail("operand follows clobber");
        llvm::Type* type=nullptr;llvm::Value* input=nullptr;
        if(constraint.Type==llvm::InlineAsm::isOutput) {
            if(inputs || output>=outputs.size())fail("output constraint does not match result type");
            type=outputs[output++];
        } else if(constraint.Type==llvm::InlineAsm::isInput) {
            inputs=true;
            if(argument>=call.arg_size())fail("missing input operand");
            input=call.getArgOperand(argument++);type=input->getType();
        } else fail("unsupported label constraint");
        if(!type->isPointerTy() && !type->isIntegerTy() && !type->isFloatingPointTy())fail("register operand must be scalar");
        unsigned bits=type->isPointerTy()?64:unsigned(type->getPrimitiveSizeInBits());
        unsigned match=0;
        if(constraint.Type==llvm::InlineAsm::isInput && !llvm::StringRef(code).getAsInteger(10,match)) {
            if(match>=outputs.size() || match>=operands.size() || operands[match].type!=type || operands[match].input)
                fail("invalid tied input constraint "+code);
            operands[match].input=input;
            operands.push_back({type,nullptr,operands[match].text,bits});continue;
        }
        if(code=="n") {
            auto* immediate=llvm::dyn_cast_or_null<llvm::ConstantInt>(input);
            if(!immediate || immediate->getBitWidth()>64)fail("immediate constraint requires a constant integer of at most 64 bits");
            operands.push_back({type,nullptr,std::to_string(immediate->getSExtValue()),bits});continue;
        }
        bool valid=(code=="h" && type->isIntegerTy(16)) || (code=="r" && type->isIntegerTy(32)) ||
            (code=="l" && (type->isIntegerTy(64) || (type->isPointerTy() && type->getPointerAddressSpace()==0))) ||
            (code=="f" && type->isFloatTy()) || (code=="d" && type->isDoubleTy());
        if(!valid)fail("register constraint "+code+" does not match operand type");
        operands.push_back({type,input,register_prefix+std::to_string(operands.size()),bits});
    }
    if(output!=outputs.size() || argument!=call.arg_size())fail("constraint count does not match signature");
    auto& module=*call.getModule();unsigned ordinal=0;std::string name;
    do {name="__cudavm_inline_ptx_"+std::to_string(ordinal++);}while(module.getNamedValue(name));
    std::string architecture;
    if(auto* abi=module.getNamedMetadata("cudavm.source.abi");abi && abi->getNumOperands()) {
        auto* row=abi->getOperand(0);
        if(row->getNumOperands()>2)if(auto* value=llvm::dyn_cast<llvm::MDString>(row->getOperand(2)))architecture=value->getString().str();
    }
    if(architecture.empty())if(auto* abi=module.getNamedMetadata("cudavm.import.architecture");abi && abi->getNumOperands())
        if(auto* value=llvm::dyn_cast<llvm::MDString>(abi->getOperand(0)->getOperand(0)))architecture=value->getString().str();
    if(architecture.empty())fail("missing source architecture");
    std::string body;auto text=assembly->getAsmString();
    for(size_t i=0;i<text.size();) {
        if(text[i]=='$' && i+1<text.size() && text[i+1]=='$'){body+='$';i+=2;}
        else if(text[i]=='$' && i+1<text.size() && std::isdigit(static_cast<unsigned char>(text[i+1]))) {
            size_t end=i+1;while(end<text.size() && std::isdigit(static_cast<unsigned char>(text[end])))++end;
            unsigned index=0;
            if(text.slice(i+1,end).getAsInteger(10,index) || index>=operands.size())fail("invalid operand reference");
            body+=operands[index].text;i=end;
        } else if(text[i]=='$' && i+1<text.size() && text[i+1]=='{')fail("unsupported operand modifier");
        else if(text.substr(i,5)==".reg."){body+=".reg .";i+=5;}
        else body+=text[i++];
    }
    // The block may branch only within its own scope. Returning from a temporary
    // wrapper would silently change a kernel return or skip result packaging.
    int depth=0;
    for(const auto& token:ptx_tokens(body)) {
        if(token.text=="{")++depth;
        if(token.text=="}" && --depth<0)fail("unbalanced assembly block");
        if(token.text=="ret" || token.text=="exit" || token.text=="call" || token.text.starts_with("call.") ||
            token.text==".entry" || token.text==".func" || token.text==".extern")fail("unsupported block control transfer or external declaration");
    }
    if(depth)fail("unbalanced assembly block");
    auto spelling=[](const operand& operand){return ".b"+std::to_string(operand.bits);};
    std::string source=".version 8.4\n.target "+architecture+"\n.address_size 64\n.func ";
    if(!outputs.empty())source+="(.param .b64 result["+std::to_string(outputs.size())+"]) ";
    source+=name+"(";
    std::vector<llvm::Value*> arguments;
    for(unsigned i=0;i<operands.size();++i)if(operands[i].input) {
        source+=(arguments.empty()?"":", ")+std::string(".param ")+spelling(operands[i])+" arg"+std::to_string(i);
        arguments.push_back(operands[i].input);
    }
    source+=") {\n";
    for(unsigned i=0;i<operands.size();++i) {
        const auto& operand=operands[i];
        if(operand.text!=register_prefix+std::to_string(i))continue;
        source+=".reg "+spelling(operand)+" "+operand.text+";\n";
        if(operand.input)source+="ld.param"+spelling(operand)+" "+operand.text+", [arg"+std::to_string(i)+"];\n";
    }
    source+="{\n"+body+"\n}\n";
    if(!outputs.empty())source+=".reg .b64 __cudavm_result;\n";
    for(unsigned i=0;i<outputs.size();++i) {
        source+=(operands[i].bits==64?"mov.b64 ":"cvt.u64.u"+std::to_string(operands[i].bits)+" ")+std::string("__cudavm_result, ")+operands[i].text+";\n";
        source+="st.param.b64 [result+"+std::to_string(i*8)+"], __cudavm_result;\n";
    }
    source+="ret; }\n";
    auto imported=import_ptx(module.getContext(),source,{},true);
    auto* original=imported->getFunction(name);original->setLinkage(llvm::GlobalValue::ExternalLinkage);
    bool convergent=call.isConvergent();
    for(const auto& function:*imported)for(const auto& block:function)for(const auto& instruction:block)
        if(auto* operation=llvm::dyn_cast<llvm::CallBase>(&instruction))convergent|=operation->isConvergent();
    // A volatile block's explicit memory operations must survive optimization,
    // including when the C++ caller discards its register results.
    if(assembly->hasSideEffects())for(auto& block:*original)for(auto& instruction:block) {
        if(auto* load=llvm::dyn_cast<llvm::LoadInst>(&instruction))load->setVolatile(true);
        if(auto* store=llvm::dyn_cast<llvm::StoreInst>(&instruction))store->setVolatile(true);
        if(auto* atomic=llvm::dyn_cast<llvm::AtomicRMWInst>(&instruction))atomic->setVolatile(true);
        if(auto* atomic=llvm::dyn_cast<llvm::AtomicCmpXchgInst>(&instruction))atomic->setVolatile(true);
    }
    // Resolve declarations explicitly, including internal numeric definitions.
    // LLVM module linking treats internal names as separate identities and can
    // otherwise rename the existing helper away from its canonical spelling.
    // Cloning also leaves source ABI/architecture metadata unchanged.
    llvm::ValueToValueMapTy values;
    for(auto& function:*imported)if(function.isDeclaration()) {
        auto* existing=module.getFunction(function.getName());
        if(existing && existing->getFunctionType()!=function.getFunctionType())fail("conflicting helper signature: "+function.getName().str());
        if(!existing && function.getName().starts_with("__cudavm_native_"))fail("missing numeric helper: "+function.getName().str());
        if(!existing) {
            existing=llvm::Function::Create(function.getFunctionType(),llvm::GlobalValue::ExternalLinkage,function.getName(),module);
            existing->copyAttributesFrom(&function);
        }
        values[&function]=existing;
    }
    for(auto& function:*imported)if(!function.isDeclaration()) {
        auto* clone=llvm::Function::Create(function.getFunctionType(),llvm::GlobalValue::InternalLinkage,function.getName(),module);
        clone->copyAttributesFrom(&function);values[&function]=clone;
        for(unsigned i=0;i<function.arg_size();++i)values[function.getArg(i)]=clone->getArg(i);
    }
    for(auto& global:imported->globals()) {
        auto* clone=new llvm::GlobalVariable(module,global.getValueType(),global.isConstant(),llvm::GlobalValue::InternalLinkage,
            nullptr,global.getName(),nullptr,global.getThreadLocalMode(),global.getAddressSpace());
        clone->copyAttributesFrom(&global);values[&global]=clone;
    }
    for(auto& global:imported->globals())if(global.hasInitializer())
        llvm::cast<llvm::GlobalVariable>(values[&global])->setInitializer(llvm::MapValue(global.getInitializer(),values));
    for(auto& function:*imported)if(!function.isDeclaration()) {
        llvm::SmallVector<llvm::ReturnInst*> returns;
        llvm::CloneFunctionInto(llvm::cast<llvm::Function>(values[&function]),&function,values,
            llvm::CloneFunctionChangeType::DifferentModule,returns);
    }
    auto* function=llvm::cast<llvm::Function>(values[original]);
    function->addFnAttr(llvm::Attribute::AlwaysInline);
    llvm::IRBuilder<> b(&call);
    for(auto*& value:arguments) {
        auto* type=llvm::Type::getIntNTy(module.getContext(),value->getType()->isPointerTy()?64:unsigned(value->getType()->getPrimitiveSizeInBits()));
        value=value->getType()->isPointerTy()?b.CreatePtrToInt(value,type):b.CreateBitCast(value,type);
    }
    if(memory)b.CreateFence(llvm::AtomicOrdering::SequentiallyConsistent,llvm::SyncScope::SingleThread);
    auto* invocation=b.CreateCall(function,arguments);invocation->setDebugLoc(call.getDebugLoc());
    if(convergent){function->addFnAttr(llvm::Attribute::Convergent);invocation->setConvergent();}
    if(memory)b.CreateFence(llvm::AtomicOrdering::SequentiallyConsistent,llvm::SyncScope::SingleThread);
    if(outputs.empty())return invocation;
    llvm::Value* result=llvm::UndefValue::get(call.getType());
    for(unsigned i=0;i<outputs.size();++i) {
        llvm::Value* value=b.CreateExtractValue(invocation,i);
        value=b.CreateTruncOrBitCast(value,llvm::Type::getIntNTy(module.getContext(),operands[i].bits));
        value=outputs[i]->isPointerTy()?b.CreateIntToPtr(value,outputs[i]):b.CreateBitCast(value,outputs[i]);
        result=call.getType()->isStructTy()?b.CreateInsertValue(result,value,i):value;
    }
    return result;
}
void lower_inline_ptx(llvm::Module& module) {
    std::vector<llvm::CallInst*> calls;
    for(auto& function:module)for(auto& block:function)for(auto& instruction:block)
        if(auto* call=llvm::dyn_cast<llvm::CallInst>(&instruction);call && call->isInlineAsm())calls.push_back(call);
    for(auto* call:calls) {
        try {
            // Object image assembly retains its specialized descriptor lowering.
            // Inspect tokens so image spellings inside comments do not hide an
            // otherwise supported arithmetic block from the common importer.
            auto text=llvm::cast<llvm::InlineAsm>(call->getCalledOperand())->getAsmString();
            bool image=false;
            for(const auto& token:ptx_tokens(text.str()))
                image|=token.text.starts_with("tex.") || token.text.starts_with("suld.") || token.text.starts_with("sust.");
            if(image)continue;
            auto* result=import_inline_ptx(*call);
            if(!call->getType()->isVoidTy())call->replaceAllUsesWith(result);
            call->eraseFromParent();
        } catch(const std::exception& error) {
            auto diagnostic="inline PTX in "+call->getFunction()->getName().str()+": "+error.what();
            call->setMetadata("cudavm.inline.ptx.error",llvm::MDNode::get(module.getContext(),llvm::MDString::get(module.getContext(),diagnostic)));
        }
    }
}

}
