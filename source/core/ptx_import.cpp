#include "ptx_import.hpp"
#include "compiler_profiles.hpp"
#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <set>
#include <vector>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Linker/Linker.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/raw_ostream.h>
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
            while (at < input.size() && input[at] != '"' && input[at] != '\n' && input[at] != '\\') {
                token.text += input[at]; advance();
            }
            if (at == input.size() || input[at] != '"') ptx_fail(token, "unsupported or unterminated string literal", true);
            token.text += input[at]; advance();
        } else if (std::string_view("{}()[]<>,;:@!+-|=").find(input[at]) != std::string_view::npos) {
            token.text += input[at]; advance();
        } else {
            while (at < input.size()) {
                auto c = static_cast<unsigned char>(input[at]);
                if (!std::isalnum(c) && c != '_' && c != '$' && c != '%' && c != '.') break;
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
    llvm::Type* value(llvm::LLVMContext& context) const { return kind == 'f' ? (bits==16?llvm::Type::getHalfTy(context):llvm::Type::getFloatTy(context)) : storage(context); }
};
ptx_type ptx_scalar(const ptx_token& token, bool memory = false) {
    auto name = token.text; if (name.starts_with('.')) name.erase(0, 1);
    if (name == "f16x2" && !memory) return {32, 'h'};
    if (name == "pred" && !memory) return {1, 'p'};
    for (auto prefix : {'b', 'u', 's', 'f'}) for (unsigned bits : {8u, 16u, 32u, 64u})
        if (name == prefix + std::to_string(bits)) {
            if ((prefix == 'f' && bits != 32 && bits != 16) || (!memory && bits == 8)) break;
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
    uint64_t bytes() const { return count * (type.bits / 8); }
    llvm::Type* storage(llvm::LLVMContext& context) const {
        return array ? llvm::ArrayType::get(type.storage(context), count) : static_cast<llvm::Type*>(type.storage(context));
    }
};
struct ptx_body {
    ptx_token name;
    llvm::Function* function;
    std::vector<ptx_parameter> parameters, results;
    std::map<std::string, ptx_register> registers;
    std::map<std::string, std::pair<ptx_parameter, llvm::Value*>> slots;
    std::vector<ptx_instruction> instructions;
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
    std::map<std::string, std::pair<std::vector<ptx_parameter>, std::vector<ptx_parameter>>> signatures;
    std::vector<ptx_global> globals;
    std::map<std::string, ptx_token> symbols;
    unsigned symbol_counter = 0;
    bool relocatable = false;
    const ptx_token& peek() const { return tokens.at(cursor); }
    ptx_token take() { auto token = peek(); if (cursor + 1 < tokens.size()) ++cursor; return token; }
    bool accept(const char* text) { if (peek().text != text) return false; take(); return true; }
    ptx_token expect(const char* text) { auto token = take(); if (token.text != text) ptx_fail(token, std::string("expected '") + text + "'"); return token; }
    ptx_token identifier() { auto token = take(); if (!ptx_identifier(token.text)) ptx_fail(token, "expected identifier"); return token; }


    unsigned alignment() {
        auto token = take(); auto value = ptx_unsigned(token);
        if (!value || value > 256 || (value & (value - 1))) ptx_fail(token, "invalid storage alignment");
        return unsigned(value);
    }
    ptx_parameter declaration(bool unsized = false) {
        unsigned align = accept(".align") ? alignment() : 0;
        auto type = ptx_scalar(take(), true);
        if (peek().text == ".ptr") ptx_fail(peek(), "pointer parameter annotations are unsupported", true);
        auto name = identifier();
        ptx_parameter result{name, type, 1, align ? align : type.bits / 8, false};
        if (accept("[")) {
            result.array = true;
            result.count = peek().text == "]" ? 0 : ptx_unsigned(take());
            expect("]");
            if ((!result.count && !unsized) || result.count > (1u << 24)) ptx_fail(name, "invalid or excessive storage extent");
            if (peek().text == "[") ptx_fail(peek(), "multidimensional declarations are unsupported", true);
        }
        if (result.alignment < type.bits / 8) ptx_fail(name, "storage alignment is smaller than element size");
        return result;
    }
    std::vector<ptx_parameter> parameter_list() {
        expect("("); std::vector<ptx_parameter> result; std::set<std::string> names;
        uint64_t extent = 0;
        if (!accept(")")) for (;;) {
            expect(".param"); auto parameter = declaration();
            if (!names.insert(parameter.name.text).second) ptx_fail(parameter.name, "duplicate parameter");
            extent = llvm::alignTo(extent, parameter.alignment) + parameter.bytes();
            if (extent > 65536) ptx_fail(parameter.name, "function parameters exceed 64 KiB", true);
            result.push_back(parameter);
            if (accept(")")) break;
            expect(",");
        }
        return result;
    }
    void storage(bool external, const std::string& space, ptx_body* body = nullptr,
            std::map<std::string, std::string>* scope = nullptr) {
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
            variable->setInitializer(address_space == 3 ? llvm::UndefValue::get(data.storage(context)) : llvm::Constant::getNullValue(data.storage(context)));
            globals.push_back({data, variable, std::move(initializer)});
        }
    }
    void callable(bool kernel, bool external) {
        std::vector<ptx_parameter> results;
        if (!kernel && peek().text == "(") results = parameter_list();
        if (results.size() > 1) ptx_fail(results[1].name, "PTX ABI permits a single return parameter");
        auto name = identifier();
        if(name.text.starts_with("__cudavm_native_") || name.text.starts_with("_ZN6cudavm"))ptx_fail(name,"reserved compiler symbol",true);
        auto parameters = parameter_list();
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
        auto same = [](const auto& a, const auto& b) {
            if (a.size() != b.size()) return false;
            for (unsigned i = 0; i < a.size(); ++i)
                if (a[i].bytes() != b[i].bytes() || a[i].alignment != b[i].alignment || a[i].array != b[i].array) return false;
            return true;
        };
        if (auto found = signatures.find(name.text); found != signatures.end())
            if (!same(found->second.first, parameters) || !same(found->second.second, results)) ptx_fail(name, "conflicting function parameter layout");
        if (!function) function = llvm::Function::Create(type, llvm::GlobalValue::ExternalLinkage, name.text, *module);
        if (kernel) function->addFnAttr("cudavm.kernel");
        signatures[name.text] = {parameters, results};
        for (unsigned i = 0; i < parameters.size(); ++i) {
            if (kernel && inputs[i]->isPointerTy()) {
                function->addParamAttr(i, llvm::Attribute::getWithByValType(context, parameters[i].storage(context)));
                function->addParamAttr(i, llvm::Attribute::getWithAlignment(context, llvm::Align(parameters[i].alignment)));
            }
            function->getArg(i)->setName(parameters[i].name.text);
        }
        if (accept(";")) {
            if (kernel) ptx_fail(name, "kernel declarations require a definition", true);
            return;
        }
        if (external) ptx_fail(name, "extern function cannot have a body");
        if (!function->isDeclaration()) ptx_fail(name, "duplicate function definition");
        if (peek().text != "{") ptx_fail(peek(), "unsupported kernel directive or declaration", true);
        expect("{");
        ptx_body body{name, function, parameters, results, {}, {}, {}};
        auto* entry = llvm::BasicBlock::Create(context, "entry", function); llvm::IRBuilder<> allocations(entry);
        std::vector<std::map<std::string, std::string>> scopes(1);
        auto add_slot = [&](const ptx_parameter& parameter, llvm::Value* input = nullptr, bool byval = false) {
            if (body.slots.contains(parameter.name.text)) ptx_fail(parameter.name, "duplicate parameter");
            llvm::Value* slot = input;
            if (!byval) {
                auto* storage = allocations.CreateAlloca(parameter.storage(context)); storage->setAlignment(llvm::Align(parameter.alignment)); slot = storage;
                if (input) allocations.CreateAlignedStore(input, slot, llvm::Align(parameter.alignment));
            }
            body.slots.emplace(parameter.name.text, std::make_pair(parameter, slot));
            scopes.back()[parameter.name.text] = parameter.name.text;
        };
        for (unsigned i = 0; i < parameters.size(); ++i) add_slot(parameters[i], function->getArg(i), kernel && inputs[i]->isPointerTy());
        for (const auto& parameter : results) add_slot(parameter);
        auto resolve = [&](ptx_token token) {
            for (auto scope = scopes.rbegin(); scope != scopes.rend(); ++scope)
                if (auto found = scope->find(token.text); found != scope->end()) { token.text = found->second; break; }
            return token;
        };
        while (true) {
            if (accept("}")) { if (scopes.size() == 1) break; scopes.pop_back(); continue; }
            if (accept("{")) { scopes.emplace_back(); continue; }
            if (peek().text == "<eof>") ptx_fail(peek(), "unterminated function body");
            if (accept(".pragma")) {
                auto hint = take(); if (hint.text != "\"nounroll\"") ptx_fail(hint, "unsupported pragma", true);
                expect(";"); continue;
            }
            if (accept(".param")) {
                auto parameter = declaration(); auto original = parameter.name.text;
                if (scopes.back().contains(original)) ptx_fail(parameter.name, "duplicate local parameter");
                parameter.name.text = name.text + "$param$" + std::to_string(symbol_counter++);
                add_slot(parameter); scopes.back().erase(parameter.name.text); scopes.back()[original] = parameter.name.text;
                expect(";"); continue;
            }
            bool external_storage = accept(".extern");
            if (peek().text == ".shared") { take(); storage(external_storage, ".shared", &body, &scopes.back()); continue; }
            if (external_storage) ptx_fail(peek(), "unsupported external body declaration", true);
            if (accept(".reg")) {
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
                        body.registers.emplace(unique, ptx_register{type, allocations.CreateAlloca(type.storage(context))});
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
        if(op=="mov" && parts.size()==2 && (operands.size()==2) && (operands[0].size()>1 || operands[1].size()>1)) {
            // mov.b32 packs/unpacks two independent 16-bit lanes.
            if(type.bits!=32 || type.kind!='b')return false;
            auto fields=[&](const auto& v){return v.size()==5 && v[0].text=="{" && v[2].text=="," && v[4].text=="}";};
            if(fields(operands[1]))return done(b.CreateOr(b.CreateZExt(read({operands[1][1]},{16,'b'}),b.getInt32Ty()),b.CreateShl(b.CreateZExt(read({operands[1][3]},{16,'b'}),b.getInt32Ty()),16)),type);
            if(fields(operands[0])) {
                auto* v=read(operands[1],type);
                write({operands[0][1]},b.CreateTrunc(v,b.getInt16Ty()),{16,'b'},false);
                write({operands[0][3]},b.CreateTrunc(b.CreateLShr(v,16),b.getInt16Ty()),{16,'b'},false);return true;
            }
            return false;
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
            arity(2);auto* v=word(read(operands[1],type));auto sign=type.kind=='h'?0x80008000u:type.bits==16?0x8000u:0x80000000u;
            auto* mask=llvm::ConstantInt::get(v->getType(),sign);
            v=op=="neg"?b.CreateXor(v,mask):b.CreateAnd(v,b.CreateNot(mask));
            return done(type.kind=='f'?b.CreateBitCast(v,type.value(context)):v,type);
        }
        if((op=="min" || op=="max") && (type.kind=='f' || type.kind=='h') &&
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
                    ((type.bits==16 || type.kind=='h') && ((parts.size()==3 && parts[1]=="sat") || (parts.size()==4 && parts[1]=="rn" && parts[2]=="sat"))) ||
                    (op=="fma" && (type.bits==16 || type.kind=='h') && parts.size()==4 && parts[1]=="rn" && parts[2]=="relu"))) {
            bool fused=op=="fma";arity(fused?4:3);
            auto* a=read(operands[1],type);auto* c=read(operands[2],type);
            auto compute=[&](llvm::Value* a,llvm::Value* c,llvm::Value* d)->llvm::Value* {
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
            if(type.kind=='h') {
                auto lane=[&](unsigned shift){auto half=[&](llvm::Value* x){return b.CreateBitCast(b.CreateTrunc(b.CreateLShr(x,shift),b.getInt16Ty()),b.getHalfTy());};return b.CreateZExt(word(compute(half(a),half(c),d?half(d):nullptr)),b.getInt32Ty());};
                return done(b.CreateOr(lane(0),b.CreateShl(lane(16),16)),type);
            }
            return done(compute(a,c,d),type);
        }
        if(op=="div" && type.kind=='f' && type.bits==32 && parts==std::vector<std::string>{"div","rn","f32"}) {
            arity(3);return done(b.CreateFDiv(read(operands[1],type),read(operands[2],type)),type);
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
    void lower(ptx_body& body) {
        auto& function = *body.function; auto* entry = &function.getEntryBlock();
        const auto& registers = body.registers;
        std::vector<ptx_instruction> instructions;
        // CUDA compilers use vector parameter operations for aggregate fields.
        // Expand them into scalar accesses before SSA construction; each lane
        // retains the original predicate and its byte offset.
        for (const auto& instruction : body.instructions) {
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
        auto reg = [&](const ptx_token& token) -> const ptx_register& {
            auto found = registers.find(token.text);
            if (found == registers.end()) ptx_fail(token, "undeclared register");
            return found->second;
        };
        auto scalar = [&](const std::vector<ptx_token>& tokens, ptx_type type, bool truncate_memory = false) -> llvm::Value* {
            if (tokens.empty() || tokens.size() > 2) ptx_fail(tokens.empty() ? ptx_token{"operand", 0, 0} : tokens.front(), "expected scalar operand");
            bool negative = tokens.size() == 2;
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
                    if (!value) ptx_fail(token, token.text.find('.') == std::string::npos ? "undeclared register" : "unsupported special register",
                        token.text.find('.') != std::string::npos);
                }
            } else if (auto* global = module->getGlobalVariable(token.text, true)) {
                if (negative || (type.bits != 32 && type.bits != 64) || type.kind == 'f') ptx_fail(token, "invalid symbol address operand");
                value = builder.CreatePtrToInt(global, type.storage(context));
            } else if (token.text.starts_with("0f") && token.text.size() == 10) {
                if (type.bits != 32 || negative) ptx_fail(token, "invalid binary32 literal");
                uint64_t bits = 0;
                if (llvm::StringRef(token.text).drop_front(2).getAsInteger(16, bits)) ptx_fail(token, "invalid binary32 literal");
                value = builder.getInt32(uint32_t(bits));
            } else {
                if (type.kind == 'f') ptx_fail(token, "floating literals require PTX 0f binary32 encoding", true);
                uint64_t integer = token.text == "WARP_SZ" ? 32 : ptx_unsigned(token);
                value = llvm::ConstantInt::get(type.storage(context), negative ? uint64_t(0) - integer : integer);
            }
            return type.kind == 'f' ? builder.CreateBitCast(value, type.value(context)) : value;
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
            if (opcode.text == "ret" || opcode.text == "exit") {
                arity(0);
                if (opcode.text == "exit" && !function.hasFnAttribute("cudavm.kernel")) ptx_fail(opcode, "exit in device functions is unsupported", true);
                llvm::Value* result = nullptr;
                for (unsigned i = 0; i < body.results.size(); ++i) {
                    const auto& parameter = body.results[i];
                    auto* value = builder.CreateAlignedLoad(parameter.storage(context), body.slots.at(parameter.name.text).second, llvm::Align(parameter.alignment));
                    result = value;
                }
                if (result) builder.CreateRet(result); else builder.CreateRetVoid();
            } else if (opcode.text == "bar.sync") {
                if (instruction.operands.empty() || instruction.operands.size() > 2) ptx_fail(opcode, "wrong operand count");
                std::vector<llvm::Value*> args;
                for (unsigned i = 0; i < instruction.operands.size(); ++i) {
                    auto* value = operand(i, {32, 'u'});
                    if (auto* constant = llvm::dyn_cast<llvm::ConstantInt>(value)) {
                        auto number = constant->getZExtValue();
                        if ((!i && number >= 16) || (i && (!number || number % 32))) ptx_fail(opcode, "invalid CTA barrier operand");
                    }
                    args.push_back(value);
                }
                auto callee = module->getOrInsertFunction(args.size() == 1 ? "llvm.nvvm.barrier.cta.sync.aligned.all" : "llvm.nvvm.barrier.cta.sync.aligned.count",
                    llvm::FunctionType::get(builder.getVoidTy(), std::vector<llvm::Type*>(args.size(), builder.getInt32Ty()), false));
                llvm::cast<llvm::Function>(callee.getCallee())->addFnAttr(llvm::Attribute::Convergent);
                builder.CreateCall(callee, args)->setConvergent();
            } else if (opcode.text == "call" || opcode.text == "call.uni") {
                auto list = [&](const std::vector<ptx_token>& tokens) {
                    std::vector<ptx_token> names;
                    if (tokens.size() < 2 || tokens.front().text != "(" || tokens.back().text != ")") ptx_fail(opcode, "expected call parameter list");
                    for (size_t i = 1; i + 1 < tokens.size(); ++i) {
                        if (i % 2 == 0) { if (tokens[i].text != ",") ptx_fail(tokens[i], "expected call parameter separator"); }
                        else names.push_back(tokens[i]);
                    }
                    if (tokens.size() > 2 && tokens[tokens.size() - 2].text == ",") ptx_fail(opcode, "empty call parameter");
                    return names;
                };
                unsigned target_index = !instruction.operands.empty() && instruction.operands[0].front().text == "(" ? 1 : 0;
                if (instruction.operands.size() != target_index + 2) ptx_fail(opcode, "unsupported indirect call or call prototype", true);
                const auto& target = instruction.operands[target_index];
                if (target.size() != 1 || !ptx_identifier(target[0].text)) ptx_fail(opcode, "indirect calls are unsupported", true);
                auto* callee = module->getFunction(target[0].text);
                if (!callee || (!relocatable && callee->isDeclaration())) ptx_fail(target[0], "unresolved device function");
                if (callee->hasFnAttribute("cudavm.kernel")) ptx_fail(target[0], "cannot call a kernel");
                const auto& signature = signatures.at(target[0].text);
                auto inputs = list(instruction.operands[target_index + 1]);
                auto outputs = target_index ? list(instruction.operands[0]) : std::vector<ptx_token>{};
                if (inputs.size() != signature.first.size() || outputs.size() != signature.second.size()) ptx_fail(opcode, "call parameter count mismatch");
                auto slot = [&](const ptx_token& token, const ptx_parameter& expected, bool output) -> llvm::Value* {
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
                for (unsigned i = 0; i < inputs.size(); ++i) arguments.push_back(builder.CreateAlignedLoad(signature.first[i].storage(context), slot(inputs[i], signature.first[i], false), llvm::Align(signature.first[i].alignment)));
                auto* call = builder.CreateCall(callee, arguments); call->setConvergent();
                for (unsigned i = 0; i < outputs.size(); ++i) {
                    builder.CreateAlignedStore(call, slot(outputs[i], signature.second[i], true), llvm::Align(signature.second[i].alignment));
                }
            }
            else if (parts.size() >= 2) {
                const auto& op = parts[0];
                auto type = ptx_scalar({parts.back(), opcode.line, opcode.column}, op == "ld" || op == "st");
                if (type.kind == 'p') ptx_fail(opcode, "unsupported predicate instruction", true);
                llvm::Value* value = nullptr;
                if (numeric_instruction(parts,instruction,builder,scalar,write)) {}
                else if (op == "ld" || op == "st") {
                    arity(2);
                    bool load = op == "ld", parameter = parts.size() == 3 && parts[1] == "param";
                    bool global = (parts.size() == 3 && parts[1] == "global") ||
                        (load && parts.size() == 4 && parts[1] == "global" && parts[2] == "nc");
                    bool shared = parts.size() == 3 && parts[1] == "shared";
                    bool constant = parts.size() == 3 && parts[1] == "const";
                    bool generic = parts.size() == 2;
                    if ((!parameter && !global && !shared && !constant && !generic) || (!load && constant)) ptx_fail(opcode, "unsupported memory instruction", true);
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
                        if (shared && registers.contains(address[1].text)) width = registers.at(address[1].text).type.bits;
                        if (width != 32 && width != 64) ptx_fail(address[1], "invalid address register width");
                        auto* pointer_bits = scalar({address[1]}, {width, 'u'});
                        if (displacement) pointer_bits = builder.CreateAdd(pointer_bits, llvm::ConstantInt::get(pointer_bits->getType(), uint64_t(displacement)));
                        pointer = builder.CreateIntToPtr(pointer_bits, llvm::PointerType::get(context, shared ? 3 : constant ? 4 : global ? 1 : 0));
                    }
                    if (load) value = builder.CreateAlignedLoad(type.storage(context), pointer, llvm::Align(type.bits / 8));
                    else {
                        auto* data = scalar(instruction.operands[1], type, true);
                        if (type.kind == 'f') data = builder.CreateBitCast(data, type.storage(context));
                        builder.CreateAlignedStore(data, pointer, llvm::Align(type.bits / 8));
                    }
                    if (load) write(instruction.operands[0], value, type, true);
                } else if (op == "cvta" && (type.bits == 32 || type.bits == 64) && type.kind == 'u' &&
                        (parts.size() == 3 || (parts.size() == 4 && parts[1] == "to"))) {
                    arity(2); bool to = parts.size() == 4; auto space = parts[to ? 2 : 1];
                    if (space != "global" && space != "shared" && space != "const") ptx_fail(opcode, "unsupported address space conversion", true);
                    auto* specific = llvm::PointerType::get(context, space == "shared" ? 3 : space == "const" ? 4 : 1);
                    auto* generic = llvm::PointerType::get(context, 0);
                    auto* pointer = builder.CreateIntToPtr(operand(1, type), to ? generic : specific);
                    value = builder.CreatePtrToInt(builder.CreateAddrSpaceCast(pointer, to ? specific : generic), type.storage(context));
                    write(instruction.operands[0], value, type);
                } else if (op == "cvt" && parts.size() == 3 && type.kind != 'f' && type.kind != 'b' && type.bits >= 16) {
                    arity(2); auto destination = ptx_scalar({parts[1], opcode.line, opcode.column});
                    if (destination.kind != 's' && destination.kind != 'u') ptx_fail(opcode, "unsupported conversion", true);
                    auto* input = scalar(instruction.operands[1], type, true);
                    value = type.signed_integer() ? builder.CreateSExtOrTrunc(input, destination.storage(context)) : builder.CreateZExtOrTrunc(input, destination.storage(context));
                    write(instruction.operands[0], value, destination, true);
                } else if (op == "setp" && parts.size() == 3 && (type.kind == 's' || type.kind == 'u') && type.bits >= 16) {
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
                } else if ((op == "mul" || op == "mad") && parts.size() == 3 && (type.kind == 's' || type.kind == 'u') &&
                        (parts[1] == "lo" || (parts[1] == "wide" && type.bits == 32))) {
                    arity(op == "mad" ? 4 : 3); auto* left = operand(1, type); auto* right = operand(2, type);
                    auto result_type = type;
                    if (parts[1] == "wide") {
                        result_type.bits *= 2;
                        left = type.signed_integer() ? builder.CreateSExt(left, result_type.storage(context)) : builder.CreateZExt(left, result_type.storage(context));
                        right = type.signed_integer() ? builder.CreateSExt(right, result_type.storage(context)) : builder.CreateZExt(right, result_type.storage(context));
                    }
                    value = builder.CreateMul(left, right);
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
        expect(".target"); auto target = take();
        const std::map<std::string, unsigned> targets{{"sm_50", 60}, {"sm_52", 60}, {"sm_53", 60}, {"sm_60", 60},
            {"sm_61", 60}, {"sm_62", 60}, {"sm_70", 60}, {"sm_72", 61}, {"sm_75", 63},
            {"sm_80", 70}, {"sm_86", 71}, {"sm_87", 74}, {"sm_89", 78}, {"sm_90", 78}};
        if (!targets.contains(target.text)) ptx_fail(target, "unsupported PTX target", true);
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
            accept(".visible"); bool external = accept(".extern");
            if (accept(".entry")) callable(true, external);
            else if (accept(".func")) callable(false, external);
            else if (peek().text == ".global" || peek().text == ".const" || peek().text == ".shared") {
                auto space = take(); storage(external, space.text);
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
                auto* target = module->getGlobalVariable(tokens[0].text, true);
                if (target) {
                    if (data.type.bits != 64 || data.type.kind == 'f' || (target->getAddressSpace() != 1 && target->getAddressSpace() != 4)) ptx_fail(tokens[0], "symbol initializer requires a 64-bit global address", true);
                    llvm::Constant* address = target;
                    if (tokens.size() == 3 && (tokens[1].text == "+" || tokens[1].text == "-")) {
                        auto offset = ptx_unsigned(tokens[2]);
                        if (tokens[1].text == "-" || offset > module->getDataLayout().getTypeAllocSize(target->getValueType()).getFixedValue()) ptx_fail(tokens[0], "initializer relocation is out of bounds");
                        address = llvm::ConstantExpr::getGetElementPtr(llvm::Type::getInt8Ty(context), target, llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), offset));
                    } else if (tokens.size() != 1) ptx_fail(tokens[0], "unsupported initializer expression", true);
                    elements[i] = llvm::ConstantExpr::getPtrToInt(address, data.type.storage(context));
                } else {
                    bool negative = tokens.size() == 2 && tokens[0].text == "-";
                    if (tokens.size() != (negative ? 2u : 1u)) ptx_fail(tokens[0], "unsupported initializer expression", true);
                    const auto& literal = tokens.back(); uint64_t bits;
                    if (literal.text.starts_with("0f") && literal.text.size() == 10 && data.type.bits == 32 && !negative) {
                        if (llvm::StringRef(literal.text).drop_front(2).getAsInteger(16, bits)) ptx_fail(literal, "invalid binary32 literal");
                    } else {
                        if (data.type.kind == 'f') ptx_fail(literal, "floating initializers require PTX 0f binary32 encoding", true);
                        bits = ptx_unsigned(literal); if (negative) bits = uint64_t(0) - bits;
                    }
                    elements[i] = llvm::ConstantInt::get(data.type.storage(context), bits);
                }
            }
            global.variable->setInitializer(data.array ? llvm::ConstantArray::get(llvm::cast<llvm::ArrayType>(data.storage(context)), elements) : elements[0]);
        }
        for (auto& body : bodies) lower(body);
        // Inlining is the shader call mechanism. Reject cycles before entering
        // LLVM so recursive input never becomes a target failure or an abort.
        std::map<llvm::Function*, unsigned> colors;
        std::function<void(llvm::Function*)> visit = [&](llvm::Function* function) {
            if (colors[function] == 2) return;
            if (colors[function] == 1) ptx_fail(symbols.at(function->getName().str()), "recursive device calls are unsupported", true);
            colors[function] = 1;
            for (auto& block : *function) for (auto& instruction : block)
                if (auto* call = llvm::dyn_cast<llvm::CallInst>(&instruction))
                    if (auto* callee = call->getCalledFunction(); callee && !callee->isDeclaration()) visit(callee);
            colors[function] = 2;
        };
        for (auto& body : bodies) visit(body.function);
        for (auto& global : module->globals())
            if (!relocatable && global.isDeclaration() && global.getAddressSpace() != 3 && !global.use_empty()) ptx_fail(symbols.at(global.getName().str()), "unresolved device storage");
        std::string diagnostic; llvm::raw_string_ostream stream(diagnostic);
        if (llvm::verifyModule(*module, &stream)) throw ptx_import_error(false, "invalid imported PTX IR: " + diagnostic);
        return std::move(module);
    }
};
}
std::unique_ptr<llvm::Module> import_ptx(llvm::LLVMContext& context, std::string_view input, std::string_view architecture, bool relocatable) {
    return ptx_parser(context, input).parse(architecture, relocatable);
}
llvm::Value* import_numeric_inline_ptx(llvm::CallInst& call) {
    auto* assembly=llvm::cast<llvm::InlineAsm>(call.getCalledOperand());
    auto constraints=assembly->ParseConstraints();
    if(constraints.empty() || constraints[0].Type!=llvm::InlineAsm::isOutput || constraints[0].isIndirect || !call.getType()->isSingleValueType())
        throw std::runtime_error("numeric PTX requires one scalar register result");
    if(constraints.size()!=call.arg_size()+1)throw std::runtime_error("unsupported numeric PTX constraints");
    auto width=[](llvm::Type* t){return unsigned(t->getPrimitiveSizeInBits());};
    std::vector<llvm::Type*> types{call.getType()};
    for(unsigned i=0;i<call.arg_size();++i) {
        if(constraints[i+1].Type!=llvm::InlineAsm::isInput || constraints[i+1].isIndirect)throw std::runtime_error("unsupported numeric PTX input constraint");
        types.push_back(call.getArgOperand(i)->getType());
    }
    auto spelling=[&](unsigned i){auto bits=width(types[i]);if(bits!=16 && bits!=32 && bits!=64)throw std::runtime_error("unsupported numeric PTX register width");return ".b"+std::to_string(bits);};
    auto& module=*call.getModule();unsigned ordinal=0;std::string name;
    do {name="__cudavm_inline_numeric_"+std::to_string(ordinal++);}while(module.getNamedValue(name));
    std::string source=".version 8.4\n.target sm_75\n.address_size 64\n.func (.param "+spelling(0)+" result) "+name+"(";
    for(unsigned i=1;i<types.size();++i)source+=(i>1?", ":"")+std::string(".param ")+spelling(i)+" arg"+std::to_string(i);
    source+=") {\n";
    for(unsigned i=0;i<types.size();++i)source+=".reg "+spelling(i)+" %cv"+std::to_string(i)+";\n";
    for(unsigned i=1;i<types.size();++i)source+="ld.param"+spelling(i)+" %cv"+std::to_string(i)+", [arg"+std::to_string(i)+"];\n";
    if(constraints[0].hasMatchingInput())source+="mov"+spelling(0)+" %cv0, %cv"+std::to_string(constraints[0].MatchingInput)+";\n";
    auto text=assembly->getAsmString();
    for(size_t i=0;i<text.size();) {
        if(text[i]=='$' && i+1<text.size() && text[i+1]=='$'){source+='$';i+=2;}
        else if(text[i]=='$' && i+1<text.size() && std::isdigit(static_cast<unsigned char>(text[i+1]))) {
            size_t end=i+1;while(end<text.size() && std::isdigit(static_cast<unsigned char>(text[end])))++end;
            auto index=std::stoul(text.substr(i+1,end-i-1).str());if(index>=types.size())throw std::runtime_error("invalid numeric PTX operand reference");
            source+="%cv"+std::to_string(index);i=end;
        } else if(text.substr(i,5)==".reg."){source+=".reg .";i+=5;}
        else source+=text[i++];
    }
    source+="\nst.param"+spelling(0)+" [result], %cv0; ret; }\n.entry __cudavm_inline_unused() {ret;}\n";
    auto imported=import_ptx(module.getContext(),source);
    auto* original=imported->getFunction(name);
    auto* function=llvm::Function::Create(original->getFunctionType(),llvm::GlobalValue::InternalLinkage,name,module);
    llvm::ValueToValueMapTy values;
    for(auto& f:*imported)if(f.isDeclaration()) {
        auto* existing=module.getFunction(f.getName());
        if(f.isIntrinsic())existing=llvm::cast<llvm::Function>(module.getOrInsertFunction(f.getName(),f.getFunctionType()).getCallee());
        else if(!existing || existing->isDeclaration() || existing->getFunctionType()!=f.getFunctionType())throw std::runtime_error("missing numeric PTX helper: "+f.getName().str());
        values[&f]=existing;
    }
    for(unsigned i=0;i<function->arg_size();++i)values[original->getArg(i)]=function->getArg(i);
    llvm::SmallVector<llvm::ReturnInst*> returns;
    llvm::CloneFunctionInto(function,original,values,llvm::CloneFunctionChangeType::DifferentModule,returns);
    function->addFnAttr(llvm::Attribute::AlwaysInline);
    llvm::IRBuilder<> b(&call);std::vector<llvm::Value*> args;
    for(unsigned i=0;i<call.arg_size();++i)args.push_back(b.CreateBitCast(call.getArgOperand(i),llvm::Type::getIntNTy(module.getContext(),width(types[i+1]))));
    return b.CreateBitCast(b.CreateCall(function,args),call.getType());
}

}
