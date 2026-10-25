#include "ptx_import.hpp"
#include "compiler.hpp"
#include "compiler_profiles.hpp"
#include "device_conversions.hpp"
#include "integer_ir.hpp"
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
#include <llvm/IR/Operator.h>
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
    std::vector<ptx_global> globals;
    std::map<std::string, ptx_token> symbols;
    unsigned symbol_counter = 0;
    unsigned source_sm = 0;
    unsigned source_version = 0;
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
        if (!kernel) function->addFnAttr("cudavm.ptx.signature", ptx_signature_key(signatures.at(name.text)));
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
        function->setLinkage(linkage);
        if (peek().text != "{") ptx_fail(peek(), "unsupported kernel directive or declaration", true);
        expect("{");
        ptx_body body{name, function, parameters, results, {}, {}, {}, {}, {}};
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
        if ((op == "setp" || op == "set") && type.kind == 'f' && type.bits == 32) {
            if (parts.size() < (op == "setp" ? 3u : 4u)) ptx_fail(instruction.opcode, "unsupported FP32 comparison modifier", true);
            unsigned at = 2;
            auto boolean = parts[at] == "and" || parts[at] == "or" || parts[at] == "xor" ? parts[at++] : std::string{};
            bool flush = parts[at] == "ftz"; if (flush) ++at;
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
                    for (auto special : {"laneid", "warpsize", "lanemask.eq", "lanemask.lt", "lanemask.le", "lanemask.gt", "lanemask.ge"})
                        if (token.text == std::string("%") + special)
                            value = builder.CreateCall(module->getOrInsertFunction(std::string("llvm.nvvm.read.ptx.sreg.") + special,
                                llvm::FunctionType::get(builder.getInt32Ty(), {}, false)));
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
                if (!relocatable && target->isDeclaration()) ptx_fail(token, "unresolved device function");
                value = builder.CreatePtrToInt(target, type.storage(context));
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
            if (opcode.text == "ret" || opcode.text == "exit") {
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
            } else if (opcode.text == "bar.sync" || opcode.text == "bar.arrive") {
                bool arrive = opcode.text == "bar.arrive";
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
                auto name = arrive ? "llvm.nvvm.barrier.cta.arrive.aligned.count" :
                    args.size() == 1 ? "llvm.nvvm.barrier.cta.sync.aligned.all" : "llvm.nvvm.barrier.cta.sync.aligned.count";
                auto callee = module->getOrInsertFunction(name,
                    llvm::FunctionType::get(builder.getVoidTy(), std::vector<llvm::Type*>(args.size(), builder.getInt32Ty()), false));
                llvm::cast<llvm::Function>(callee.getCallee())->addFnAttr(llvm::Attribute::Convergent);
                builder.CreateCall(callee, args)->setConvergent();
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
                bool floating = type.kind == 'f' && type.bits == 32;
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
                        (operation == "add" && type.bits == 64 && type.kind != 'u') ||
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
                        if (!function || (!relocatable && function->isDeclaration())) ptx_fail(token, "unresolved device function");
                        if (function->hasFnAttribute("cudavm.kernel")) ptx_fail(token, "cannot call a kernel");
                        if (targets.empty()) signature = signatures.at(token.text);
                        else if (ptx_signature_key(signature) != ptx_signature_key(signatures.at(token.text))) ptx_fail(token, "call target signature mismatch");
                        targets.push_back(llvm::ValueAsMetadata::get(function));
                    }
                } else {
                    if (!callee || (!relocatable && callee->isDeclaration())) ptx_fail(target[0], "unresolved device function");
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
                    if (parts.size() != 2 || (op != "and" && op != "or" && op != "xor" && op != "not"))
                        ptx_fail(opcode, "unsupported predicate instruction", true);
                    arity(op == "not" ? 2 : 3); auto* left = operand(1, type);
                    value = op == "not" ? builder.CreateNot(left) : op == "and" ? builder.CreateAnd(left, operand(2, type)) :
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
            bool visible = accept(".visible"), weak = !visible && accept(".weak");
            bool external = !visible && !weak && accept(".extern");
            auto linkage = weak ? llvm::GlobalValue::WeakAnyLinkage : visible || external || !relocatable
                ? llvm::GlobalValue::ExternalLinkage : llvm::GlobalValue::InternalLinkage;
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
                    if (function && !relocatable && function->isDeclaration()) ptx_fail(tokens[0], "unresolved device function");
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
