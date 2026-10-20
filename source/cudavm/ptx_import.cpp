#include "ptx_import.hpp"
#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <vector>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>

namespace cudavm {
namespace {
struct ptx_token { std::string text; unsigned line, column; };
[[noreturn]] void ptx_fail(const ptx_token& token, const std::string& reason, bool unsupported = false) {
    throw ptx_import_error(unsupported, "PTX " + std::to_string(token.line) + ":" +
        std::to_string(token.column) + ": " + reason + " (" + token.text + ")");
}
std::vector<ptx_token> ptx_tokens(std::string_view input) {
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
        } else if (std::string_view("{}()[]<>,;:@!+-|").find(input[at]) != std::string_view::npos) {
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
    llvm::Type* value(llvm::LLVMContext& context) const { return kind == 'f' ? llvm::Type::getFloatTy(context) : storage(context); }
};
ptx_type ptx_scalar(const ptx_token& token, bool memory = false) {
    auto name = token.text; if (name.starts_with('.')) name.erase(0, 1);
    if (name == "pred" && !memory) return {1, 'p'};
    for (auto prefix : {'b', 'u', 's', 'f'}) for (unsigned bits : {8u, 16u, 32u, 64u})
        if (name == prefix + std::to_string(bits)) {
            if ((prefix == 'f' && bits != 32) || (!memory && bits == 8)) break;
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
struct ptx_parameter { ptx_token name; ptx_type type; };

class ptx_parser {
    std::vector<ptx_token> tokens;
    size_t cursor = 0;
    llvm::LLVMContext& context;
    std::unique_ptr<llvm::Module> module;
    const ptx_token& peek() const { return tokens.at(cursor); }
    ptx_token take() { auto token = peek(); if (cursor + 1 < tokens.size()) ++cursor; return token; }
    bool accept(const char* text) { if (peek().text != text) return false; take(); return true; }
    ptx_token expect(const char* text) { auto token = take(); if (token.text != text) ptx_fail(token, std::string("expected '") + text + "'"); return token; }
    ptx_token identifier() { auto token = take(); if (!ptx_identifier(token.text)) ptx_fail(token, "expected identifier"); return token; }

    void kernel() {
        auto name = identifier();
        if (module->getNamedValue(name.text)) ptx_fail(name, "duplicate kernel");
        if (module->size() >= 4096) ptx_fail(name, "kernel resource limit exceeded", true);
        expect("("); std::vector<ptx_parameter> parameters; std::set<std::string> parameter_names;
        uint64_t parameter_bytes = 0;
        if (!accept(")")) for (;;) {
            expect(".param"); unsigned alignment = 0;
            if (accept(".align")) {
                auto token = take(); auto value = ptx_unsigned(token);
                if (!value || value > 256 || (value & (value - 1))) ptx_fail(token, "invalid parameter alignment");
                alignment = unsigned(value);
            }
            auto type_token = take(); auto type = ptx_scalar(type_token, true);
            if (alignment && alignment != type.bits / 8) ptx_fail(type_token, "non-natural scalar parameter alignment is unsupported", true);
            if (peek().text == ".ptr") ptx_fail(peek(), "pointer parameter annotations are unsupported", true);
            auto parameter = identifier();
            if (peek().text == "[") ptx_fail(peek(), "aggregate kernel parameters are unsupported", true);
            if (!parameter_names.insert(parameter.text).second) ptx_fail(parameter, "duplicate parameter");
            parameters.push_back({parameter, type});
            parameter_bytes = (parameter_bytes + type.bits / 8 - 1) & ~uint64_t(type.bits / 8 - 1);
            parameter_bytes += type.bits / 8;
            if (parameter_bytes > 65536) ptx_fail(parameter, "kernel parameters exceed 64 KiB", true);
            if (accept(")")) break;
            expect(",");
        }
        // Execution constraints must be implemented, never discarded as hints.
        if (peek().text != "{") ptx_fail(peek(), "unsupported kernel directive or declaration", true);
        expect("{");
        std::vector<llvm::Type*> signature;
        for (const auto& parameter : parameters) signature.push_back(parameter.type.storage(context));
        auto* function = llvm::Function::Create(llvm::FunctionType::get(llvm::Type::getVoidTy(context), signature, false),
            llvm::GlobalValue::ExternalLinkage, name.text, *module);
        function->addFnAttr("cudavm.kernel");
        llvm::SmallVector<llvm::Metadata*> layout;
        uint64_t offset = 0;
        for (unsigned i = 0; i < parameters.size(); ++i) {
            auto size = parameters[i].type.bits / 8;
            offset = (offset + size - 1) & ~uint64_t(size - 1);
            llvm::SmallVector<llvm::Metadata*> fields;
            for (uint64_t value : {uint64_t(size), uint64_t(size), offset})
                fields.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), value)));
            layout.push_back(llvm::MDNode::get(context, fields)); offset += size;
            function->getArg(i)->setName(parameters[i].name.text);
        }
        function->setMetadata("cudavm.kernel.layout", llvm::MDNode::get(context, layout));
        auto* entry = llvm::BasicBlock::Create(context, "entry", function);
        llvm::IRBuilder<> allocations(entry);
        std::map<std::string, ptx_register> registers;
        std::vector<ptx_instruction> instructions;
        while (!accept("}")) {
            if (peek().text == "<eof>") ptx_fail(peek(), "unterminated kernel body");
            if (accept(".pragma")) {
                auto hint = take();
                if (hint.text != "\"nounroll\"") ptx_fail(hint, "unsupported pragma", true);
                expect(";"); continue;
            }
            if (accept(".reg")) {
                auto type_token = take(); auto type = ptx_scalar(type_token);
                for (;;) {
                    auto reg = take();
                    if (!reg.text.starts_with('%') || !ptx_identifier(reg.text.substr(1))) ptx_fail(reg, "invalid register name");
                    unsigned count = 0;
                    if (accept("<")) {
                        auto token = take(); auto value = ptx_unsigned(token);
                        if (!value || value > 65536) ptx_fail(token, "register resource limit exceeded", true);
                        count = unsigned(value); expect(">");
                    }
                    for (unsigned i = 0; i < std::max(1u, count); ++i) {
                        auto symbol = reg.text + (count ? std::to_string(i) : "");
                        if (registers.contains(symbol)) ptx_fail(reg, "duplicate register");
                        if (registers.size() >= 65536) ptx_fail(reg, "register resource limit exceeded", true);
                        registers.emplace(symbol, ptx_register{type, allocations.CreateAlloca(type.storage(context), nullptr, symbol.substr(1))});
                    }
                    if (accept(";")) break;
                    expect(",");
                }
                continue;
            }
            ptx_instruction instruction{};
            if (accept("@")) { instruction.negate = accept("!"); instruction.predicate = take().text; }
            instruction.opcode = take();
            if (accept(":")) {
                if (!instruction.predicate.empty() || !ptx_identifier(instruction.opcode.text)) ptx_fail(instruction.opcode, "invalid label");
                instruction.label = true;
            } else {
                if (instruction.opcode.text.starts_with('.')) ptx_fail(instruction.opcode, "unsupported body directive", true);
                if (!accept(";")) {
                    instruction.operands.emplace_back();
                    while (!accept(";")) {
                        if (peek().text == "}" || peek().text == "<eof>") ptx_fail(peek(), "expected instruction terminator");
                        if (accept(",")) instruction.operands.emplace_back();
                        else instruction.operands.back().push_back(take());
                    }
                    for (const auto& operand : instruction.operands) if (operand.empty()) ptx_fail(instruction.opcode, "empty operand");
                }
            }
            instructions.push_back(std::move(instruction));
            if (instructions.size() > 65536) ptx_fail(peek(), "instruction resource limit exceeded", true);
        }
        lower(*function, entry, registers, parameters, instructions);
    }

    void lower(llvm::Function& function, llvm::BasicBlock* entry, const std::map<std::string, ptx_register>& registers,
            const std::vector<ptx_parameter>& parameters, const std::vector<ptx_instruction>& instructions) {
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
            if (token.text.starts_with('%')) {
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
            if (value->getType()->isFloatTy()) value = builder.CreateBitCast(value, builder.getInt32Ty());
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
            if (opcode.text == "ret" || opcode.text == "exit") { arity(0); builder.CreateRetVoid(); }
            else if (parts.size() >= 2) {
                const auto& op = parts[0];
                auto type = ptx_scalar({parts.back(), opcode.line, opcode.column}, op == "ld" || op == "st");
                if (type.kind == 'p') ptx_fail(opcode, "unsupported predicate instruction", true);
                llvm::Value* value = nullptr;
                if (op == "ld" || op == "st") {
                    arity(2);
                    bool load = op == "ld", parameter = parts.size() == 3 && parts[1] == "param";
                    bool global = (parts.size() == 3 && parts[1] == "global") ||
                        (load && parts.size() == 4 && parts[1] == "global" && parts[2] == "nc");
                    if ((!parameter && !global) || (!load && parameter)) ptx_fail(opcode, "unsupported memory instruction", true);
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
                    if (parameter) {
                        auto found = std::find_if(parameters.begin(), parameters.end(), [&](const auto& p) { return p.name.text == address[1].text; });
                        if (found == parameters.end()) ptx_fail(address[1], "unknown kernel parameter");
                        if (displacement < 0 || uint64_t(displacement) + type.bits / 8 > found->type.bits / 8 || displacement % (type.bits / 8))
                            ptx_fail(address[1], "parameter load is out of bounds or misaligned");
                        value = function.getArg(unsigned(found - parameters.begin()));
                        if (displacement) value = builder.CreateLShr(value, uint64_t(displacement) * 8);
                        value = builder.CreateTruncOrBitCast(value, type.storage(context));
                    } else {
                        auto* pointer_bits = scalar({address[1]}, {64, 'u'});
                        if (displacement) pointer_bits = builder.CreateAdd(pointer_bits, builder.getInt64(uint64_t(displacement)));
                        auto* pointer = builder.CreateIntToPtr(pointer_bits, llvm::PointerType::get(context, 1));
                        if (load) value = builder.CreateAlignedLoad(type.storage(context), pointer, llvm::Align(type.bits / 8));
                        else {
                            auto* data = scalar(instruction.operands[1], type, true);
                            if (type.kind == 'f') data = builder.CreateBitCast(data, builder.getInt32Ty());
                            builder.CreateAlignedStore(data, pointer, llvm::Align(type.bits / 8));
                        }
                    }
                    if (load) write(instruction.operands[0], value, type, true);
                } else if (op == "cvta" && parts == std::vector<std::string>{"cvta", "to", "global", "u64"}) {
                    arity(2); write(instruction.operands[0], operand(1, type), type);
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
        llvm::PromotePass().run(function, analyses);
    }
public:
    ptx_parser(llvm::LLVMContext& owner, std::string_view input) : tokens(ptx_tokens(input)), context(owner), module(std::make_unique<llvm::Module>("imported.ptx", owner)) {}
    std::unique_ptr<llvm::Module> parse() {
        expect(".version"); auto version = take();
        // Bound accepted language versions independently of opcode support.
        // Later versions remain explicit obligations even for old opcodes.
        const std::set<std::string> versions{"6.0", "6.1", "6.2", "6.3", "6.4", "6.5", "7.0", "7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8", "8.0", "8.1", "8.2", "8.3", "8.4"};
        if (!versions.contains(version.text))
            ptx_fail(version, "unsupported PTX version (expected 6.0-6.5, 7.0-7.8 or 8.0-8.4)", true);
        expect(".target"); auto target = take();
        const std::map<std::string, unsigned> targets{{"sm_50", 60}, {"sm_52", 60}, {"sm_53", 60}, {"sm_60", 60},
            {"sm_61", 60}, {"sm_62", 60}, {"sm_70", 60}, {"sm_72", 61}, {"sm_75", 63},
            {"sm_80", 70}, {"sm_86", 71}, {"sm_87", 74}, {"sm_89", 78}, {"sm_90", 78}};
        if (!targets.contains(target.text)) ptx_fail(target, "unsupported PTX target", true);
        if (unsigned((version.text[0] - '0') * 10 + version.text[2] - '0') < targets.at(target.text))
            ptx_fail(target, "target is unavailable in this PTX version");
        if (peek().text == ",") ptx_fail(peek(), "unsupported target option", true);
        expect(".address_size"); auto address = take();
        if (address.text != "64") ptx_fail(address, "PTX import requires 64-bit addresses", true);
        module->setTargetTriple(llvm::Triple("nvptx64-nvidia-cuda"));
        module->setDataLayout("e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64");
        module->getOrInsertNamedMetadata("cudavm.ir.stage")->addOperand(llvm::MDNode::get(context, llvm::MDString::get(context, "cudavm.portable")));
        while (peek().text != "<eof>") {
            accept(".visible");
            if (!accept(".entry")) ptx_fail(peek(), "unsupported module declaration or directive", true);
            kernel();
        }
        if (module->empty()) ptx_fail(peek(), "PTX module contains no kernels");
        std::string diagnostic; llvm::raw_string_ostream stream(diagnostic);
        if (llvm::verifyModule(*module, &stream)) throw ptx_import_error(false, "invalid imported PTX IR: " + diagnostic);
        return std::move(module);
    }
};
}
std::unique_ptr<llvm::Module> import_ptx(llvm::LLVMContext& context, std::string_view input) {
    return ptx_parser(context, input).parse();
}
}
