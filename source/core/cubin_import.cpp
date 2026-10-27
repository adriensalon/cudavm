#include "cubin_import.hpp"
#include "compiler.hpp"
#include "ptx_import.hpp"
#include "../nvfatbin/container_api.hpp"
#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <set>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalAlias.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

namespace cudavm {
namespace {
[[noreturn]] void cubin_fail(const std::string& reason, bool unsupported = false) {
    throw compiler_error(unsupported ? compiler_status::unsupported : compiler_status::invalid, ("cubin: " + reason).c_str());
}
void cubin_check(bool valid, const std::string& reason, bool unsupported = false) {
    if (!valid) cubin_fail(reason, unsupported);
}
struct cubin_section {
    std::string name;
    uint32_t type, link, info;
    uint64_t flags, offset, size, stride, alignment;
};
struct cubin_parameter { unsigned offset, size; };
bool cubin_symbol_spelling(const std::string& name) {
    return !name.empty() && name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.$") == name.npos
        && !(name.front() >= '0' && name.front() <= '9');
}
struct cubin_symbol {
    std::string name;
    unsigned type, binding, other, section;
    uint64_t offset, size;
};
struct cubin_relocation {
    size_t section, offset, target_section;
    uint64_t target_offset;
    unsigned type;
};
struct cubin_reader {
    std::string_view bytes;
    std::vector<cubin_section> sections;
    std::map<size_t, std::vector<cubin_symbol>> symbols;
    std::map<size_t, std::string> storage;
    std::vector<cubin_relocation> relocations;
    uint64_t number(size_t at, unsigned width = 4) const { return fatbin_reader{bytes}.number(at, width); }
    std::string_view slice(uint64_t at, uint64_t size) const {
        cubin_check(at <= bytes.size() && size <= bytes.size() - at, "truncated ELF payload");
        return bytes.substr(size_t(at), size_t(size));
    }
    std::string string(std::string_view table, size_t at) const {
        cubin_check(at < table.size(), "invalid ELF string offset");
        auto end = table.find('\0', at); cubin_check(end != table.npos, "unterminated ELF string");
        return std::string(table.substr(at, end - at));
    }
    std::string_view data(const cubin_section& section) const { return slice(section.offset, section.size); }
    const cubin_section& named(const std::string& name) const {
        auto found = std::find_if(sections.begin(), sections.end(), [&](const auto& section) { return section.name == name; });
        cubin_check(found != sections.end(), "missing section " + name); return *found;
    }
    explicit cubin_reader(std::string_view image) : bytes(image) {
        cubin_check(bytes.size() <= fatbin_limit, "excessive ELF extent");
        unsigned architecture = 0;
        try { architecture = fatbin_elf_architecture(bytes); }
        catch (const std::exception& error) { cubin_fail(error.what()); }
        cubin_check(number(6, 1) == 1 && number(52, 2) == 64, "invalid ELF header");
        cubin_check(number(7, 1) == 51 && number(8, 1) == 7 && number(20) == 124, "only CUDA 12.4 ELF ABI 7 images are supported", true);
        cubin_check(number(16, 2) == 2, "only completed ET_EXEC images are supported", true);
        cubin_check(architecture == 75 && (number(48) & 0xff00) == 0x500, "only 64-bit SM75 images are supported", true);
        auto table = number(40, 8), width = number(58, 2), count = number(60, 2), names = number(62, 2);
        cubin_check(count <= 4096 && names < count, "invalid section table");
        cubin_check(number(table + names * width + 4) == 3, "invalid section name table");
        auto strings = slice(number(table + names * width + 24, 8), number(table + names * width + 32, 8));
        std::set<std::string> unique;
        for (size_t i = 0; i < count; ++i) {
            auto at = table + i * width;
            cubin_section section{string(strings, size_t(number(at))), uint32_t(number(at + 4)), uint32_t(number(at + 40)),
                uint32_t(number(at + 44)), number(at + 8, 8), number(at + 24, 8), number(at + 32, 8), number(at + 56, 8), number(at + 48, 8)};
            cubin_check(unique.insert(section.name).second, "duplicate section name");
            cubin_check(section.type != 11, "dynamic symbols are unsupported", true);
            sections.push_back(std::move(section));
        }
        for (size_t index = 0; index < sections.size(); ++index) {
            const auto& section = sections[index];
            if (section.name.starts_with(".nv.shared.")) {
                cubin_check(section.type == 8 && section.flags == 0x43 && section.link == 0 && section.stride == 0,
                    "invalid static shared section");
                cubin_check(section.size <= 32768, "unsupported static shared extent", true);
                cubin_check(section.alignment && section.alignment <= 256 && !(section.alignment & (section.alignment - 1)),
                    "invalid static shared alignment");
            } else if ((section.flags & 2) && !section.name.starts_with(".text.") && !section.name.starts_with(".nv.constant0.")) {
                bool global = section.name == ".nv.global" || section.name == ".nv.global.init";
                bool constant = section.name.starts_with(".nv.constant") && section.name.size() > 12 &&
                    section.name[12] >= '1' && section.name[12] <= '9';
                cubin_check(global || constant || section.size == 0, "unsupported allocated storage section", true);
                if (!section.size) continue;
                cubin_check((section.type == 1 || (global && section.type == 8)) && section.flags == (global ? 3u : 2u)
                    && section.stride == 0, "invalid global/constant storage section");
                cubin_check(section.alignment && !(section.alignment & (section.alignment - 1)) && section.alignment <= 65536
                    && section.size <= fatbin_limit, "invalid storage extent/alignment");
                if (section.type != 8) (void)data(section);
                storage.emplace(index, "__cudavm_cubin_storage_" + std::to_string(index));
            }
        }
        for (size_t index = 0; index < sections.size(); ++index) {
            const auto& table = sections[index];
            if (table.type != 2) continue;
            cubin_check(table.link < sections.size() && sections[table.link].type == 3 && table.stride == 24 && table.size % 24 == 0,
                "invalid symbol table");
            auto strings = data(sections[table.link]); fatbin_reader values{data(table)};
            auto& entries = symbols[index];
            for (size_t at = 0; at < table.size; at += 24) {
                unsigned info = unsigned(values.number(at + 4, 1));
                entries.push_back({string(strings, size_t(values.number(at))), info & 15, info >> 4,
                    unsigned(values.number(at + 5, 1)), unsigned(values.number(at + 6, 2)), values.number(at + 8, 8), values.number(at + 16, 8)});
                const auto& symbol = entries.back();
                if (symbol.section < sections.size() && storage.contains(symbol.section)) {
                    cubin_check(symbol.offset <= sections[symbol.section].size && symbol.size <= sections[symbol.section].size - symbol.offset,
                        "storage symbol exceeds section");
                    cubin_check(symbol.type == 1 || symbol.type == 3, "unsupported storage symbol", true);
                    cubin_check(symbol.binding <= 1 && !(symbol.other & ~3u), "unsupported storage binding", true);
                }
            }
        }
        std::map<size_t, std::set<size_t>> occupied;
        for (const auto& table : sections) if ((table.type == 4 || table.type == 9) && table.size) {
            cubin_check(table.info < sections.size() && symbols.contains(table.link), "invalid relocation target/symbol table");
            cubin_check(table.stride == (table.type == 4 ? 24u : 16u) && table.size % table.stride == 0, "invalid relocation table");
            fatbin_reader values{data(table)};
            const auto& target = sections[table.info];
            for (size_t at = 0; at < table.size; at += size_t(table.stride)) {
                auto offset = values.number(at, 8), info = values.number(at + 8, 8);
                cubin_check((info >> 32) < symbols.at(table.link).size(), "invalid relocation symbol");
                if (target.name.starts_with(".debug")) continue;
                unsigned type = unsigned(info);
                bool instruction = target.name.starts_with(".text.");
                cubin_check(instruction ? type == 56 || type == 57 : type == 2 || type == 4, "unsupported relocation encoding", true);
                cubin_check(instruction || storage.contains(table.info), "unsupported relocation target section", true);
                cubin_check(target.type == 1 && offset <= target.size && 8 <= target.size - offset &&
                    offset % (instruction ? 16 : 8) == 0, "relocation exceeds/misaligns target");
                const auto& symbol = symbols.at(table.link)[size_t(info >> 32)];
                cubin_check(storage.contains(symbol.section) && (symbol.type == 1 || symbol.type == 3), "unresolved/unsupported relocation symbol", true);
                int64_t addend = table.type == 4 ? int64_t(values.number(at + 16, 8)) :
                    instruction ? 0 : int64_t(number(size_t(target.offset + offset), 8));
                if (instruction && table.type == 9)
                    cubin_check(number(size_t(target.offset + offset + 4)) == 0, "implicit instruction relocation addend is unsupported", true);
                cubin_check(symbol.offset <= INT64_MAX && (addend >= 0 ? uint64_t(addend) <= sections[symbol.section].size - symbol.offset :
                    uint64_t(-(addend + 1)) + 1 <= symbol.offset), "relocation addend exceeds storage");
                auto address = addend < 0 ? symbol.offset - (uint64_t(-(addend + 1)) + 1) : symbol.offset + uint64_t(addend);
                cubin_check(occupied[table.info].insert(size_t(offset)).second, "overlapping relocations");
                relocations.push_back({table.info, size_t(offset), symbol.section, address, type});
            }
        }
    }
    template<class Visit> void records(const cubin_section& section, Visit visit) const {
        auto content = data(section); fatbin_reader reader{content};
        for (size_t at = 0; at < content.size();) {
            auto format = reader.number(at, 1), attribute = reader.number(at + 1, 1), size = reader.number(at + 2, 2);
            cubin_check(format >= 1 && format <= 4, "invalid information record");
            auto payload = format == 4 ? size : 0;
            cubin_check(at + 4 <= content.size() && payload <= content.size() - at - 4, "truncated information record");
            visit(unsigned(format), unsigned(attribute), unsigned(size), fatbin_reader{content.substr(at + 4, size_t(payload))});
            at += size_t(4 + payload);
        }
    }
};
std::string cubin_storage_declarations(const cubin_reader& input) {
    std::string result;
    for (const auto& [index, name] : input.storage) {
        const auto& section = input.sections[index];
        result += std::string(section.name.starts_with(".nv.constant") ? ".const" : ".global") + " .align " +
            std::to_string(section.alignment) + " .b8 " + name + "[" + std::to_string(section.size) + "];\n";
    }
    return result;
}
void materialize_cubin_storage(llvm::Module& module, const cubin_reader& input) {
    auto& context = module.getContext(); auto* byte = llvm::Type::getInt8Ty(context); auto* word = llvm::Type::getInt64Ty(context);
    std::map<size_t, llvm::GlobalVariable*> storage;
    for (const auto& [index, name] : input.storage) storage[index] = module.getGlobalVariable(name, true);
    for (const auto& [index, name] : input.storage) {
        const auto& section = input.sections[index];
        std::map<size_t, const cubin_relocation*> relocations;
        for (const auto& relocation : input.relocations) if (relocation.section == index) relocations.emplace(relocation.offset, &relocation);
        std::vector<llvm::Constant*> fields;
        for (size_t at = 0; at < section.size;) {
            if (auto found = relocations.find(at); found != relocations.end()) {
                const auto& relocation = *found->second;
                auto* address = llvm::ConstantExpr::getGetElementPtr(byte, storage.at(relocation.target_section),
                    llvm::ConstantInt::get(word, relocation.target_offset));
                fields.push_back(llvm::ConstantExpr::getPtrToInt(address, word)); at += 8;
            } else {
                auto next = relocations.upper_bound(at); auto end = next == relocations.end() ? size_t(section.size) : next->first;
                auto* type = llvm::ArrayType::get(byte, end - at);
                fields.push_back(section.type == 8 ? llvm::ConstantAggregateZero::get(type) :
                    static_cast<llvm::Constant*>(llvm::ConstantDataArray::getString(context, llvm::StringRef(input.data(section).data() + at, end - at), false)));
                at = end;
            }
        }
        std::vector<llvm::Type*> types; for (auto* field : fields) types.push_back(field->getType());
        auto* type = llvm::StructType::get(context, types, true); auto* old = storage.at(index);
        auto* variable = new llvm::GlobalVariable(module, type, false, llvm::GlobalValue::InternalLinkage,
            llvm::ConstantStruct::get(type, fields), "", nullptr, llvm::GlobalVariable::NotThreadLocal, old->getAddressSpace());
        variable->setAlignment(llvm::Align(section.alignment)); variable->takeName(old);
        old->replaceAllUsesWith(variable); old->eraseFromParent(); storage[index] = variable;
    }
    llvm::SmallVector<llvm::GlobalValue*> exports;
    for (const auto& [table, symbols] : input.symbols) for (const auto& symbol : symbols) {
        if (symbol.type != 1 || !input.storage.contains(symbol.section) || !symbol.size) continue;
        cubin_check(!symbol.name.empty() && !module.getNamedValue(symbol.name), "duplicate/reserved storage symbol");
        auto* variable = storage.at(symbol.section);
        auto* address = llvm::ConstantExpr::getGetElementPtr(byte, variable, llvm::ConstantInt::get(word, symbol.offset));
        auto* alias = llvm::GlobalAlias::create(llvm::ArrayType::get(byte, symbol.size), variable->getAddressSpace(),
            symbol.binding ? llvm::GlobalValue::ExternalLinkage : llvm::GlobalValue::InternalLinkage, symbol.name, address, &module);
        exports.push_back(alias);
    }
    // CUDA exposes named objects through module queries even when their ELF
    // binding is local and machine instructions reference only section storage.
    // Keep these views alive through device-link dead stripping.
    llvm::appendToCompilerUsed(module, exports);
}
struct cubin_function { std::string name; size_t offset, size; };
struct cubin_kernel {
    std::string name;
    const cubin_section* text;
    std::vector<cubin_parameter> parameters;
    unsigned base = 0, size = 0, shared_size = 0, shared_alignment = 1;
    bool shared_storage = false;
    std::map<size_t, unsigned> warp_barriers;
    std::map<size_t, cubin_function> functions;
    unsigned stack_size = 0;
};
std::vector<cubin_kernel> cubin_kernels(const cubin_reader& input) {
    std::vector<cubin_kernel> kernels; std::set<std::string> names;
    for (const auto& table : input.sections) if (table.type == 2) {
        cubin_check(table.link < input.sections.size() && table.stride == 24 && table.size % 24 == 0, "invalid symbol table");
        cubin_check(input.sections[table.link].type == 3, "invalid symbol name table");
        auto strings = input.data(input.sections[table.link]); fatbin_reader symbols{input.data(table)};
        for (size_t at = 0; at < table.size; at += 24) if ((symbols.number(at + 4, 1) & 15) == 2) {
            if (!(symbols.number(at + 5, 1) & 0x10)) continue;
            auto section = symbols.number(at + 6, 2); cubin_check(section < input.sections.size(), "invalid function section");
            auto name = input.string(strings, size_t(symbols.number(at)));
            cubin_check(cubin_symbol_spelling(name), "unsupported kernel symbol spelling", true);
            const auto& text = input.sections[section];
            cubin_check(text.type == 1 && (text.flags == 6 || text.flags == 0x100006), "unsupported code section", true);
            cubin_check(text.name == ".text." + name && symbols.number(at + 8, 8) == 0 && symbols.number(at + 16, 8) == text.size,
                "unsupported function extent", true);
            cubin_check(text.size && text.size % 16 == 0 && text.size <= 65536, "invalid/excessive SM75 code extent");
            cubin_check(names.insert(name).second && kernels.size() < 256, "duplicate/excessive kernels");
            cubin_kernel kernel{}; kernel.name = name; kernel.text = &text;
            std::map<unsigned, cubin_parameter> layout; bool bank = false, extent = false; unsigned declared_size = 0;
            std::optional<std::vector<unsigned>> cooperative_offsets, cooperative_masks;
            bool internal_warp_offsets = false;
            input.records(input.named(".nv.info." + name), [&](unsigned format, unsigned attr, unsigned size, const fatbin_reader& record) {
                if (attr == 0xa) {
                    cubin_check(format == 4 && size == 8 && !bank, "invalid parameter bank"); bank = true;
                    // EIATTR_PARAM_CBANK refers to a symbol, not a section.
                    auto index = record.number(0); cubin_check(index < table.size / 24, "invalid parameter bank symbol");
                    auto bank_section = symbols.number(size_t(index) * 24 + 6, 2);
                    cubin_check((symbols.number(size_t(index) * 24 + 4, 1) & 15) == 3 && bank_section < input.sections.size()
                        && input.sections[bank_section].name == ".nv.constant0." + name, "invalid parameter constant section");
                    kernel.base = unsigned(record.number(4, 2)); kernel.size = unsigned(record.number(6, 2));
                    cubin_check(kernel.base == 0x160 && kernel.size <= 4096, "unsupported parameter bank ABI", true);
                    cubin_check(input.sections[bank_section].type == 1 && input.sections[bank_section].size == kernel.base + kernel.size,
                        "parameter bank extent mismatch");
                } else if (attr == 0x19) {
                    cubin_check(format == 3 && !extent, "invalid parameter size record"); extent = true; declared_size = size;
                } else if (attr == 0x17) {
                    cubin_check(format == 4 && size == 12, "invalid kernel parameter record");
                    auto flags = unsigned(record.number(8)); unsigned width = flags >> 18;
                    cubin_check((width == 4 || width == 8) && (flags & 0x3ffff) == 0x1f000 && record.number(0) == 0,
                        "only scalar 32/64-bit parameters are supported", true);
                    cubin_check(layout.emplace(unsigned(record.number(4, 2)), cubin_parameter{unsigned(record.number(6, 2)), width}).second, "duplicate parameter ordinal");
                } else if (attr == 0x36 || attr == 0x37) {
                    cubin_check(format == 4 && size == 4, "invalid producer information");
                    cubin_check(record.number(0) == (attr == 0x36 ? 1u : 124u), "unsupported producer information", true);
                } else if (attr == 0x1b) cubin_check(format == 3, "invalid register limit");
                else if (attr == 0x1e) {
                    cubin_check(format == 4 && size == 4, "invalid reconvergence stack information");
                    cubin_check(record.number(0) == 0, "reconvergence stack spills are unsupported", true);
                } else if (attr == 0x4) cubin_check(format == 1 && size == 0, "invalid CTAID.Z usage record");
                else if (attr == 0x1c) {
                    cubin_check(format == 4 && size % 4 == 0, "invalid exit offsets");
                    for (unsigned offset = 0; offset < size; offset += 4)
                        cubin_check(record.number(offset) < text.size && record.number(offset) % 16 == 0, "invalid exit offset");
                } else if (attr == 0x31) {
                    cubin_check(format == 4 && size % 4 == 0 && !internal_warp_offsets, "invalid internal warp instruction metadata");
                    internal_warp_offsets = true;
                    std::set<unsigned> offsets;
                    for (unsigned at = 0; at < size; at += 4) {
                        auto offset = unsigned(record.number(at));
                        cubin_check(offset < text.size && offset % 16 == 0 && offsets.insert(offset).second,
                            "invalid internal warp instruction offset");
                        auto lo = input.number(text.offset + offset, 8);
                        auto hi = input.number(text.offset + offset + 8, 8) & ((uint64_t(1) << 41) - 1);
                        cubin_check((lo & 0xfff) == 0x806 && !(lo >> 24) && (hi & ~uint64_t(0x78e0000)) == 0x100,
                            "unsupported internal warp instruction", true);
                    }
                } else if (attr == 0x28 || attr == 0x29) {
                    auto& count = attr == 0x28 ? cooperative_offsets : cooperative_masks;
                    cubin_check(format == 4 && size % 4 == 0 && !count, "invalid cooperative instruction metadata");
                    count.emplace();
                    for (unsigned at = 0; at < size; at += 4) {
                        auto value = record.number(at);
                        count->push_back(unsigned(value));
                        if (attr == 0x28) {
                            cubin_check(value < text.size && value % 16 == 0, "invalid cooperative instruction offset");
                            auto opcode = input.number(text.offset + value, 2) & 0xfff;
                            cubin_check(opcode == 0x389 || opcode == 0x589 || opcode == 0x989 || opcode == 0xf89 ||
                                opcode == 0x806 || opcode == 0x348 || opcode == 0x948 || opcode == 0x918,
                                "invalid cooperative instruction opcode");
                        }
                    }
                } else cubin_fail("unsupported kernel information attribute " + std::to_string(attr), true);
            });
            cubin_check(bank && extent, "missing parameter metadata");
            cubin_check(bool(cooperative_offsets) == bool(cooperative_masks) &&
                (!cooperative_offsets || cooperative_offsets->size() == cooperative_masks->size()), "inconsistent cooperative instruction metadata");
            if (cooperative_offsets) for (unsigned i = 0; i < cooperative_offsets->size(); ++i) {
                auto lo = input.number(text.offset + (*cooperative_offsets)[i], 8);
                unsigned mask = (*cooperative_masks)[i];
                if ((lo & 0xfff) == 0x918) {
                    // CUDA 12.4 can encode warp sync as a NOP carrying its
                    // mask in these paired records. It is not padding.
                    cubin_check(mask == UINT32_MAX || ((mask & 0xffffff00u) == 0x05000000u && (mask & 255) != 1),
                        "unsupported cooperative mask register", true);
                    cubin_check(kernel.warp_barriers.emplace((*cooperative_offsets)[i], mask).second, "duplicate cooperative instruction offset");
                } else {
                    auto expected = (lo & 0xfff) == 0x348 ? 0x05000000u | (unsigned(lo >> 32) & 255) : UINT32_MAX;
                    cubin_check(mask == expected, "unsupported cooperative mask register", true);
                }
            }
            unsigned end = 0;
            for (auto [ordinal, parameter] : layout) {
                cubin_check(ordinal == kernel.parameters.size() && parameter.offset == ((end + parameter.size - 1) & ~(parameter.size - 1)), "invalid scalar parameter layout");
                kernel.parameters.push_back(parameter); end = parameter.offset + parameter.size;
            }
            cubin_check(end == kernel.size && end == declared_size, "parameter extent mismatch");
            kernels.push_back(std::move(kernel));
        }
    }
    cubin_check(!kernels.empty(), "no kernel symbols");
    for (const auto& section : input.sections) if (section.name.starts_with(".text."))
        cubin_check(names.contains(section.name.substr(6)), "text section lacks a kernel symbol");
    for (const auto& [table, symbols] : input.symbols) for (const auto& symbol : symbols) {
        if (symbol.type != 2 || (symbol.other & 0x10)) continue;
        cubin_check(symbol.section < input.sections.size(), "invalid device function section");
        auto owner = std::find_if(kernels.begin(), kernels.end(), [&](const auto& kernel) { return kernel.text == &input.sections[symbol.section]; });
        cubin_check(owner != kernels.end() && symbol.binding == 0 && symbol.other == 0, "unsupported device function binding", true);
        cubin_check(cubin_symbol_spelling(symbol.name), "unsupported device function spelling", true);
        cubin_check(symbol.offset && symbol.offset % 16 == 0 && symbol.size && symbol.size % 16 == 0 &&
            symbol.offset < owner->text->size && symbol.size <= owner->text->size - symbol.offset, "invalid device function extent");
        cubin_check(owner->functions.emplace(size_t(symbol.offset), cubin_function{symbol.name, size_t(symbol.offset), size_t(symbol.size)}).second,
            "duplicate device function entry");
    }
    for (const auto& kernel : kernels) {
        size_t end = kernel.functions.empty() ? 0 : kernel.functions.begin()->first;
        for (const auto& [offset, function] : kernel.functions) {
            cubin_check(offset >= end, "overlapping device functions"); end = offset + function.size;
        }
    }
    for (const auto& section : input.sections) if (section.name.starts_with(".nv.shared.")) {
        auto found = std::find_if(kernels.begin(), kernels.end(), [&](const auto& kernel) { return kernel.name == section.name.substr(11); });
        cubin_check(found != kernels.end() && section.info < input.sections.size() && &input.sections[section.info] == found->text,
            "invalid static shared owner");
        found->shared_size = unsigned(section.size); found->shared_alignment = unsigned(section.alignment);
        found->shared_storage = true;
    }
    input.records(input.named(".nv.info"), [&](unsigned format, unsigned attr, unsigned size, const fatbin_reader& record) {
        cubin_check(format == 4 && size == 8 && (attr == 0x2f || attr == 0x11 || attr == 0x12), "unsupported module information", true);
        auto index = record.number(0);
        cubin_check(input.symbols.size() == 1 && index < input.symbols.begin()->second.size(), "invalid function resource symbol");
        const auto& symbol = input.symbols.begin()->second[size_t(index)];
        cubin_check(symbol.type == 2, "invalid function resource owner");
        if (attr != 0x2f) {
            auto owner = std::find_if(kernels.begin(), kernels.end(), [&](const auto& kernel) { return symbol.section < input.sections.size()
                && kernel.text == &input.sections[symbol.section]; });
            cubin_check(owner != kernels.end(), "invalid stack owner");
            cubin_check(record.number(4) <= 16384, "machine stack exceeds private budget", true);
            owner->stack_size = std::max(owner->stack_size, unsigned(record.number(4)));
        }
    });
    for (const auto& kernel : kernels) if (!kernel.functions.empty()) {
        std::map<size_t, std::set<size_t>> edges;
        auto calls = [&](size_t begin, size_t end) {
            auto& targets = edges[begin];
            for (size_t pc = begin; pc < end; pc += 16) {
                auto lo = input.number(size_t(kernel.text->offset) + pc, 8);
                if ((lo & 0xfff) != 0x944) continue;
                int64_t target = int64_t(pc) + 16 + int32_t(lo >> 32);
                cubin_check(target >= 0 && kernel.functions.contains(size_t(target)), "call target lacks a device function entry", true);
                targets.insert(size_t(target));
            }
        };
        calls(0, kernel.functions.begin()->first);
        for (const auto& [offset, function] : kernel.functions) calls(offset, offset + function.size);
        std::map<size_t, size_t> incoming;
        for (const auto& [offset, targets] : edges) { incoming.try_emplace(offset, 0); for (auto target : targets) ++incoming[target]; }
        std::vector<size_t> ready;
        for (const auto& [offset, count] : incoming) if (!count) ready.push_back(offset);
        size_t complete = 0;
        while (!ready.empty()) {
            auto offset = ready.back(); ready.pop_back(); ++complete;
            for (auto target : edges.at(offset)) if (!--incoming[target]) ready.push_back(target);
        }
        cubin_check(complete == edges.size(), "recursive machine calls require a separate stack contract", true);
    }
    return kernels;
}
std::string lift_cubin_kernel(const cubin_reader& input, const cubin_kernel& kernel, const cubin_function* function = nullptr) {
    size_t start = function ? function->offset : 0;
    size_t extent = function ? function->size : kernel.functions.empty() ? size_t(kernel.text->size) : kernel.functions.begin()->first;
    auto code = input.data(*kernel.text).substr(start, extent); fatbin_reader words{code};
    cubin_check(!code.empty() && code.size() % 16 == 0 && code.size() <= 65536, "invalid/excessive SM75 code extent");
    std::set<size_t> targets;
    auto branch_target = [&](size_t pc, uint64_t lo, uint64_t hi) {
        // BRA.U tags the displacement's low bit; it promises a uniform
        // predicate but retains the same per-invocation control-flow result.
        auto encoded = uint32_t(lo >> 32);
        if ((lo & 0xfff) == 0x947) encoded &= ~1u;
        auto displacement = int32_t(encoded);
        cubin_check((hi & 0x3ffff) == (displacement < 0 ? 0x3ffffu : 0), "unsupported branch displacement", true);
        int64_t target = int64_t(pc) + 16 + displacement;
        cubin_check(target >= 0 && uint64_t(target) < code.size() && target % 16 == 0, "invalid branch target");
        return size_t(target);
    };
    for (size_t pc = 0; pc < code.size(); pc += 16) if ((words.number(pc, 8) & 0xfff) == 0x947)
        targets.insert(branch_target(pc, words.number(pc, 8), words.number(pc + 8, 8)));
    std::set<size_t> reachable;
    std::vector<size_t> pending{0};
    while (!pending.empty()) {
        auto pc = pending.back(); pending.pop_back();
        cubin_check(pc < code.size(), "kernel control flow falls beyond its code extent");
        if (!reachable.insert(pc).second) continue;
        auto lo = words.number(pc, 8), hi = words.number(pc + 8, 8);
        auto op = unsigned(lo & 0xfff), predicate = unsigned(lo >> 12) & 15;
        if (op == 0x947) pending.push_back(branch_target(pc, lo, hi));
        if ((op != 0x947 && op != 0x94d && op != 0x950) || predicate != 7) pending.push_back(pc + 16);
    }
    // Retain the validated lexical token scopes as explicit member masks in
    // common IR. Branch ballots partition the current cohort; BSYNC reunites
    // the saved BSSY cohort through the shared warp-barrier implementation.
    std::map<size_t, size_t> regions;
    bool warp_operations = !kernel.warp_barriers.empty();
    size_t uniform_prefix = code.size();
    for (auto pc : reachable) {
        auto lo = words.number(pc, 8), hi = words.number(pc + 8, 8) & ((uint64_t(1) << 41) - 1);
        unsigned opcode = unsigned(lo & 0xfff);
        warp_operations |= opcode == 0x806 || opcode == 0x389 || opcode == 0x589 ||
            opcode == 0x989 || opcode == 0xf89 || opcode == 0x348 || opcode == 0x948;
        if ((lo & 0xfff) == 0x947) uniform_prefix = std::min({uniform_prefix, pc, branch_target(pc, lo, hi)});
        if ((lo & 0xfff) != 0x945) continue;
        unsigned barrier = unsigned(lo >> 16) & 255;
        cubin_check(uint32_t(lo) == (barrier << 16 | 0x7945) && barrier < 16 &&
            (hi & ~uint64_t(0x3ffff)) == 0x3800000, "unsupported BSSY encoding", true);
        auto end = branch_target(pc, lo, hi);
        cubin_check(end > pc + 16 && words.number(end - 16, 8) == (uint64_t(barrier) << 16 | 0x7941) &&
            (words.number(end - 8, 8) & ((uint64_t(1) << 41) - 1)) == 0x3800000,
            "invalid reconvergence target");
        regions.emplace(pc, end);
    }
    auto scope = [&](size_t pc) {
        std::vector<size_t> active;
        for (const auto& [begin, end] : regions) if (begin < pc && pc < end) active.push_back(begin);
        return active;
    };
    for (auto pc : reachable) {
        auto lo = words.number(pc, 8), hi = words.number(pc + 8, 8) & ((uint64_t(1) << 41) - 1);
        unsigned op = unsigned(lo & 0xfff), predicate = unsigned(lo >> 12) & 15;
        auto active = scope(pc);
        if (op == 0xb1d)
            cubin_check(active.empty(), "CTA barrier inside a reconvergence region is unsupported", true);
        if (op == 0x945) {
            auto barrier = (lo >> 16) & 255;
            for (auto begin : active) cubin_check(((words.number(begin, 8) >> 16) & 255) != barrier,
                "reused active reconvergence barrier");
            active.push_back(pc);
        } else if (op == 0x941) {
            cubin_check(!active.empty() && regions.at(active.back()) == pc + 16 &&
                lo == ((words.number(active.back(), 8) & 0xff0000) | 0x7941) && hi == 0x3800000,
                "invalid reconvergence sync");
            active.pop_back();
        } else if (op == 0x355) {
            auto barrier = (lo >> 24) & 255;
            cubin_check(lo == (barrier << 24 | 0xff7355) && barrier < 16 && hi == 0x100000,
                "unsupported reconvergence clear", true);
            for (auto begin : active) cubin_check(((words.number(begin, 8) >> 16) & 255) != barrier,
                "clearing active reconvergence barrier");
        }
        auto edge = [&](size_t next) { cubin_check(active == scope(next), "branch crosses a reconvergence region"); };
        if (op == 0x947) edge(branch_target(pc, lo, hi));
        if (op == 0x94d || op == 0x950) cubin_check(active.empty(), "exit bypasses reconvergence sync");
        if ((op != 0x947 && op != 0x94d && op != 0x950) || predicate != 7) edge(pc + 16);
    }
    bool calls = !kernel.functions.empty();
    unsigned register_count = unsigned(kernel.text->info >> 24);
    cubin_check(!calls || (register_count >= 2 && register_count <= 255), "invalid call register extent");
    unsigned state_size = (register_count + 63 + 7 + 1) * 4;
    std::string output = function ? ".func (.param .align 4 .b32 machine_result[" + std::to_string(state_size / 4) + "]) " +
        function->name + "(.param .align 4 .b32 machine_registers[" + std::to_string(state_size / 4) + "],.param .b64 machine_stack,"
        ".param .b32 machine_shared,.param .b32 machine_dynamic" : ".visible .entry " + kernel.name + "(";
    for (unsigned i = 0; i < kernel.parameters.size(); ++i)
        output += (i || function ? "," : "") + std::string(".param .b") + std::to_string(kernel.parameters[i].size * 8) + " arg" + std::to_string(i);
    output += "){ .reg .b32 %r<255>,%u<63>,%scratch,%constant,%shared_base,%shared_dynamic,%shared_static_address,%shared_address;"
        " .reg .b64 %a,%b,%c,%bank_address,%local_address,%local_base; .reg .f32 %f<3>; .reg .pred %p<7>,%pt,%shared_is_static;\nsetp.eq.u32 %pt,0,0;\n";
    bool track_cohort = warp_operations || !regions.empty();
    if (calls) for (size_t pc = 0; pc < kernel.text->size; pc += 16) {
        unsigned op = unsigned(input.number(size_t(kernel.text->offset) + pc, 8)) & 0xfff;
        track_cohort |= op == 0x945 || op == 0x806 || op == 0x389 || op == 0x589 || op == 0x989 || op == 0xf89 || op == 0x348 || op == 0x948;
    }
    if (track_cohort || calls) {
        output += ".reg .b32 %cohort,%members,%taken,%fallthrough,%token<16>; .reg .pred %branch_taken;\n";
        if (!function) output += track_cohort ? "activemask.b32 %cohort;\n" : "mov.u32 %cohort,0;\n";
    }
    if (function) output += "ld.param.u64 %local_base,[machine_stack];\n"
        "ld.param.u32 %shared_base,[machine_shared];\nld.param.u32 %shared_dynamic,[machine_dynamic];\n";
    if (!function && kernel.shared_size) output += ".shared .align " + std::to_string(kernel.shared_alignment) + " .b8 shared_storage[" +
        std::to_string(kernel.shared_size) + "];\nmov.u32 %shared_base,shared_storage;\n";
    if (!function && kernel.shared_storage) output += ".extern .shared .align " + std::to_string(std::max(16u, kernel.shared_alignment)) +
        " .b8 dynamic_storage[];\nmov.u32 %shared_dynamic,dynamic_storage;\n";
    if (!function && !kernel.shared_size) output += "mov.u32 %shared_base,0;\n";
    if (!function && !kernel.shared_storage) output += "mov.u32 %shared_dynamic,0;\n";
    if (!function && (kernel.stack_size || calls)) output += ".local .align 16 .b8 machine_stack[" +
        std::to_string(std::max(16u, kernel.stack_size)) + "];\nmov.u64 %local_base,machine_stack;\n";
    auto machine_state = [&](bool load, const std::string& slot) {
        std::string text;
        auto word = [&](const std::string& reg, unsigned offset) {
            auto address = "[" + slot + "+" + std::to_string(offset) + "]";
            text += load ? "ld.param.u32 " + reg + "," + address + ";\n" : "st.param.u32 " + address + "," + reg + ";\n";
        };
        for (unsigned i = 0; i < register_count; ++i) word("%r" + std::to_string(i), i * 4);
        for (unsigned i = 0; i < 63; ++i) word("%u" + std::to_string(i), (register_count + i) * 4);
        for (unsigned i = 0; i < 7; ++i) {
            if (!load) text += "selp.u32 %scratch,1,0,%p" + std::to_string(i) + ";\n";
            word("%scratch", (register_count + 63 + i) * 4);
            if (load) text += "setp.ne.u32 %p" + std::to_string(i) + ",%scratch,0;\n";
        }
        word("%cohort", state_size - 4); return text;
    };
    if (calls) {
        output += ".param .align 4 .b32 call_registers[" + std::to_string(state_size / 4) + "]; .param .align 4 .b32 call_results[" +
            std::to_string(state_size / 4) + "]; .param .b64 call_stack; .param .b32 call_shared; .param .b32 call_dynamic;\n";
        if (function) output += machine_state(true, "machine_registers");
        else {
            for (unsigned i = 0; i < register_count; ++i) output += "mov.u32 %r" + std::to_string(i) + ",0;\n";
            for (unsigned i = 0; i < 63; ++i) output += "mov.u32 %u" + std::to_string(i) + ",0;\n";
            for (unsigned i = 0; i < 7; ++i) output += "setp.eq.u32 %p" + std::to_string(i) + ",0,1;\n";
        }
    }
    std::array<std::optional<uint32_t>, 256> known{}; known[255] = 0;
    auto reg = [&](unsigned id) {
        cubin_check(id <= 255 && (id != 1 || kernel.stack_size || calls), "stack-register use is unsupported", true);
        cubin_check(!calls || id == 255 || id < register_count, "register exceeds machine call ABI extent");
        return id == 255 ? std::string("0") : "%r" + std::to_string(id);
    };
    uint64_t uniform_defined = function ? UINT64_MAX : uint64_t(1) << 63;
    auto uniform = [&](unsigned id, bool write = false) {
        cubin_check(id <= 63 && (!write || id != 63), "invalid uniform register");
        if (write) uniform_defined |= uint64_t(1) << id;
        else cubin_check((uniform_defined & (uint64_t(1) << id)) != 0, "uninitialized uniform register", true);
        return id == 63 ? std::string("0") : "%u" + std::to_string(id);
    };
    auto pred = [](unsigned id) { return id == 7 ? std::string("%pt") : "%p" + std::to_string(id); };
    auto constant_load = [&](const std::string& dest, const std::string& next, unsigned offset, unsigned size, const std::string& guard,
            unsigned bank = 0, const std::string& index = "") {
        if (bank) {
            auto name = ".nv.constant" + std::to_string(bank);
            auto local = std::find_if(input.sections.begin(), input.sections.end(), [&](const auto& section) { return section.name == name + "." + kernel.name; });
            if (local == input.sections.end()) local = std::find_if(input.sections.begin(), input.sections.end(), [&](const auto& section) { return section.name == name; });
            cubin_check(local != input.sections.end(), "unsupported constant-bank operand", true);
            const auto& section = *local;
            auto section_index = size_t(&section - input.sections.data());
            cubin_check(input.storage.contains(section_index) && offset <= section.size && size <= section.size - offset,
                "constant load exceeds storage");
            std::string result = guard + "mov.u64 %bank_address," + input.storage.at(section_index) + ";\n" + guard +
                "add.u64 %bank_address,%bank_address," + std::to_string(offset) + ";\n";
            if (!index.empty()) result += guard + "cvt.u64.u32 %a," + index + ";\n" + guard + "add.u64 %bank_address,%bank_address,%a;\n";
            if (size == 4) return result + guard + "ld.const.b32 " + dest + ",[%bank_address];\n";
            return result + guard + "ld.const.b64 %a,[%bank_address];\n" + guard + "cvt.u32.u64 " + dest + ",%a;\n" +
                guard + "shr.u64 %a,%a,32;\n" + guard + "cvt.u32.u64 " + next + ",%a;\n";
        }
        // CUDA 12.4's SM75 launch ABI stores block and grid dimensions before
        // the parameter area. Never treat other runtime-owned bank bytes as
        // ordinary constants from the (zero-filled) ELF section.
        if (size == 4 && offset < 24 && offset % 4 == 0) {
            const char* dimensions[]{"%ntid.x", "%ntid.y", "%ntid.z", "%nctaid.x", "%nctaid.y", "%nctaid.z"};
            return guard + "mov.u32 " + dest + "," + dimensions[offset / 4] + ";\n";
        }
        cubin_check(offset >= kernel.base, "unsupported constant-bank read", true); offset -= kernel.base;
        for (unsigned i = 0; i < kernel.parameters.size(); ++i) {
            auto parameter = kernel.parameters[i];
            if (offset >= parameter.offset && offset - parameter.offset <= parameter.size && size <= parameter.size - (offset - parameter.offset)) {
                cubin_check(offset % size == 0, "unaligned constant load");
                auto address = "[arg" + std::to_string(i) + "+" + std::to_string(offset - parameter.offset) + "]";
                if (size == 4) return guard + "ld.param.b32 " + dest + "," + address + ";\n";
                return guard + "ld.param.b64 %a," + address + ";\n" + guard + "cvt.u32.u64 " + dest + ",%a;\n" +
                    guard + "shr.u64 %b,%a,32;\n" + guard + "cvt.u32.u64 " + next + ",%b;\n";
            }
        }
        cubin_fail("constant read exceeds a parameter", true);
    };
    for (size_t pc = 0; pc < code.size(); pc += 16) {
        if (targets.contains(pc)) { known.fill({}); known[255] = 0; }
        auto lo = words.number(pc, 8), hi = words.number(pc + 8, 8) & ((uint64_t(1) << 41) - 1);
        unsigned op = unsigned(lo & 0xfff), predicate = unsigned(lo >> 12) & 15, dest = unsigned(lo >> 16) & 255,
            left = unsigned(lo >> 24) & 255, right = unsigned(lo >> 32) & 255;
        auto fail = [&](const std::string& reason) { cubin_fail(kernel.name + "+" + std::to_string(pc) + ": " + reason, true); };
        // Only the acyclic entry prefix may define uniform state in a branched
        // kernel. Those definitions dominate every later read and cannot be
        // revisited by a loop or overwritten by a divergent arm.
        if (pc >= uniform_prefix && (op == 0xab9 || op == 0x9c3 || op == 0x2a4 || op == 0x899))
            fail("uniform writes across branches are unsupported");
        std::string guard = predicate == 7 ? "" : "@" + std::string(predicate & 8 ? "!" : "") + pred(predicate & 7) + " ";
        auto bank_index = [&] {
            if (lo >> 59) fail("unsupported constant-bank operand");
            return unsigned(lo >> 54) & 31;
        };
        auto bank_offset = [&] {
            (void)bank_index();
            return unsigned(lo >> 40 & 0x3fff) * 4;
        };
        auto bank_word = [&] {
            if (right) fail("unsupported indexed constant-bank operand");
            output += constant_load("%constant", "", bank_offset(), 4, guard, bank_index());
            return std::string("%constant");
        };
        auto join = [&](const std::string& low, const std::string& high, const std::string& name) {
            output += guard + "cvt.u64.u32 " + name + "," + low + ";\n" + guard + "cvt.u64.u32 %c," + high +
                ";\n" + guard + "shl.b64 %c,%c,32;\n" + guard + "or.b64 " + name + "," + name + ",%c;\n";
        };
        auto shared_address = [&](unsigned shift) {
            if (!kernel.shared_storage) fail("missing shared storage");
            output += guard + (shift ? "shl.b32 %shared_address," : "mov.u32 %shared_address,") + reg(left) +
                (shift ? "," + std::to_string(shift) + ";\n" : ";\n");
            auto displacement = int32_t(lo >> 32) >> 8;
            output += guard + "add.u32 %shared_address,%shared_address," + std::to_string(uint32_t(displacement)) + ";\n";
            // SASS addresses one contiguous static/dynamic segment. Common IR
            // may place synchronization scratch between those allocations, so
            // translate each side separately instead of assuming adjacency.
            if (kernel.shared_size) {
                output += guard + "setp.lt.u32 %shared_is_static,%shared_address," + std::to_string(kernel.shared_size) + ";\n" +
                    guard + "add.u32 %shared_static_address,%shared_base,%shared_address;\n";
            }
            output += guard + "sub.u32 %shared_address,%shared_address," + std::to_string(kernel.shared_size) + ";\n" +
                guard + "add.u32 %shared_address,%shared_dynamic,%shared_address;\n";
            if (kernel.shared_size) output += guard + "selp.u32 %shared_address,%shared_static_address,%shared_address,%shared_is_static;\n";
        };
        auto warp_members = [&] {
            // SASS participants are the current cohort filtered by instruction
            // predication, not the whole warp or the backend's subgroup.
            auto test = predicate == 7 ? "%pt" : std::string(predicate & 8 ? "!" : "") + pred(predicate & 7);
            output += "vote.sync.ballot.b32 %members," + test + ",%cohort;\n";
        };
        auto memory_transfer = [&](bool load, const std::string& space, unsigned kind, unsigned first, const std::string& address) {
            if (kind > 6) fail("unsupported memory width");
            unsigned lanes = kind < 5 ? 1 : kind == 5 ? 2 : 4;
            if (first != 255 && (first + lanes > 255 || (lanes > 1 && first % lanes))) fail("invalid memory register group");
            if (first == 255 && lanes > 1) fail("invalid memory register group");
            if (load) for (unsigned i = 1; i < lanes; ++i) known[first + i].reset();
            const char* types[]{"u8", "s8", "u16", "s16", "b32", "b32", "b32"};
            std::string registers;
            for (unsigned i = 0; i < lanes; ++i) registers += (i ? "," : "") +
                (load && first == 255 ? std::string("%scratch") : reg(first + i));
            if (lanes > 1) registers = "{" + registers + "}";
            return std::string(load ? "ld." : "st.") + space + (lanes > 1 ? ".v" + std::to_string(lanes) : "") + "." +
                (load || kind >= 4 ? types[kind] : kind < 2 ? "u8" : "u16") + " " +
                (load ? registers + "," + address : address + "," + registers) + ";";
        };
        output += "L" + std::to_string(pc) + ":\n";
        std::string instruction; std::optional<uint32_t> constant;
        auto relocation = std::find_if(input.relocations.begin(), input.relocations.end(), [&](const auto& item) {
            return &input.sections[item.section] == kernel.text && item.offset == start + pc;
        });
        if (relocation != input.relocations.end()) {
            if (!((op == 0x802 && hi == 0xf00 && left == 0) || (op == 0x424 && hi == 0x78e00ff && left == 255)))
                fail("unsupported instruction relocation field");
            output += guard + "mov.u64 %a," + input.storage.at(relocation->target_section) + ";\n" + guard +
                "add.u64 %a,%a," + std::to_string(relocation->target_offset) + ";\n";
            if (relocation->type == 57) output += guard + "shr.u64 %a,%a,32;\n";
            instruction = "cvt.u32.u64 " + reg(dest) + ",%a;";
        } else if (auto barrier = kernel.warp_barriers.find(start + pc); barrier != kernel.warp_barriers.end()) {
            if (lo != 0x7918 || hi != 0) fail("unsupported cooperative warp barrier encoding");
            instruction = "bar.warp.sync " + (barrier->second == UINT32_MAX ? std::to_string(UINT32_MAX) : reg(barrier->second & 255)) + ";";
        } else if (op == 0x202 && hi == 0xf00 && (lo & 0xffffff00ff000000ull) == 0) {
            instruction = "mov.b32 " + reg(dest) + "," + reg(right) + ";"; constant = known[right];
        } else if (op == 0xc02 && hi == 0x8000f00 && !(lo >> 40) && left == 0) {
            instruction = "mov.b32 " + reg(dest) + "," + uniform(right) + ";";
        } else if (op == 0x802 && hi == 0xf00 && left == 0) {
            instruction = "mov.b32 " + reg(dest) + "," + std::to_string(uint32_t(lo >> 32)) + ";"; constant = uint32_t(lo >> 32);
        } else if (op == 0xa02 && hi == 0xf00 && (lo & 0xf80000ffff000000ull) == 0) {
            unsigned offset = bank_offset();
            if (pc == 0 && !function && predicate == 7 && dest == 1 && offset == 0x28 && !bank_index()) {
                if (kernel.stack_size || calls) instruction = "mov.u32 %r1," + std::to_string(kernel.stack_size) + ";";
            } else output += constant_load(reg(dest), "", offset, 4, guard, bank_index());
        } else if (op == 0xb82 && (hi == 0x800 || hi == 0xa00) && (lo & 0xf80000ff00000000ull) == 0) {
            if (!bank_index() && !known[left]) fail("dynamic constant-bank indexing is unsupported");
            unsigned width = hi == 0xa00 ? 8 : 4;
            auto offset = uint64_t(bank_offset()) + (bank_index() ? 0 : *known[left]);
            if (offset > UINT32_MAX) fail("constant-bank offset overflow");
            cubin_check(dest + width / 4 <= 255, "invalid constant destination");
            output += constant_load(reg(dest), width == 8 ? reg(dest + 1) : "", unsigned(offset), width, guard,
                bank_index(), bank_index() ? reg(left) : "");
            if (width == 8) known[dest + 1].reset();
        } else if (op == 0xab9 && (hi == 0x800 || hi == 0xa00) && left == 0 && right == 0 && predicate == 7) {
            unsigned width = hi == 0xa00 ? 8 : 4;
            if (dest + width / 4 > 63 || (width == 8 && dest % 2)) fail("invalid uniform constant destination");
            output += constant_load(uniform(dest, true), width == 8 ? uniform(dest + 1, true) : "", bank_offset(), width, guard, bank_index());
        } else if ((op == 0x919 || op == 0x9c3) && (lo >> 24) == 0 && (hi & ~uint64_t(0xff00)) == 0) {
            unsigned special = unsigned(hi >> 8);
            const char* component = special >= 0x21 && special <= 0x23 ? "%tid." :
                special >= 0x25 && special <= 0x27 ? "%ctaid." : nullptr;
            if (special == 0 && op == 0x919) instruction = "mov.u32 " + reg(dest) + ",%laneid;";
            else {
                if (!component) fail("unsupported special register " + std::to_string(special));
                if (op == 0x9c3 && (special < 0x25 || predicate != 7)) fail("unsupported uniform special register");
                unsigned axis = special >= 0x25 ? special - 0x25 : special - 0x21;
                instruction = "mov.u32 " + (op == 0x9c3 ? uniform(dest, true) : reg(dest)) + "," + component + "xyz"[axis] + ";";
            }
        } else if (op == 0x389 || op == 0x589 || op == 0x989 || op == 0xf89) {
            bool immediate_offset = op == 0x989 || op == 0xf89, immediate_control = op == 0x589 || op == 0xf89;
            uint64_t fields = uint64_t(3) << 58 | (immediate_offset ? uint64_t(31) << 53 : uint64_t(255) << 32) |
                (immediate_control ? uint64_t(0x1f1f) << 40 : 0);
            if ((lo & ~(fields | uint64_t(0xffffffff))) != 0 || (hi & ~uint64_t(immediate_control ? 0xe0000 : 0xe00ff)) != 0)
                fail("unsupported shuffle encoding");
            auto offset = immediate_offset ? std::to_string(unsigned(lo >> 53) & 31) : reg(right);
            auto control = immediate_control ? std::to_string(unsigned(lo >> 40) & 0x1f1f) : reg(unsigned(hi) & 255);
            unsigned destination = unsigned(hi >> 17) & 7;
            const char* modes[]{"idx", "up", "down", "bfly"};
            warp_members();
            instruction = "shfl.sync." + std::string(modes[unsigned(lo >> 58) & 3]) + ".b32 " +
                (dest == 255 ? std::string("_") : reg(dest)) + (destination == 7 ? "" : "|" + pred(destination)) + "," +
                reg(left) + "," + offset + "," + control + ",%members;";
        } else if (op == 0x806) {
            // VOTE.ANY writes a ballot and optionally its any predicate.
            if ((lo >> 24) != 0 || (hi & ~uint64_t(0x78e0000)) != 0x100)
                fail("unsupported vote encoding");
            warp_members();
            auto test = std::string(hi & 0x4000000 ? "!" : "") + pred(unsigned(hi >> 23) & 7);
            unsigned destination = unsigned(hi >> 17) & 7;
            auto target = dest == 255 ? std::string("%scratch") : reg(dest);
            // Compute before writing: the source and result predicates may alias.
            output += guard + "vote.sync.ballot.b32 %scratch," + test + ",%members;\n";
            instruction = "mov.b32 " + target + ",%scratch;";
            if (destination != 7) {
                // The result predicate can also be this instruction's guard.
                // Publish the register result before changing that guard.
                output += guard + instruction + "\n";
                instruction = "setp.ne.u32 " + pred(destination) + ",%scratch,0;";
            }
        } else if (op == 0x348 || op == 0x948) {
            if ((uint32_t(lo) & 0xffff0000u) != 0 || (op == 0x348 && (lo >> 40)) || hi != 0x3800000)
                fail("unsupported warp barrier encoding");
            instruction = "bar.warp.sync " + (op == 0x948 ? std::to_string(uint32_t(lo >> 32)) : reg(right)) + ";";
        } else if (op == 0x992) {
            if ((lo >> 16) || (hi & ~uint64_t(0xa000))) fail("unsupported memory fence encoding");
            instruction = std::string("fence.") + (hi & 0x8000 ? "acq_rel." : "sc.") + (hi & 0x2000 ? "gpu;" : "cta;");
        } else if (op == 0x98f) {
            if ((lo >> 16) != 0xff00 || hi != 0x2000000) fail("unsupported cache invalidation encoding");
            // The acquire half of SM75 device-scope sequences. Keep a fence
            // even when control flow separates the strong access from IVALL.
            instruction = "fence.acq_rel.gpu;";
        } else if (op == 0x9ab) {
            if ((lo >> 16) || hi) fail("unsupported error barrier encoding");
            // ERRBAR completes the GPU release sequence's error observation;
            // common execution observes faults at dispatch completion. Retain
            // its ordering conservatively in the shared memory representation.
            instruction = "fence.acq_rel.gpu;";
        } else if (op == 0x946) {
            if ((lo >> 16) || hi != 0x3800000) fail("unsupported yield encoding");
            // A scheduling hint in ptxas polling loops. Continuations already
            // provide lane progress; the hint has no memory-ordering effect.
        } else if (op == 0x381 || op == 0x981 || op == 0x986 || op == 0x386) {
            bool uniform_address = op == 0x981 || op == 0x986, load = op == 0x381 || op == 0x981;
            auto mode = hi & 0xc000000;
            auto strength = hi & 0xe000;
            bool strong = strength == 0 || strength == 0x4000;
            auto modifiers = uint64_t(0xe00) | (strong ? uint64_t(0xe000) : 0) |
                (uniform_address ? uint64_t(0xc000000) | (load ? 0 : 63) : 0);
            if ((hi & ~modifiers) != (strong ? (load ? 0x1f0100u : 0x110100u) : (load ? 0x1ee100u : 0x10e100u)) ||
                (uniform_address && mode != 0x8000000 && mode != 0xc000000)) fail("unsupported global memory encoding");
            bool address_pair = !uniform_address || mode == 0xc000000;
            if (address_pair && left >= 254 && !(uniform_address && left == 255)) fail("unsupported extended address register");
            if (!load && dest != 0) fail("unsupported global store modifier");
            if (left == 255) output += guard + "mov.u64 %a,0;\n";
            else if (address_pair) join(reg(left), reg(left + 1), "%a");
            else output += guard + "cvt.u64.u32 %a," + reg(left) + ";\n";
            if (uniform_address) {
                auto base = load ? right : unsigned(hi) & 63;
                if (base >= 62 || base % 2) fail("invalid uniform address pair");
                join(uniform(base), uniform(base + 1), "%b");
                output += guard + "add.u64 %a,%a,%b;\n";
            }
            // STG puts the source in bits 32..39 and its signed displacement
            // in bits 40..63; LDG uses the full high word for displacement.
            auto displacement = load && !uniform_address ? int32_t(lo >> 32) : int32_t(lo >> 32) >> 8;
            auto address = "[%a" + std::string(displacement < 0 ? "-" : "+") + std::to_string(displacement < 0 ? -int64_t(displacement) : displacement) + "]";
            if (strong) {
                if (((hi >> 9) & 7) != 4) fail("unsupported strong memory width/address");
                // Strong CTA reads provide the acquire behavior for which
                // ptxas emits no separate IVALL. GPU reads pair with IVALL.
                auto scope = strength ? "gpu.global" : "cta.global";
                instruction = std::string(load ? (strength ? "ld.relaxed." : "ld.acquire.") : "st.relaxed.") + scope + ".u32 " +
                    (load ? (dest == 255 ? std::string("%scratch") : reg(dest)) + "," + address : address + "," + reg(right)) + ";";
            } else instruction = memory_transfer(load, "global", unsigned(hi >> 9) & 7, load ? dest : right, address);
        } else if (op == 0x984 || op == 0x388) {
            bool load = op == 0x984;
            if ((hi & ~uint64_t(0xce00)) != (load ? 0x1000u : 0) ||
                (load ? right != 0 : dest != 0))
                fail("unsupported shared memory encoding");
            auto scale = unsigned(hi >> 14) & 3;
            shared_address(scale ? scale + 1 : 0);
            instruction = memory_transfer(load, "shared", unsigned(hi >> 9) & 7, load ? dest : right, "[%shared_address]");
        } else if (op == 0x983 || op == 0x387) {
            bool load = op == 0x983;
            if ((hi & ~uint64_t(0xe00)) != 0x100000 || (!load && dest != 0) || (!kernel.stack_size && !calls))
                fail("unsupported local memory encoding");
            auto displacement = load ? int32_t(lo >> 32) : int32_t(lo >> 32) >> 8;
            output += guard + "add.u32 %scratch," + reg(left) + "," + std::to_string(uint32_t(displacement)) + ";\n" +
                guard + "cvt.u64.u32 %local_address,%scratch;\n" + guard + "add.u64 %local_address,%local_base,%local_address;\n";
            instruction = memory_transfer(load, "local", unsigned(hi >> 9) & 7, load ? dest : right, "[%local_address]");
        } else if (op == 0x38c || op == 0x38d) {
            bool cas = op == 0x38d;
            if ((hi & ~uint64_t(0x40ff)) != 0 || (!cas && (hi & 255) != 255))
                fail("unsupported shared atomic encoding");
            shared_address(hi & 0x4000 ? 2 : 0);
            instruction = cas ? "atom.relaxed.cta.shared.cas.b32 " + (dest == 255 ? std::string("%scratch") : reg(dest)) +
                ",[%shared_address]," + reg(right) + "," + reg(unsigned(hi) & 255) + ";" :
                dest == 255 ? "red.relaxed.cta.shared.add.u32 [%shared_address]," + reg(right) + ";" :
                "atom.relaxed.cta.shared.add.u32 " + reg(dest) + ",[%shared_address]," + reg(right) + ";";
        } else if (op == 0x3a8 || op == 0x9a8 || op == 0x3a9 || op == 0x98e) {
            bool cas = op == 0x3a9, reduction = op == 0x98e;
            bool cta = !(hi & 0x4000);
            auto scoped = hi | 0x4000;
            bool uniform_address = op == 0x9a8 || (reduction && (hi & 0xc000000));
            bool valid = uniform_address ?
                (scoped & ~uint64_t(63)) == (reduction ? 0xc114100u : 0x81f4140u) && left == 255 :
                cas ? (scoped & ~uint64_t(255)) == 0x1f4100 : scoped == (reduction ? 0x114100u : 0x1f41ffu);
            if (!valid || (reduction && dest != 0) || (!uniform_address && (left >= 254 || left % 2)))
                fail("unsupported global atomic encoding");
            if (uniform_address) {
                auto base = unsigned(hi) & 63;
                if (base >= 62 || base % 2) fail("invalid uniform atomic address pair");
                join(uniform(base), uniform(base + 1), "%a");
            } else join(reg(left), reg(left + 1), "%a");
            auto displacement = int32_t(lo >> 32) >> 8;
            auto address = "[%a" + std::string(displacement < 0 ? "-" : "+") +
                std::to_string(displacement < 0 ? -int64_t(displacement) : displacement) + "]";
            auto atomic = std::string(cta ? "atom.acquire.cta.global." : "atom.relaxed.gpu.global.");
            instruction = cas ? atomic + "cas.b32 " + (dest == 255 ? std::string("%scratch") : reg(dest)) +
                "," + address + "," + reg(right) + "," + reg(unsigned(hi) & 255) + ";" :
                reduction ? std::string(cta ? "red.relaxed.cta.global.add.u32 " : "red.relaxed.gpu.global.add.u32 ") + address + "," + reg(right) + ";" :
                atomic + "add.u32 " + (dest == 255 ? std::string("%scratch") : reg(dest)) + "," + address + "," + reg(right) + ";";
        } else if (op == 0xb1d) {
            if (lo != 0x7b1d || hi != 0 || kernel.text->flags != 0x100006)
                fail("unsupported full-block barrier encoding");
            instruction = "bar.sync 0;";
        } else if (op == 0x216 || op == 0x816 || op == 0xa16) {
            if ((hi & ~uint64_t(255)) != 0 || (op == 0x216 && (lo >> 40))) fail("unsupported byte permutation encoding");
            auto control = op == 0xa16 ? bank_word() : op == 0x816 ? std::to_string(uint32_t(lo >> 32)) : reg(right);
            instruction = "prmt.b32 " + reg(dest) + "," + reg(left) + "," + reg(unsigned(hi) & 255) + "," + control + ";";
        } else if ((op == 0x212 || op == 0x812 || op == 0xa12) &&
                   (hi & ~uint64_t(0xeffff)) == 0x7800000u &&
                   (op == 0x812 || op == 0xa12 || !(lo >> 40))) {
            auto second = op == 0xa12 ? bank_word() :
                op == 0x812 ? std::to_string(uint32_t(lo >> 32)) : reg(right);
            output += guard + "lop3.b32 %scratch," + reg(left) + "," + second + "," + reg(unsigned(hi) & 255) + "," +
                std::to_string(unsigned(hi >> 8) & 255) + ";\n";
            unsigned destination = unsigned(hi >> 17) & 7;
            instruction = "mov.b32 " + (dest == 255 ? std::string("%scratch") : reg(dest)) + ",%scratch;";
            if (destination != 7) {
                output += guard + instruction + "\n";
                instruction = "setp.ne.u32 " + pred(destination) + ",%scratch,0;";
            }
        } else if ((op == 0x210 || op == 0x810 || op == 0xa10) && (hi & ~uint64_t(0xe0000)) == 0x7f1e0ff &&
                   ((hi >> 17) & 7) != 7 && predicate == 7 && (op != 0x210 || !(lo >> 40))) {
            // IADD3 Rd, Pn, Ra, Rb/imm32, RZ: retain the carry used by the
            // following IADD3.X when ptxas forms a 64-bit address.
            auto source = op == 0xa10 ? bank_word() : op == 0x210 ? reg(right) : std::to_string(uint32_t(lo >> 32));
            output += "cvt.u64.u32 %a," + reg(left) + ";\ncvt.u64.u32 %b," + source + ";\nadd.u64 %a,%a,%b;\n"
                "setp.gt.u64 " + pred(unsigned(hi >> 17) & 7) + ",%a,4294967295;\n";
            instruction = "cvt.u32.u64 " + reg(dest) + ",%a;";
        } else if ((op == 0x210 || op == 0xa10) && (hi & ~uint64_t(0x3800000)) == 0x7fe4ff && (op != 0x210 || !(lo >> 40))) {
            // IADD3.X Rd, Ra, Rb, RZ, Pn, !PT.
            auto source = op == 0xa10 ? bank_word() : reg(right);
            output += guard + "selp.u32 %scratch,1,0," + pred(unsigned(hi >> 23) & 7) + ";\n" + guard + "add.u32 %scratch,%scratch," + reg(left) + ";\n";
            instruction = "add.u32 " + reg(dest) + ",%scratch," + source + ";";
        } else if ((op == 0x210 || op == 0x810 || op == 0xa10) && (hi & ~uint64_t(0x1ff)) == 0x7ffe000) {
            if (op == 0x210 && (lo >> 40)) fail("unsupported IADD3 register modifier");
            auto source = op == 0xa10 ? bank_word() : op == 0x210 ? reg(right) : std::to_string(uint32_t(lo >> 32));
            auto first = reg(left);
            if (hi & 0x100) { output += guard + "sub.u32 %scratch,0," + first + ";\n"; first = "%scratch"; }
            output += guard + "add.u32 %scratch," + first + "," + source + ";\n";
            instruction = "add.u32 " + reg(dest) + ",%scratch," + reg(unsigned(hi) & 255) + ";";
        } else if (op == 0x2a4 && (hi & ~uint64_t(63)) == 0xf8e0200 && !(lo >> 40) && predicate == 7) {
            // Uniform values retain a distinct register file. Representing each
            // value per invocation preserves the checked, unpredicated forms.
            auto a = uniform(left), b = uniform(right), c = uniform(unsigned(hi) & 63);
            instruction = "mad.lo.u32 " + uniform(dest, true) + "," + a + "," + b + "," + c + ";";
        } else if (op == 0x224 || op == 0x424 || op == 0x824 || op == 0xa24 || op == 0xc24 || op == 0xe24 || op == 0x624 ||
                   op == 0x225 || op == 0x825 || op == 0xa25 || op == 0x625 || op == 0x227 || op == 0x827) {
            bool wide = (op & 15) == 5, high_product = (op & 15) == 7;
            unsigned form = op & 0xe00, addend = unsigned(hi) & 255;
            uint64_t expected = form == 0xc00 || form == 0xe00 ? 0xf8e0000 : 0x78e0000;
            if ((hi & ~uint64_t(0x2ff)) != expected || ((wide || high_product) && (hi & 0x200)))
                fail("unsupported IMAD modifier");
            if ((form == 0x200 || form == 0xc00 || form == 0xe00) && (lo >> 40)) fail("unsupported IMAD register modifier");
            if (high_product && addend != 255) fail("unsupported IMAD high-product addend");
            if (pc == 0 && !function && predicate == 7 && dest == 1 && left == 255 && form == 0x600 && addend == 255 &&
                    bank_offset() == 0x28 && !bank_index()) {
                if (kernel.stack_size || calls) instruction = "mov.u32 %r1," + std::to_string(kernel.stack_size) + ";";
            } else {
                auto source = form == 0xa00 ? bank_word() : form == 0x400 ? reg(addend) : form == 0x800 ? std::to_string(uint32_t(lo >> 32)) :
                    form == 0xc00 ? uniform(right) : form == 0x600 || form == 0xe00 ? reg(addend) : reg(right);
                if (wide) {
                    if (dest >= 254 || dest % 2) fail("invalid wide IMAD destination");
                    if (form == 0x600) {
                        if (right) fail("unsupported indexed constant-bank operand");
                        output += constant_load("%constant", "%scratch", bank_offset(), 8, guard, bank_index());
                        join("%constant", "%scratch", "%b");
                    } else if (addend == 255) output += guard + "mov.u64 %b,0;\n";
                    else {
                        if (addend >= 254 || addend % 2) fail("invalid wide IMAD addend");
                        join(reg(addend), reg(addend + 1), "%b");
                    }
                    output += guard + "mul.wide.u32 %a," + reg(left) + "," + source + ";\n" + guard + "add.u64 %a,%a,%b;\n";
                    instruction = "cvt.u32.u64 " + reg(dest) + ",%a;";
                    // Split only after reading every source, including aliased
                    // input/output pairs used by NVCC's address calculations.
                    output += guard + instruction + "\n" + guard + "shr.u64 %a,%a,32;\n";
                    instruction = "cvt.u32.u64 " + reg(dest + 1) + ",%a;";
                    known[dest + 1].reset();
                } else {
                    auto third = form == 0x600 ? bank_word() : form == 0xe00 ? uniform(right) : form == 0x400 ? std::to_string(uint32_t(lo >> 32)) : reg(addend);
                    instruction = std::string(high_product ? "mad.hi.u32 " : "mad.lo.u32 ") + reg(dest) + "," + reg(left) + "," + source + "," + third + ";";
                }
            }
        } else if (op == 0xa11) {
            // Checked unsigned index*4 + bank pointer pair, with P0 carry.
            bool high = hi == 0xf14ff;
            if ((!high && hi != 0x78010ff) || right || predicate != 7)
                fail("unsupported LEA modifier");
            output += "cvt.u64.u32 %a," + reg(left) + ";\nshl.b64 %a,%a,2;\n";
            if (high) {
                output += guard + "shr.u64 %a,%a,32;\n" + guard + "selp.u32 %scratch,1,0,%p0;\n" +
                    guard + "cvt.u32.u64 %constant,%a;\n" + guard + "add.u32 %scratch,%scratch,%constant;\n";
                auto source = bank_word();
                instruction = "add.u32 " + reg(dest) + ",%scratch," + source + ";";
            } else {
                auto source = bank_word();
                output += guard + "cvt.u32.u64 %scratch,%a;\n" + guard + "cvt.u64.u32 %a,%scratch;\n" +
                    guard + "cvt.u64.u32 %b," + source + ";\n" + guard + "add.u64 %a,%a,%b;\n";
                output += guard + "setp.gt.u64 %p0,%a,4294967295;\n";
                instruction = "cvt.u32.u64 " + reg(dest) + ",%a;";
            }
        } else if (op == 0x211 && (hi & ~uint64_t(0xf800)) == 0x78e00ff && !(lo >> 40)) {
            unsigned shift = unsigned(hi >> 11) & 31;
            output += guard + "shl.b32 %scratch," + reg(left) + "," + std::to_string(shift) + ";\n";
            instruction = "add.u32 " + reg(dest) + ",%scratch," + reg(right) + ";";
        } else if ((op == 0x819 || op == 0x899) && left == (op == 0x899 ? 63u : 255u) &&
                   (hi & ~uint64_t(255)) == (op == 0x899 ? 0x8011600u : 0x11600u) && uint32_t(lo >> 32) < 32) {
            unsigned source = unsigned(hi) & 255;
            if (op == 0x899 && predicate != 7) fail("unsupported uniform shift predicate");
            auto value = op == 0x899 ? uniform(source) : reg(source);
            instruction = "shr.u32 " + (op == 0x899 ? uniform(dest, true) : reg(dest)) + "," + value + "," + std::to_string(uint32_t(lo >> 32)) + ";";
        } else if (op == 0x819 && hi == 0x6ff && uint32_t(lo >> 32) < 32) {
            instruction = "shl.b32 " + reg(dest) + "," + reg(left) + "," + std::to_string(uint32_t(lo >> 32)) + ";";
        } else if (op == 0x305 || op == 0x306 || op == 0x307) {
            bool to_integer = op == 0x305, from_integer = op == 0x306;
            auto signed_bit = to_integer ? uint64_t(0x100) : from_integer ? uint64_t(0x400) : 0;
            if ((lo >> 40) || left || (hi & ~(uint64_t(0xc000) | signed_bit)) != (to_integer ? 0x203000u : 0x201000u))
                fail("unsupported numeric conversion encoding");
            const char* modes[]{"rn", "rm", "rp", "rz"};
            auto mode = std::string(modes[(hi >> 14) & 3]);
            auto integer = hi & signed_bit ? "s32" : "u32";
            auto destination = dest == 255 ? std::string("%scratch") : reg(dest);
            if (!from_integer) output += guard + "mov.b32 %f0," + reg(right) + ";\n";
            output += guard + "cvt." + mode + (from_integer ? ".f32." + std::string(integer) + " %f0," + reg(right) :
                "i." + std::string(to_integer ? integer : "f32") + ".f32 " + (to_integer ? destination : "%f0") + ",%f0") + ";\n";
            if (!to_integer) instruction = "mov.b32 " + destination + ",%f0;";
        } else if (op == 0x221 || op == 0x421 || op == 0xa21 || op == 0x220 || op == 0x820 || op == 0xa20 || op == 0x223 || op == 0x823 || op == 0xa23) {
            bool add = op == 0x221 || op == 0x421 || op == 0xa21, multiply = op == 0x220 || op == 0x820 || op == 0xa20;
            bool bank = (op & 0xe00) == 0xa00;
            bool immediate = op == 0x421 || op == 0x820 || op == 0x823;
            uint64_t modifiers = 0x10000 | (add ? 0x300 : multiply ? 0 : 255);
            if ((hi & ~modifiers) != (multiply ? 0x400000u : 0u) ||
                (!immediate && !bank && ((lo >> 40) & ~(add ? uint64_t(0xc00000) : uint64_t(0)))))
                fail("unsupported FP32 arithmetic modifier");
            auto second = bank ? bank_word() : immediate ? std::to_string(uint32_t(lo >> 32)) : reg(right);
            auto source = [&](unsigned slot, const std::string& bits, bool absolute = false, bool negate = false) {
                auto name = "%f" + std::to_string(slot);
                output += guard + "mov.b32 " + name + "," + bits + ";\n";
                if (absolute) output += guard + "abs.f32 " + name + "," + name + ";\n";
                if (negate) output += guard + "neg.f32 " + name + "," + name + ";\n";
            };
            source(0, reg(left), add && (hi & 0x200), add && (hi & 0x100));
            source(1, second, add && !immediate && !bank && (lo & (uint64_t(1) << 62)), add && !immediate && !bank && (lo & (uint64_t(1) << 63)));
            if (!add && !multiply) source(2, reg(unsigned(hi) & 255));
            output += guard + std::string(add ? "add" : multiply ? "mul" : "fma") + ".rn" +
                (hi & 0x10000 ? ".ftz" : "") + ".f32 %f0,%f0,%f1" + (!add && !multiply ? ",%f2" : "") + ";\n";
            instruction = "mov.b32 " + reg(dest) + ",%f0;";
        } else if (op == 0x20b && dest == 0 && !(lo >> 40)) {
            // Checked register FSETP with AND PT and a discarded complementary
            // destination. Predicate register numbers are independent of Rn.
            unsigned comparison = unsigned(hi >> 12) & 15, destination = unsigned(hi >> 17) & 7;
            if ((hi & ~uint64_t(0xff000)) != 0x3f00000 || comparison == 0 || comparison == 15 || destination == 7)
                fail("unsupported FP32 comparison modifier");
            const char* comparisons[]{"", "lt", "eq", "le", "gt", "ne", "ge", "num", "nan", "ltu", "equ", "leu", "gtu", "neu", "geu"};
            instruction = "setp." + std::string(comparisons[comparison]) + (hi & 0x10000 ? ".ftz" : "") +
                ".f32 " + pred(destination) + "," + reg(left) + "," + reg(right) + ";";
        } else if ((op == 0x208 || op == 0x207 || op == 0x807) && (op == 0x807 || !(lo >> 40)) && (hi & ~uint64_t(0x7800000)) == 0) {
            auto positive = reg(left), negative = op == 0x807 ? std::to_string(uint32_t(lo >> 32)) : reg(right);
            if (hi & 0x4000000) std::swap(positive, negative);
            instruction = "selp.b32 " + reg(dest) + "," + positive + "," + negative + "," + pred(unsigned(hi >> 23) & 7) + ";";
        } else if ((op == 0x20c || op == 0x80c || op == 0xa0c || op == 0xc0c) && dest == 0 && ((op != 0x20c && op != 0xc0c) || !(lo >> 40)) &&
                   (hi & ~uint64_t(0xe7200)) == (op == 0xc0c ? 0xbf00070u : 0x3f00070u)) {
            unsigned comparison = unsigned(hi >> 12) & 7, destination = unsigned(hi >> 17) & 7;
            const char* comparisons[]{"", "lt", "eq", "le", "gt", "ne", "ge", ""};
            if (comparison == 0 || comparison == 7 || destination == 7) fail("unsupported integer comparison modifier");
            auto source = op == 0xa0c ? bank_word() : op == 0x80c ? std::to_string(uint32_t(lo >> 32)) : op == 0xc0c ? uniform(right) : reg(right);
            instruction = "setp." + std::string(comparisons[comparison]) + (hi & 0x200 ? ".s32 " : ".u32 ") + pred(destination) + "," + reg(left) + "," + source + ";";
        } else if (op == 0x81c && !(lo >> 16)) {
            auto form = hi & ~uint64_t(0x38e0000);
            if (form != 0x70e170 && (form & ~uint64_t(0xe000)) != 0x701070) fail("unsupported predicate LUT");
            auto destination = unsigned(hi >> 17) & 7, source = unsigned(hi >> 23) & 7;
            if (destination == 7) fail("unsupported predicate LUT destination");
            instruction = std::string(form == 0x70e170 ? "not.pred " : "and.pred ") + pred(destination) + "," + pred(source) +
                (form == 0x70e170 ? ";" : "," + pred(unsigned(hi >> 13) & 7) + ";");
        } else if (op == 0x944) {
            if (!calls || hi != 0x3c00000 || uint32_t(lo) != 0x7944) fail("unsupported direct call encoding");
            int64_t target = int64_t(start + pc) + 16 + int32_t(lo >> 32);
            if (target < 0 || !kernel.functions.contains(size_t(target))) fail("call target lacks a device function entry");
            output += machine_state(false, "call_registers") + "st.param.b64 [call_stack],%local_base;\n"
                "st.param.b32 [call_shared],%shared_base;\nst.param.b32 [call_dynamic],%shared_dynamic;\ncall (call_results)," +
                kernel.functions.at(size_t(target)).name + ",(call_registers,call_stack,call_shared,call_dynamic";
            for (unsigned i = 0; i < kernel.parameters.size(); ++i) output += ",arg" + std::to_string(i);
            output += ");\n" + machine_state(true, "call_results");
            known.fill({}); known[255] = 0;
        } else if (op == 0x950) {
            auto displacement = int32_t(lo >> 32);
            if (!function || uint32_t(lo) != (uint32_t(left) << 24 | 0x7950) || left >= 254 || left % 2 ||
                hi != (0x3c00000u | (displacement < 0 ? 0x3ffffu : 0)) || int64_t(start + pc) + 16 + displacement != 0)
                fail("unsupported relative return ABI");
            (void)reg(left); (void)reg(left + 1);
            output += machine_state(false, "machine_result"); instruction = "ret;";
        } else if (op == 0x947 && (hi & ~uint64_t(0x3ffff)) == 0x3800000 && (lo & 0xffff0000) == 0) {
            if (track_cohort && predicate != 7 && reachable.contains(pc)) {
                output += std::string(predicate & 8 ? "not.pred " : "and.pred ") + "%branch_taken," + pred(predicate & 7) +
                    (predicate & 8 ? ";\n" : ",%pt;\n");
                output += "vote.sync.ballot.b32 %taken,%branch_taken,%cohort;\nxor.b32 %fallthrough,%cohort,%taken;\n"
                    "selp.b32 %cohort,%taken,%fallthrough,%branch_taken;\n";
            }
            instruction = "bra L" + std::to_string(branch_target(pc, lo, hi)) + ";";
        } else if (op == 0x945 || op == 0x941 || op == 0x355) {
            // Reachable token operations were checked against the CFG above.
            cubin_check(reachable.contains(pc), "unreachable reconvergence instruction", true);
            auto token = "%token" + std::to_string(unsigned(lo >> 16) & 255);
            if (op == 0x945) instruction = "mov.b32 " + token + ",%cohort;";
            if (op == 0x941) instruction = "bar.warp.sync " + token + ";\nmov.b32 %cohort," + token + ";";
        } else if (op == 0x94d && hi == 0x3800000 && !(lo >> 16) && !function) instruction = "ret;";
        else if (op == 0x918 && hi == 0 && !(lo >> 16)) { }
        else fail("unsupported SM75 instruction encoding " + std::to_string(lo) + "/" + std::to_string(hi));
        if (!instruction.empty()) output += guard + instruction + "\n";
        if (dest != 255 && op != 0x386 && op != 0x986 && op != 0x388 && op != 0x387 && op != 0x98e && op != 0xb1d && op != 0x947 && op != 0x944 && op != 0x950 && op != 0x94d && op != 0x918 && op != 0x20c && op != 0x80c && op != 0xa0c && op != 0xc0c && op != 0x20b && op != 0x81c &&
            op != 0xab9 && op != 0x9c3 && op != 0x2a4 && op != 0x899 && op != 0x945 && op != 0x941 && op != 0x355 &&
            op != 0x992 && op != 0x98f && op != 0x9ab && op != 0x946)
            known[dest] = predicate == 7 ? constant : std::nullopt;
    }
    // ptxas pads the section with unreachable NOPs after its exit/stop loop.
    // The reachability check above proves that this synthetic terminator cannot
    // turn a real machine-code fallthrough into a successful return.
    return output + "ret;\n}\n";
}
}
std::unique_ptr<llvm::Module> import_cubin(llvm::LLVMContext& context, std::string_view bytes, std::string_view architecture) {
    cubin_reader input(bytes); auto kernels = cubin_kernels(input);
    std::string semantics = ".version 7.0\n.target sm_75\n.address_size 64\n" + cubin_storage_declarations(input);
    for (const auto& kernel : kernels) {
        semantics += lift_cubin_kernel(input, kernel);
        for (const auto& [offset, function] : kernel.functions) semantics += lift_cubin_kernel(input, kernel, &function);
    }
    // Share register SSA, predication, ABI construction and semantic identities
    // with PTX import. This text is reconstructed from decoded machine code;
    // no PTX member, disassembler executable or NVIDIA tool is used at runtime.
    auto module = import_ptx(context, semantics, architecture);
    materialize_cubin_storage(*module, input);
    for (const auto& kernel : kernels) for (const auto& [offset, function] : kernel.functions) {
        auto* helper = module->getFunction(function.name);
        helper->setLinkage(llvm::GlobalValue::InternalLinkage);
        helper->addFnAttr(llvm::Attribute::AlwaysInline);
    }
    std::string diagnostic; llvm::raw_string_ostream stream(diagnostic);
    if (llvm::verifyModule(*module, &stream)) cubin_fail("invalid lifted storage IR: " + diagnostic);
    module->getOrInsertNamedMetadata("cudavm.import.cubin.sm75");
    return module;
}
}
