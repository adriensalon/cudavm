#include "cubin_import.hpp"
#include "compiler.hpp"
#include "ptx_import.hpp"
#include "shader_packet.hpp"
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
struct cubin_parameter { unsigned offset, size, alignment = 1; };
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
    std::map<size_t, unsigned> warp_masks;
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
            cubin_check(text.type == 1 && (text.flags & ~uint64_t(0x500000)) == 6, "unsupported code section", true);
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
                    cubin_check(width && width <= 4096 && (flags & 0x3ffff) == 0x1f000 && record.number(0) == 0,
                        "unsupported kernel parameter encoding", true);
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
                        auto opcode = unsigned(lo & 0xfff);
                        cubin_check(opcode == 0x806 || opcode == 0x886 || opcode == 0x389 || opcode == 0x589 || opcode == 0x989 ||
                            opcode == 0xf89 || opcode == 0x3a1 || opcode == 0x348 || opcode == 0x948,
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
                                opcode == 0x806 || opcode == 0x886 || opcode == 0x348 || opcode == 0x948 || opcode == 0x918 || opcode == 0x3a1 || opcode == 0x944,
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
                } else if ((lo & 0xfff) == 0x3a1) {
                    cubin_check(mask == UINT32_MAX || (mask & 0xffffff00u) == 0x05000000u,
                        "unsupported match mask register", true);
                    kernel.warp_masks.emplace((*cooperative_offsets)[i], mask);
                } else {
                    auto expected = (lo & 0xfff) == 0x348 ? 0x05000000u | (unsigned(lo >> 32) & 255) : UINT32_MAX;
                    cubin_check(mask == expected, "unsupported cooperative mask register", true);
                }
            }
            unsigned end = 0;
            for (auto [ordinal, parameter] : layout) {
                cubin_check(ordinal == kernel.parameters.size() && parameter.offset >= end && parameter.offset <= kernel.size &&
                    parameter.size <= kernel.size - parameter.offset, "invalid parameter layout");
                // EIATTR_KPARAM_INFO retains byte extents and offsets. Recover
                // the least alignment reproducing that layout, without
                // inventing source types which machine metadata has erased.
                unsigned alignment = parameter.size == 1 || parameter.size == 2 || parameter.size == 4 || parameter.size == 8 ? parameter.size : 1;
                while (alignment <= 4096 && ((end + alignment - 1) & ~(alignment - 1)) < parameter.offset) alignment *= 2;
                cubin_check(alignment <= 4096 && ((end + alignment - 1) & ~(alignment - 1)) == parameter.offset, "invalid parameter alignment");
                parameter.alignment = alignment;
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
        cubin_check(owner != kernels.end() && symbol.binding <= 2 && symbol.other == 0, "unsupported device function binding", true);
        cubin_check(cubin_symbol_spelling(symbol.name), "unsupported device function spelling", true);
        cubin_check(symbol.offset && symbol.offset % 16 == 0 && symbol.size && symbol.size % 16 == 0 &&
            symbol.offset < owner->text->size && symbol.size <= owner->text->size - symbol.offset, "invalid device function extent");
        cubin_check(owner->functions.emplace(size_t(symbol.offset), cubin_function{owner->name + "$" + symbol.name, size_t(symbol.offset), size_t(symbol.size)}).second,
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
    auto unconditional = [](uint64_t lo, uint64_t hi) {
        return ((lo >> 12) & 15) == 7 && ((lo & 0xfff) != 0x947 || ((hi >> 23) & 15) == 7);
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
        auto op = unsigned(lo & 0xfff);
        if (op == 0x947) pending.push_back(branch_target(pc, lo, hi));
        if ((op != 0x947 && op != 0x94d && op != 0x950) || !unconditional(lo, hi)) pending.push_back(pc + 16);
    }
    // Retain the validated lexical token scopes as explicit member masks in
    // common IR. Branch ballots partition the current cohort; BSYNC reunites
    // the saved BSSY cohort through the shared warp-barrier implementation.
    std::map<size_t, size_t> regions;
    bool warp_operations = !kernel.warp_barriers.empty();
    for (auto pc : reachable) {
        auto lo = words.number(pc, 8), hi = words.number(pc + 8, 8) & ((uint64_t(1) << 41) - 1);
        unsigned opcode = unsigned(lo & 0xfff);
        warp_operations |= opcode == 0x3a1 || opcode == 0x886 || opcode == 0x806 || opcode == 0x389 || opcode == 0x589 ||
            opcode == 0x989 || opcode == 0xf89 || opcode == 0x348 || opcode == 0x948;
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
        unsigned op = unsigned(lo & 0xfff);
        auto active = scope(pc);
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
        if ((op != 0x947 && op != 0x94d && op != 0x950) || !unconditional(lo, hi)) edge(pc + 16);
    }
    bool calls = !kernel.functions.empty();
    unsigned register_count = unsigned(kernel.text->info >> 24);
    cubin_check(!calls || (register_count >= 2 && register_count <= 255), "invalid call register extent");
    unsigned state_size = (register_count + 63 + 14 + 1) * 4;
    std::string output = function ? ".func (.param .align 4 .b32 machine_result[" + std::to_string(state_size / 4) + "]) " +
        function->name + "(.param .align 4 .b32 machine_registers[" + std::to_string(state_size / 4) + "],.param .b64 machine_stack,"
        ".param .b32 machine_shared,.param .b32 machine_dynamic" : ".visible .entry " + kernel.name + "(";
    for (unsigned i = 0; i < kernel.parameters.size(); ++i) {
        const auto& parameter = kernel.parameters[i];
        bool scalar = (parameter.size == 1 || parameter.size == 2 || parameter.size == 4 || parameter.size == 8) && parameter.alignment == parameter.size;
        output += (i || function ? "," : "") + std::string(scalar ? ".param .b" + std::to_string(parameter.size * 8) :
            ".param .align " + std::to_string(parameter.alignment) + " .b8") + " arg" + std::to_string(i) +
            (scalar ? "" : "[" + std::to_string(parameter.size) + "]");
    }
    output += "){ .reg .b32 %r<255>,%u<63>,%scratch,%constant,%shared_base,%shared_dynamic,%shared_static_address,%shared_address;"
        " .reg .b64 %a,%b,%c,%bank_address,%local_address,%local_base,%wide<8>; .reg .f32 %f<3>,%fp<4>;"
        " .reg .f64 %dfp<4>; .reg .b32 %tmp<12>; .reg .b16 %half<4>;"
        " .reg .u32 %barrier_result; .reg .pred %barrier_pred;"
        " .reg .pred %p<7>,%up<7>,%pt,%upt,%shared_is_static,%instruction_guard,%test<4>;\nsetp.eq.u32 %pt,0,0;\nsetp.eq.u32 %upt,0,0;\n";
    bool track_cohort = warp_operations || !regions.empty();
    if (calls) for (size_t pc = 0; pc < kernel.text->size; pc += 16) {
        unsigned op = unsigned(input.number(size_t(kernel.text->offset) + pc, 8)) & 0xfff;
        track_cohort |= op == 0x3a1 || op == 0x886 || op == 0x945 || op == 0x806 || op == 0x389 || op == 0x589 || op == 0x989 || op == 0xf89 || op == 0x348 || op == 0x948;
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
            if (!load) text += "selp.u32 %scratch,1,0,%up" + std::to_string(i) + ";\n";
            word("%scratch", (register_count + 70 + i) * 4);
            if (load) text += "setp.ne.u32 %up" + std::to_string(i) + ",%scratch,0;\n";
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
            for (unsigned i = 0; i < 7; ++i) output += "setp.eq.u32 %up" + std::to_string(i) + ",0,1;\n";
        }
    }
    std::array<std::optional<uint32_t>, 256> known{}; known[255] = 0;
    size_t instruction_pc = 0;
    auto reg = [&](unsigned id) {
        cubin_check(id <= 255 && (id != 1 || kernel.stack_size || calls), kernel.name + "+" + std::to_string(instruction_pc) + ": stack-register use is unsupported", true);
        cubin_check(!calls || id == 255 || id < register_count, "register exceeds machine call ABI extent");
        return id == 255 ? std::string("0") : "%r" + std::to_string(id);
    };
    uint64_t uniform_defined = function ? UINT64_MAX : uint64_t(1) << 63;
    std::array<std::optional<size_t>, 64> uniform_definitions{};
    std::array<std::optional<size_t>, 7> uniform_predicate_definitions{};
    unsigned uniform_predicates_defined = function ? 127 : 0;
    auto dominates = [&](size_t definition, size_t use) {
        if (definition == use || definition == 0) return true;
        std::set<size_t> visited;
        std::vector<size_t> work{0};
        while (!work.empty()) {
            auto pc = work.back(); work.pop_back();
            if (pc == definition || !reachable.contains(pc) || !visited.insert(pc).second) continue;
            if (pc == use) return false;
            auto lo = words.number(pc, 8), hi = words.number(pc + 8, 8);
            auto opcode = lo & 0xfff;
            if (opcode == 0x947) work.push_back(branch_target(pc, lo, hi));
            if ((opcode != 0x947 && opcode != 0x94d && opcode != 0x950) || !unconditional(lo, hi)) work.push_back(pc + 16);
        }
        return true;
    };
    auto uniform_write = [&] {
        for (auto pc : reachable) {
            auto lo = words.number(pc, 8);
            if ((lo & 0xfff) != 0x947) continue;
            auto target = branch_target(pc, lo, words.number(pc + 8, 8));
            cubin_check(target > pc || instruction_pc < target || instruction_pc > pc,
                "uniform writes across branches with a backedge are unsupported", true);
        }
    };
    auto uniform = [&](unsigned id, bool write = false) {
        cubin_check(id <= 63 && (!write || id != 63), "invalid uniform register");
        if (write) {uniform_write();uniform_defined |= uint64_t(1) << id;uniform_definitions[id] = instruction_pc;}
        else {
            cubin_check((uniform_defined & (uint64_t(1) << id)) != 0, "uninitialized uniform register", true);
            cubin_check(!uniform_definitions[id] || dominates(*uniform_definitions[id], instruction_pc), "uniform register definition does not dominate its read", true);
        }
        return id == 63 ? std::string("0") : "%u" + std::to_string(id);
    };
    auto pred = [](unsigned id) { return id == 7 ? std::string("%pt") : "%p" + std::to_string(id); };
    auto uniform_predicate = [&](unsigned id, bool write = false) {
        cubin_check(id < 8 && (!write || id != 7), "invalid uniform predicate");
        if (id == 7) return std::string("%upt");
        if (write) {uniform_write();uniform_predicates_defined |= 1u << id;uniform_predicate_definitions[id] = instruction_pc;}
        else {
            cubin_check(uniform_predicates_defined & (1u << id), "uninitialized uniform predicate", true);
            cubin_check(!uniform_predicate_definitions[id] || dominates(*uniform_predicate_definitions[id], instruction_pc), "uniform predicate definition does not dominate its read", true);
        }
        return "%up" + std::to_string(id);
    };
    auto constant_load = [&](const std::string& dest, const std::string& next, unsigned offset, unsigned size, const std::string& guard,
            unsigned bank = 0, const std::string& index = "", bool sign = false) {
        auto type = std::string(sign ? "s" : "b") + std::to_string(size * 8);
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
            if (size <= 4) return result + guard + "ld.const." + type + " " + dest + ",[%bank_address];\n";
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
        // cvta.const adds the SM75 runtime's generic constant-space base.
        // Rebind the checked user-constant bank to this imported module's
        // storage, rather than reading runtime-owned bytes from the ELF image.
        if ((offset == 0x50 && (size == 4 || size == 8)) || (offset == 0x54 && size == 4)) {
            auto section = std::find_if(input.sections.begin(), input.sections.end(), [&](const auto& item) {
                return item.name == ".nv.constant3." + kernel.name;
            });
            if (section == input.sections.end()) section = std::find_if(input.sections.begin(), input.sections.end(),
                [](const auto& item) { return item.name == ".nv.constant3"; });
            cubin_check(section != input.sections.end() && input.storage.contains(size_t(section - input.sections.begin())),
                "generic constant address requires defined user-constant storage", true);
            auto result = guard + "cvta.const.u64 %bank_address," + input.storage.at(size_t(section - input.sections.begin())) + ";\n";
            if (offset == 0x54) result += guard + "shr.u64 %bank_address,%bank_address,32;\n";
            result += guard + "cvt.u32.u64 " + dest + ",%bank_address;\n";
            if (size == 8) result += guard + "shr.u64 %bank_address,%bank_address,32;\n" + guard + "cvt.u32.u64 " + next + ",%bank_address;\n";
            return result;
        }
        cubin_check(offset >= kernel.base, "unsupported constant-bank read", true); offset -= kernel.base;
        for (unsigned i = 0; i < kernel.parameters.size(); ++i) {
            auto parameter = kernel.parameters[i];
            if (offset >= parameter.offset && offset - parameter.offset <= parameter.size && size <= parameter.size - (offset - parameter.offset)) {
                cubin_check(offset % size == 0, "unaligned constant load");
                auto address = "[arg" + std::to_string(i) + "+" + std::to_string(offset - parameter.offset) + "]";
                if (size <= 4) return guard + "ld.param." + type + " " + dest + "," + address + ";\n";
                return guard + "ld.param.b64 %a," + address + ";\n" + guard + "cvt.u32.u64 " + dest + ",%a;\n" +
                    guard + "shr.u64 %b,%a,32;\n" + guard + "cvt.u32.u64 " + next + ",%b;\n";
            }
        }
        cubin_fail("constant read exceeds a parameter", true);
    };
    for (size_t pc = 0; pc < code.size(); pc += 16) {
        instruction_pc = pc;
        if (targets.contains(pc)) { known.fill({}); known[255] = 0; }
        auto lo = words.number(pc, 8), hi = words.number(pc + 8, 8) & ((uint64_t(1) << 41) - 1);
        unsigned op = unsigned(lo & 0xfff), predicate = unsigned(lo >> 12) & 15, dest = unsigned(lo >> 16) & 255,
            left = unsigned(lo >> 24) & 255, right = unsigned(lo >> 32) & 255;
        auto fail = [&](const std::string& reason) { cubin_fail((function ? function->name : kernel.name) + "+" + std::to_string(pc) + ": " + reason, true); };
        std::string guard = predicate == 7 ? "" : "@" + std::string(predicate & 8 ? "!" : "") + pred(predicate & 7) + " ";
        auto bank_index = [&](bool negated = false) {
            if ((lo >> 59) & ~(negated ? uint64_t(0x10) : uint64_t(0))) fail("unsupported constant-bank operand");
            return unsigned(lo >> 54) & 31;
        };
        auto bank_offset = [&](bool negated = false) {
            (void)bank_index(negated);
            return unsigned(lo >> 40 & 0x3fff) * 4;
        };
        auto bank_word = [&](bool negated = false) {
            if (right) fail("unsupported indexed constant-bank operand");
            output += constant_load("%constant", "", bank_offset(negated), 4, guard, bank_index(negated));
            if (negated) output += guard + "sub.u32 %constant,0,%constant;\n";
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
            if (load && first == 255 && lanes > 1) fail("invalid memory register group");
            if (load) for (unsigned i = 1; i < lanes; ++i) known[first + i].reset();
            const char* types[]{"u8", "s8", "u16", "s16", "b32", "b32", "b32"};
            std::string registers;
            for (unsigned i = 0; i < lanes; ++i) registers += (i ? "," : "") +
                (first == 255 ? std::string(load ? "%scratch" : "0") : reg(first + i));
            if (lanes > 1) registers = "{" + registers + "}";
            return std::string(load ? "ld." : "st.") + space + (lanes > 1 ? ".v" + std::to_string(lanes) : "") + "." +
                (load || kind >= 4 ? types[kind] : kind < 2 ? "u8" : "u16") + " " +
                (load ? registers + "," + address : address + "," + registers) + ";";
        };
        // Decode machine operations into the existing PTX semantic vocabulary.
        // Every source is read before publishing any aliased destination word.
        auto extended_instruction = [&]() -> std::optional<std::string> {
            switch (op) {
            case 0x219: case 0x819: case 0x226: case 0x238: case 0x309: case 0x301: case 0x300: case 0x213:
            case 0x217: case 0x817: case 0xa17: case 0x21b: case 0x21a: case 0x81a:
            case 0x230: case 0x232: case 0x231: case 0x228: case 0x828: case 0x229: case 0x429: case 0x22b: case 0x42b: case 0x82b:
            case 0x209: case 0x809: case 0xa09: case 0x308: case 0x908: case 0x81c:
            case 0x20c: case 0x80c: case 0xa0c: case 0xc0c:
            case 0x20b: case 0x80b: case 0xa0b: case 0x22a: case 0x42a: case 0xa2a:
            case 0x302: case 0x304: case 0x305: case 0x306: case 0x906: case 0x310: case 0x311: case 0x312: break;
            default: return std::nullopt;
            }
            // Predicate-producing instructions encode Pn in fields otherwise
            // used for Rn. Validate a data destination only on a data path.
            auto destination = dest == 255 ? std::string("%scratch") : "%r" + std::to_string(dest);
            switch (op) {
            case 0x81c: case 0x20c: case 0x80c: case 0xa0c: case 0xc0c:
            case 0x20b: case 0x80b: case 0xa0b: case 0x22a: case 0x42a: case 0xa2a: case 0x302: break;
            default: (void)reg(dest); break;
            }
            auto pair = [&](unsigned id, unsigned slot) {
                if (id != 255 && (id >= 254 || id % 2)) fail("invalid 64-bit register group");
                auto name = "%wide" + std::to_string(slot);
                join(reg(id), id == 255 ? "0" : reg(id + 1), name);
                return name;
            };
            auto split = [&](const std::string& value) {
                if (dest != 255 && (dest >= 254 || dest % 2)) fail("invalid 64-bit destination");
                output += guard + "cvt.u32.u64 " + destination + "," + value + ";\n";
                if (dest != 255) {
                    output += guard + "shr.u64 %wide7," + value + ",32;\n";
                    known[dest + 1].reset();
                    return "cvt.u32.u64 " + reg(dest + 1) + ",%wide7;";
                }
                return std::string();
            };
            if (op == 0x304 || op == 0x305 || op == 0x306 || op == 0x906 || op == 0x310 || op == 0x311 || op == 0x312) {
                bool to_integer = op == 0x305 || op == 0x311, from_integer = op == 0x306 || op == 0x906 || op == 0x312;
                unsigned source_code = unsigned(hi >> 20) & 3, destination_code = unsigned(hi >> 11) & 3;
                auto allowed = uint64_t(0x301800 | 0xc000) | (to_integer ? 0x12100u : from_integer ? 0x400u : 0u);
                bool long_form = op == 0x310 || op == 0x311 || op == 0x312;
                bool immediate = op == 0x906;
                if (left || (!immediate && (lo >> 40)) || !source_code || !destination_code || (hi & ~allowed) ||
                        (immediate && (source_code != 2 || destination_code != 2)) ||
                        (long_form != (source_code == 3 || destination_code == 3)) ||
                        (to_integer && bool(hi & 0x2000) != (op == 0x305)) || ((hi & 0x10000) && source_code != 2))
                    fail("unsupported numeric conversion encoding");
                unsigned source_bits = 8u << source_code, destination_bits = 8u << destination_code;
                auto source_type = std::string(from_integer ? (hi & 0x400 ? "s" : "u") : "f") + std::to_string(source_bits);
                auto destination_type = std::string(to_integer ? (hi & 0x100 ? "s" : "u") : "f") + std::to_string(destination_bits);
                std::string input;
                if (source_bits == 64) {
                    input = pair(right, 0);
                    if (!from_integer) {output += guard + "mov.b64 %dfp0," + input + ";\n";input = "%dfp0";}
                } else if (source_bits == 16) {
                    output += guard + "cvt.u16.u32 %half0," + reg(right) + ";\n";input = "%half0";
                } else {
                    input = from_integer ? "%tmp0" : "%fp0";
                    output += guard + "mov.b32 " + input + "," + (immediate ? std::to_string(uint32_t(lo >> 32)) : reg(right)) + ";\n";
                }
                auto result = destination_bits == 64 ? (to_integer ? "%wide1" : "%dfp1") : destination_bits == 16 ? "%half1" : to_integer ? "%tmp1" : "%fp1";
                const char* modes[]{"rn", "rm", "rp", "rz"};
                bool widening = !to_integer && !from_integer && destination_bits > source_bits;
                auto rounding = widening ? std::string() : "." + std::string(modes[(hi >> 14) & 3]) + (to_integer ? "i" : "");
                output += guard + "cvt" + rounding + (hi & 0x10000 ? ".ftz" : "") + "." + destination_type + "." + source_type +
                    " " + result + "," + input + ";\n";
                if (destination_bits == 64) {
                    if (!to_integer) output += guard + "mov.b64 %wide1,%dfp1;\n";
                    return split("%wide1");
                }
                if (destination_bits == 16) return std::string(to_integer && (hi & 0x100) ? "cvt.s32.s16 " : "cvt.u32.u16 ") + destination + ",%half1;";
                return "mov.b32 " + destination + "," + result + ";";
            }
            if (op == 0x81c && !(lo >> 16) && !(hi & ~uint64_t(0x7fffff7)) && ((hi >> 20) & 7) == 7) {
                auto predicate_value = [&](unsigned field) { return std::string(field & 8 ? "!" : "") + pred(field & 7); };
                output += guard + "selp.b32 %tmp0,-1,0," + predicate_value(unsigned(hi >> 23)) + ";\n" +
                    guard + "selp.b32 %tmp1,-1,0," + predicate_value(unsigned(hi >> 13)) + ";\n" +
                    guard + "selp.b32 %tmp2,-1,0," + predicate_value(unsigned(hi >> 4)) + ";\n";
                unsigned lut = unsigned(hi & 7) | unsigned(hi >> 5 & 248), target = unsigned(hi >> 17) & 7;
                output += guard + "lop3.b32 %tmp3,%tmp0,%tmp1,%tmp2," + std::to_string(lut) + ";\n";
                return target == 7 ? "" : "setp.ne.u32 " + pred(target) + ",%tmp3,0;";
            }
            bool integer_comparison = op == 0x20c || op == 0x80c || op == 0xa0c || op == 0xc0c;
            bool floating_comparison = op == 0x20b || op == 0x80b || op == 0xa0b || op == 0x22a || op == 0x42a || op == 0xa2a;
            if ((integer_comparison || floating_comparison) && dest == 0) {
                bool wide = op == 0x22a || op == 0x42a || op == 0xa2a, extended = integer_comparison && (hi & 0x100);
                unsigned boolean = unsigned(hi >> 10) & 3, comparison = unsigned(hi >> 12) & (integer_comparison ? 7 : 15);
                uint64_t allowed = integer_comparison ? 0x7fe7ff0 : 0x7fffc00;
                if (floating_comparison) allowed |= 0x300;
                if (op == 0xc0c) allowed |= 0x8000000;
                if ((hi & ~allowed) || boolean > 2 || (wide && (hi & 0x10000)) || (integer_comparison && !extended && (hi & 0xf0) != 0x70) ||
                        ((hi & 0x8000000) != (op == 0xc0c ? 0x8000000u : 0u)) ||
                        ((op == 0x20c || op == 0xc0c || op == 0x20b || op == 0x22a) &&
                            ((lo >> 40) & ~(op == 0x20b || op == 0x22a ? uint64_t(0xc00000) : uint64_t(0)))))
                    fail("unsupported comparison encoding");
                const char* integer_names[]{"", "lt", "eq", "le", "gt", "ne", "ge", ""};
                const char* floating_names[]{"", "lt", "eq", "le", "gt", "ne", "ge", "num", "nan", "ltu", "equ", "leu", "gtu", "neu", "geu", ""};
                const char* booleans[]{"and", "or", "xor"};
                auto first = reg(left);
                auto form = op & 0xe00;
                auto second = form == 0x800 || form == 0x400 ? std::to_string(uint32_t(lo >> 32)) :
                    form == 0xa00 ? bank_word() : op == 0xc0c ? uniform(right) : reg(right);
                auto type = integer_comparison ? (hi & 0x200 ? "s32" : "u32") : wide ? "f64" : "f32";
                if (wide) {
                    first = pair(left, 0);
                    if (form == 0x400) second = std::to_string(uint64_t(uint32_t(lo >> 32)) << 32);
                    else if (form == 0xa00) {
                        output += constant_load("%constant", "%scratch", bank_offset(), 8, guard, bank_index());
                        join("%constant", "%scratch", "%wide1");second = "%wide1";
                    } else second = pair(right, 1);
                    output += guard + "mov.b64 %dfp0," + first + ";\n" + guard + "mov.b64 %dfp1," + second + ";\n";
                    if (hi & 0x200) output += guard + "abs.f64 %dfp0,%dfp0;\n";
                    if (hi & 0x100) output += guard + "neg.f64 %dfp0,%dfp0;\n";
                    if (op == 0x22a && (lo & (uint64_t(1) << 62))) output += guard + "abs.f64 %dfp1,%dfp1;\n";
                    if (op == 0x22a && (lo & (uint64_t(1) << 63))) output += guard + "neg.f64 %dfp1,%dfp1;\n";
                    output += guard + "mov.b64 %wide0,%dfp0;\n" + guard + "mov.b64 %wide1,%dfp1;\n";
                    first = "%dfp0"; second = "%dfp1";
                } else if (floating_comparison) {
                    output += guard + "mov.b32 %fp0," + first + ";\n" + guard + "mov.b32 %fp1," + second + ";\n";
                    if (hi & 0x200) output += guard + "abs.f32 %fp0,%fp0;\n";
                    if (hi & 0x100) output += guard + "neg.f32 %fp0,%fp0;\n";
                    if (op == 0x20b && (lo & (uint64_t(1) << 62))) output += guard + "abs.f32 %fp1,%fp1;\n";
                    if (op == 0x20b && (lo & (uint64_t(1) << 63))) output += guard + "neg.f32 %fp1,%fp1;\n";
                    first = "%fp0"; second = "%fp1";
                }
                auto incoming = std::string(hi & 0x4000000 ? "!" : "") + pred(unsigned(hi >> 23) & 7);
                bool extrema = wide && (comparison == 0 || comparison == 15);
                if (extrema) {
                    output += guard + "testp.notanumber.f64 %test0,%dfp0;\n" + guard + "testp.notanumber.f64 %test1,%dfp1;\n" +
                        guard + "and.pred %test3,%test0,%test1;\n" + guard + "not.pred %test0,%test0;\n" +
                        guard + "and.pred %test1,%test0,%test1;\n" + guard + "setp." + (comparison ? "gt" : "lt") + ".f64 %test0,%dfp0,%dfp1;\n" +
                        guard + "or.pred %test0,%test0,%test1;\n" + guard + "and.b64 %wide2,%wide0,0x7fffffffffffffff;\n" +
                        guard + "and.b64 %wide3,%wide1,0x7fffffffffffffff;\n" + guard + "or.b64 %wide2,%wide2,%wide3;\n" +
                        guard + "setp.eq.u64 %test1,%wide2,0;\n" + guard + "setp." + (comparison ? "ge" : "lt") + ".s64 %test2,%wide0,0;\n" +
                        guard + "and.pred %test1,%test1,%test2;\n" + guard + "or.pred %test0,%test0,%test1;\n" +
                        guard + "and.pred %test1,%test3,%pt;\n";
                } else if (comparison == 0 || comparison == (integer_comparison ? 7u : 15u)) {
                    if (!integer_comparison) fail("unsupported floating min/max comparison encoding");
                    output += guard + std::string(comparison ? "setp.eq.u32" : "setp.ne.u32") + " %test0,0,0;\n";
                } else output += guard + "setp." + (integer_comparison ? integer_names[comparison] : floating_names[comparison]) +
                    (floating_comparison && !wide && (hi & 0x10000) ? ".ftz" : "") + "." + type + " %test0," + first + "," + second + ";\n";
                if (extended) {
                    auto previous = std::string(hi & 0x80 ? "!" : "") + pred(unsigned(hi >> 4) & 7);
                    if (comparison == 2 || comparison == 5) output += guard + std::string(comparison == 2 ? "and.pred" : "or.pred") + " %test0,%test0," + previous + ";\n";
                    else {
                        auto strict = comparison == 3 ? "lt" : comparison == 6 ? "gt" : integer_names[comparison];
                        output += guard + "setp." + strict + "." + type + " %test0," + first + "," + second + ";\n" +
                            guard + "setp.eq.u32 %test1," + first + "," + second + ";\n" +
                            guard + "and.pred %test1,%test1," + previous + ";\n" + guard + "or.pred %test0,%test0,%test1;\n";
                    }
                }
                if (!extrema) output += guard + "not.pred %test1,%test0;\n";
                output += guard + booleans[boolean] + ".pred %test0,%test0," + incoming + ";\n" +
                    guard + booleans[boolean] + ".pred %test1,%test1," + incoming + ";\n";
                unsigned positive = unsigned(hi >> 17) & 7, negative = unsigned(hi >> 20) & 7;
                if (positive != 7) output += guard + "and.pred " + pred(positive) + ",%test0,%pt;\n";
                return negative == 7 ? "" : "and.pred " + pred(negative) + ",%test1,%pt;";
            }
            if ((op == 0x209 || op == 0x809 || op == 0xa09) && (hi & ~uint64_t(0x7800000)) == 0 && (op != 0x209 || !(lo >> 40))) {
                auto second = op == 0x209 ? reg(right) : op == 0x809 ? std::to_string(uint32_t(lo >> 32)) : bank_word();
                if (op == 0x809) { output += guard + "mov.b32 %fp2," + second + ";\n"; second = "%fp2"; }
                output += guard + "min.f32 %fp0," + reg(left) + "," + second + ";\n" + guard + "max.f32 %fp1," + reg(left) + "," + second + ";\n" +
                    guard + "mov.b32 %tmp0,%fp0;\n" + guard + "mov.b32 %tmp1,%fp1;\n";
                return "selp.b32 " + destination + ",%tmp0,%tmp1," + (hi & 0x4000000 ? "!" : "") + pred(unsigned(hi >> 23) & 7) + ";";
            }
            if ((op == 0x308 || op == 0x908) && left == 0 && (op == 0x908 || !(lo >> 40)) && (hi & 0x3ff) == 0 && hi / 1024 <= 9) {
                const char* operations[]{"cos", "sin", "ex2", "lg2", "rcp", "rsqrt", "", "", "sqrt", "tanh"};
                auto operation = unsigned(hi / 1024);
                auto source = op == 0x908 ? std::to_string(uint32_t(lo >> 32)) : reg(right);
                if (operation == 6 || operation == 7) {
                    output += guard + "cvt.u64.u32 %wide0," + source + ";\n" + guard + "shl.b64 %wide0,%wide0,32;\n" +
                        guard + "mov.b64 %dfp0,%wide0;\n";
                    if (operation == 7) output += guard + "sqrt.rn.f64 %dfp0,%dfp0;\n";
                    output += guard + "div.rn.f64 %dfp0,0d3ff0000000000000,%dfp0;\n" + guard + "mov.b64 %wide0,%dfp0;\n" +
                        guard + "shr.u64 %wide0,%wide0,32;\n";
                    return "cvt.u32.u64 " + destination + ",%wide0;";
                }
                output += guard + "mov.b32 %fp0," + source + ";\n";
                if (operation < 2) output += guard + "mul.rn.f32 %fp0,%fp0,0f40c90fdb;\n";
                output += guard + operations[operation] + (operation == 9 ? ".approx.f32" : ".approx.ftz.f32") + " %fp0,%fp0;\n";
                return "mov.b32 " + destination + ",%fp0;";
            }
            if (op == 0x302 && hi == 0 && dest < 8 && !(lo >> 40)) {
                // The SM75 divide range check is a predicate over biased
                // exponents. Its conservative fast range also leaves room for
                // the reciprocal/refinement intermediates, not only the result.
                output += guard + "shr.u32 %tmp0," + reg(left) + ",23;\n" + guard + "and.b32 %tmp0,%tmp0,255;\n" +
                    guard + "shr.u32 %tmp1," + reg(right) + ",23;\n" + guard + "and.b32 %tmp1,%tmp1,255;\n" +
                    guard + "sub.s32 %tmp2,%tmp0,%tmp1;\n" +
                    guard + "setp.lt.u32 %test0,%tmp0,25;\n" + guard + "setp.eq.u32 %test1,%tmp0,255;\n" +
                    guard + "or.pred %test0,%test0,%test1;\n" + guard + "setp.lt.u32 %test1,%tmp1,2;\n" +
                    guard + "or.pred %test0,%test0,%test1;\n" + guard + "setp.ge.u32 %test1,%tmp1,252;\n" +
                    guard + "or.pred %test0,%test0,%test1;\n" + guard + "setp.lt.s32 %test1,%tmp2,-124;\n" +
                    guard + "or.pred %test0,%test0,%test1;\n" + guard + "setp.gt.s32 %test2,%tmp2,126;\n";
                return "or.pred " + pred(dest & 7) + ",%test0,%test2;";
            }
            if (op == 0x226 && (hi & ~uint64_t(0x36ff)) == 0 && !(lo >> 40) &&
                    (!(hi & 0x2000) || (hi & 0x1000))) {
                bool two = hi & 0x1000;
                return std::string(two ? (hi & 0x2000 ? "dp2a.hi" : "dp2a.lo") : "dp4a") +
                    (hi & 0x200 ? ".s32" : ".u32") + (hi & 0x400 ? ".s32 " : ".u32 ") +
                    destination + "," + reg(left) + "," + reg(right) + "," + reg(unsigned(hi) & 255) + ";";
            }
            if (op == 0x238 && (hi & ~uint64_t(0x3000)) == 0 && left == 0 && !(lo >> 40)) {
                unsigned width = hi & 0x2000 ? 16 : 8;
                bool sign = hi & 0x1000;
                auto minimum = sign ? -int(1u << (width - 1)) : 0;
                auto maximum = (1u << (width - unsigned(sign))) - 1;
                output += guard + "max.s32 %tmp0," + reg(right) + "," + std::to_string(minimum) + ";\n";
                return "min.s32 " + destination + ",%tmp0," + std::to_string(maximum) + ";";
            }
            if ((op == 0x219 || op == 0x819) && (hi & ~uint64_t(0x11eff)) == 0 &&
                    (op == 0x819 || !(lo >> 40))) {
                bool right_shift = hi & 0x1000, high = hi & 0x10000, wrap = hi & 0x800, sign = !(hi & 0x200);
                unsigned width = hi & 0x400 ? 32 : 64;
                join(reg(left), reg(unsigned(hi) & 255), "%wide0");
                auto count = op == 0x819 ? std::to_string(uint32_t(lo >> 32)) : reg(right);
                output += guard + std::string(wrap ? "and.b32" : "min.u32") + " %tmp0," + count + "," + std::to_string(wrap ? width - 1 : width) + ";\n";
                output += guard + std::string(right_shift ? (sign ? "shr.s64" : "shr.u64") : "shl.b64") + " %wide0,%wide0,%tmp0;\n";
                if (high) output += guard + "shr.u64 %wide0,%wide0,32;\n";
                return "cvt.u32.u64 " + destination + ",%wide0;";
            }
            if (op == 0x309 && hi == 0 && left == 0 && !(lo >> 40)) return "popc.b32 " + destination + "," + reg(right) + ";";
            if (op == 0x301 && hi == 0 && left == 0 && !(lo >> 40)) return "brev.b32 " + destination + "," + reg(right) + ";";
            if (op == 0x300 && (hi & ~uint64_t(0x200)) == 0xe0000 && left == 0 && !(lo >> 40))
                return std::string(hi & 0x200 ? "bfind.s32 " : "bfind.u32 ") + destination + "," + reg(right) + ";";
            if (op == 0x213 && hi == 0 && left == 0 && !(lo >> 40)) return "abs.s32 " + destination + "," + reg(right) + ";";
            if ((op == 0x217 || op == 0x817 || op == 0xa17) && (hi & ~uint64_t(0x7800200)) == 0 &&
                    (op != 0x217 || !(lo >> 40))) {
                auto source = op == 0x217 ? reg(right) : op == 0x817 ? std::to_string(uint32_t(lo >> 32)) : bank_word();
                auto type = std::string(hi & 0x200 ? "s32" : "u32");
                output += guard + "min." + type + " %tmp0," + reg(left) + "," + source + ";\n" +
                    guard + "max." + type + " %tmp1," + reg(left) + "," + source + ";\n";
                return "selp.b32 " + destination + ",%tmp0,%tmp1," + (hi & 0x4000000 ? "!" : "") + pred(unsigned(hi >> 23) & 7) + ";";
            }
            if (op == 0x21b && hi == 0 && !(lo >> 40))
                return "bfi.b32 " + destination + ",-1,0," + reg(left) + "," + reg(right) + ";";
            if ((op == 0x21a || op == 0x81a) && (hi & ~uint64_t(0x200)) == 0 && (op != 0x21a || !(lo >> 40))) {
                auto count = op == 0x21a ? reg(right) : std::to_string(uint32_t(lo >> 32));
                return std::string(hi & 0x200 ? "bfe.s32 " : "bfe.u32 ") + destination + "," + reg(left) + ",0," + count + ";";
            }
            if ((op == 0x230 || op == 0x232 || op == 0x231) &&
                    (hi & ~uint64_t(0x66cff)) == 0 && (lo & 0x4fffff0000000000ull) == 0) {
                bool fma = op == 0x231;
                bool float_result = hi & 0x4000;
                // ptxas lowers FP32-to-half truncation to this HADD2 form.
                // Both result halves contain the conversion; the negative
                // zero addend preserves the source's signed zero.
                if (op == 0x230 && hi == 0x400 && right == 255 && (lo >> 60) == 10) {
                    output += guard + "mov.b32 %fp0," + reg(left) + ";\n" + guard + "cvt.rz.f16.f32 %half0,%fp0;\n" +
                        guard + "cvt.u32.u16 %tmp0,%half0;\n" + guard + "shl.b32 %tmp1,%tmp0,16;\n";
                    return "or.b32 " + destination + ",%tmp0,%tmp1;";
                }
                if (float_result && (fma || ((hi >> 10) & 3) < 2 || ((lo >> 60) & 3) < 2)) fail("unsupported half FP32-result operand selection");
                auto source = [&](unsigned id, unsigned slot, unsigned selection) {
                    if (selection == 1) fail("unsupported mixed FP32/half operand");
                    auto name = "%tmp" + std::to_string(slot);
                    output += guard + "mov.b32 " + name + "," + reg(id) + ";\n";
                    if (selection) {
                        if (selection == 3) output += guard + "shr.u32 " + name + "," + name + ",16;\n";
                        output += guard + "and.b32 " + name + "," + name + ",65535;\n" +
                            guard + "shl.b32 %tmp4," + name + ",16;\n" + guard + "or.b32 " + name + "," + name + ",%tmp4;\n";
                    }
                    return name;
                };
                auto a = source(left, 0, unsigned(hi >> 10) & 3), b = source(right, 1, unsigned(lo >> 60) & 3),
                    c = fma ? source(unsigned(hi) & 255, 2, unsigned(hi >> 17) & 3) : "";
                if (lo & (uint64_t(1) << 63)) output += guard + "xor.b32 " + b + "," + b + ",0x80008000;\n";
                if (float_result) {
                    output += guard + "cvt.u16.u32 %half0," + a + ";\n" + guard + "cvt.u16.u32 %half1," + b + ";\n" +
                        guard + "cvt.f32.f16 %fp0,%half0;\n" + guard + "cvt.f32.f16 %fp1,%half1;\n" +
                        guard + (op == 0x230 ? "add" : "mul") + ".rn" + (hi & 0x2000 ? ".sat" : "") + ".f32 %fp2,%fp0,%fp1;\n";
                    return "mov.b32 " + destination + ",%fp2;";
                }
                output += guard + std::string(op == 0x230 ? "add" : fma ? "fma" : "mul") + ".rn" +
                    (hi & 0x2000 ? ".sat" : "") + ".f16x2 %tmp3," + a + "," + b + (fma ? "," + c : "") + ";\n";
                return "mov.b32 " + destination + ",%tmp3;";
            }
            if ((op == 0x228 || op == 0x828 || op == 0x229 || op == 0x429 || op == 0x22b || op == 0x42b || op == 0x82b) &&
                    (hi & ~uint64_t(op == 0x229 ? 0xcdff : op == 0x429 ? 0xc100 : op == 0x22b || op == 0x42b || op == 0x82b ? 0xc9ff : 0xc000)) == 0 &&
                    (op == 0x828 || op == 0x429 || op == 0x42b || op == 0x82b || !((lo >> 40) & ~(op == 0x22b ? uint64_t(0x800000) : uint64_t(0))))) {
                bool add = op == 0x229 || op == 0x429, fma = op == 0x22b || op == 0x42b || op == 0x82b;
                auto a = pair(left, 0), b = op == 0x828 || op == 0x429 || op == 0x82b ? std::to_string(uint64_t(uint32_t(lo >> 32)) << 32) : pair(add || op == 0x42b ? unsigned(hi) & 255 : right, 1);
                output += guard + "mov.b64 %dfp0," + a + ";\n" + guard + "mov.b64 %dfp1," + b + ";\n";
                if ((add || fma) && (hi & 0x100)) output += guard + "neg.f64 %dfp0,%dfp0;\n";
                if (add && (hi & 0x400)) output += guard + "abs.f64 %dfp1,%dfp1;\n";
                if (((add || op == 0x42b) && (hi & 0x800)) || (op == 0x22b && (lo >> 63))) output += guard + "neg.f64 %dfp1,%dfp1;\n";
                if (fma) {
                    auto c = op == 0x42b ? std::to_string(uint64_t(uint32_t(lo >> 32)) << 32) : pair(unsigned(hi) & 255, 2);
                    output += guard + "mov.b64 %dfp2," + c + ";\n";
                    if (op != 0x42b && (hi & 0x800)) output += guard + "neg.f64 %dfp2,%dfp2;\n";
                }
                const char* modes[]{"rn", "rm", "rp", "rz"};
                output += guard + std::string(add ? "add" : fma ? "fma" : "mul") + "." + modes[(hi >> 14) & 3] + ".f64 %dfp3,%dfp0,%dfp1" + (fma ? ",%dfp2" : "") + ";\n" +
                    guard + "mov.b64 %wide3,%dfp3;\n";
                return split("%wide3");
            }
            return std::nullopt;
        };
        output += "L" + std::to_string(pc) + ":\n";
        if (predicate != 7) {
            output += "and.pred %instruction_guard," + std::string(predicate & 8 ? "!" : "") + pred(predicate & 7) + ",%pt;\n";
            guard = "@%instruction_guard ";
        }
        if (op == 0x947 && ((hi >> 23) & 15) != 7) {
            output += "and.pred %instruction_guard," + std::string(predicate == 7 ? "%pt" : "%instruction_guard") + "," +
                (hi & 0x4000000 ? "!" : "") + pred(unsigned(hi >> 23) & 7) + ";\n";
            guard = "@%instruction_guard ";
        }
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
        } else if (auto extended = extended_instruction()) {
            instruction = *extended;
        } else if (op == 0x202 && hi == 0xf00 && (lo & 0xffffff00ff000000ull) == 0) {
            instruction = "mov.b32 " + reg(dest) + "," + reg(right) + ";"; constant = known[right];
        } else if (op == 0xc02 && hi == 0x8000f00 && !(lo >> 40) && left == 0) {
            instruction = "mov.b32 " + reg(dest) + "," + uniform(right) + ";";
        } else if (op == 0xc82 && hi == 0x8000000 && !(lo >> 38) && left == 0 && predicate == 7) {
            auto source = uniform(right);
            instruction = "mov.b32 " + uniform(dest, true) + "," + source + ";";
        } else if (op == 0x802 && hi == 0xf00 && left == 0) {
            instruction = "mov.b32 " + reg(dest) + "," + std::to_string(uint32_t(lo >> 32)) + ";"; constant = uint32_t(lo >> 32);
        } else if (op == 0xa02 && hi == 0xf00 && (lo & 0xf80000ffff000000ull) == 0) {
            unsigned offset = bank_offset();
            if (pc == 0 && !function && predicate == 7 && dest == 1 && offset == 0x28 && !bank_index()) {
                if (kernel.stack_size || calls) instruction = "mov.u32 %r1," + std::to_string(kernel.stack_size) + ";";
            } else output += constant_load(reg(dest), "", offset, 4, guard, bank_index());
        } else if (op == 0xb82 && (hi & ~uint64_t(0xe00)) == 0 && ((hi >> 9) & 7) <= 5 && (lo & 0xf800003f00000000ull) == 0) {
            if (!bank_index() && !known[left]) fail("dynamic constant-bank indexing is unsupported");
            unsigned kind = unsigned(hi >> 9) & 7, width = kind < 2 ? 1 : kind < 4 ? 2 : kind == 4 ? 4 : 8;
            auto offset = uint64_t(unsigned(lo >> 38) & 65535) + (bank_index() ? 0 : *known[left]);
            if (offset > UINT32_MAX) fail("constant-bank offset overflow");
            cubin_check(dest + width / 4 <= 255, "invalid constant destination");
            output += constant_load(reg(dest), width == 8 ? reg(dest + 1) : "", unsigned(offset), width, guard,
                bank_index(), bank_index() ? reg(left) : "", kind == 1 || kind == 3);
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
                if (op == 0x919 && special >= 0x38 && special <= 0x3c) {
                    const char* masks[]{"eq", "lt", "le", "gt", "ge"};
                    instruction = "mov.u32 " + reg(dest) + ",%lanemask_" + masks[special - 0x38] + ";";
                } else {
                if (!component) fail("unsupported special register " + std::to_string(special));
                if (op == 0x9c3 && (special < 0x25 || predicate != 7)) fail("unsupported uniform special register");
                unsigned axis = special >= 0x25 ? special - 0x25 : special - 0x21;
                instruction = "mov.u32 " + (op == 0x9c3 ? uniform(dest, true) : reg(dest)) + "," + component + "xyz"[axis] + ";";
                }
            }
        } else if (op == 0x805 && hi == 0x1ff00 && !(lo >> 24)) {
            if (dest >= 254 || dest % 2) fail("invalid zero special-register destination");
            output += guard + "mov.b32 " + reg(dest) + ",0;\n";
            instruction = "mov.b32 " + reg(dest + 1) + ",0;";known[dest + 1] = predicate == 7 ? std::optional<uint32_t>(0) : std::nullopt;constant = 0;
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
        } else if (op == 0x806 || op == 0x886) {
            // VOTE.ANY writes a ballot and optionally its any predicate.
            if ((lo >> 24) != 0 || (hi & ~uint64_t(0x78e0000)) != 0x100)
                fail("unsupported vote encoding");
            warp_members();
            auto test = std::string(hi & 0x4000000 ? "!" : "") + pred(unsigned(hi >> 23) & 7);
            unsigned destination = unsigned(hi >> 17) & 7;
            if (op == 0x886 && predicate != 7) fail("unsupported uniform vote predicate");
            auto target = op == 0x886 ? uniform(dest, true) : dest == 255 ? std::string("%scratch") : reg(dest);
            // Compute before writing: the source and result predicates may alias.
            output += guard + "vote.sync.ballot.b32 %scratch," + test + ",%members;\n";
            instruction = "mov.b32 " + target + ",%scratch;";
            if (destination != 7) {
                // The result predicate can also be this instruction's guard.
                // Publish the register result before changing that guard.
                output += guard + instruction + "\n";
                instruction = "setp.ne.u32 " + (op == 0x886 ? uniform_predicate(destination, true) : pred(destination)) + ",%scratch,0;";
            }
        } else if (op == 0x2bd && hi == 0x80e0000 && left == 0 && !(lo >> 38) && predicate == 7) {
            auto source = uniform(right);
            instruction = "bfind.u32 " + uniform(dest, true) + "," + source + ";";
        } else if (op == 0x2bf && hi == 0x8000000 && left == 0 && !(lo >> 38) && predicate == 7) {
            auto source = uniform(right);
            instruction = "popc.b32 " + uniform(dest, true) + "," + source + ";";
        } else if (op == 0xd00 && hi == 0x80e0000 && left == 0 && !(lo >> 38)) {
            instruction = "bfind.u32 " + reg(dest) + "," + uniform(right) + ";";
        } else if (op == 0xd09 && hi == 0x8000000 && left == 0 && !(lo >> 38)) {
            instruction = "popc.b32 " + reg(dest) + "," + uniform(right) + ";";
        } else if (op == 0x292 && predicate == 7 && !(lo >> 38) &&
                   (hi & ~uint64_t(0xff3f)) == 0xf8e0000) {
            // Register ULOP3.LUT with discarded predicate output. As with the
            // other uniform operations, read aliased inputs before the write.
            auto first = uniform(left), second = uniform(right), third = uniform(unsigned(hi) & 63);
            instruction = "lop3.b32 " + uniform(dest, true) + "," + first + "," + second + "," + third + "," +
                std::to_string(unsigned(hi >> 8) & 255) + ";";
        } else if ((op == 0x290 || op == 0x890) && predicate == 7) {
            auto source = op == 0x890 ? std::to_string(uint32_t(lo >> 32)) : uniform(right);
            auto a = uniform(left);
            if (hi == 0xfffe03f && (op == 0x890 || !(lo >> 38))) {
                instruction = "add.u32 " + uniform(dest, true) + "," + a + "," + source + ";";
            } else if ((hi & ~uint64_t(0xe0000)) == 0xff1e03f && ((hi >> 17) & 7) != 7) {
                output += "cvt.u64.u32 %a," + a + ";\ncvt.u64.u32 %b," + source + ";\nadd.u64 %a,%a,%b;\n"
                    "setp.gt.u64 " + uniform_predicate(unsigned(hi >> 17) & 7, true) + ",%a,4294967295;\n";
                instruction = "cvt.u32.u64 " + uniform(dest, true) + ",%a;";
            } else if (op == 0x290 && (hi & ~uint64_t(0x3800000)) == 0x87fe43f && !(lo >> 38)) {
                output += "selp.u32 %scratch,1,0," + uniform_predicate(unsigned(hi >> 23) & 7) + ";\nadd.u32 %scratch,%scratch," + a + ";\n";
                instruction = "add.u32 " + uniform(dest, true) + ",%scratch," + source + ";";
            } else fail("unsupported uniform addition encoding");
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
        } else if (op == 0xb60 || op == 0xb66 || op == 0x361 || op == 0x367 || op == 0xb99 || op == 0xb9d || op == 0x99a || op == 0x99e) {
            // Register-bound handles retain a 32-bit descriptor ID; parameter
            // handles use a constant-bank pair. Reconstruct the existing image
            // operations, preserving descriptor formats and sampler flags.
            bool texture = op == 0xb60 || op == 0xb66 || op == 0x361 || op == 0x367;
            bool fetch = op == 0xb66 || op == 0x367, load = op != 0xb9d && op != 0x99e;
            bool register_handle = op == 0x361 || op == 0x367 || op == 0x99a || op == 0x99e;
            unsigned bank = unsigned(lo >> 54) & 31, offset = unsigned(lo >> 40) & 0x3fff;
            if (bank) fail("unsupported image handle bank");
            auto handle = [&] {
                if (register_handle) output += guard + "cvt.u64.u32 %wide0," + reg(texture ? right : unsigned(hi) & 255) + ";\n" +
                    guard + "or.b64 %wide0,%wide0," + std::to_string(packet_image_base) + ";\n";
                else {
                    output += constant_load("%constant", "%scratch", offset * 4, 8, guard);
                    join("%constant", "%scratch", "%wide0");
                }
            };
            if (texture) {
                auto fields = (register_handle ? uint64_t(0) : uint64_t(0x3fff) << 40) | uint64_t(0xffffff) << 16;
                auto fixed = (fetch ? uint64_t(0x1000000000000000) : uint64_t(0x3000000000000000)) |
                    (register_handle ? uint64_t(1) << 59 : 0) | uint64_t(op) | uint64_t(predicate) << 12;
                unsigned lod = register_handle ? right + 1 : right;
                if ((lo & ~fields) != fixed || (hi & ~uint64_t(255)) != (fetch ? 0x9e0f00u : 0x19e0f00u) ||
                        (fetch && !register_handle && right != 255) ||
                        (!fetch && (lod > 255 || !known[lod] || (*known[lod] != 0xc2000000u && *known[lod] != 0))))
                    fail("unsupported texture instruction encoding");
                unsigned upper = unsigned(hi) & 255;
                if (dest >= 254 || dest % 2 || upper >= 254 || upper % 2 || dest == upper ||
                        (!fetch && left != 255 && left >= 254)) fail("invalid texture register group");
                handle();
                auto coordinate = reg(left);
                if (!fetch) coordinate += "," + (left == 255 ? std::string("0") : reg(left + 1));
                output += guard + std::string(fetch ? "tex.1d.v4.f32.s32" : "tex.2d.v4.f32.f32") +
                    " {%tmp0,%tmp1,%tmp2,%tmp3},[%wide0,{" + coordinate + "}];\n";
                for (unsigned i = 0; i < 4; ++i) {
                    auto id = i < 2 ? dest + i : upper + i - 2;
                    output += guard + "mov.b32 " + reg(id) + ",%tmp" + std::to_string(i) + ";\n";known[id].reset();
                }
            } else {
                unsigned kind = unsigned(hi >> 9) & 7, boundary = unsigned(lo >> 59) & 3;
                auto fields = uint64_t(3) << 59 | (register_handle ? uint64_t(0) : uint64_t(0x3fff) << 40) | uint64_t(255) << 24 |
                    uint64_t(255) << (load ? 16 : 32);
                auto fixed = uint64_t(0x6000000000000000) | uint64_t(op) | uint64_t(predicate) << 12;
                if ((lo & ~fields) != fixed || (hi & ~uint64_t(register_handle ? 0xeff : 0xe00)) != (load ? 0x1f0100u : 0x110100u) ||
                        (kind != 0 && kind != 2 && kind != 4 && kind != 5 && kind != 6) || boundary > 2 || (left != 255 && left >= 254))
                    fail("unsupported surface instruction encoding");
                unsigned lanes = kind < 4 ? 1 : 1u << (kind - 4), first = load ? dest : right;
                if (first != 255 && (first + lanes > 255 || (lanes > 1 && first % lanes))) fail("invalid surface register group");
                handle();
                std::string registers;
                for (unsigned i = 0; i < lanes; ++i) {
                    registers += (i ? "," : "") + (first == 255 ? (load ? "%tmp" + std::to_string(i) : "0") : reg(first + i));
                    if (load && first != 255) known[first + i].reset();
                }
                if (kind < 4) {
                    if (!load) output += guard + "cvt.u16.u32 %half0," + registers + ";\n";
                    registers = "%half0";
                } else if (lanes > 1) registers = "{" + registers + "}";
                const char* modes[]{"zero", "clamp", "trap"};
                auto operation = std::string(load ? "suld.b.2d" : "sust.b.2d") + (lanes > 1 ? ".v" + std::to_string(lanes) : "") +
                    (kind == 0 ? ".b8." : kind == 2 ? ".b16." : ".b32.") + modes[boundary];
                auto address = "[%wide0,{" + reg(left) + "," + (left == 255 ? std::string("0") : reg(left + 1)) + "}]";
                instruction = operation + " " + (load ? registers + "," + address : address + "," + registers) + ";";
                if (load && kind < 4 && first != 255) {
                    output += guard + instruction + "\n";
                    instruction = "cvt.u32.u16 " + reg(first) + ",%half0;";
                }
            }
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
                unsigned width = unsigned(hi >> 9) & 7;
                if (width != 4 && width != 5) fail("unsupported strong memory width/address");
                // Strong CTA reads provide the acquire behavior for which
                // ptxas emits no separate IVALL. GPU reads pair with IVALL.
                auto scope = strength ? "gpu.global" : "cta.global";
                auto kind = std::string(load ? (strength ? "ld.relaxed." : "ld.acquire.") : "st.relaxed.") + scope;
                if (width == 4) instruction = kind + ".u32 " +
                    (load ? (dest == 255 ? std::string("%scratch") : reg(dest)) + "," + address : address + "," + reg(right)) + ";";
                else {
                    unsigned first = load ? dest : right;
                    if (first >= 254 || first % 2) fail("invalid strong 64-bit register group");
                    if (load) {
                        output += guard + kind + ".u64 %wide0," + address + ";\n" + guard + "cvt.u32.u64 " + reg(first) + ",%wide0;\n" +
                            guard + "shr.u64 %wide0,%wide0,32;\n";
                        instruction = "cvt.u32.u64 " + reg(first + 1) + ",%wide0;"; known[first + 1].reset();
                    } else { join(reg(first), reg(first + 1), "%wide0"); instruction = kind + ".u64 " + address + ",%wide0;"; }
                }
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
            if ((hi & ~uint64_t(0xe00)) != 0x100000 || (load ? right != 0 : dest != 0) || (!kernel.stack_size && !calls))
                fail("unsupported local memory encoding");
            // LDL and STL both carry a signed 24-bit displacement in bits
            // 40..63. LDL differs here from the ordinary global-load form.
            auto displacement = int32_t(lo >> 32) >> 8;
            output += guard + "add.u32 %scratch," + reg(left) + "," + std::to_string(uint32_t(displacement)) + ";\n" +
                guard + "cvt.u64.u32 %local_address,%scratch;\n" + guard + "add.u64 %local_address,%local_base,%local_address;\n";
            instruction = memory_transfer(load, "local", unsigned(hi >> 9) & 7, load ? dest : right, "[%local_address]");
        } else if (op == 0x38c || op == 0x38d || op == 0x3a8 || op == 0x9a8 || op == 0x3a9 || op == 0x98e) {
            bool shared = op == 0x38c || op == 0x38d, cas = op == 0x38d || op == 0x3a9, reduction = op == 0x98e;
            bool cta = shared || !(hi & 0x4000);
            unsigned type = unsigned(hi >> 9) & 7, operation = cas ? 0 : unsigned(hi >> 23) & (reduction ? 7 : 15);
            const char* operations[]{"add", "min", "max", "inc", "dec", "and", "or", "xor", "exch"};
            bool spin = shared && cas && (hi & 0x7800000) == 0x1800000;
            if (operation >= std::size(operations) || (type != 0 && type != 1 && type != 2 && type != 3 && type != 5 && type != 6) ||
                    (shared && type != 0 && type != 1 && type != 2) || (cas && ((type != 0 && type != 2) || (!spin && (hi & 0x7800000)))) ||
                    ((type == 3 || type == 6) && operation != 0) || ((operation == 3 || operation == 4) && type != 0))
                fail(shared ? "unsupported shared atomic encoding" : "unsupported global atomic encoding");
            bool bits = cas || operation >= 5, wide = type == 2 || type == 5 || type == 6, floating = type == 3 || type == 6;
            std::string value_type = std::string(bits ? "b" : floating ? "f" : type == 1 || type == 5 ? "s" : "u") + (wide ? "64" : "32");
            bool uniform_address = op == 0x9a8 || (reduction && (hi & 0x8000000));
            std::string address;
            if (shared) {
                if ((hi & ~uint64_t(0x78046ff)) != 0 || (!cas && (hi & 255) != 255)) fail("unsupported shared atomic encoding");
                shared_address(hi & 0x4000 ? 2 : 0); address = "[%shared_address]";
            } else {
                auto scoped = (hi | 0x4000) & ~uint64_t(reduction ? 0x3800e00 : 0x7800e00);
                bool valid = uniform_address ? (scoped & ~uint64_t(63)) == (reduction ? 0xc114100u : 0x81f4140u) && left == 255 :
                    cas ? (scoped & ~uint64_t(255)) == 0x1f4100 : scoped == (reduction ? 0x114100u : 0x1f41ffu);
                if (!valid || (reduction && (dest != 0 || cas || operation == 8)) || (!uniform_address && (left >= 254 || left % 2)))
                    fail("unsupported global atomic encoding");
                if (uniform_address) {
                    auto base = unsigned(hi) & 63; if (base >= 62 || base % 2) fail("invalid uniform atomic address pair");
                    join(uniform(base), uniform(base + 1), "%a");
                } else join(reg(left), reg(left + 1), "%a");
                auto displacement = int32_t(lo >> 32) >> 8;
                address = "[%a" + std::string(displacement < 0 ? "-" : "+") +
                    std::to_string(displacement < 0 ? -int64_t(displacement) : displacement) + "]";
            }
            auto source = [&](unsigned id, unsigned slot) {
                auto value = "%tmp" + std::to_string(slot);
                if (wide) {
                    if (id != 255 && (id >= 254 || id % 2)) fail("invalid atomic 64-bit register group");
                    value = "%wide" + std::to_string(slot); join(reg(id), id == 255 ? "0" : reg(id + 1), value);
                } else output += guard + "mov.b32 " + value + "," + reg(id) + ";\n";
                if (floating) {
                    auto fp = std::string(wide ? "%dfp" : "%fp") + std::to_string(slot);
                    output += guard + (wide ? "mov.b64 " : "mov.b32 ") + fp + "," + value + ";\n"; return fp;
                }
                return value;
            };
            auto first = source(right, 0), second = cas ? source(unsigned(hi) & 255, 1) : "";
            auto result = floating ? (wide ? "%dfp2" : "%fp2") : wide ? "%wide2" : "%tmp2";
            auto prefix = std::string(reduction ? "red.relaxed." : cta && !shared ? "atom.acquire." : "atom.relaxed.") +
                (cta ? "cta." : "gpu.") + (shared ? "shared." : "global.");
            output += guard + prefix + (cas ? "cas" : operations[operation]) + "." + value_type + " " +
                (reduction ? "" : std::string(result) + ",") + address + "," + first + (cas ? "," + second : "") + ";\n";
            if (spin) output += guard + "setp.eq." + std::string(wide ? "u64" : "u32") + " %test0," + result + "," + first + ";\n" +
                guard + "selp." + (wide ? "u64 " : "u32 ") + result + ",1,0,%test0;\n";
            if (!reduction && dest != 255) {
                if (floating) output += guard + (wide ? "mov.b64 %wide2," : "mov.b32 %tmp2,") + std::string(result) + ";\n";
                if (wide) {
                    if (dest >= 254 || dest % 2) fail("invalid atomic 64-bit destination");
                    output += guard + "cvt.u32.u64 " + reg(dest) + ",%wide2;\n" + guard + "shr.u64 %wide2,%wide2,32;\n";
                    instruction = "cvt.u32.u64 " + reg(dest + 1) + ",%wide2;"; known[dest + 1].reset();
                } else instruction = "mov.b32 " + reg(dest) + ",%tmp2;";
            }
        } else if (op == 0xb1d) {
            auto mode = hi & ~uint64_t(0x7800000);
            if ((lo & ~uint64_t(0x3fffff000000000)) != 0x7b1d ||
                    (mode != 0 && mode != 0x2000 && mode != 0x4000 && mode != 0x4400 && mode != 0x4800) ||
                    (mode < 0x4000 && hi != mode)) fail("unsupported CTA barrier encoding");
            unsigned id = unsigned(lo >> 54) & 15, count = unsigned(lo >> 42) & 4095;
            auto operands = std::to_string(id) + (count ? "," + std::to_string(count) : "");
            if (mode >= 0x4000) {
                auto reduction = mode == 0x4000 ? "popc.u32 %barrier_result," : mode == 0x4400 ? "and.pred %barrier_pred," : "or.pred %barrier_pred,";
                instruction = std::string("bar.red.") + reduction + operands + "," + (hi & 0x4000000 ? "!" : "") + pred(unsigned(hi >> 23) & 7) + ";";
                if (mode == 0x4000) instruction += "\n" + guard + "setp.ne.u32 %barrier_pred,%barrier_result,0;";
            } else instruction = std::string(mode ? "bar.arrive " : "bar.sync ") + operands + ";";
        } else if (op == 0x31c && (lo >> 24) == 0 && (hi & ~uint64_t(0xe0000)) == 0x4000) {
            if (dest != 255) output += guard + "mov.b32 " + reg(dest) + ",%barrier_result;\n";
            unsigned target = unsigned(hi >> 17) & 7;
            if (target != 7) instruction = "and.pred " + pred(target) + ",%barrier_pred,%pt;";
        } else if (op == 0x3a1 && hi == 0xe8000 && !(lo >> 40)) {
            warp_members();
            if (auto mask = kernel.warp_masks.find(start + pc); mask != kernel.warp_masks.end() && mask->second != UINT32_MAX)
                output += guard + "and.b32 %members,%members," + reg(mask->second & 255) + ";\n";
            instruction = "match.any.sync.b32 " + (dest == 255 ? std::string("%scratch") : reg(dest)) + "," + reg(left) + ",%members;";
        } else if (op == 0x215) {
            if ((lo >> 40) || (hi & ~uint64_t(0x8ff)) != 0xe0000)
                fail("unsupported byte absolute-difference encoding");
            auto addend = unsigned(hi) & 255;
            if (!(hi & 0x800) && addend != 255 && known[addend] != 0)
                fail("byte absolute-difference merge requires a known zero operand");
            instruction = std::string("vabsdiff4.u32.u32.u32") + (hi & 0x800 ? ".add " : " ") +
                reg(dest) + "," + reg(left) + "," + reg(right) + "," + reg(addend) + ";";
        } else if (op == 0x216 || op == 0x816 || op == 0xa16) {
            if ((hi & ~uint64_t(255)) != 0 || (op == 0x216 && (lo >> 40))) fail("unsupported byte permutation encoding");
            auto control = op == 0xa16 ? bank_word() : op == 0x816 ? std::to_string(uint32_t(lo >> 32)) : reg(right);
            instruction = "prmt.b32 " + reg(dest) + "," + reg(left) + "," + reg(unsigned(hi) & 255) + "," + control + ";";
        } else if ((op == 0x212 || op == 0x812 || op == 0xa12 || op == 0xc12) &&
                   (hi & ~uint64_t(0xeffff)) == (op == 0xc12 ? 0xf800000u : 0x7800000u) &&
                   (op == 0x812 || op == 0xa12 || !(lo >> 40))) {
            auto second = op == 0xc12 ? uniform(right) : op == 0xa12 ? bank_word() :
                op == 0x812 ? std::to_string(uint32_t(lo >> 32)) : reg(right);
            output += guard + "lop3.b32 %scratch," + reg(left) + "," + second + "," + reg(unsigned(hi) & 255) + "," +
                std::to_string(unsigned(hi >> 8) & 255) + ";\n";
            unsigned destination = unsigned(hi >> 17) & 7;
            instruction = "mov.b32 " + (dest == 255 ? std::string("%scratch") : reg(dest)) + ",%scratch;";
            if (destination != 7) {
                output += guard + instruction + "\n";
                instruction = "setp.ne.u32 " + pred(destination) + ",%scratch,0;";
            }
        } else if ((op == 0x210 || op == 0x810 || op == 0xa10 || op == 0xc10) &&
                   (hi & ~uint64_t(0xe0000)) == (op == 0xc10 ? 0xff1e0ffu : 0x7f1e0ffu) &&
                   ((hi >> 17) & 7) != 7 && predicate == 7 &&
                   (op == 0x210 ? !((lo >> 40) & ~uint64_t(0x800000)) : op != 0xc10 || !(lo >> 40))) {
            // IADD3 Rd, Pn, Ra, +/-Rb/imm32, RZ. A negated register uses
            // a + ~b + 1, retaining the no-borrow carry for the high word.
            auto source = op == 0xa10 ? bank_word() : op == 0xc10 ? uniform(right) : op == 0x210 ? reg(right) : std::to_string(uint32_t(lo >> 32));
            output += "cvt.u64.u32 %a," + reg(left) + ";\ncvt.u64.u32 %b," + source + ";\n";
            if (op == 0x210 && (lo & (uint64_t(1) << 63))) output += "sub.u64 %b,4294967296,%b;\n";
            output += "add.u64 %a,%a,%b;\nsetp.gt.u64 " + pred(unsigned(hi >> 17) & 7) + ",%a,4294967295;\n";
            instruction = "cvt.u32.u64 " + reg(dest) + ",%a;";
        } else if ((op == 0x210 || op == 0xa10 || op == 0xc10) &&
                   (hi & ~uint64_t(0x3800000)) == (op == 0xc10 ? 0x87fe4ffu : 0x7fe4ffu) &&
                   (op == 0xa10 || (op == 0x210 ? !((lo >> 40) & ~uint64_t(0x800000)) : !(lo >> 40)))) {
            // IADD3.X Rd, Ra, Rb/~Rb, RZ, Pn, !PT.
            auto source = op == 0xa10 ? bank_word() : op == 0xc10 ? uniform(right) : reg(right);
            if (op == 0x210 && (lo & (uint64_t(1) << 63))) {
                output += guard + "not.b32 %tmp0," + source + ";\n"; source = "%tmp0";
            }
            output += guard + "selp.u32 %scratch,1,0," + pred(unsigned(hi >> 23) & 7) + ";\n" + guard + "add.u32 %scratch,%scratch," + reg(left) + ";\n";
            instruction = "add.u32 " + reg(dest) + ",%scratch," + source + ";";
        } else if ((op == 0x210 || op == 0x810 || op == 0xa10 || op == 0xc10) &&
                   (hi & ~uint64_t(0x9ff)) == (op == 0xc10 ? 0xfffe000u : 0x7ffe000u)) {
            if (op == 0x210 && ((lo >> 40) & ~uint64_t(0x800000))) fail("unsupported IADD3 register modifier");
            if (op == 0xc10 && (lo >> 40)) fail("unsupported IADD3 uniform operand modifier");
            auto source = op == 0xa10 ? bank_word() : op == 0xc10 ? uniform(right) : op == 0x210 ? reg(right) : std::to_string(uint32_t(lo >> 32));
            if (op == 0x210 && (lo & (uint64_t(1) << 63))) { output += guard + "sub.u32 %tmp0,0," + source + ";\n"; source = "%tmp0"; }
            auto first = reg(left);
            if (hi & 0x100) { output += guard + "sub.u32 %scratch,0," + first + ";\n"; first = "%scratch"; }
            auto second_known = op == 0x810 ? std::optional<uint32_t>(uint32_t(lo >> 32)) : op == 0x210 ? known[right] : std::nullopt;
            if (known[left] && second_known && known[unsigned(hi) & 255]) {
                uint32_t a = *known[left], b = *second_known;
                if (hi & 0x100) a = 0 - a;
                if (op == 0x210 && (lo & (uint64_t(1) << 63))) b = 0 - b;
                auto c = *known[unsigned(hi) & 255];if (hi & 0x800) c = 0 - c;
                constant = a + b + c;
            }
            auto third = reg(unsigned(hi) & 255);
            if (hi & 0x800) {output += guard + "sub.u32 %tmp1,0," + third + ";\n";third = "%tmp1";}
            output += guard + "add.u32 %scratch," + first + "," + source + ";\n";
            instruction = "add.u32 " + reg(dest) + ",%scratch," + third + ";";
        } else if (op == 0x2a4 && (hi & ~uint64_t(63)) == 0xf8e0200 && !(lo >> 40) && predicate == 7) {
            // Uniform values retain a distinct register file. Representing each
            // value per invocation preserves the checked, unpredicated forms.
            auto a = uniform(left), b = uniform(right), c = uniform(unsigned(hi) & 63);
            instruction = "mad.lo.u32 " + uniform(dest, true) + "," + a + "," + b + "," + c + ";";
        } else if (op == 0x8a5 && (hi & ~uint64_t(0x23f)) == 0xf8e0000 && predicate == 7) {
            auto source = uniform(left);
            auto addend = unsigned(hi) & 63;
            if (dest >= 62 || dest % 2 || (addend != 63 && (addend >= 62 || addend % 2))) fail("invalid uniform wide register group");
            join(uniform(addend), addend == 63 ? "0" : uniform(addend + 1), "%b");
            output += "mul.wide." + std::string(hi & 0x200 ? "s32" : "u32") + " %a," + source + "," + std::to_string(uint32_t(lo >> 32)) + ";\n"
                "add.u64 %a,%a,%b;\ncvt.u32.u64 " + uniform(dest, true) + ",%a;\nshr.u64 %a,%a,32;\n";
            instruction = "cvt.u32.u64 " + uniform(dest + 1, true) + ",%a;";
        } else if ((op == 0x224 || op == 0x824) && (hi & ~uint64_t(0x38002ff)) == 0xe0400 && (op == 0x824 || !(lo >> 40))) {
            auto a = reg(left), c = reg(unsigned(hi) & 255);
            auto b = op == 0x824 ? std::to_string(uint32_t(lo >> 32)) : reg(right);
            output += guard + "mad.lo.u32 %tmp0," + a + "," + b + "," + c + ";\n" +
                guard + "selp.u32 %tmp1,1,0," + pred(unsigned(hi >> 23) & 7) + ";\n";
            instruction = "add.u32 " + reg(dest) + ",%tmp0,%tmp1;";
        } else if (op == 0x224 || op == 0x424 || op == 0x824 || op == 0xa24 || op == 0xc24 || op == 0xe24 || op == 0x624 ||
                   op == 0x225 || op == 0x825 || op == 0xa25 || op == 0x625 || op == 0xc25 || op == 0xe25 || op == 0x227 || op == 0x827) {
            bool wide = (op & 15) == 5, high_product = (op & 15) == 7;
            unsigned form = op & 0xe00, addend = unsigned(hi) & 255;
            uint64_t expected = form == 0xc00 || form == 0xe00 ? 0xf8e0000 : 0x78e0000;
            auto modifiers = uint64_t(0x2ff) | (!wide && !high_product && (form == 0x200 || form == 0x800) ? 0x800u : 0u);
            bool wide_carry_out = wide && form == 0x200 && (hi & ~uint64_t(0xe00ff)) == 0x7800000;
            bool wide_carry_in = wide && form == 0x200 && (hi & ~uint64_t(0x78000ff)) == 0xe0400;
            if ((hi & ~modifiers) != expected && !wide_carry_out && !wide_carry_in)
                fail("unsupported IMAD modifier");
            if ((form == 0x200 || form == 0xc00 || form == 0xe00) && (lo >> 40)) fail("unsupported IMAD register modifier");
            if (pc == 0 && !function && predicate == 7 && dest == 1 && left == 255 && form == 0x600 && addend == 255 &&
                    bank_offset() == 0x28 && !bank_index()) {
                if (kernel.stack_size || calls) instruction = "mov.u32 %r1," + std::to_string(kernel.stack_size) + ";";
            } else {
                auto source = form == 0xa00 ? bank_word() : form == 0x400 ? reg(addend) : form == 0x800 ? std::to_string(uint32_t(lo >> 32)) :
                    form == 0xc00 ? uniform(right) : form == 0x600 || form == 0xe00 ? reg(addend) : reg(right);
                if (wide || high_product) {
                    if (wide && (dest >= 254 || dest % 2)) fail("invalid wide IMAD destination");
                    if (form == 0x600) {
                        if (right) fail("unsupported indexed constant-bank operand");
                        output += constant_load("%constant", "%scratch", bank_offset(), 8, guard, bank_index());
                        join("%constant", "%scratch", "%b");
                    } else if (addend == 255) output += guard + "mov.u64 %b,0;\n";
                    else {
                        if (addend >= 254 || addend % 2) fail(high_product ? "invalid IMAD high-product addend" : "invalid wide IMAD addend");
                        join(reg(addend), reg(addend + 1), "%b");
                    }
                    output += guard + std::string(hi & 0x200 ? "mul.wide.s32 " : "mul.wide.u32 ") + "%a," + reg(left) + "," + source + ";\n";
                    if (wide_carry_out && ((hi >> 17) & 7) != 7) {
                        // The product fits 64 bits; only adding the register
                        // pair can overflow. Capture it before aliased writes.
                        output += guard + "add.u64 %c,%a,%b;\n" + guard + "setp.lt.u64 " + pred(unsigned(hi >> 17) & 7)
                            + ",%c,%a;\n" + guard + "mov.u64 %a,%c;\n";
                    } else output += guard + "add.u64 %a,%a,%b;\n";
                    if (wide_carry_in) output += guard + "selp.u64 %c,1,0," + (hi & 0x4000000 ? "!" : "")
                        + pred(unsigned(hi >> 23) & 7) + ";\n" + guard + "add.u64 %a,%a,%c;\n";
                    if (high_product) {
                        output += guard + "shr.u64 %a,%a,32;\n";
                        instruction = "cvt.u32.u64 " + reg(dest) + ",%a;";
                    } else {
                        instruction = "cvt.u32.u64 " + reg(dest) + ",%a;";
                        // Split after reading aliased input/output pairs.
                        output += guard + instruction + "\n" + guard + "shr.u64 %a,%a,32;\n";
                        instruction = "cvt.u32.u64 " + reg(dest + 1) + ",%a;";
                        known[dest + 1].reset();
                    }
                } else {
                    // Low IMAD's bank-addend form carries the addend sign in
                    // bit 63 of the first word (including IMAD.MOV -c[...]).
                    // Wide products and other bank operands keep rejecting it.
                    auto third = form == 0x600 ? bank_word(bool(lo >> 63)) : form == 0xe00 ? uniform(right) : form == 0x400 ? std::to_string(uint32_t(lo >> 32)) : reg(addend);
                    if (hi & 0x800) {output += guard + "sub.u32 %tmp0,0," + third + ";\n";third = "%tmp0";}
                    auto second_known = form == 0x400 ? known[addend] : form == 0x800 ? std::optional<uint32_t>(uint32_t(lo >> 32)) :
                        form == 0x200 ? known[right] : std::nullopt;
                    auto third_known = form == 0x400 ? std::optional<uint32_t>(uint32_t(lo >> 32)) :
                        form == 0x200 || form == 0x800 || form == 0xa00 || form == 0xc00 ? known[addend] : std::nullopt;
                    if (known[left] && second_known && third_known) {
                        auto c = *third_known;if (hi & 0x800) c = 0 - c;
                        constant = *known[left] * *second_known + c;
                    }
                    instruction = std::string("mad.lo.") + (hi & 0x200 ? "s32 " : "u32 ") + reg(dest) + "," + reg(left) + "," + source + "," + third + ";";
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
        } else if ((op == 0x211 || op == 0x811) && (hi & ~uint64_t(0xf800)) == 0x78e00ff && (op == 0x811 || !(lo >> 40))) {
            unsigned shift = unsigned(hi >> 11) & 31;
            output += guard + "shl.b32 %scratch," + reg(left) + "," + std::to_string(shift) + ";\n";
            instruction = "add.u32 " + reg(dest) + ",%scratch," + (op == 0x811 ? std::to_string(uint32_t(lo >> 32)) : reg(right)) + ";";
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
            if ((lo >> 40) || left || (hi & ~(uint64_t(0xc000) | signed_bit | (to_integer ? uint64_t(0x10000) : 0))) != (to_integer ? 0x203000u : 0x201000u))
                fail("unsupported numeric conversion encoding");
            const char* modes[]{"rn", "rm", "rp", "rz"};
            auto mode = std::string(modes[(hi >> 14) & 3]);
            auto integer = hi & signed_bit ? "s32" : "u32";
            auto destination = dest == 255 ? std::string("%scratch") : reg(dest);
            if (!from_integer) output += guard + "mov.b32 %f0," + reg(right) + ";\n";
            output += guard + "cvt." + mode + (from_integer ? ".f32." + std::string(integer) + " %f0," + reg(right) :
                "i." + std::string(to_integer && (hi & 0x10000) ? "ftz." : "") + (to_integer ? integer : "f32") + ".f32 " + (to_integer ? destination : "%f0") + ",%f0") + ";\n";
            if (!to_integer) instruction = "mov.b32 " + destination + ",%f0;";
        } else if (op == 0x221 || op == 0x421 || op == 0xa21 || op == 0x220 || op == 0x820 || op == 0xa20 || op == 0x223 || op == 0x423 || op == 0x823 || op == 0xa23) {
            bool add = op == 0x221 || op == 0x421 || op == 0xa21, multiply = op == 0x220 || op == 0x820 || op == 0xa20;
            bool fma = !add && !multiply, immediate_addend = op == 0x423;
            bool bank = (op & 0xe00) == 0xa00;
            bool immediate = op == 0x421 || op == 0x820 || op == 0x823 || immediate_addend;
            uint64_t modifiers = 0x1e000 | (add ? 0x300 : multiply ? 0 : 0x9ff);
            if ((hi & ~modifiers) != (multiply ? 0x400000u : 0u) ||
                (!immediate && !bank && ((lo >> 40) & ~(add ? uint64_t(0xc00000) : fma ? uint64_t(0x800000) : uint64_t(0)))))
                fail("unsupported FP32 arithmetic modifier");
            auto second = bank ? bank_word() : immediate_addend ? reg(unsigned(hi) & 255) : immediate ? std::to_string(uint32_t(lo >> 32)) : reg(right);
            auto source = [&](unsigned slot, const std::string& bits, bool absolute = false, bool negate = false) {
                auto name = "%f" + std::to_string(slot);
                output += guard + "mov.b32 " + name + "," + bits + ";\n";
                if (absolute) output += guard + "abs.f32 " + name + "," + name + ";\n";
                if (negate) output += guard + "neg.f32 " + name + "," + name + ";\n";
            };
            source(0, reg(left), add && (hi & 0x200), (add || fma) && (hi & 0x100));
            source(1, second, add && !immediate && !bank && (lo & (uint64_t(1) << 62)),
                (!immediate && !bank && (lo & (uint64_t(1) << 63))) || (immediate_addend && (hi & 0x800)));
            if (fma) source(2, immediate_addend ? std::to_string(uint32_t(lo >> 32)) : reg(unsigned(hi) & 255), false, !immediate_addend && (hi & 0x800));
            const char* modes[]{"rn", "rm", "rp", "rz"};
            output += guard + std::string(add ? "add" : multiply ? "mul" : "fma") + "." + modes[(hi >> 14) & 3] +
                (hi & 0x10000 ? ".ftz" : "") + (hi & 0x2000 ? ".sat" : "") + ".f32 %f0,%f0,%f1" + (!add && !multiply ? ",%f2" : "") + ";\n";
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
        } else if ((op == 0x208 || op == 0x808 || op == 0x207 || op == 0x807) && (op == 0x807 || op == 0x808 || !(lo >> 40)) && (hi & ~uint64_t(0x7800000)) == 0) {
            auto positive = reg(left), negative = op == 0x807 || op == 0x808 ? std::to_string(uint32_t(lo >> 32)) : reg(right);
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
        } else if (op == 0x947 && (hi & ~uint64_t(0x783ffff)) == 0 && (lo & 0xffff0000) == 0) {
            if (track_cohort && !unconditional(lo, hi) && reachable.contains(pc)) {
                output += "and.pred %branch_taken,%instruction_guard,%pt;\n";
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
