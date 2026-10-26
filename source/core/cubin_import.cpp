#include "cubin_import.hpp"
#include "compiler.hpp"
#include "ptx_import.hpp"
#include "../nvfatbin/container_api.hpp"
#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <set>
#include <llvm/IR/Module.h>

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
    uint64_t flags, offset, size, stride;
};
struct cubin_parameter { unsigned offset, size; };
struct cubin_reader {
    std::string_view bytes;
    std::vector<cubin_section> sections;
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
                uint32_t(number(at + 44)), number(at + 8, 8), number(at + 24, 8), number(at + 32, 8), number(at + 56, 8)};
            cubin_check(unique.insert(section.name).second, "duplicate section name");
            cubin_check(section.type != 11, "dynamic symbols are unsupported", true);
            sections.push_back(std::move(section));
        }
        for (const auto& section : sections) {
            if ((section.type == 4 || section.type == 9) && section.size) {
                cubin_check(section.info < sections.size(), "invalid relocation target");
                cubin_check(sections[section.info].name.starts_with(".debug"), "code/data relocations are unsupported", true);
            }
            if ((section.flags & 2) && !section.name.starts_with(".text.") && !section.name.starts_with(".nv.constant0."))
                cubin_check(section.size == 0, "global, shared and additional constant storage are unsupported", true);
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
struct cubin_kernel {
    std::string name;
    const cubin_section* text;
    std::vector<cubin_parameter> parameters;
    unsigned base = 0, size = 0;
};
std::vector<cubin_kernel> cubin_kernels(const cubin_reader& input) {
    std::vector<cubin_kernel> kernels; std::set<std::string> names;
    for (const auto& table : input.sections) if (table.type == 2) {
        cubin_check(table.link < input.sections.size() && table.stride == 24 && table.size % 24 == 0, "invalid symbol table");
        cubin_check(input.sections[table.link].type == 3, "invalid symbol name table");
        auto strings = input.data(input.sections[table.link]); fatbin_reader symbols{input.data(table)};
        for (size_t at = 0; at < table.size; at += 24) if ((symbols.number(at + 4, 1) & 15) == 2) {
            cubin_check((symbols.number(at + 5, 1) & 0x10) != 0, "device functions are unsupported", true);
            auto section = symbols.number(at + 6, 2); cubin_check(section < input.sections.size(), "invalid function section");
            auto name = input.string(strings, size_t(symbols.number(at)));
            cubin_check(!name.empty() && name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.$") == name.npos
                && !(name.front() >= '0' && name.front() <= '9'), "unsupported kernel symbol spelling", true);
            const auto& text = input.sections[section];
            cubin_check(text.type == 1 && text.flags == 6, "unsupported code section", true);
            cubin_check(text.name == ".text." + name && symbols.number(at + 8, 8) == 0 && symbols.number(at + 16, 8) == text.size,
                "unsupported function extent", true);
            cubin_check(names.insert(name).second && kernels.size() < 256, "duplicate/excessive kernels");
            cubin_kernel kernel{name, &text, {}}; std::map<unsigned, cubin_parameter> layout; bool bank = false, extent = false; unsigned declared_size = 0;
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
                else if (attr == 0x4) cubin_check(format == 1 && size == 0, "invalid CTAID.Z usage record");
                else if (attr == 0x1c) {
                    cubin_check(format == 4 && size % 4 == 0, "invalid exit offsets");
                    for (unsigned offset = 0; offset < size; offset += 4)
                        cubin_check(record.number(offset) < text.size && record.number(offset) % 16 == 0, "invalid exit offset");
                } else cubin_fail("unsupported kernel information attribute " + std::to_string(attr), true);
            });
            cubin_check(bank && extent, "missing parameter metadata");
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
    input.records(input.named(".nv.info"), [&](unsigned format, unsigned attr, unsigned size, const fatbin_reader& record) {
        cubin_check(format == 4 && size == 8 && (attr == 0x2f || attr == 0x11 || attr == 0x12), "unsupported module information", true);
        if (attr != 0x2f) cubin_check(record.number(4) == 0, "stack frames are unsupported", true);
    });
    return kernels;
}
std::string lift_cubin_kernel(const cubin_reader& input, const cubin_kernel& kernel) {
    auto code = input.data(*kernel.text); fatbin_reader words{code};
    cubin_check(!code.empty() && code.size() % 16 == 0 && code.size() <= 65536, "invalid/excessive SM75 code extent");
    std::set<size_t> targets;
    auto branch_target = [&](size_t pc, uint64_t lo, uint64_t hi) {
        auto displacement = int32_t(lo >> 32);
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
        if ((op != 0x947 && op != 0x94d) || predicate != 7) pending.push_back(pc + 16);
    }
    bool has_branch = std::any_of(reachable.begin(), reachable.end(), [&](size_t pc) { return (words.number(pc, 8) & 0xfff) == 0x947; });
    std::string output = ".visible .entry " + kernel.name + "(";
    for (unsigned i = 0; i < kernel.parameters.size(); ++i)
        output += (i ? "," : "") + std::string(".param .b") + std::to_string(kernel.parameters[i].size * 8) + " arg" + std::to_string(i);
    output += "){ .reg .b32 %r<255>,%u<63>,%scratch,%constant; .reg .b64 %a,%b,%c; .reg .f32 %f<3>; .reg .pred %p<7>,%pt;\nsetp.eq.u32 %pt,0,0;\n";
    std::array<std::optional<uint32_t>, 256> known{}; known[255] = 0;
    auto reg = [&](unsigned id) { cubin_check(id <= 255 && id != 1, "stack-register use is unsupported", true); return id == 255 ? std::string("0") : "%r" + std::to_string(id); };
    auto uniform = [&](unsigned id) { cubin_check(id <= 63, "invalid uniform register"); return id == 63 ? std::string("0") : "%u" + std::to_string(id); };
    auto pred = [](unsigned id) { return id == 7 ? std::string("%pt") : "%p" + std::to_string(id); };
    auto constant_load = [&](const std::string& dest, const std::string& next, unsigned offset, unsigned size, const std::string& guard) {
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
        // Per-invocation copies implement uniform values along the checked
        // straight-line/early-exit paths. A uniform write in one divergent arm
        // can affect other lanes after reconvergence; that needs a separate
        // warp-state contract. Do not silently treat it as a thread register.
        if (has_branch && (op == 0xab9 || op == 0x9c3 || op == 0x2a4 || op == 0xc24 || op == 0x981 || op == 0x986))
            fail("uniform registers across branches are unsupported");
        std::string guard = predicate == 7 ? "" : "@" + std::string(predicate & 8 ? "!" : "") + pred(predicate & 7) + " ";
        auto bank_offset = [&] {
            if (lo & 0xffc0000000000000ull) fail("unsupported constant-bank operand");
            return unsigned(lo >> 40) * 4;
        };
        auto bank_word = [&] {
            if (right) fail("unsupported indexed constant-bank operand");
            output += constant_load("%constant", "", bank_offset(), 4, guard);
            return std::string("%constant");
        };
        auto join = [&](const std::string& low, const std::string& high, const std::string& name) {
            output += guard + "cvt.u64.u32 " + name + "," + low + ";\n" + guard + "cvt.u64.u32 %c," + high +
                ";\n" + guard + "shl.b64 %c,%c,32;\n" + guard + "or.b64 " + name + "," + name + ",%c;\n";
        };
        output += "L" + std::to_string(pc) + ":\n";
        std::string instruction; std::optional<uint32_t> constant;
        if (op == 0x202 && hi == 0xf00 && (lo & 0xffffff00ff000000ull) == 0) {
            instruction = "mov.b32 " + reg(dest) + "," + reg(right) + ";"; constant = known[right];
        } else if (op == 0x802 && hi == 0xf00 && left == 0) {
            instruction = "mov.b32 " + reg(dest) + "," + std::to_string(uint32_t(lo >> 32)) + ";"; constant = uint32_t(lo >> 32);
        } else if (op == 0xa02 && hi == 0xf00 && (lo & 0xffc000ffff000000ull) == 0) {
            unsigned offset = unsigned(lo >> 40) * 4;
            if (pc == 0 && predicate == 7 && dest == 1 && offset == 0x28) { /* dead stack-pointer prologue; all reads of R1 reject */ }
            else output += constant_load(reg(dest), "", offset, 4, guard);
        } else if (op == 0xb82 && (hi == 0x800 || hi == 0xa00) && (lo & 0xffc000ff00000000ull) == 0) {
            if (!known[left]) fail("dynamic constant-bank indexing is unsupported");
            unsigned width = hi == 0xa00 ? 8 : 4;
            auto offset = uint64_t(unsigned(lo >> 40)) * 4 + *known[left];
            if (offset > UINT32_MAX) fail("constant-bank offset overflow");
            cubin_check(dest + width / 4 <= 255, "invalid constant destination");
            output += constant_load(reg(dest), width == 8 ? reg(dest + 1) : "", unsigned(offset), width, guard);
            if (width == 8) known[dest + 1].reset();
        } else if (op == 0xab9 && (hi == 0x800 || hi == 0xa00) && left == 0 && right == 0 && predicate == 7) {
            unsigned width = hi == 0xa00 ? 8 : 4;
            if (dest + width / 4 > 63 || (width == 8 && dest % 2)) fail("invalid uniform constant destination");
            output += constant_load(uniform(dest), width == 8 ? uniform(dest + 1) : "", bank_offset(), width, guard);
        } else if ((op == 0x919 || op == 0x9c3) && (lo >> 24) == 0 && (hi & ~uint64_t(0xff00)) == 0) {
            unsigned special = unsigned(hi >> 8);
            const char* component = special >= 0x21 && special <= 0x23 ? "%tid." :
                special >= 0x25 && special <= 0x27 ? "%ctaid." : nullptr;
            if (!component) fail("unsupported special register " + std::to_string(special));
            if (op == 0x9c3 && (special < 0x25 || predicate != 7)) fail("unsupported uniform special register");
            unsigned axis = special >= 0x25 ? special - 0x25 : special - 0x21;
            instruction = "mov.u32 " + (op == 0x9c3 ? uniform(dest) : reg(dest)) + "," + component + "xyz"[axis] + ";";
        } else if ((op == 0x381 && hi == 0x1ee900) || (op == 0x981 && hi == 0xc1ee900) ||
                   (op == 0x986 && (hi & ~uint64_t(63)) == 0xc10e900) || (op == 0x386 && hi == 0x10e900)) {
            bool uniform_address = op == 0x981 || op == 0x986, load = op == 0x381 || op == 0x981;
            if (left >= 254 && !(uniform_address && left == 255)) fail("unsupported extended address register");
            if (!load && dest != 0) fail("unsupported global store modifier");
            if (left == 255) output += guard + "mov.u64 %a,0;\n";
            else join(reg(left), reg(left + 1), "%a");
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
            instruction = load ? "ld.global.u32 " + reg(dest) + "," + address + ";" : "st.global.u32 " + address + "," + reg(right) + ";";
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
        } else if ((op == 0x210 || op == 0x810 || op == 0xa10) && (hi & ~uint64_t(255)) == 0x7ffe000) {
            if (op == 0x210 && (lo >> 40)) fail("unsupported IADD3 register modifier");
            auto source = op == 0xa10 ? bank_word() : op == 0x210 ? reg(right) : std::to_string(uint32_t(lo >> 32));
            output += guard + "add.u32 %scratch," + reg(left) + "," + source + ";\n";
            instruction = "add.u32 " + reg(dest) + ",%scratch," + reg(unsigned(hi) & 255) + ";";
        } else if (op == 0x2a4 && (hi & ~uint64_t(63)) == 0xf8e0200 && !(lo >> 40) && predicate == 7) {
            // Uniform values retain a distinct register file. Representing each
            // value per invocation preserves the checked, unpredicated forms.
            instruction = "mad.lo.u32 " + uniform(dest) + "," + uniform(left) + "," + uniform(right) + "," + uniform(unsigned(hi) & 63) + ";";
        } else if (op == 0x224 || op == 0x824 || op == 0xa24 || op == 0xc24 || op == 0x624 ||
                   op == 0x225 || op == 0x825 || op == 0xa25 || op == 0x625 || op == 0x227 || op == 0x827) {
            bool wide = (op & 15) == 5, high_product = (op & 15) == 7;
            unsigned form = op & 0xe00, addend = unsigned(hi) & 255;
            uint64_t expected = form == 0xc00 ? 0xf8e0000 : 0x78e0000;
            if ((hi & ~uint64_t(0x2ff)) != expected || ((wide || high_product) && (hi & 0x200)))
                fail("unsupported IMAD modifier");
            if ((form == 0x200 || form == 0xc00) && (lo >> 40)) fail("unsupported IMAD register modifier");
            if (high_product && addend != 255) fail("unsupported IMAD high-product addend");
            if (pc == 0 && predicate == 7 && dest == 1 && left == 255 && form == 0x600 && addend == 255 && bank_offset() == 0x28) {
                // IMAD.MOV is another dead stack-pointer prologue. R1 reads
                // still reject, including through a paired address operand.
            } else {
                auto source = form == 0xa00 ? bank_word() : form == 0x800 ? std::to_string(uint32_t(lo >> 32)) :
                    form == 0xc00 ? uniform(right) : form == 0x600 ? reg(addend) : reg(right);
                if (wide) {
                    if (dest >= 254 || dest % 2) fail("invalid wide IMAD destination");
                    if (form == 0x600) {
                        if (right) fail("unsupported indexed constant-bank operand");
                        output += constant_load("%constant", "%scratch", bank_offset(), 8, guard);
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
                    auto third = form == 0x600 ? bank_word() : reg(addend);
                    instruction = std::string(high_product ? "mad.hi.u32 " : "mad.lo.u32 ") + reg(dest) + "," + reg(left) + "," + source + "," + third + ";";
                }
            }
        } else if (op == 0x819 && hi == 0x6ff && uint32_t(lo >> 32) < 32) {
            instruction = "shl.b32 " + reg(dest) + "," + reg(left) + "," + std::to_string(uint32_t(lo >> 32)) + ";";
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
        } else if ((op == 0x20c || op == 0xa0c) && dest == 0 && (op != 0x20c || !(lo >> 40)) &&
                   (hi & ~uint64_t(0xe7000)) == 0x3f00070) {
            unsigned comparison = unsigned(hi >> 12) & 7, destination = unsigned(hi >> 17) & 7;
            const char* comparisons[]{"", "lt", "eq", "le", "gt", "ne", "ge", ""};
            if (comparison == 0 || comparison == 7 || destination == 7) fail("unsupported integer comparison modifier");
            auto source = op == 0xa0c ? bank_word() : reg(right);
            instruction = "setp." + std::string(comparisons[comparison]) + ".u32 " + pred(destination) + "," + reg(left) + "," + source + ";";
        } else if (op == 0x81c && !(lo >> 16)) {
            auto form = hi & ~uint64_t(0x38e0000);
            if (form != 0x70e170 && form != 0x70f070) fail("unsupported predicate LUT");
            auto destination = unsigned(hi >> 17) & 7, source = unsigned(hi >> 23) & 7;
            if (destination == 7) fail("unsupported predicate LUT destination");
            instruction = std::string(form == 0x70e170 ? "not.pred " : "and.pred ") + pred(destination) + "," + pred(source) +
                (form == 0x70e170 ? ";" : ",%pt;");
        } else if (op == 0x947 && (hi & ~uint64_t(0x3ffff)) == 0x3800000 && (lo & 0xffff0000) == 0) {
            instruction = "bra L" + std::to_string(branch_target(pc, lo, hi)) + ";";
        } else if (op == 0x94d && hi == 0x3800000 && !(lo >> 16)) instruction = "ret;";
        else if (op == 0x918 && hi == 0 && !(lo >> 16)) { }
        else fail("unsupported SM75 instruction encoding " + std::to_string(lo) + "/" + std::to_string(hi));
        if (!instruction.empty()) output += guard + instruction + "\n";
        if (op != 0x386 && op != 0x986 && op != 0x947 && op != 0x94d && op != 0x918 && op != 0x20c && op != 0xa0c && op != 0x20b && op != 0x81c &&
            op != 0xab9 && op != 0x9c3 && op != 0x2a4)
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
    std::string semantics = ".version 6.5\n.target sm_75\n.address_size 64\n";
    for (const auto& kernel : kernels) semantics += lift_cubin_kernel(input, kernel);
    // Share register SSA, predication, ABI construction and semantic identities
    // with PTX import. This text is reconstructed from decoded machine code;
    // no PTX member, disassembler executable or NVIDIA tool is used at runtime.
    auto module = import_ptx(context, semantics, architecture);
    module->getOrInsertNamedMetadata("cudavm.import.cubin.sm75");
    return module;
}
}
