#pragma once
#include "compiler.hpp"
#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Lex/PPCallbacks.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Lex/MacroInfo.h>
#include <clang/Lex/Lexer.h>

namespace cudavm {
inline std::string source_literal(const std::string& value) {
    std::string out = "\"";
    for (char c : value) {
        if (c == '\\' || c == '"') out += '\\';
        out += c;
    }
    return out + '"';
}

// AST ranges address the expanded tokens, rather than text in macro definitions.
// User headers/conditionals are resolved once by this very same preprocessor.
// System declarations remain ordinary includes for the eventual host compiler.
struct host_source : clang::PPCallbacks {
    clang::Preprocessor& pp;
    clang::SourceManager& sources;
    host_extraction& result;
    std::string host;
    std::filesystem::path builtin;
    struct item { clang::SourceLocation location; std::string text; bool directive = false, unsupported = false; };
    struct edit { size_t begin, end; std::string text; };
    std::vector<item> items;
    std::vector<edit> edits;
    std::unordered_set<size_t> omitted_attributes;
    std::unordered_map<unsigned, size_t> locations;
    std::map<std::string, std::string> macros;
    std::map<const clang::FileEntry*, size_t> headers;
    mutable std::unordered_map<const clang::FileEntry*, bool> real_files;
    host_source(clang::CompilerInstance& compiler, host_extraction& out, const compiler_options& values)
        : pp(compiler.getPreprocessor()), sources(compiler.getSourceManager()), result(out),
          host(std::filesystem::absolute(values.host).generic_string()),
          builtin(std::filesystem::absolute(values.resources.resource_dir).lexically_normal()) {
        for (const auto& define : values.defines) {
            auto equal = define.find('=');
            auto name = define.substr(0, equal);
            // -D may define a function-like macro. Macro history and undef/
            // push/pop directives address its identifier, not its parameters.
            if (auto parameters = name.find('('); parameters != std::string::npos) name.resize(parameters);
            macros[name] = "#define " + name + " " + (equal == std::string::npos ? "1" : define.substr(equal + 1));
        }
        pp.setTokenWatcher([this](const clang::Token& token) {
            if (token.isAnnotation() || token.is(clang::tok::eof) || !user(token.getLocation())) return;
            locations.emplace(token.getLocation().getRawEncoding(), items.size());
            items.push_back({token.getLocation(), pp.getSpelling(token)});
        });
    }
    bool embedded(clang::FileEntryRef file) const {
        auto relative = std::filesystem::absolute(file.getName().str()).lexically_normal().lexically_relative(builtin);
        return !relative.empty() && *relative.begin() != "..";
    }
    bool user(clang::SourceLocation location) const {
        if (location.isInvalid()) return false;
        auto expansion = sources.getExpansionLoc(location);
        // The overwhelming majority of CUDA/std tokens are system declarations.
        // Do not normalize a filesystem path for every one, especially in Debug.
        if (sources.isInSystemHeader(expansion)) return false;
        auto file = sources.getFileEntryRefForID(sources.getFileID(expansion));
        if (!file) return false;
        auto [entry, added] = real_files.emplace(&file->getFileEntry(), false);
        if (added) entry->second = !embedded(*file);
        return entry->second;
    }
    // Freeze user macro state only around a preserved directive. Expanded user
    // code and generated registration names must never be macro-expanded again.
    std::string directive(const std::string& text, clang::SourceLocation location) const {
        std::string output;
        for (const auto& [name, value] : macros) {
            (void)value;
            output += "#pragma push_macro(" + source_literal(name) + ")\n#undef " + name + "\n";
            auto* info = pp.getMacroDefinitionAtLoc(pp.getIdentifierInfo(name), sources.getExpansionLoc(location)).getMacroInfo();
            if (info) {
                if (info->isBuiltinMacro()) throw std::runtime_error("user-modified builtin macro state around host directives is pending: " + name);
                output += definition(name, *info) + "\n";
            }
        }
        output += text + "\n";
        for (auto it = macros.rbegin(); it != macros.rend(); ++it)
            output += "#pragma pop_macro(" + source_literal(it->first) + ")\n";
        return output;
    }
    void FileChanged(clang::SourceLocation location, FileChangeReason reason,
                     clang::SrcMgr::CharacteristicKind kind, clang::FileID) override {
        if (reason != EnterFile || kind != clang::SrcMgr::C_User || !user(location)) return;
        auto id = sources.getFileID(location);
        auto file = *sources.getFileEntryRefForID(id);
        result.dependencies.push_back(file.getName().str());
        if (id != sources.getMainFileID() && headers.emplace(&file.getFileEntry(), headers.size()).second)
            result.headers.push_back({std::filesystem::absolute(file.getName().str()).lexically_normal().generic_string(), host});
    }
    void InclusionDirective(clang::SourceLocation hash, const clang::Token&, llvm::StringRef filename,
                            bool angled, clang::CharSourceRange, clang::OptionalFileEntryRef file,
                            llvm::StringRef, llvm::StringRef, const clang::Module*, bool,
                            clang::SrcMgr::CharacteristicKind kind) override {
        if (!file || !user(hash)) return;
        if (embedded(*file)) {
            // Standard compiler headers belong to the host compiler when the
            // generated C++ is built. Never publish our virtual resource path.
            items.push_back({hash, "#include " + (angled ? "<" + filename.str() + ">" : source_literal(filename.str())), true});
        } else if (kind == clang::SrcMgr::C_User) result.dependencies.push_back(file->getName().str());
        else items.push_back({hash, "#include " + source_literal(
            std::filesystem::absolute(file->getName().str()).lexically_normal().generic_string()), true});
    }
    std::string definition(const std::string& key, const clang::MacroInfo& info) const {
        std::string value = "#define " + key;
        if (info.isFunctionLike()) {
            value += '(';
            auto params = info.params();
            for (size_t i = 0; i < params.size(); ++i) {
                if (i) value += ',';
                if (info.isC99Varargs() && i + 1 == params.size()) value += "...";
                else {
                    value += params[i]->getName().str();
                    if (info.isGNUVarargs() && i + 1 == params.size()) value += "...";
                }
            }
            value += ')';
        }
        value += ' ';
        // Keep original token adjacency: stringification, pasting and literal
        // suffixes must retain the definition seen by an included system file.
        for (const auto& token : info.tokens()) {
            if (token.hasLeadingSpace()) value += ' ';
            value += pp.getSpelling(token);
        }
        return value;
    }
    void MacroDefined(const clang::Token& name, const clang::MacroDirective*) override {
        if (user(name.getLocation())) macros[name.getIdentifierInfo()->getName().str()] = "";
    }
    void MacroUndefined(const clang::Token& name, const clang::MacroDefinition&, const clang::MacroDirective*) override {
        if (user(name.getLocation())) macros[name.getIdentifierInfo()->getName().str()] = "";
    }
    void PragmaDirective(clang::SourceLocation location, clang::PragmaIntroducerKind kind) override {
        if (!user(location)) return;
        if (kind != clang::PIK_HashPragma || location.isMacroID()) {
            // A device body can erase this event. Diagnose it only if it would
            // survive into host output; do not silently discard host ABI pragmas.
            items.push_back({location, "macro-generated host pragmas are not supported by this preprocessing profile", true, true});
            return;
        }
        auto [file, offset] = sources.getDecomposedLoc(location);
        auto buffer = sources.getBufferData(file);
        auto end = buffer.find('\n', offset);
        auto line = buffer.slice(offset, end == llvm::StringRef::npos ? buffer.size() : end).trim();
        if (line.consume_front("#")) line = line.ltrim();
        if (line.consume_front("pragma")) line = line.ltrim();
        // These are preprocessing state, already reflected by the collected
        // tokens and live macro snapshots, rather than host ABI directives.
        if (line == "once" || line.starts_with("push_macro") || line.starts_with("pop_macro")) return;
        items.push_back({location, "#pragma " + line.str(), true});
    }
    std::pair<size_t, size_t> span(clang::SourceRange range) const {
        if (range.isInvalid()) throw std::runtime_error("dependent/default expression extraction requires a concrete source range (pending)");
        auto begin = locations.find(range.getBegin().getRawEncoding());
        auto end = locations.find(range.getEnd().getRawEncoding());
        if (end == locations.end()) {
            // Clang splits a lexed >> while parsing nested template closers.
            // The watcher still has the original token at the preceding byte.
            auto split = locations.find(range.getEnd().getLocWithOffset(-1).getRawEncoding());
            if (split != locations.end() && items[split->second].text == ">>") end = split;
        }
        if (begin == locations.end() || end == locations.end())
            throw std::runtime_error("AST source range cannot be mapped to expanded user tokens");
        if (begin->second > end->second) throw std::runtime_error("reversed expanded source range");
        return {begin->second, end->second + 1};
    }
    std::string text(clang::SourceRange range) const {
        auto [begin, end] = span(range);
        std::string out;
        for (size_t i = begin; i < end;) {
            auto change = std::find_if(edits.begin(), edits.end(), [&](const edit& value) { return value.begin == i && value.end <= end; });
            if (change != edits.end()) { out += change->text + ' '; i = change->end; }
            else { if (!items[i].directive) out += items[i].text + ' '; ++i; }
        }
        return out;
    }
    void replace(clang::SourceRange range, const std::string& value) {
        auto [begin, end] = span(range);
        for (const auto& change : edits) {
            if (change.begin == begin && change.end == end) {
                if (change.text == value) return;
                throw std::runtime_error("context-dependent template extraction is pending");
            }
            if (change.begin < end && begin < change.end && !(begin <= change.begin && change.end <= end))
                throw std::runtime_error("overlapping expanded AST edits cannot be published");
        }
        std::erase_if(edits, [&](const edit& change) { return begin <= change.begin && change.end <= end; });
        edits.push_back({begin, end, value});
    }
    void omit_device_alias(clang::SourceRange range) {
        auto [begin, end] = span(range);
        if (items[begin].text != "alias" && items[begin].text != "__alias__")
            throw std::runtime_error("CUDA-only helper alias extraction requires a GNU alias attribute");
        omitted_attributes.insert(begin);
    }
    void omit_device_weak(clang::SourceRange range) {
        auto [begin, end] = span(range);
        if (items[begin].text != "weak" && items[begin].text != "__weak__")
            throw std::runtime_error("CUDA weak storage extraction requires a GNU weak attribute");
        omitted_attributes.insert(begin);
    }
    std::string joined(size_t begin, size_t end) const {
        std::string output;
        for (size_t i = begin; i < end; ++i) output += items[i].text + ' ';
        return output;
    }
    // CUDA execution/storage attributes have no ordinary-host counterpart.
    // Preserve other attributes; translate the CUDA wrapper's Windows alignment
    // and inlining spellings to ones understood by both Clang and MSVC.
    bool attribute(size_t& position, std::string& output) const {
        auto start = position;
        bool gnu = items[start].text == "__attribute__" || items[start].text == "__attribute";
        bool declspec = items[start].text == "__declspec";
        if (!gnu && !declspec) return false;
        size_t first = start + (gnu ? 3 : 2);
        if (first >= items.size() || items[start + 1].text != "(" || (gnu && items[start + 2].text != "(")) return false;
        size_t end = first;
        int depth = 0;
        for (; end < items.size(); ++end) {
            if (items[end].text == "(" ) ++depth;
            else if (items[end].text == ")") { if (!depth) break; --depth; }
        }
        if (end == items.size() || (gnu && (end + 1 == items.size() || items[end + 1].text != ")")))
            throw std::runtime_error("unterminated expanded attribute");
        std::vector<std::pair<size_t, size_t>> parts;
        size_t part = first;
        depth = 0;
        for (size_t i = first; i <= end; ++i) {
            if (i == end || (items[i].text == "," && !depth)) { if (part != i) parts.push_back({part, i}); part = i + 1; }
            else if (items[i].text == "(") ++depth;
            else if (items[i].text == ")") --depth;
        }
        static const std::unordered_set<std::string> cuda = {"global", "host", "device", "shared", "constant", "managed",
            "grid_constant", "launch_bounds", "device_builtin", "cudart_builtin", "cluster_dims", "blocks_per_cluster"};
        for (auto [begin, last] : parts) {
            if (omitted_attributes.contains(begin)) continue;
            auto name = items[begin].text;
            name.erase(0, name.find_first_not_of('_'));
            auto tail = name.find_last_not_of('_');
            if (tail != std::string::npos) name.resize(tail + 1);
            if (cuda.contains(name)) continue;
#ifdef _WIN32
            if (gnu && name == "aligned" && last > begin + 2) {
                output += "__declspec(align(" + joined(begin + 2, last - 1) + ")) ";
                continue;
            }
            if (gnu && name == "noinline") { output += "__declspec(noinline) "; continue; }
            if (gnu && name == "always_inline") { output += "__forceinline "; continue; }
#endif
            output += (gnu ? "__attribute__((" : "__declspec(") + joined(begin, last) + (gnu ? ")) " : ") ");
        }
        position = end + (gnu ? 2 : 1);
        return true;
    }
    std::string render() {
        std::sort(edits.begin(), edits.end(), [](const edit& a, const edit& b) { return a.begin < b.begin; });
        std::string out;
        auto clear_macros = [&] {
            // Protect expanded tokens and generated registration names against
            // both input defines and macros introduced by preserved includes.
            for (const auto& [name, value] : macros) { (void)value; out += "\n#undef " + name + "\n"; }
        };
        clear_macros();
        std::string filename;
        unsigned line = 0;
        size_t change = 0;
        auto emit = [&](clang::SourceLocation location, const std::string& text, bool directive) {
            auto origin = sources.getPresumedLoc(sources.getExpansionLoc(location));
            if (origin.isInvalid()) throw std::runtime_error("expanded user token lost its source location");
            if (directive || filename != origin.getFilename() || line != origin.getLine()) {
                out += "\n#line " + std::to_string(origin.getLine()) + " " + source_literal(origin.getFilename()) + "\n";
                filename = origin.getFilename(); line = origin.getLine();
            }
            out += text;
            if (directive) { out += '\n'; line = 0; }
            else { out += ' '; line += unsigned(std::count(text.begin(), text.end(), '\n')); }
        };
        for (size_t i = 0; i < items.size();) {
            if (change < edits.size() && edits[change].begin == i) {
                emit(items[i].location, edits[change].text, false);
                i = edits[change++].end;
                continue;
            }
            if (items[i].unsupported) {
                auto origin = sources.getPresumedLoc(sources.getExpansionLoc(items[i].location));
                throw std::runtime_error(std::string(origin.isValid() ? origin.getFilename() : "<user source>") + ":"
                    + std::to_string(origin.isValid() ? origin.getLine() : 0) + ": " + items[i].text);
            }
            auto location = items[i].location;
            std::string attrs;
            if (!items[i].directive && attribute(i, attrs)) { if (!attrs.empty()) emit(location, attrs, false); continue; }
            emit(location, items[i].directive ? directive(items[i].text, location) : items[i].text, items[i].directive);
            if (items[i].directive) { clear_macros(); line = 0; }
            ++i;
        }
        return out + '\n';
    }
};
}
