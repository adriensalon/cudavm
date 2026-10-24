#pragma once
#include <set>
#include <string>
#include <vector>
#include <llvm/Object/Archive.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Support/MemoryBuffer.h>

namespace cudavm {
// Read symbol tables only. The native linker still owns the final host link;
// generated registration references feed selected device members back into
// this closure before the module is emitted.
struct host_link {
    struct member {
        std::set<std::string> definitions, references;
        bool selected;
    };
    std::vector<member> members;
    explicit host_link(const std::vector<std::string>& paths) {
        auto add = [&](llvm::MemoryBufferRef bytes, const std::string& origin, bool selected) {
            auto object = llvm::object::ObjectFile::createObjectFile(bytes);
            if (!object) throw std::runtime_error("host selection requires native object code " + origin + ": " + llvm::toString(object.takeError()));
            member result{{}, {}, selected};
            for (const auto& symbol : (*object)->symbols()) {
                auto flags = symbol.getFlags();
                if (!flags) throw std::runtime_error("invalid host symbol flags: " + llvm::toString(flags.takeError()));
                if (!(*flags & llvm::object::SymbolRef::SF_Global)) continue;
                auto name = symbol.getName();
                if (!name) throw std::runtime_error("invalid host symbol name: " + llvm::toString(name.takeError()));
                if (*flags & llvm::object::SymbolRef::SF_Undefined) {
                    // An unresolved weak reference does not extract an object.
                    if (!(*flags & llvm::object::SymbolRef::SF_Weak)) result.references.insert(name->str());
                } else result.definitions.insert(name->str());
            }
            members.push_back(std::move(result));
        };
        for (const auto& path : paths) {
            auto file = llvm::MemoryBuffer::getFile(path);
            if (!file) throw std::runtime_error("cannot read host input " + path + ": " + file.getError().message());
            auto bytes = (*file)->getMemBufferRef();
            if (bytes.getBuffer().starts_with("!<thin>\n"))
                throw std::runtime_error("thin host libraries are unsupported: " + path);
            if (!bytes.getBuffer().starts_with("!<arch>\n")) { add(bytes, path, true); continue; }
            auto archive = llvm::object::Archive::create(bytes);
            if (!archive) throw std::runtime_error("invalid host library " + path + ": " + llvm::toString(archive.takeError()));
            llvm::Error error = llvm::Error::success();
            for (const auto& child : (*archive)->children(error)) {
                auto name = child.getName();
                if (!name) throw std::runtime_error("invalid host library member: " + llvm::toString(name.takeError()));
                auto buffer = child.getMemoryBufferRef();
                if (!buffer) throw std::runtime_error("invalid host library member: " + llvm::toString(buffer.takeError()));
                add(*buffer, path + "(" + name->str() + ")", false);
            }
            if (error) throw std::runtime_error("invalid host library " + path + ": " + llvm::toString(std::move(error)));
        }
    }
    std::set<std::string> select(const std::set<std::string>& device_units) {
        if (members.empty()) return {};
        std::set<std::string> definitions, references;
        for (const auto& id : device_units) references.insert("__cudavm_register_" + id);
        auto retain = [&](const member& value) {
            definitions.insert(value.definitions.begin(), value.definitions.end());
            references.insert(value.references.begin(), value.references.end());
        };
        for (const auto& value : members) if (value.selected) retain(value);
        bool changed;
        do {
            changed = false;
            for (auto& value : members) {
                if (value.selected) continue;
                for (const auto& name : value.definitions) {
                    if (!definitions.contains(name) && references.contains(name)) {
                        value.selected = true; retain(value); changed = true; break;
                    }
                }
            }
        } while (changed);
        std::set<std::string> result;
        constexpr llvm::StringLiteral prefix = "__cudavm_register_";
        for (const auto& name : definitions) {
            llvm::StringRef symbol(name);
            if (!symbol.starts_with(prefix)) continue;
            auto id = symbol.drop_front(prefix.size());
            if (id.size() == 64 && id.find_first_not_of("0123456789abcdef") == llvm::StringRef::npos)
                result.insert(id.str());
        }
        return result;
    }
};
}
