#pragma once
#include "backend.hpp"
#include <regex>
#include <string_view>
#include <unordered_map>

namespace cudavm {
inline std::vector<std::string> ptx_entry_names(std::string_view input) {
    auto text = std::regex_replace(std::string(input), std::regex(R"(/\*[\s\S]*?\*/|//[^\r\n]*)"), " ");
    std::regex entries(R"(\.entry\s+([A-Za-z_$][\w$.]*)\s*\()");
    std::vector<std::string> result;
    for (std::sregex_iterator it(text.begin(), text.end(), entries), end; it != end; ++it) result.push_back((*it)[1]);
    std::sort(result.begin(), result.end()); result.erase(std::unique(result.begin(), result.end()), result.end()); return result;
}
// Read declarations only. Execution remains NVIDIA's responsibility; an
// unsupported declaration is omitted rather than assigned a guessed layout.
inline std::unordered_map<std::string, std::vector<kernel_parameter>> ptx_parameter_layouts(std::string_view input) {
    std::string text(input);
    text = std::regex_replace(text, std::regex(R"(/\*[\s\S]*?\*/|//[^\r\n]*)"), " ");
    std::regex entries(R"(\.entry\s+([A-Za-z_$][\w$.]*)\s*\(([^)]*)\))");
    std::regex declaration(R"(^\s*\.param\s+(?:\.align\s+(\d+)\s+)?(?:\.v(2|4)\s+)?\.(?:b|u|s|f)(8|16|32|64|128)\s+(?:\.ptr(?:\s+\.(?:global|const|shared|local|generic))?(?:\s+\.align\s+\d+)?\s+)?[A-Za-z_$][\w$.]*(?:\s*\[\s*(\d+)\s*\])?\s*$)");
    std::unordered_map<std::string, std::vector<kernel_parameter>> result;
    for (std::sregex_iterator entry(text.begin(), text.end(), entries), end; entry != end; ++entry) {
        std::vector<kernel_parameter> layout;
        auto params = (*entry)[2].str(); size_t position = 0, offset = 0; bool supported = true;
        if (params.find_first_not_of(" \r\n\t") != params.npos) {
            do {
                auto comma = params.find(',', position);
                auto item = params.substr(position, comma == params.npos ? params.npos : comma - position);
                std::smatch match;
                if (!std::regex_match(item, match, declaration)) { supported = false; break; }
                try {
                    size_t unit = std::stoull(match[3].str()) / 8;
                    size_t lanes = match[2].matched ? std::stoull(match[2].str()) : 1;
                    size_t count = match[4].matched ? std::stoull(match[4].str()) : 1;
                    size_t alignment = match[1].matched ? std::stoull(match[1].str()) : unit * lanes;
                    if (!alignment || alignment > 65536 || (alignment & (alignment - 1)) || !count || count > 65536 / (unit * lanes)) { supported = false; break; }
                    offset = (offset + alignment - 1) & ~(alignment - 1);
                    auto size = unit * lanes * count;
                    if (offset > 65536 || size > 65536 - offset || layout.size() == 4096) { supported = false; break; }
                    layout.push_back({offset, size}); offset += size;
                } catch (const std::exception&) { supported = false; break; }
                if (comma == params.npos) break;
                position = comma + 1;
            } while (true);
        }
        if (supported) result.insert_or_assign((*entry)[1].str(), std::move(layout));
    }
    return result;
}
}
