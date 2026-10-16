#pragma once
#include <charconv>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cudavm {
struct data_view_descriptor { std::string target; uint64_t offset, bytes; };
// Independent PTX comments keep V1/V2/V3 shader images compatible. Targets are
// canonical storage names, never other views, so lookup cannot recurse in a cycle.
inline std::map<std::string, data_view_descriptor> decode_data_views(std::string_view text) {
    constexpr std::string_view marker = "// CUDAVM-DATA-VIEW-V1 ";
    std::map<std::string, data_view_descriptor> result;
    auto fail = [] { throw std::runtime_error("invalid device data view descriptor"); };
    auto symbol = [&](std::string_view value) {
        if (value.empty() || value.size() % 2) fail();
        std::string output;
        auto nibble = [&](char c) { if (c >= '0' && c <= '9') return c - '0'; if (c >= 'a' && c <= 'f') return c - 'a' + 10; fail(); return 0; };
        for (size_t i = 0; i < value.size(); i += 2) {
            auto byte = char((nibble(value[i]) << 4) | nibble(value[i+1]));
            if (!byte) fail();
            output += byte;
        }
        return output;
    };
    auto number = [&](std::string_view value) {
        uint64_t output = 0;
        auto parsed = std::from_chars(value.data(), value.data() + value.size(), output);
        if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) fail();
        return output;
    };
    while (!text.empty()) {
        auto end = text.find('\n');
        auto line = text.substr(0, end);
        text = end == text.npos ? std::string_view{} : text.substr(end + 1);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        if (!line.starts_with(marker)) continue;
        line.remove_prefix(marker.size());
        std::string_view fields[4];
        for (unsigned i = 0; i < 3; ++i) {
            auto space = line.find(' ');
            if (space == line.npos) fail();
            fields[i] = line.substr(0, space); line.remove_prefix(space + 1);
        }
        fields[3] = line;
        auto name = symbol(fields[0]);
        data_view_descriptor view{symbol(fields[1]), number(fields[2]), number(fields[3])};
        if (!view.bytes || name == view.target || result.size() >= 4096 || !result.emplace(name, std::move(view)).second) fail();
    }
    for (const auto& [name, view] : result) if (result.contains(view.target)) fail();
    return result;
}
}
