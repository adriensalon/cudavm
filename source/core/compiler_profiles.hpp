#pragma once
#include <array>
#include <string_view>

namespace cudavm {
// Implemented compiler profiles, separate from the pinned public header surface.
inline constexpr std::array<int, 2> source_architectures{75, 80};
inline constexpr std::array<int, 6> ptx_compile_architectures{75, 80, 86, 87, 89, 90};
inline constexpr std::array<std::string_view, 5> nvvm_compile_architectures{"compute_75", "compute_80", "compute_87", "compute_89", "compute_90"};
inline constexpr std::array<std::array<int, 2>, 20> ptx_language_versions{{
    {6,0}, {6,1}, {6,2}, {6,3}, {6,4}, {6,5},
    {7,0}, {7,1}, {7,2}, {7,3}, {7,4}, {7,5}, {7,6}, {7,7}, {7,8},
    {8,0}, {8,1}, {8,2}, {8,3}, {8,4}}};
inline bool supported_ptx_version(std::string_view value) {
    for (auto version : ptx_language_versions)
        if (value.size() == 3 && value[0] == '0' + version[0] && value[1] == '.' && value[2] == '0' + version[1]) return true;
    return false;
}
}
