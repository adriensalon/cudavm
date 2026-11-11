#include "test_fixtures.hpp"
#include <iostream>

namespace cudavm {
using half2_format = numerical_format<10, 5>;
constexpr std::array<const char*, 12> half2_profiles{"representation", "widen", "narrow", "arithmetic", "fused", "saturation", "divide", "operators", "comparisons", "predicates", "divide_operator", "divide_assign"};
unsigned half2_profile;
uint32_t half2_pack(uint32_t low, uint32_t high) { return low | (high << 16); }
bool half2_equivalent(uint32_t actual, uint32_t expected) {
    for (unsigned shift : {0u, 16u}) {
        const auto a = (actual >> shift) & 0xffff, b = (expected >> shift) & 0xffff;
        if (half2_format::is_nan(b) ? !half2_format::is_nan(a) : a != b) return false;
    }
    return true;
}
std::vector<std::array<uint32_t, 3>> half2_inputs() {
    constexpr uint32_t edges[]{0, 0x8000, 1, 0x8001, 3, 0x03ff, 0x0400, 0x3555, 0x3800, 0xb800,
        0x3bff, 0x3c00, 0x3c01, 0xbc00, 0x4000, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe01, 0x7fff};
    std::vector<std::array<uint32_t, 3>> result;
    for (auto a : edges) for (auto b : edges) for (auto c : edges)
        result.push_back({half2_pack(a, b), half2_pack(b, c), half2_pack(c, a)});
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        const auto word = half2_pack(bits, (bits * 40503 + 17) & 0xffff);
        const auto a = half2_pack(edges[bits % std::size(edges)], edges[(bits + 7) % std::size(edges)]);
        const auto b = half2_pack(edges[(bits / std::size(edges)) % std::size(edges)], edges[(bits + 13) % std::size(edges)]);
        result.push_back({word, a, b}); result.push_back({a, word, b}); result.push_back({a, b, word});
        // Equal pairs exercise both-lane equality and agreement predicates.
        result.push_back({word, word, a});
        if (half2_profile == 8 || half2_profile == 9) {
            const auto high = word >> 16;
            // Keep one lane equal while its neighbor changes; both-lane
            // predicates must reject a result that only satisfies one lane.
            result.push_back({word, half2_pack(bits, (high + 1) & 0xffff), b});
            result.push_back({word, half2_pack((bits + 1) & 0xffff, high), a});
            result.push_back({word, word ^ 0x80008000u, b});
        }
    }
    for (uint32_t offset : {1u, 2u, 3u, 7u, 31u, 127u, 255u})
        result.push_back({half2_pack(0x3c00 + offset, 0x3c00 - offset), half2_pack(0x3c00 - 2 * offset, 0x3c00 + 2 * offset), 0xbc00bc00});
    return result;
}
uint32_t half2_narrow_bits(uint32_t bits) {
    if (numerical_binary32::is_nan(bits)) return uint32_t(half2_format::nan);
    if (numerical_binary32::is_infinite(bits)) return ((bits >> 16) & 0x8000) | uint32_t(half2_format::infinity);
    return uint32_t(half2_format::encode(numerical_binary32::decode(bits), 0));
}
uint32_t half2_widen_bits(uint32_t bits) {
    if (half2_format::is_nan(bits)) return uint32_t(numerical_binary32::nan);
    if (half2_format::is_infinite(bits)) return ((bits & 0x8000) << 16) | uint32_t(numerical_binary32::infinity);
    return uint32_t(numerical_binary32::encode(half2_format::decode(bits), 0));
}
std::vector<std::array<uint32_t, 3>> half2_narrow_inputs() {
    std::vector<uint32_t> values;
    auto append = [&](uint32_t bits) { values.push_back(bits); values.push_back(bits | 0x80000000u); };
    for (uint32_t exponent = 0; exponent < 256; ++exponent)
        for (uint32_t fraction : {0u, 1u, 4095u, 4096u, 4097u, 8191u, 0x3fffffu, 0x400000u, 0x7fffffu}) append((exponent << 23) | fraction);
    for (uint32_t half = 0; half < 0x7c00; ++half) {
        const auto upper = half == 0x7bff ? numerical_dyadic{numerical_integer(1), 16, false} : half2_format::decode(half + 1);
        auto midpoint = numerical_sum(half2_format::decode(half), upper, 0); --midpoint.scale;
        const auto bits = uint32_t(numerical_binary32::encode(midpoint, 0));
        for (auto value : {bits - 1, bits, bits + 1}) append(value);
    }
    std::vector<std::array<uint32_t, 3>> result;
    for (size_t i = 0; i < values.size(); ++i) result.push_back({values[i], values[(i + 113) % values.size()], 0});
    return result;
}
std::array<bool, 12> half2_relations(uint32_t a, uint32_t b) {
    const bool unordered = half2_format::is_nan(a) || half2_format::is_nan(b);
    const bool equal = !unordered && (a == b || (!(a & 0x7fff) && !(b & 0x7fff)));
    const bool less = !unordered && !equal && ((a ^ b) & 0x8000 ? (a & 0x8000) != 0 : a & 0x8000 ? a > b : a < b);
    const bool greater = !unordered && !equal && !less;
    return {equal, !unordered && !equal, less, equal || less, greater, equal || greater,
        unordered || equal, unordered || !equal, unordered || less, unordered || equal || less, unordered || greater, unordered || equal || greater};
}
uint32_t half2_saturate(uint64_t bits) {
    return half2_format::is_nan(bits) || (bits & 0x8000) ? 0 : uint32_t(std::min(bits, uint64_t(0x3c00)));
}
void execute_half2_operations(const loaded_program& program) {
    const auto values = half2_profile == 2 ? half2_narrow_inputs() : half2_inputs();
    auto exact = [](uint32_t actual, uint32_t expected) { return actual == expected; };
    auto arithmetic = [](const auto& lane) {
        const auto sum = uint32_t(half2_format::add(lane[0], lane[1], 0));
        const auto difference = uint32_t(half2_format::add(lane[0], lane[1] ^ 0x8000, 0));
        const auto product = uint32_t(half2_format::multiply(lane[0], lane[1], 0));
        return std::array<uint32_t, 6>{sum, difference, product, sum, difference, product};
    };
    if (half2_profile == 0)
        execute_numerical_rows<uint32_t, 9>(program, "half2_representation", values, [](const auto& input) {
            const auto a = input[0], b = input[1], low = a & 0xffff, high = a >> 16;
            return std::array<uint32_t, 9>{a, half2_pack(high, low), half2_pack(low, b & 0xffff), half2_pack(high, b >> 16),
                half2_pack(low, low), half2_pack(high, high), half2_pack(low, low), a, half2_pack(b & 0xffff, high)};
        }, exact);
    else if (half2_profile == 1)
        execute_numerical_rows<uint32_t, 4>(program, "half2_widen", values, [](const auto& input) {
            const auto low = half2_widen_bits(input[0] & 0xffff), high = half2_widen_bits(input[0] >> 16);
            return std::array<uint32_t, 4>{low, high, low, high};
        }, [](uint32_t actual, uint32_t expected) { return numerical_binary32::is_nan(expected) ? numerical_binary32::is_nan(actual) : actual == expected; });
    else if (half2_profile == 2)
        execute_numerical_rows<uint32_t, 4>(program, "half2_narrow", values, [](const auto& input) {
            const auto a = half2_narrow_bits(input[0]), b = half2_narrow_bits(input[1]), pair = half2_pack(a, b);
            return std::array<uint32_t, 4>{pair, pair, half2_pack(a, a), half2_pack(b, b)};
        }, half2_equivalent);
    else if (half2_profile == 3 || half2_profile == 7) {
        const auto kernel = half2_profile == 3 ? "half2_arithmetic" : "half2_operators";
        execute_numerical_rows<uint32_t, 6>(program, kernel, values, [&](const auto& input) {
            return numerical_packed16_rows<6>(input, arithmetic);
        }, half2_equivalent);
    } else if (half2_profile == 4)
        execute_numerical_rows<uint32_t, 2>(program, "half2_fused", values, [](const auto& input) {
            return numerical_packed16_rows<2>(input, [](const auto& lane) {
                return std::array<uint32_t, 2>{uint32_t(half2_format::fused(lane[0], lane[1], lane[2], 0)),
                    uint32_t(half2_format::add(half2_format::multiply(lane[0], lane[1], 0), lane[2], 0))};
            });
        }, half2_equivalent);
    else if (half2_profile == 5)
        execute_numerical_rows<uint32_t, 5>(program, "half2_saturation", values, [](const auto& input) {
            return numerical_packed16_rows<5>(input, [](const auto& lane) {
                const auto fused = half2_format::fused(lane[0], lane[1], lane[2], 0);
                return std::array<uint32_t, 5>{half2_saturate(half2_format::add(lane[0], lane[1], 0)),
                    half2_saturate(half2_format::add(lane[0], lane[1] ^ 0x8000, 0)), half2_saturate(half2_format::multiply(lane[0], lane[1], 0)),
                    half2_saturate(fused), half2_format::is_nan(fused) ? 0x7fffu : fused & 0x8000 ? 0u : uint32_t(fused)};
            });
        }, exact);
    else if (half2_profile == 6 || half2_profile >= 10) {
        const auto kernel = std::string("half2_") + half2_profiles[half2_profile];
        execute_numerical_rows<uint32_t, 1>(program, kernel.c_str(), values, [](const auto& input) {
            return numerical_packed16_rows<1>(input, [](const auto& lane) {
                const auto quotient = uint32_t(numerical_divide<half2_format>(lane[0], lane[1]));
                return std::array<uint32_t, 1>{quotient};
            });
        }, half2_equivalent);
    } else if (half2_profile == 8)
        execute_numerical_rows<uint32_t, 12>(program, "half2_comparisons", values, [](const auto& input) {
            return numerical_packed16_rows<12>(input, [](const auto& lane) {
                const auto relations = half2_relations(lane[0], lane[1]); std::array<uint32_t, 12> row{};
                for (size_t i = 0; i < row.size(); ++i) row[i] = relations[i] ? 0x3c00 : 0; return row;
            });
        }, exact);
    else execute_numerical_rows<uint32_t, 12>(program, "half2_predicates", values, [](const auto& input) {
        const auto low = half2_relations(input[0] & 0xffff, input[1] & 0xffff), high = half2_relations(input[0] >> 16, input[1] >> 16);
        std::array<uint32_t, 12> row{}; for (size_t i = 0; i < row.size(); ++i) row[i] = low[i] && high[i]; return row;
    }, exact);
}
}
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: <executable> <backend> cu <half2-profile>\n"; return 1; }
    const auto& profiles = cudavm::half2_profiles;
    const auto selected = std::find(profiles.begin(), profiles.end(), std::string_view(argv[3]));
    if (selected == profiles.end()) { std::cerr << "unknown packed half profile\n"; return 1; }
    cudavm::half2_profile = unsigned(selected - profiles.begin());
    const auto source = "#define CUDAVM_HALF2_PROFILE " + std::to_string(cudavm::half2_profile) + "\n" + std::string(cudavm::fixtures.source);
    auto fixtures = cudavm::fixtures; fixtures.source = source;
    return cudavm::run_test(3, argv, fixtures, cudavm::execute_half2_operations);
}
