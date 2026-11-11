#include "test_fixtures.hpp"
#include <iostream>

namespace cudavm {
using bfloat162_format = numerical_format<7, 8>;
constexpr std::array<const char*, 13> bfloat162_profiles{"representation", "widen", "narrow", "arithmetic", "fused", "saturation", "divide", "operators", "comparisons", "predicates", "divide_operator", "divide_assign", "masks"};
unsigned bfloat162_profile;
uint32_t bfloat162_pack(uint32_t low, uint32_t high) { return low | (high << 16); }
bool bfloat162_equivalent(uint32_t actual, uint32_t expected) {
    for (unsigned shift : {0u, 16u}) {
        const auto a = (actual >> shift) & 0xffff, b = (expected >> shift) & 0xffff;
        if (bfloat162_format::is_nan(b) ? !bfloat162_format::is_nan(a) : a != b) return false;
    }
    return true;
}
std::vector<std::array<uint32_t, 3>> bfloat162_inputs() {
    constexpr uint32_t edges[]{0, 0x8000, 1, 0x8001, 3, 0x007f, 0x0080, 0x3eab, 0x3f00, 0xbf00,
        0x3f7f, 0x3f80, 0x3f81, 0xbf80, 0x4000, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7f81, 0xffc1, 0x7fff};
    std::vector<std::array<uint32_t, 3>> result;
    for (auto a : edges) for (auto b : edges) for (auto c : edges)
        result.push_back({bfloat162_pack(a, b), bfloat162_pack(b, c), bfloat162_pack(c, a)});
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        const auto word = bfloat162_pack(bits, (bits * 40503 + 17) & 0xffff);
        const auto a = bfloat162_pack(edges[bits % std::size(edges)], edges[(bits + 7) % std::size(edges)]);
        const auto b = bfloat162_pack(edges[(bits / std::size(edges)) % std::size(edges)], edges[(bits + 13) % std::size(edges)]);
        result.push_back({word, a, b}); result.push_back({a, word, b}); result.push_back({a, b, word});
        // Equal pairs exercise both-lane equality and agreement predicates.
        result.push_back({word, word, a});
        if (bfloat162_profile == 8 || bfloat162_profile == 9 || bfloat162_profile == 12) {
            const auto high = word >> 16;
            // Keep one lane equal while its neighbor changes; both-lane
            // predicates must reject a result that only satisfies one lane.
            result.push_back({word, bfloat162_pack(bits, (high + 1) & 0xffff), b});
            result.push_back({word, bfloat162_pack((bits + 1) & 0xffff, high), a});
            result.push_back({word, word ^ 0x80008000u, b});
        }
    }
    for (uint32_t offset : {1u, 2u, 3u, 7u, 15u, 31u, 63u})
        result.push_back({bfloat162_pack(0x3f80 + offset, 0x3f80 - offset), bfloat162_pack(0x3f80 - 2 * offset, 0x3f80 + 2 * offset), 0xbf80bf80});
    if (bfloat162_profile == 4 || bfloat162_profile == 5)
        for (uint32_t a = 0x3f80; a < 0x4000; ++a) for (uint32_t b = 0x3f80; b < 0x4000; ++b)
            result.push_back({bfloat162_pack(a, b), bfloat162_pack(b, a), bfloat162_pack(1, 0x8001)});
    return result;
}
uint32_t bfloat162_narrow_bits(uint32_t bits) {
    if (numerical_binary32::is_nan(bits)) return uint32_t(bfloat162_format::nan);
    if (numerical_binary32::is_infinite(bits)) return ((bits >> 16) & 0x8000) | uint32_t(bfloat162_format::infinity);
    return uint32_t(bfloat162_format::encode(numerical_binary32::decode(bits), 0));
}
uint32_t bfloat162_widen_bits(uint32_t bits) {
    if (bfloat162_format::is_nan(bits)) return uint32_t(numerical_binary32::nan);
    if (bfloat162_format::is_infinite(bits)) return ((bits & 0x8000) << 16) | uint32_t(numerical_binary32::infinity);
    return uint32_t(numerical_binary32::encode(bfloat162_format::decode(bits), 0));
}
std::vector<std::array<uint32_t, 3>> bfloat162_narrow_inputs() {
    std::vector<uint32_t> values;
    auto append = [&](uint32_t bits) { values.push_back(bits); values.push_back(bits | 0x80000000u); };
    for (uint32_t exponent = 0; exponent < 256; ++exponent)
        for (uint32_t fraction : {0u, 1u, 0x7fffu, 0x8000u, 0x8001u, 0xffffu, 0x3fffffu, 0x400000u, 0x7fffffu}) append((exponent << 23) | fraction);
    for (uint32_t bits16 = 0; bits16 < 0x7f80; ++bits16) {
        const auto upper = bits16 == 0x7f7f ? numerical_dyadic{numerical_integer(1), 128, false} : bfloat162_format::decode(bits16 + 1);
        auto midpoint = numerical_sum(bfloat162_format::decode(bits16), upper, 0); --midpoint.scale;
        const auto bits = uint32_t(numerical_binary32::encode(midpoint, 0));
        for (auto value : {bits - 1, bits, bits + 1}) append(value);
    }
    std::vector<std::array<uint32_t, 3>> result;
    for (size_t i = 0; i < values.size(); ++i) result.push_back({values[i], values[(i + 113) % values.size()], 0});
    return result;
}
std::array<bool, 12> bfloat162_relations(uint32_t a, uint32_t b) {
    const bool unordered = bfloat162_format::is_nan(a) || bfloat162_format::is_nan(b);
    const bool equal = !unordered && (a == b || (!(a & 0x7fff) && !(b & 0x7fff)));
    const bool less = !unordered && !equal && ((a ^ b) & 0x8000 ? (a & 0x8000) != 0 : a & 0x8000 ? a > b : a < b);
    const bool greater = !unordered && !equal && !less;
    return {equal, !unordered && !equal, less, equal || less, greater, equal || greater,
        unordered || equal, unordered || !equal, unordered || less, unordered || equal || less, unordered || greater, unordered || equal || greater};
}
uint32_t bfloat162_saturate(uint64_t bits) {
    return bfloat162_format::is_nan(bits) || (bits & 0x8000) ? 0 : uint32_t(std::min(bits, uint64_t(0x3f80)));
}
void execute_bfloat162_operations(const loaded_program& program) {
    const auto values = bfloat162_profile == 2 ? bfloat162_narrow_inputs() : bfloat162_inputs();
    auto exact = [](uint32_t actual, uint32_t expected) { return actual == expected; };
    auto arithmetic = [](const auto& lane) {
        const auto sum = uint32_t(bfloat162_format::add(lane[0], lane[1], 0));
        const auto difference = uint32_t(bfloat162_format::add(lane[0], lane[1] ^ 0x8000, 0));
        const auto product = uint32_t(bfloat162_format::multiply(lane[0], lane[1], 0));
        return std::array<uint32_t, 6>{sum, difference, product, sum, difference, product};
    };
    if (bfloat162_profile == 0)
        execute_numerical_rows<uint32_t, 9>(program, "bfloat162_representation", values, [](const auto& input) {
            const auto a = input[0], b = input[1], low = a & 0xffff, high = a >> 16;
            return std::array<uint32_t, 9>{a, bfloat162_pack(high, low), bfloat162_pack(low, b & 0xffff), bfloat162_pack(high, b >> 16),
                bfloat162_pack(low, low), bfloat162_pack(high, high), bfloat162_pack(low, low), a, bfloat162_pack(b & 0xffff, high)};
        }, exact);
    else if (bfloat162_profile == 1)
        execute_numerical_rows<uint32_t, 4>(program, "bfloat162_widen", values, [](const auto& input) {
            const auto low = bfloat162_widen_bits(input[0] & 0xffff), high = bfloat162_widen_bits(input[0] >> 16);
            return std::array<uint32_t, 4>{low, high, low, high};
        }, [](uint32_t actual, uint32_t expected) { return numerical_binary32::is_nan(expected) ? numerical_binary32::is_nan(actual) : actual == expected; });
    else if (bfloat162_profile == 2)
        execute_numerical_rows<uint32_t, 4>(program, "bfloat162_narrow", values, [](const auto& input) {
            const auto a = bfloat162_narrow_bits(input[0]), b = bfloat162_narrow_bits(input[1]), pair = bfloat162_pack(a, b);
            return std::array<uint32_t, 4>{pair, pair, bfloat162_pack(a, a), bfloat162_pack(b, b)};
        }, bfloat162_equivalent);
    else if (bfloat162_profile == 3 || bfloat162_profile == 7) {
        const auto kernel = bfloat162_profile == 3 ? "bfloat162_arithmetic" : "bfloat162_operators";
        execute_numerical_rows<uint32_t, 6>(program, kernel, values, [&](const auto& input) {
            return numerical_packed16_rows<6>(input, arithmetic);
        }, bfloat162_equivalent);
    } else if (bfloat162_profile == 4)
        execute_numerical_rows<uint32_t, 2>(program, "bfloat162_fused", values, [](const auto& input) {
            return numerical_packed16_rows<2>(input, [](const auto& lane) {
                return std::array<uint32_t, 2>{uint32_t(bfloat162_format::fused(lane[0], lane[1], lane[2], 0)),
                    uint32_t(bfloat162_format::add(bfloat162_format::multiply(lane[0], lane[1], 0), lane[2], 0))};
            });
        }, bfloat162_equivalent);
    else if (bfloat162_profile == 5)
        execute_numerical_rows<uint32_t, 5>(program, "bfloat162_saturation", values, [](const auto& input) {
            return numerical_packed16_rows<5>(input, [](const auto& lane) {
                const auto fused = bfloat162_format::fused(lane[0], lane[1], lane[2], 0);
                return std::array<uint32_t, 5>{bfloat162_saturate(bfloat162_format::add(lane[0], lane[1], 0)),
                    bfloat162_saturate(bfloat162_format::add(lane[0], lane[1] ^ 0x8000, 0)), bfloat162_saturate(bfloat162_format::multiply(lane[0], lane[1], 0)),
                    bfloat162_saturate(fused), bfloat162_format::is_nan(fused) ? 0x7fffu : fused & 0x8000 ? 0u : uint32_t(fused)};
            });
        }, exact);
    else if (bfloat162_profile == 6 || bfloat162_profile == 10 || bfloat162_profile == 11) {
        const auto kernel = std::string("bfloat162_") + bfloat162_profiles[bfloat162_profile];
        execute_numerical_rows<uint32_t, 1>(program, kernel.c_str(), values, [](const auto& input) {
            return numerical_packed16_rows<1>(input, [](const auto& lane) {
                const auto quotient = uint32_t(numerical_divide<bfloat162_format>(lane[0], lane[1]));
                return std::array<uint32_t, 1>{quotient};
            });
        }, bfloat162_equivalent);
    } else if (bfloat162_profile == 8 || bfloat162_profile == 12) {
        const auto kernel = bfloat162_profile == 8 ? "bfloat162_comparisons" : "bfloat162_masks";
        execute_numerical_rows<uint32_t, 12>(program, kernel, values, [](const auto& input) {
            return numerical_packed16_rows<12>(input, [](const auto& lane) {
                const auto relations = bfloat162_relations(lane[0], lane[1]); std::array<uint32_t, 12> row{};
                for (size_t i = 0; i < row.size(); ++i) row[i] = relations[i] ? (bfloat162_profile == 12 ? 0xffff : 0x3f80) : 0; return row;
            });
        }, exact);
    } else execute_numerical_rows<uint32_t, 12>(program, "bfloat162_predicates", values, [](const auto& input) {
        const auto low = bfloat162_relations(input[0] & 0xffff, input[1] & 0xffff), high = bfloat162_relations(input[0] >> 16, input[1] >> 16);
        std::array<uint32_t, 12> row{}; for (size_t i = 0; i < row.size(); ++i) row[i] = low[i] && high[i]; return row;
    }, exact);
}
}
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: <executable> <backend> cu <bfloat162-profile>\n"; return 1; }
    const auto& profiles = cudavm::bfloat162_profiles;
    const auto selected = std::find(profiles.begin(), profiles.end(), std::string_view(argv[3]));
    if (selected == profiles.end()) { std::cerr << "unknown packed bfloat16 profile\n"; return 1; }
    cudavm::bfloat162_profile = unsigned(selected - profiles.begin());
    const auto source = "#define CUDAVM_BFLOAT162_PROFILE " + std::to_string(cudavm::bfloat162_profile) + "\n" + std::string(cudavm::fixtures.source);
    auto fixtures = cudavm::fixtures; fixtures.source = source;
    return cudavm::run_test(3, argv, fixtures, cudavm::execute_bfloat162_operations);
}
