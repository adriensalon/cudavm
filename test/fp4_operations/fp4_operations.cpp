#include "test_fixtures.hpp"
#include <iostream>

namespace cudavm {
constexpr std::array<const char*, 6> fp4_profiles{"widen", "float", "double", "half", "bfloat", "representation"};
unsigned fp4_profile;

numerical_dyadic fp4_value(uint64_t bits) {
    return numerical_format<1, 2>::decode(bits);
}
// Finite-only destinations: bracket exact values, clamp before rounding and
// map NaNs to the pinned positive MAXNORM independently of their source sign.
template<class Source> uint64_t fp4_narrow(uint64_t bits, bool nearest) {
    const uint64_t sign = (bits & Source::sign) ? 8 : 0;
    if (Source::is_nan(bits)) return 7;
    if (Source::is_infinite(bits)) return sign | 7;
    auto value = Source::decode(bits); value.negative = false;
    if (elementary_compare(value, fp4_value(7)) >= 0) return sign | 7;
    uint64_t low = 0, high = 7;
    while (high - low > 1) {
        const auto middle = (low + high) / 2;
        if (elementary_compare(value, fp4_value(middle)) >= 0) low = middle;
        else high = middle;
    }
    if (!nearest) return sign | low;
    auto midpoint = numerical_sum(fp4_value(low), fp4_value(high), 0); --midpoint.scale;
    const auto order = elementary_compare(value, midpoint);
    return sign | (order > 0 || (!order && (low & 1)) ? high : low);
}
template<class Destination> uint64_t fp4_widen(uint64_t bits) {
    return Destination::encode(fp4_value(bits), 0);
}
template<class Source, unsigned Fraction, unsigned Exponent>
std::vector<std::array<uint64_t, 3>> fp4_narrow_inputs() {
    std::vector<uint64_t> words;
    auto append = [&](uint64_t bits) { words.push_back(bits); words.push_back(bits | Source::sign); };
    if constexpr (Fraction + Exponent == 15) {
        for (uint64_t bits = 0; bits < 65536; ++bits) words.push_back(bits);
    } else {
        for (uint64_t exponent = 0; exponent < (1u << Exponent); ++exponent)
            for (uint64_t fraction : {uint64_t(0), uint64_t(1), (uint64_t(1) << (Fraction - 1)) - 1,
                    uint64_t(1) << (Fraction - 1), (uint64_t(1) << Fraction) - 2, (uint64_t(1) << Fraction) - 1})
                append((exponent << Fraction) | fraction);
        
        for (uint64_t low = 0; low < 7; ++low) {
            auto midpoint = numerical_sum(fp4_value(low), fp4_value(low + 1), 0); --midpoint.scale;
            const auto bits = Source::encode(midpoint, 0);
            append(bits - 1); append(bits); append(bits + 1);
            const auto exact = Source::encode(fp4_value(low), 0);
            if (exact) append(exact - 1);
            append(exact); append(exact + 1);
        }
        const auto maximum = Source::encode(fp4_value(7), 0);
        append(maximum - 1); append(maximum); append(maximum + 1);
    }
    std::vector<std::array<uint64_t, 3>> result;
    for (size_t i = 0; i < words.size(); ++i) result.push_back({words[i], words[(i + 113) % words.size()], words[(i + 271) % words.size()]});
    return result;
}
template<class Source> std::array<uint64_t, 7> fp4_narrow_row(const std::array<uint64_t, 3>& input) {
    
    const auto a = fp4_narrow<Source>(input[0], true), b = fp4_narrow<Source>(input[1], true);
    const auto x = fp4_narrow<Source>(input[0], false), y = fp4_narrow<Source>(input[1], false);
    const auto c = fp4_narrow<Source>(input[2], true), d = fp4_narrow<Source>(input[1] ^ Source::sign, true);
    return {a, x, a | (b << 4), x | (y << 4), a, a | (b << 4), a | (b << 4) | (c << 8) | (d << 12)};
}
bool fp4_narrow_equivalent(uint64_t actual, uint64_t expected, size_t) { return actual == expected; }
void execute_fp4(const loaded_program& program) {
    const unsigned source = fp4_profile;
    const auto kernel = std::string("fp4_") + fp4_profiles[fp4_profile];
    if (source > 0 && fp4_profile < 5) {
        if (source == 1) execute_numerical_rows<uint64_t, 7>(program, kernel.c_str(), fp4_narrow_inputs<numerical_binary32, 23, 8>(), fp4_narrow_row<numerical_binary32>, fp4_narrow_equivalent);
        if (source == 2) execute_numerical_rows<uint64_t, 7>(program, kernel.c_str(), fp4_narrow_inputs<numerical_binary64, 52, 11>(), fp4_narrow_row<numerical_binary64>, fp4_narrow_equivalent);
        if (source == 3) execute_numerical_rows<uint64_t, 7>(program, kernel.c_str(), fp4_narrow_inputs<numerical_format<10, 5>, 10, 5>(), fp4_narrow_row<numerical_format<10, 5>>, fp4_narrow_equivalent);
        if (source == 4) execute_numerical_rows<uint64_t, 7>(program, kernel.c_str(), fp4_narrow_inputs<numerical_format<7, 8>, 7, 8>(), fp4_narrow_row<numerical_format<7, 8>>, fp4_narrow_equivalent);
        return;
    }
    std::vector<std::array<uint64_t, 3>> values;
    for (uint64_t bits = 0; bits < 65536; ++bits) values.push_back({bits, 0, 0});
    if (fp4_profile == 5) {
        execute_numerical_rows<uint64_t, 3>(program, kernel.c_str(), values, [](const auto& input) {
            const auto bits = input[0];
            return std::array<uint64_t, 3>{bits & 15, bits & 255, bits};
        }, [](uint64_t a, uint64_t b) { return a == b; });
        return;
    }
    execute_numerical_rows<uint64_t, 12>(program, kernel.c_str(), values, [](const auto& input) {
        const auto bits = input[0]; 
        const auto a = bits & 15, b = (bits >> 4) & 15;
        return std::array<uint64_t, 12>{fp4_widen<numerical_format<10, 5>>(a), fp4_widen<numerical_format<7, 8>>(a),
            fp4_widen<numerical_binary32>(a), fp4_widen<numerical_binary64>(a),
            fp4_widen<numerical_format<10, 5>>(a) | (fp4_widen<numerical_format<10, 5>>(b) << 16),
            fp4_widen<numerical_format<7, 8>>(a) | (fp4_widen<numerical_format<7, 8>>(b) << 16),
            fp4_widen<numerical_binary32>(a), fp4_widen<numerical_binary32>(b),
            fp4_widen<numerical_binary32>(a), fp4_widen<numerical_binary32>(b),
            fp4_widen<numerical_binary32>((bits >> 8) & 15), fp4_widen<numerical_binary32>((bits >> 12) & 15)};
    }, [](uint64_t actual, uint64_t expected) { return actual == expected; });
}
}
int main(int argc, char** argv) {
    if (argc != 4) return 1;
    const auto& names = cudavm::fp4_profiles;
    const auto found = std::find(names.begin(), names.end(), std::string_view(argv[3]));
    if (found == names.end()) return 1;
    cudavm::fp4_profile = unsigned(found - names.begin());
    const auto source = "#define CUDAVM_FP4_PROFILE " + std::to_string(cudavm::fp4_profile) + "\n" + std::string(cudavm::fixtures.source);
    auto fixtures = cudavm::fixtures; fixtures.source = source;
    return cudavm::run_conformance(3, argv, fixtures, cudavm::execute_fp4);
}

