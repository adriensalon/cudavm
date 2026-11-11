#include "test_fixtures.hpp"
#include <iostream>

namespace cudavm {
constexpr std::array<const char*, 11> fp6_profiles{"e2m3_widen", "e3m2_widen", "e2m3_float", "e3m2_float", "e2m3_double", "e3m2_double", "e2m3_half", "e3m2_half", "e2m3_bfloat", "e3m2_bfloat", "representation"};
unsigned fp6_profile;
bool fp6_e2m3() { return !(fp6_profile & 1); }
numerical_dyadic fp6_value(uint64_t bits, bool e2) {
    return e2 ? numerical_format<3, 2>::decode(bits) : numerical_format<2, 3>::decode(bits);
}
// Finite-only destinations: bracket exact values, clamp before rounding and
// map NaNs to the pinned positive MAXNORM independently of their source sign.
template<class Source> uint64_t fp6_narrow(uint64_t bits, bool e2, bool nearest) {
    const uint64_t sign = (bits & Source::sign) ? 32 : 0;
    if (Source::is_nan(bits)) return 31;
    if (Source::is_infinite(bits)) return sign | 31;
    auto value = Source::decode(bits); value.negative = false;
    if (elementary_compare(value, fp6_value(31, e2)) >= 0) return sign | 31;
    uint64_t low = 0, high = 31;
    while (high - low > 1) {
        const auto middle = (low + high) / 2;
        if (elementary_compare(value, fp6_value(middle, e2)) >= 0) low = middle;
        else high = middle;
    }
    if (!nearest) return sign | low;
    auto midpoint = numerical_sum(fp6_value(low, e2), fp6_value(high, e2), 0); --midpoint.scale;
    const auto order = elementary_compare(value, midpoint);
    return sign | (order > 0 || (!order && (low & 1)) ? high : low);
}
template<class Destination> uint64_t fp6_widen(uint64_t bits, bool e2) {
    return Destination::encode(fp6_value(bits, e2), 0);
}
template<class Source, unsigned Fraction, unsigned Exponent>
std::vector<std::array<uint64_t, 3>> fp6_narrow_inputs() {
    std::vector<uint64_t> words;
    auto append = [&](uint64_t bits) { words.push_back(bits); words.push_back(bits | Source::sign); };
    if constexpr (Fraction + Exponent == 15) {
        for (uint64_t bits = 0; bits < 65536; ++bits) words.push_back(bits);
    } else {
        for (uint64_t exponent = 0; exponent < (1u << Exponent); ++exponent)
            for (uint64_t fraction : {uint64_t(0), uint64_t(1), (uint64_t(1) << (Fraction - 1)) - 1,
                    uint64_t(1) << (Fraction - 1), (uint64_t(1) << Fraction) - 2, (uint64_t(1) << Fraction) - 1})
                append((exponent << Fraction) | fraction);
        const bool e2 = fp6_e2m3();
        for (uint64_t low = 0; low < 31; ++low) {
            auto midpoint = numerical_sum(fp6_value(low, e2), fp6_value(low + 1, e2), 0); --midpoint.scale;
            const auto bits = Source::encode(midpoint, 0);
            append(bits - 1); append(bits); append(bits + 1);
            const auto exact = Source::encode(fp6_value(low, e2), 0);
            if (exact) append(exact - 1);
            append(exact); append(exact + 1);
        }
        const auto maximum = Source::encode(fp6_value(31, e2), 0);
        append(maximum - 1); append(maximum); append(maximum + 1);
    }
    std::vector<std::array<uint64_t, 3>> result;
    for (size_t i = 0; i < words.size(); ++i) result.push_back({words[i], words[(i + 113) % words.size()], words[(i + 271) % words.size()]});
    return result;
}
template<class Source> std::array<uint64_t, 7> fp6_narrow_row(const std::array<uint64_t, 3>& input) {
    const bool e2 = fp6_e2m3();
    const auto a = fp6_narrow<Source>(input[0], e2, true), b = fp6_narrow<Source>(input[1], e2, true);
    const auto x = fp6_narrow<Source>(input[0], e2, false), y = fp6_narrow<Source>(input[1], e2, false);
    const auto c = fp6_narrow<Source>(input[2], e2, true), d = fp6_narrow<Source>(input[1] ^ Source::sign, e2, true);
    return {a, x, a | (b << 8), x | (y << 8), a, a | (b << 8), a | (b << 8) | (c << 16) | (d << 24)};
}
bool fp6_narrow_equivalent(uint64_t actual, uint64_t expected, size_t) { return actual == expected; }
template<class Format> bool fp6_real_equivalent(uint64_t actual, uint64_t expected) {
    return actual < Format::sign * 2 && (Format::is_nan(expected) ? Format::is_nan(actual) : actual == expected);
}
void execute_fp6(const loaded_program& program) {
    const unsigned source = fp6_profile / 2;
    const auto kernel = std::string("fp6_") + fp6_profiles[fp6_profile];
    if (source > 0 && fp6_profile < 10) {
        if (source == 1) execute_numerical_rows<uint64_t, 7>(program, kernel.c_str(), fp6_narrow_inputs<numerical_binary32, 23, 8>(), fp6_narrow_row<numerical_binary32>, fp6_narrow_equivalent);
        if (source == 2) execute_numerical_rows<uint64_t, 7>(program, kernel.c_str(), fp6_narrow_inputs<numerical_binary64, 52, 11>(), fp6_narrow_row<numerical_binary64>, fp6_narrow_equivalent);
        if (source == 3) execute_numerical_rows<uint64_t, 7>(program, kernel.c_str(), fp6_narrow_inputs<numerical_format<10, 5>, 10, 5>(), fp6_narrow_row<numerical_format<10, 5>>, fp6_narrow_equivalent);
        if (source == 4) execute_numerical_rows<uint64_t, 7>(program, kernel.c_str(), fp6_narrow_inputs<numerical_format<7, 8>, 7, 8>(), fp6_narrow_row<numerical_format<7, 8>>, fp6_narrow_equivalent);
        return;
    }
    std::vector<std::array<uint64_t, 3>> values;
    for (uint64_t bits = 0; bits < 4096; ++bits) {
        const auto high = (bits * 4051 + 17) & 4095;
        const auto word = (bits & 63) | ((bits >> 6) << 8) | ((high & 63) << 16) | ((high >> 6) << 24);
        values.push_back({word, 0, 0});
    }
    if (fp6_profile == 10) {
        execute_numerical_rows<uint64_t, 6>(program, kernel.c_str(), values, [](const auto& input) {
            const auto bits = input[0];
            return std::array<uint64_t, 6>{bits & 255, bits & 65535, bits, bits & 255, bits & 65535, bits};
        }, [](uint64_t a, uint64_t b) { return a == b; });
        return;
    }
    execute_numerical_rows<uint64_t, 12>(program, kernel.c_str(), values, [](const auto& input) {
        const auto bits = input[0]; const bool e2 = fp6_e2m3();
        const auto a = bits & 255, b = (bits >> 8) & 255;
        return std::array<uint64_t, 12>{fp6_widen<numerical_format<10, 5>>(a, e2), fp6_widen<numerical_format<7, 8>>(a, e2),
            fp6_widen<numerical_binary32>(a, e2), fp6_widen<numerical_binary64>(a, e2),
            fp6_widen<numerical_format<10, 5>>(a, e2) | (fp6_widen<numerical_format<10, 5>>(b, e2) << 16),
            fp6_widen<numerical_format<7, 8>>(a, e2) | (fp6_widen<numerical_format<7, 8>>(b, e2) << 16),
            fp6_widen<numerical_binary32>(a, e2), fp6_widen<numerical_binary32>(b, e2),
            fp6_widen<numerical_binary32>(a, e2), fp6_widen<numerical_binary32>(b, e2),
            fp6_widen<numerical_binary32>((bits >> 16) & 255, e2), fp6_widen<numerical_binary32>(bits >> 24, e2)};
    }, [](uint64_t actual, uint64_t expected, size_t field) {
        if (field == 3) return numerical_binary64::is_nan(expected) ? numerical_binary64::is_nan(actual) : actual == expected;
        if (field == 0) return fp6_real_equivalent<numerical_format<10, 5>>(actual, expected);
        if (field == 1) return fp6_real_equivalent<numerical_format<7, 8>>(actual, expected);
        if (field == 4 || field == 5) {
            if (actual >> 32) return false;
            for (unsigned shift : {0u, 16u}) {
                const auto a = (actual >> shift) & 65535, e = (expected >> shift) & 65535;
                if (field == 4 ? !fp6_real_equivalent<numerical_format<10, 5>>(a, e) : !fp6_real_equivalent<numerical_format<7, 8>>(a, e)) return false;
            }
            return true;
        }
        return fp6_real_equivalent<numerical_binary32>(actual, expected);
    });
}
}
int main(int argc, char** argv) {
    if (argc != 4) return 1;
    const auto& names = cudavm::fp6_profiles;
    const auto found = std::find(names.begin(), names.end(), std::string_view(argv[3]));
    if (found == names.end()) return 1;
    cudavm::fp6_profile = unsigned(found - names.begin());
    const auto source = "#define CUDAVM_FP6_PROFILE " + std::to_string(cudavm::fp6_profile) + "\n" + std::string(cudavm::fixtures.source);
    auto fixtures = cudavm::fixtures; fixtures.source = source;
    return cudavm::run_test(3, argv, fixtures, cudavm::execute_fp6);
}
