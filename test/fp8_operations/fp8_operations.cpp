#include "fixtures.hpp"
#include <iostream>

namespace cudavm {
constexpr std::array<const char*, 11> fp8_profiles{"e4m3_widen", "e5m2_widen", "e4m3_float", "e5m2_float", "e4m3_double", "e5m2_double", "e4m3_half", "e5m2_half", "e4m3_bfloat", "e5m2_bfloat", "representation"};
unsigned fp8_profile;
bool fp8_e4m3() { return !(fp8_profile & 1); }
bool fp8_nan(uint64_t bits, bool e4) { return (bits & 127) > (e4 ? 126u : 124u); }
numerical_dyadic fp8_value(uint64_t bits, bool e4) {
    return e4 ? numerical_format<3, 4>::decode(bits) : numerical_format<2, 5>::decode(bits);
}
// Enumerate adjacent destination values and compare exact midpoints. The
// finite-only format's last conceptual value is 480, with midpoint 464.
template<class Source> uint64_t fp8_narrow(uint64_t bits, bool e4, bool saturate) {
    const auto sign = (bits & Source::sign) ? 128u : 0u;
    const uint64_t maximum = e4 ? 126 : 123, overflow = e4 ? 127 : 124;
    if (Source::is_nan(bits)) return 127;
    if (Source::is_infinite(bits)) return sign | (saturate ? maximum : overflow);
    auto value = Source::decode(bits); value.negative = false;
    if (saturate && elementary_compare(value, fp8_value(maximum, e4)) >= 0) return sign | maximum;
    uint64_t low = 0, high = maximum + 1;
    while (high - low > 1) {
        const auto middle = (low + high) / 2;
        if (elementary_compare(value, fp8_value(middle, e4)) >= 0) low = middle;
        else high = middle;
    }
    auto midpoint = numerical_sum(fp8_value(low, e4), fp8_value(high, e4), 0); --midpoint.scale;
    const auto order = elementary_compare(value, midpoint);
    const auto rounded = order > 0 || (!order && (low & 1)) ? high : low;
    return sign | (rounded > maximum ? overflow : rounded);
}
template<class Destination> uint64_t fp8_widen(uint64_t bits, bool e4) {
    if (fp8_nan(bits, e4)) return Destination::nan;
    if (!e4 && (bits & 127) == 124) return (bits & 128 ? Destination::sign : 0) | Destination::infinity;
    return Destination::encode(fp8_value(bits, e4), 0);
}
template<class Source, unsigned Fraction, unsigned Exponent>
std::vector<std::array<uint64_t, 3>> fp8_narrow_inputs() {
    std::vector<uint64_t> words;
    auto append = [&](uint64_t bits) { words.push_back(bits); words.push_back(bits | Source::sign); };
    if constexpr (Fraction + Exponent == 15) {
        for (uint64_t bits = 0; bits < 65536; ++bits) words.push_back(bits);
    } else {
        for (uint64_t exponent = 0; exponent < (1u << Exponent); ++exponent)
            for (uint64_t fraction : {uint64_t(0), uint64_t(1), (uint64_t(1) << (Fraction - 1)) - 1,
                    uint64_t(1) << (Fraction - 1), (uint64_t(1) << Fraction) - 2, (uint64_t(1) << Fraction) - 1})
                append((exponent << Fraction) | fraction);
        const bool e4 = fp8_e4m3();
        for (uint64_t low = 0; low <= (e4 ? 126u : 123u); ++low) {
            auto midpoint = numerical_sum(fp8_value(low, e4), fp8_value(low + 1, e4), 0); --midpoint.scale;
            const auto bits = Source::encode(midpoint, 0);
            append(bits - 1); append(bits); append(bits + 1);
        }
    }
    std::vector<std::array<uint64_t, 3>> result;
    for (size_t i = 0; i < words.size(); ++i) result.push_back({words[i], words[(i + 113) % words.size()], words[(i + 271) % words.size()]});
    return result;
}
template<class Source> std::array<uint64_t, 7> fp8_narrow_row(const std::array<uint64_t, 3>& input) {
    const bool e4 = fp8_e4m3();
    const auto a = fp8_narrow<Source>(input[0], e4, false), b = fp8_narrow<Source>(input[1], e4, false);
    const auto x = fp8_narrow<Source>(input[0], e4, true), y = fp8_narrow<Source>(input[1], e4, true);
    const auto z = fp8_narrow<Source>(input[2], e4, true), w = fp8_narrow<Source>(input[1] ^ Source::sign, e4, true);
    return {a, x, a | (b << 8), x | (y << 8), x, x | (y << 8), x | (y << 8) | (z << 16) | (w << 24)};
}
bool fp8_narrow_equivalent(uint64_t actual, uint64_t expected, size_t field) {
    const unsigned lanes = field == 6 ? 4 : field == 2 || field == 3 || field == 5 ? 2 : 1;
    if (actual >> (lanes * 8)) return false;
    for (unsigned lane = 0; lane < lanes; ++lane) {
        const auto a = (actual >> (lane * 8)) & 255, e = (expected >> (lane * 8)) & 255;
        if (fp8_nan(e, fp8_e4m3()) ? !fp8_nan(a, fp8_e4m3()) : a != e) return false;
    }
    return true;
}
template<class Format> bool fp8_real_equivalent(uint64_t actual, uint64_t expected) {
    return actual < Format::sign * 2 && (Format::is_nan(expected) ? Format::is_nan(actual) : actual == expected);
}
void execute_fp8(const loaded_program& program) {
    const unsigned source = fp8_profile / 2;
    const auto kernel = std::string("fp8_") + fp8_profiles[fp8_profile];
    if (source > 0 && fp8_profile < 10) {
        if (source == 1) execute_numerical_rows<uint64_t, 7>(program, kernel.c_str(), fp8_narrow_inputs<numerical_binary32, 23, 8>(), fp8_narrow_row<numerical_binary32>, fp8_narrow_equivalent);
        if (source == 2) execute_numerical_rows<uint64_t, 7>(program, kernel.c_str(), fp8_narrow_inputs<numerical_binary64, 52, 11>(), fp8_narrow_row<numerical_binary64>, fp8_narrow_equivalent);
        if (source == 3) execute_numerical_rows<uint64_t, 7>(program, kernel.c_str(), fp8_narrow_inputs<numerical_format<10, 5>, 10, 5>(), fp8_narrow_row<numerical_format<10, 5>>, fp8_narrow_equivalent);
        if (source == 4) execute_numerical_rows<uint64_t, 7>(program, kernel.c_str(), fp8_narrow_inputs<numerical_format<7, 8>, 7, 8>(), fp8_narrow_row<numerical_format<7, 8>>, fp8_narrow_equivalent);
        return;
    }
    std::vector<std::array<uint64_t, 3>> values;
    for (uint64_t bits = 0; bits < 65536; ++bits) values.push_back({bits | (((bits * 40503 + 17) & 65535) << 16), 0, 0});
    if (fp8_profile == 10) {
        execute_numerical_rows<uint64_t, 6>(program, kernel.c_str(), values, [](const auto& input) {
            const auto bits = input[0];
            return std::array<uint64_t, 6>{bits & 255, bits & 65535, bits, bits & 255, bits & 65535, bits};
        }, [](uint64_t a, uint64_t b) { return a == b; });
        return;
    }
    execute_numerical_rows<uint64_t, 12>(program, kernel.c_str(), values, [](const auto& input) {
        const auto bits = input[0]; const bool e4 = fp8_e4m3();
        const auto a = bits & 255, b = (bits >> 8) & 255;
        return std::array<uint64_t, 12>{fp8_widen<numerical_format<10, 5>>(a, e4), fp8_widen<numerical_format<7, 8>>(a, e4),
            fp8_widen<numerical_binary32>(a, e4), fp8_widen<numerical_binary64>(a, e4),
            fp8_widen<numerical_format<10, 5>>(a, e4) | (fp8_widen<numerical_format<10, 5>>(b, e4) << 16),
            fp8_widen<numerical_format<7, 8>>(a, e4) | (fp8_widen<numerical_format<7, 8>>(b, e4) << 16),
            fp8_widen<numerical_binary32>(a, e4), fp8_widen<numerical_binary32>(b, e4),
            fp8_widen<numerical_binary32>(a, e4), fp8_widen<numerical_binary32>(b, e4),
            fp8_widen<numerical_binary32>((bits >> 16) & 255, e4), fp8_widen<numerical_binary32>(bits >> 24, e4)};
    }, [](uint64_t actual, uint64_t expected, size_t field) {
        if (field == 3) return numerical_binary64::is_nan(expected) ? numerical_binary64::is_nan(actual) : actual == expected;
        if (field == 0) return fp8_real_equivalent<numerical_format<10, 5>>(actual, expected);
        if (field == 1) return fp8_real_equivalent<numerical_format<7, 8>>(actual, expected);
        if (field == 4 || field == 5) {
            if (actual >> 32) return false;
            for (unsigned shift : {0u, 16u}) {
                const auto a = (actual >> shift) & 65535, e = (expected >> shift) & 65535;
                if (field == 4 ? !fp8_real_equivalent<numerical_format<10, 5>>(a, e) : !fp8_real_equivalent<numerical_format<7, 8>>(a, e)) return false;
            }
            return true;
        }
        return fp8_real_equivalent<numerical_binary32>(actual, expected);
    });
}
}
int main(int argc, char** argv) {
    if (argc != 4) return 1;
    const auto& names = cudavm::fp8_profiles;
    const auto found = std::find(names.begin(), names.end(), std::string_view(argv[3]));
    if (found == names.end()) return 1;
    cudavm::fp8_profile = unsigned(found - names.begin());
    const auto source = "#define CUDAVM_FP8_PROFILE " + std::to_string(cudavm::fp8_profile) + "\n" + std::string(cudavm::fixtures.source);
    auto fixtures = cudavm::fixtures; fixtures.source = source;
    return cudavm::run_conformance(3, argv, fixtures, cudavm::execute_fp8);
}
