#include "test_fixtures.hpp"
#include <cmath>
#include <iostream>

namespace cudavm {
constexpr std::array<const char*, 9> float_transcendental_profiles{"sin", "cos", "exp", "exp2", "exp10", "log", "log2", "log10", "tanh"};
unsigned float_transcendental_profile;
uint32_t float_transcendental_oracle(uint32_t bits) {
    // Use integer format conversion so a graphics driver's host FTZ/DAZ state
    // cannot silently flush a subnormal source or expected binary32 result.
    const auto wide = numerical_binary32::is_nan(bits) ? numerical_binary64::nan
        : numerical_binary32::is_infinite(bits) ? (uint64_t(bits >> 31) << 63) | numerical_binary64::infinity
        : numerical_binary64::encode(numerical_binary32::decode(bits), 0);
    const double x = std::bit_cast<double>(wide);
    double result;
    switch (float_transcendental_profile) {
    case 0: result = std::sin(x); break;
    case 1: result = std::cos(x); break;
    case 2: result = std::exp(x); break;
    case 3: result = std::exp2(x); break;
    case 4: result = std::pow(10.0, x); break;
    case 5: result = std::log(x); break;
    case 6: result = std::log2(x); break;
    case 7: result = std::log10(x); break;
    default: result = std::tanh(x); break;
    }
    const auto answer = std::bit_cast<uint64_t>(result);
    if (numerical_binary64::is_nan(answer)) return uint32_t(numerical_binary32::nan);
    if (numerical_binary64::is_infinite(answer))
        return uint32_t(((answer >> 32) & numerical_binary32::sign) | numerical_binary32::infinity);
    return uint32_t(numerical_binary32::encode(numerical_binary64::decode(answer), 0));
}
std::vector<std::array<uint32_t, 3>> float_transcendental_inputs() {
    auto result = elementary_inputs<uint32_t, 23, 8>();
    // Encodings adjacent to reduction switches, pi multiples, logarithm
    // cancellation and exponential normal/subnormal/overflow boundaries.
    constexpr uint32_t centers[]{0x39800000, 0x3f000000, 0x3f3504f3, 0x3f490fdb, 0x3f800000,
        0x3fb504f3, 0x3fc90fdb, 0x40490fdb, 0x40c90fdb, 0x41100000, 0x41200000, 0x42000000,
        0x421a209b, 0x42aeac50, 0x42b17218, 0x42ce8ed0, 0x42cff1b4, 0x42fc0000, 0x43000000, 0x43160000};
    for (auto center : centers) for (int step = -16; step <= 16; ++step)
        for (uint32_t sign : {0u, 0x80000000u}) result.push_back({(center + step) | sign, 0, 0});
    // Quotient/summation regressions distinguished by the public one-ULP
    // log/log2 and two-ULP log10 contracts.
    for (uint32_t bits : {0x3fce91efu, 0x3f3504e7u, 0x3faf7635u, 0x3f7fb5bfu, 0x3fa674dbu, 0x3f800009u})
        result.push_back({bits, 0, 0});
    // Sample the finite exponential and hyperbolic ranges densely with exact
    // dyadics rather than mostly overflowing full-width random arguments.
    for (int integer = -1200; integer <= 1200; ++integer)
        result.push_back({uint32_t(numerical_binary32::encode({numerical_integer(unsigned(integer < 0 ? -integer : integer)), -3, integer < 0}, 0)), 0, 0});
    return result;
}
void execute_float_transcendental_math(const loaded_program& program) {
    const auto kernel = std::string("float_transcendental_") + float_transcendental_profiles[float_transcendental_profile];
    execute_numerical_rows<uint32_t, 1>(program, kernel.c_str(), float_transcendental_inputs(), [](const auto& input) {
        return std::array<uint32_t, 1>{float_transcendental_oracle(input[0])};
    }, [](uint32_t actual, uint32_t expected) {
        if (numerical_binary32::is_nan(expected)) return numerical_binary32::is_nan(actual);
        if (numerical_binary32::is_infinite(expected) || numerical_binary32::is_zero(expected)) return actual == expected;
        if ((actual & 0x7fffffffu) >= 0x7f800000u) return false;
        auto order = [](uint32_t bits) { return bits & 0x80000000u ? ~bits : bits | 0x80000000u; };
        const auto a = order(actual), b = order(expected), distance = a > b ? a - b : b - a;
        const unsigned bound = float_transcendental_profile == 5 || float_transcendental_profile == 6 ? 1 : 2;
        return distance <= bound;
    });
}
}
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: <executable> <backend> cu <function>\n"; return 1; }
    const auto& profiles = cudavm::float_transcendental_profiles;
    const auto selected = std::find(profiles.begin(), profiles.end(), std::string_view(argv[3]));
    if (selected == profiles.end()) { std::cerr << "unknown binary32 transcendental function\n"; return 1; }
    cudavm::float_transcendental_profile = unsigned(selected - profiles.begin());
    const auto source = "#define CUDAVM_TRANSCENDENTAL_PROFILE " + std::to_string(cudavm::float_transcendental_profile) + "\n"
        + std::string(cudavm::fixtures.source);
    auto fixtures = cudavm::fixtures; fixtures.source = source;
    return cudavm::run_conformance(3, argv, fixtures, cudavm::execute_float_transcendental_math);
}
