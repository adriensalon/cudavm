#include "test_fixtures.hpp"
#include <iostream>

namespace cudavm {
#include "reference.hpp"
constexpr std::array<const char*, 6> double_transcendental_profiles{"exp", "exp2", "exp10", "log", "log2", "log10"};
unsigned double_transcendental_profile;
void execute_double_transcendental_math(const loaded_program& program) {
    const bool logarithm = double_transcendental_profile >= 3;
    const auto* rows = logarithm ? double_logarithmic_reference : double_exponential_reference;
    const auto size = logarithm ? std::size(double_logarithmic_reference) : std::size(double_exponential_reference);
    std::vector<std::array<uint64_t, 3>> values;
    for (size_t i = 0; i != size; ++i) values.push_back({rows[i][0], 0, 0});
    const auto kernel = std::string("double_transcendental_") + double_transcendental_profiles[double_transcendental_profile];
    execute_numerical_rows<uint64_t, 1>(program, kernel.c_str(), values, [rows, size](const auto& input) {
        const auto* row = std::lower_bound(rows, rows + size, input[0], [](const auto& candidate, uint64_t bits) { return candidate[0] < bits; });
        require(row != rows + size && (*row)[0] == input[0], "missing binary64 reference row");
        return std::array<uint64_t, 1>{(*row)[double_transcendental_profile % 3 + 1]};
    }, [](uint64_t actual, uint64_t expected) {
        if (numerical_binary64::is_nan(expected)) return numerical_binary64::is_nan(actual);
        if (numerical_binary64::is_zero(expected) || numerical_binary64::is_infinite(expected)) return actual == expected;
        if (numerical_binary64::is_nan(actual) || numerical_binary64::is_infinite(actual)) return false;
        auto order = [](uint64_t bits) { return bits & numerical_binary64::sign ? ~bits : bits | numerical_binary64::sign; };
        const auto a = order(actual), b = order(expected);
        return (a > b ? a - b : b - a) <= 1;
    });
}
}
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: <executable> <backend> cu <function>\n"; return 1; }
    const auto& profiles = cudavm::double_transcendental_profiles;
    const auto selected = std::find(profiles.begin(), profiles.end(), std::string_view(argv[3]));
    if (selected == profiles.end()) return 1;
    cudavm::double_transcendental_profile = unsigned(selected - profiles.begin());
    const auto source = "#define CUDAVM_DOUBLE_TRANSCENDENTAL_PROFILE " + std::to_string(cudavm::double_transcendental_profile) + "\n" + std::string(cudavm::fixtures.source);
    auto fixtures = cudavm::fixtures; fixtures.source = source;
    return cudavm::run_conformance(3, argv, fixtures, cudavm::execute_double_transcendental_math);
}
