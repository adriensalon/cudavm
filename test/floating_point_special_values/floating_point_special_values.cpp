#include "test_fixtures.hpp"
#include "oracle.hpp"
#include <iostream>

namespace cudavm {
constexpr std::array<const char*, 2> special_value_profiles{"float_neighbors", "double_neighbors"};
unsigned special_value_profile;
template<class Format, unsigned Fraction, unsigned Exponent>
void execute_special_neighbors(const loaded_program& program, const char* kernel) {
    execute_numerical_rows<uint64_t, 8>(program, kernel, special_neighbor_inputs<Fraction, Exponent>(), special_neighbor_oracle<Format>,
        [](uint64_t actual, uint64_t expected, size_t field) {
            if (field == 7) return actual == expected;
            // Binary32 outputs must also have a zero upper storage word.
            if constexpr (sizeof(uint64_t) * 8 > Fraction + Exponent + 1)
                if (actual >> (Fraction + Exponent + 1)) return false;
            return Format::is_nan(expected) ? Format::is_nan(actual) : actual == expected;
        });
}
void execute_floating_point_special_values(const loaded_program& program) {
    if (!special_value_profile) execute_special_neighbors<numerical_binary32, 23, 8>(program, "special_float_neighbors");
    else execute_special_neighbors<numerical_binary64, 52, 11>(program, "special_double_neighbors");
}
}
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: <executable> <backend> cu <profile>\n"; return 1; }
    const auto& profiles = cudavm::special_value_profiles;
    const auto selected = std::find(profiles.begin(), profiles.end(), std::string_view(argv[3]));
    if (selected == profiles.end()) return 1;
    cudavm::special_value_profile = unsigned(selected - profiles.begin());
    const auto source = "#define CUDAVM_SPECIAL_PROFILE " + std::to_string(cudavm::special_value_profile) + "\n" + std::string(cudavm::fixtures.source);
    auto fixtures = cudavm::fixtures; fixtures.source = source;
    return cudavm::run_conformance(3, argv, fixtures, cudavm::execute_floating_point_special_values);
}
