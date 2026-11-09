#include "test_fixtures.hpp"
#include "oracle.hpp"
#include <iostream>

namespace cudavm {
constexpr std::array<const char*, 3> fast_math_profiles{"divide_normal", "divide_large", "divide_special"};
unsigned fast_math_profile;
void execute_fast_math_intrinsics(const loaded_program& program) {
    execute_numerical_rows<uint32_t, 2>(program, "fast_math_divide", fast_divide_inputs(fast_math_profile), fast_divide_oracle,
        [](uint32_t actual, uint32_t expected, size_t field) { return fast_divide_equivalent(actual, expected, field, fast_math_profile); });
}
}
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: <executable> <backend> cu <profile>\n"; return 1; }
    const auto& profiles = cudavm::fast_math_profiles;
    const auto selected = std::find(profiles.begin(), profiles.end(), std::string_view(argv[3]));
    if (selected == profiles.end()) return 1;
    cudavm::fast_math_profile = unsigned(selected - profiles.begin());
    return cudavm::run_conformance(3, argv, cudavm::fixtures, cudavm::execute_fast_math_intrinsics);
}
