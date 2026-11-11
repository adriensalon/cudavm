#include "test_fixtures.hpp"
#include "oracle.hpp"
#include <iostream>

namespace cudavm {
constexpr std::array<const char*, 4> floating_mode_profiles{"add", "multiply", "fused", "separate"};
unsigned floating_mode_operation;
void execute_floating_point_modes(const loaded_program& program) {
    const auto kernel = "mode_" + std::string(floating_mode_profiles[floating_mode_operation]);
    execute_numerical_rows<uint32_t, 8>(program, kernel.c_str(), floating_mode_inputs(floating_mode_operation),
        [](const auto& row) { return floating_mode_oracle(row, floating_mode_operation); },
        [](uint32_t actual, uint32_t expected) {
            return numerical_binary32::is_nan(expected) ? numerical_binary32::is_nan(actual) : actual == expected;
        });
}
}
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: <executable> <backend> cu <profile>\n"; return 1; }
    const auto& profiles = cudavm::floating_mode_profiles;
    const auto selected = std::find(profiles.begin(), profiles.end(), std::string_view(argv[3]));
    if (selected == profiles.end()) return 1;
    cudavm::floating_mode_operation = unsigned(selected - profiles.begin());
    return cudavm::run_test(3, argv, cudavm::fixtures, cudavm::execute_floating_point_modes);
}
