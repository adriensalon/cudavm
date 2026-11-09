#include "test_fixtures.hpp"

namespace cudavm {
std::array<uint64_t, 10> integer_double_conversion_to_real(const std::array<uint64_t, 3>& input) {
    std::array<uint64_t, 10> row{};
    row[0] = numerical_binary64::encode(numerical_integer_value(input[0], 32, true), 0);
    row[1] = numerical_binary64::encode(numerical_integer_value(input[0], 32, false), 0);
    for (unsigned group = 0; group < 2; ++group) for (unsigned rounding = 0; rounding < 4; ++rounding)
        row[2 + group * 4 + rounding] = numerical_binary64::encode(numerical_integer_value(input[0], 64, group == 0), rounding);
    return row;
}
std::array<uint64_t, 16> integer_double_conversion_to_integer(const std::array<uint64_t, 3>& input) {
    std::array<uint64_t, 16> row; row.fill(numerical_output_guard<uint64_t>);
    for (unsigned group = 0; group < 4; ++group) if (input[1] & (1u << group))
        for (unsigned rounding = 0; rounding < 4; ++rounding) row[group * 4 + rounding] = numerical_to_integer<numerical_binary64>(input[0], rounding);
    return row;
}
void execute_integer_double_conversion(const loaded_program& program) {
    auto exact = [](uint64_t actual, uint64_t expected) { return actual == expected; };
    execute_numerical_rows<uint64_t, 10>(program, "integer_double_to_real", numerical_integer_inputs(), integer_double_conversion_to_real, exact);
    execute_numerical_rows<uint64_t, 16>(program, "integer_double_to_integer",
        numerical_real_conversion_inputs<numerical_binary64, 52, 11>(), integer_double_conversion_to_integer, exact);
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_integer_double_conversion);
}
