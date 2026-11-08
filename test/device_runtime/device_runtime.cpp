#include "fixtures.hpp"

namespace cudavm {
void execute_device_runtime(const loaded_program& program) {
    std::vector<std::array<uint32_t, 3>> rows;
    uint32_t state = 0x3c6ef372;
    auto next = [&] { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; };
    for (unsigned i = 0; i != 1025; ++i) {
        const auto before = next(), after = next(), value = next(); rows.push_back({before, after, value});
    }
    execute_numerical_rows<uint32_t, 12>(program, "device_runtime_version", rows, [](const auto& row) {
        return std::array<uint32_t, 12>{cudaSuccess, CUDART_VERSION, row[0], row[1], cudaErrorInvalidValue,
            cudaSuccess, CUDART_VERSION, row[2], row[0], CUDART_VERSION, row[1], cudaSuccess};
    }, [](uint32_t actual, uint32_t expected) { return actual == expected; });
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_device_runtime);
}
