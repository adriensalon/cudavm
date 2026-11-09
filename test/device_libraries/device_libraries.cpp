#include "test_framework.hpp"
#include "device_library_images.hpp"

namespace cudavm {
std::string_view device_libraries_case = "chain";
uint32_t device_archive_oracle(uint32_t value) {
    const uint32_t argument = (value * 3u + 5u) ^ 0x193759bdu;
    return device_libraries_case == "chain" ? argument * 13u + 24u : argument * 19u + 41u;
}
void execute_device_archive(const loaded_program& program) {
    constexpr unsigned capacity = 139, fields = 4, guards = 13;
    constexpr uint32_t guard = 0xa73b951d;
    constexpr std::array<uint32_t, 8> edges{0, 1, 0x7fffffff, 0x80000000, 0xffffffff, 0xfffffffe, 0x193759bd, 0xabcdef01};
    require(!program.contains_function("archive_unused_kernel"), "unreferenced archive kernel remains unselected");
    const auto global = program.global("archive_guard", sizeof(uint32_t));
    std::array<uint32_t, 5> global_bytes; global_bytes.fill(guard);
    CONFORMANCE_CUDA(cuMemcpyDtoH(global_bytes.data() + 2, global, 4));
    require(global_bytes[2] == 0x13579bdfu && global_bytes[0] == guard && global_bytes[1] == guard
        && global_bytes[3] == guard && global_bytes[4] == guard, "root global initializer excludes conflicting unused archive member");
    std::vector<uint32_t> input(capacity + guards * 2, guard), preserved(input.size());
    std::vector<uint32_t> expected((capacity + 3) * fields + guards * 2, guard), actual(expected.size());
    device_buffer source(input.size() * 4), output(expected.size() * 4);
    auto from = source.value + guards * 4, to = output.value + guards * 4;
    for (unsigned phase = 0; phase < 2; ++phase) {
        const uint32_t global_value = phase ? 0xffffffe7u : 0x13579bdfu;
        CONFORMANCE_CUDA(cuMemcpyHtoD(global, &global_value, sizeof(global_value)));
        for (unsigned i = 0; i < capacity; ++i)
            input[guards + i] = i < edges.size() ? edges[(i + phase) % edges.size()] : (i * 0x7654321u) ^ (phase * 0x193759bdu);
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        for (unsigned threads : {32u, 61u}) for (unsigned count : {0u, 1u, 33u, capacity}) {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
            void* arguments[]{&to, &from, &count};
            CONFORMANCE_CUDA(cuLaunchKernel(program.function("archive_results"), (capacity + threads - 1) / threads, 1, 1,
                threads, 1, 1, 0, nullptr, arguments, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
            for (unsigned i = 0; i < count; ++i) {
                const uint32_t value = input[guards + i];
                const std::array<uint32_t, fields> row{device_archive_oracle(value), device_archive_oracle(value ^ i),
                    global_value, value ^ (i * 0x193759bdu)};
                std::copy(row.begin(), row.end(), expected.begin() + guards + i * fields);
            }
            expect_words(actual, expected, "device archive extraction, dependency results and complete output guards");
            CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
            expect_words(preserved, input, "device archives preserve complete input and guards");
        }
    }
}
void execute_device_libraries() {
    require(device_libraries_case == "chain" || device_libraries_case == "cycle", "unknown device_libraries case");
    const auto images = device_libraries_case == "chain"
        ? std::array{device_library_chain_forward, device_library_chain_reverse}
        : std::array{device_library_cycle_forward, device_library_cycle_reverse};
    for (const auto* image : images) for (bool library : {false, true}) {
        loaded_program program(image, library);
        execute_device_archive(program);
        CONFORMANCE_CUDA(cuCtxSynchronize());
    }
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::device_libraries_case = argv[2]; argc = 2; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_device_libraries, 0, cudavm::probe_execution);
}
