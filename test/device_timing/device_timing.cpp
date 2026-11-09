#include "test_fixtures.hpp"

namespace cudavm {
std::string_view timing_profile = "clocks";
// Exponentiate the affine transition instead of duplicating the device loop.
uint32_t timing_state(uint32_t seed, unsigned steps) {
    uint32_t multiplier = 1664525u, increment = 1013904223u;
    while (steps) {
        if (steps & 1) seed = seed * multiplier + increment;
        increment *= multiplier + 1; multiplier *= multiplier; steps >>= 1;
    }
    return seed;
}
void timing_case(const loaded_program& program, unsigned x, unsigned y, unsigned count, unsigned sleep, unsigned nanoseconds) {
    constexpr unsigned blocks = 3, guards = 16;
    constexpr uint32_t guard = 0x8c5917a3;
    const unsigned capacity = blocks * x * y;
    std::vector<uint32_t> input(guards * 2 + capacity, guard), preserved(input.size());
    for (unsigned i = 0; i < capacity; ++i) input[guards + i] = 0x93b17d5au ^ (i * 0x10203u);
    std::vector<uint32_t> initial(guards * 2 + (capacity + 1) * 16, guard), actual(initial.size());
    device_buffer source(input.size() * 4), destination(initial.size() * 4);
    CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
    CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, initial.data(), initial.size() * 4));
    auto in = source.value + guards * 4, out = destination.value + guards * 4;
    void* arguments[]{&out, &in, &count, &sleep, &nanoseconds};
    CONFORMANCE_CUDA(cuLaunchKernel(program.function("timing_samples"), blocks, 1, 1, x, y, 1, 0, nullptr, arguments, nullptr));
    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
    for (unsigned i = 0; i < count; ++i) {
        const auto base = guards + i * 16;
        bool narrow_changed = false, wide_changed = false;
        for (unsigned phase = 0; phase < 4; ++phase) {
            require(actual[base + phase * 4 + 3] == timing_state(input[guards + i], (phase + 1) * 19),
                "device timing/sleep preserves dependent integer state");
            narrow_changed |= actual[base + phase * 4] != actual[base];
            wide_changed |= actual[base + phase * 4 + 1] != actual[base + 1] || actual[base + phase * 4 + 2] != actual[base + 2];
        }
        // Per-SM counters can wrap or move with a rescheduled thread. Check
        // observable change, without ordering separate samples or comparing SMs.
        require(narrow_changed && wide_changed, "32/64-bit device counters change across samples");
    }
    for (size_t i = 0; i < actual.size(); ++i)
        if (i < guards || i >= guards + count * 16) require(actual[i] == guard, "device timing output guards and inactive records");
    CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, input.size() * 4));
    expect_words(preserved, input, "device timing source and allocation guards are preserved");
}
void execute_device_timing(const loaded_program& program) {
    require(timing_profile == "clocks" || timing_profile == "sleep", "unknown device timing profile");
    for (auto geometry : {std::array<unsigned, 2>{1, 1}, {11, 3}, {16, 4}}) {
        const auto capacity = geometry[0] * geometry[1] * 3;
        for (unsigned count : {0u, 1u, capacity - 1, capacity}) {
            if (timing_profile == "clocks") timing_case(program, geometry[0], geometry[1], count, 0, 0);
            else for (unsigned ns : {0u, 32u, 1000u}) timing_case(program, geometry[0], geometry[1], count, 1, ns);
        }
    }
}
}
int main(int argc, char** argv) {
    if (argc == 4) { cudavm::timing_profile = argv[3]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_device_timing);
}
