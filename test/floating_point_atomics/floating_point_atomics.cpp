#include "fixtures.hpp"
#include <algorithm>
#include <bit>

namespace cudavm {
// Encode q/4 using integer arithmetic; the corpus fits exactly in either format.
template<class Word> Word atomic_quarters(int64_t q) {
    if (q == 0) return 0;
    constexpr unsigned fraction = sizeof(Word) == 4 ? 23 : 52;
    constexpr unsigned bias = sizeof(Word) == 4 ? 127 : 1023;
    const uint64_t magnitude = q < 0 ? uint64_t(-q) : uint64_t(q);
    const unsigned top = unsigned(std::bit_width(magnitude) - 1);
    const Word sign = q < 0 ? Word(1) << (sizeof(Word) * 8 - 1) : 0;
    const Word mask = (Word(1) << fraction) - 1;
    return sign | (Word(top + bias - 2) << fraction)
        | (Word(magnitude << (fraction - top)) & mask);
}

template<class Word> struct floating_atomic_observation {
    static constexpr Word guard = Word(0x89abcdef01234567ull);
    std::vector<Word> values;
    device_buffer storage;
    CUdeviceptr pointer;
    explicit floating_atomic_observation(size_t count)
        : values(count + 8, guard), storage(values.size() * sizeof(Word)),
          pointer(storage.value + 4 * sizeof(Word)) {}
    void reset() {
        std::fill(values.begin(), values.end(), guard);
        CONFORMANCE_CUDA(cuMemcpyHtoD(storage.value, values.data(), values.size() * sizeof(Word)));
    }
    void fetch(size_t written, const std::string& context) {
        CONFORMANCE_CUDA(cuMemcpyDtoH(values.data(), storage.value, values.size() * sizeof(Word)));
        for (size_t i = 0; i < values.size(); ++i)
            if (i < 4 || i >= written + 4)
                require(values[i] == guard, context + ": output guard/inactive lane " + std::to_string(i));
    }
};

template<class Word> void expect_floating_tickets(std::vector<Word> actual, int64_t seed,
                                                int64_t delta, const std::string& context) {
    std::vector<Word> expected;
    for (size_t i = 0; i < actual.size(); ++i)
        expected.push_back(atomic_quarters<Word>(seed + delta * int64_t(i)));
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    require(actual == expected, context + ": atomic old-value multiset");
}

template<class Word> void check_floating_global(const loaded_program& program, const char* name) {
    const auto function = program.function(name);
    constexpr unsigned blocks = 3;
    constexpr Word guard = floating_atomic_observation<Word>::guard;
    for (unsigned lanes : {1u, 13u, 33u, 64u}) {
        const unsigned total = blocks * lanes;
        floating_atomic_observation<Word> output(total);
        device_buffer counters(6 * sizeof(Word));
        auto state = counters.value + sizeof(Word);
        for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 7}) {
            const unsigned active = std::min(count, total);
            for (int64_t seed : {int64_t(0), int64_t(-64)}) {
                for (int64_t delta : {int64_t(1), int64_t(4), int64_t(-2)}) {
                    const auto context = std::string(name) + " lanes=" + std::to_string(lanes)
                        + " count=" + std::to_string(count) + " seed/4=" + std::to_string(seed)
                        + " delta/4=" + std::to_string(delta);
                    const std::array<Word, 6> initial{guard, atomic_quarters<Word>(seed),
                        atomic_quarters<Word>(seed), guard, guard, guard};
                    CONFORMANCE_CUDA(cuMemcpyHtoD(counters.value, initial.data(), sizeof(initial)));
                    output.reset();
                    Word operand = atomic_quarters<Word>(delta);
                    void* args[]{&output.pointer, &state, &count, &operand};
                    CONFORMANCE_CUDA(cuLaunchKernel(function, blocks, 1, 1, lanes, 1, 1,
                        0, nullptr, args, nullptr));
                    output.fetch(active, context);
                    std::array<Word, 6> actual{};
                    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), counters.value, sizeof(actual)));
                    auto expected = initial;
                    expected[1] = expected[2] = atomic_quarters<Word>(seed + delta * active);
                    require(actual == expected, context + ": returned/discarded add and state guards");
                    expect_floating_tickets(std::vector<Word>(output.values.begin() + 4,
                        output.values.begin() + 4 + active), seed, delta, context);
                }
            }
        }
    }
}

void check_floating_shared(const loaded_program& program) {
    const auto function = program.function("floating_shared_add");
    constexpr unsigned blocks = 3;
    for (unsigned lanes : {1u, 13u, 33u, 64u}) {
        const unsigned total = blocks * lanes;
        floating_atomic_observation<uint32_t> output(total), totals(blocks * 2);
        for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 7}) {
            for (int64_t seed : {int64_t(0), int64_t(-64)}) {
                for (int64_t delta : {int64_t(1), int64_t(4), int64_t(-2)}) {
                    const auto context = "shared float lanes=" + std::to_string(lanes)
                        + " count=" + std::to_string(count) + " seed/4=" + std::to_string(seed)
                        + " delta/4=" + std::to_string(delta);
                    output.reset();
                    totals.reset();
                    uint32_t initial = atomic_quarters<uint32_t>(seed);
                    uint32_t operand = atomic_quarters<uint32_t>(delta);
                    void* args[]{&output.pointer, &totals.pointer, &count, &initial, &operand};
                    CONFORMANCE_CUDA(cuLaunchKernel(function, blocks, 1, 1, lanes, 1, 1,
                        0, nullptr, args, nullptr));
                    output.fetch(std::min(count, total), context);
                    totals.fetch(blocks * 2, context);
                    for (unsigned block = 0; block < blocks; ++block) {
                        const unsigned base = block * lanes;
                        const unsigned active = count > base ? std::min(lanes, count - base) : 0;
                        const auto final = atomic_quarters<uint32_t>(seed + delta * active);
                        require(totals.values[4 + block * 2] == final
                            && totals.values[5 + block * 2] == final,
                            context + ": returned/discarded per-block final state");
                        expect_floating_tickets(std::vector<uint32_t>(output.values.begin() + 4 + base,
                            output.values.begin() + 4 + base + active), seed, delta,
                            context + " block=" + std::to_string(block));
                    }
                }
            }
        }
    }
}

void check_floating_exchange(const loaded_program& program) {
    const auto function = program.function("floating_exchange");
    // Exchange copies bits: include both zeros, subnormals, infinities and NaN payloads.
    constexpr std::array<uint32_t, 16> corpus{0, 0x80000000, 1, 0x80000001,
        0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
        0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff,
        0x7f800000, 0xff800000, 0x7fc12345, 0x7fa54321};
    constexpr uint32_t guard = floating_atomic_observation<uint32_t>::guard;
    floating_atomic_observation<uint32_t> output(1);
    device_buffer counters(4 * sizeof(uint32_t));
    auto state = counters.value + sizeof(uint32_t);
    for (unsigned i = 0; i < corpus.size(); ++i) {
        const std::array<uint32_t, 4> initial{guard, corpus[i], guard, guard};
        CONFORMANCE_CUDA(cuMemcpyHtoD(counters.value, initial.data(), sizeof(initial)));
        output.reset();
        uint32_t replacement = corpus[(i * 7 + 3) % corpus.size()];
        void* args[]{&output.pointer, &state, &replacement};
        CONFORMANCE_CUDA(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
        output.fetch(1, "float exchange");
        std::array<uint32_t, 4> actual{};
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), counters.value, sizeof(actual)));
        auto expected = initial;
        expected[1] = replacement;
        require(output.values[4] == corpus[i] && actual == expected,
            "float exchange raw old/new bits and guards case=" + std::to_string(i));
    }
}

void execute_floating_point_atomics(const loaded_program& program) {
    check_floating_global<uint32_t>(program, "floating_global_add");
    check_floating_global<uint64_t>(program, "floating_global_double_add");
    check_floating_shared(program);
    check_floating_exchange(program);
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_floating_point_atomics);
}
