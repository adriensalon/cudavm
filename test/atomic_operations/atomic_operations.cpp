#include "fixtures.hpp"
#include <algorithm>

namespace cudavm {
constexpr uint32_t atomic_guard = 0xdeadbeef;

struct atomic_observation {
    std::vector<uint32_t> values;
    device_buffer storage;
    CUdeviceptr pointer;
    explicit atomic_observation(size_t capacity)
        : values(capacity + 4, atomic_guard), storage(values.size() * 4), pointer(storage.value + 4) {}
    void reset() {
        std::fill(values.begin(), values.end(), atomic_guard);
        CONFORMANCE_CUDA(cuMemcpyHtoD(storage.value, values.data(), values.size() * 4));
    }
    void fetch(size_t written, const std::string& context) {
        CONFORMANCE_CUDA(cuMemcpyDtoH(values.data(), storage.value, values.size() * 4));
        require(values.front() == atomic_guard, context + ": output prefix guard");
        for (size_t i = written + 1; i < values.size(); ++i)
            require(values[i] == atomic_guard, context + ": output tail word " + std::to_string(i));
    }
};

void expect_atomic_tickets(std::vector<uint32_t> tickets, const std::string& context) {
    std::sort(tickets.begin(), tickets.end());
    for (size_t i = 0; i < tickets.size(); ++i)
        require(tickets[i] == i, context + ": missing or repeated atomic ticket " + std::to_string(i));
}

void execute_atomic_operations(const loaded_program& program) {
    const auto add = program.function("global_add");
    const auto cas = program.function("global_cas");
    const auto shared_cas = program.function("shared_cas");
    const auto histogram = program.function("shared_histogram");
    constexpr unsigned blocks = 3;
    for (unsigned lanes : {4u, 13u, 32u, 64u, 128u}) {
        const unsigned total = blocks * lanes;
        atomic_observation output(total + blocks * 8);
        device_buffer counters(6 * 4);
        auto state = counters.value + 4;
        const auto shape = " lanes=" + std::to_string(lanes);
        for (unsigned count : {0u, 1u, total / 2 + 1, total}) {
            const auto context = shape + " count=" + std::to_string(count);
            for (uint32_t seed : {0u, 0xfffffffdu}) {
                const std::array<uint32_t, 6> initial{atomic_guard, seed, seed, seed, seed, atomic_guard};
                CONFORMANCE_CUDA(cuMemcpyHtoD(counters.value, initial.data(), sizeof(initial)));
                output.reset();
                void* args[]{&output.pointer, &state, &count};
                CONFORMANCE_CUDA(cuLaunchKernel(add, blocks, 1, 1, lanes, 1, 1, 0, nullptr, args, nullptr));
                output.fetch(total, "global add" + context);
                std::array<uint32_t, 6> actual{};
                CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), counters.value, sizeof(actual)));
                auto expected = initial;
                for (unsigned bucket = 0; bucket < 2; ++bucket) {
                    std::vector<uint32_t> tickets;
                    const unsigned active = (count + 1 - bucket) / 2;
                    for (unsigned i = bucket; i < total; i += 2) {
                        const uint32_t ticket = output.values[i + 1] - seed;
                        require(ticket <= active, "global add old-value range" + context);
                        if (i < count) tickets.push_back(ticket);
                    }
                    expect_atomic_tickets(tickets, "global add" + context);
                    expected[bucket + 1] = seed + active;
                    expected[bucket + 3] = seed + active;
                }
                expect_words(actual, expected, "returned/discarded global add state" + context);
            }
        }

        for (uint32_t replacement : {0u, 1u, 0xfedcba98u}) {
            for (uint32_t initial_value : {0u, 0x87654321u, 0xffffffffu}) {
                const std::array<uint32_t, 6> initial{atomic_guard, initial_value, initial_value,
                    0x12345678, 0x23456789, atomic_guard};
                CONFORMANCE_CUDA(cuMemcpyHtoD(counters.value, initial.data(), sizeof(initial)));
                output.reset();
                void* args[]{&output.pointer, &state, &replacement};
                CONFORMANCE_CUDA(cuLaunchKernel(cas, blocks, 1, 1, lanes, 1, 1, 0, nullptr, args, nullptr));
                output.fetch(total, "global CAS" + shape);
                std::array<uint32_t, 6> actual{};
                CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), counters.value, sizeof(actual)));
                auto expected = initial;
                expected[1] = expected[2] = initial_value == 0 ? replacement : initial_value;
                expect_words(actual, expected, "returned/discarded global CAS state" + shape);
                unsigned zeros = 0;
                for (unsigned i = 0; i < total; ++i) {
                    const uint32_t old = output.values[i + 1];
                    zeros += old == 0;
                    require(initial_value == 0 ? (old == 0 || old == replacement) : old == initial_value,
                        "global CAS old-value domain" + shape);
                }
                const unsigned expected_zeros = initial_value != 0 ? 0 : replacement == 0 ? total : 1;
                require(zeros == expected_zeros, "global CAS winner count" + shape);
            }
            output.reset();
            unsigned unused = 0;
            void* args[]{&output.pointer, &replacement, &unused};
            CONFORMANCE_CUDA(cuLaunchKernel(shared_cas, blocks, 1, 1, lanes, 1, 1, 8, nullptr, args, nullptr));
            output.fetch(total + blocks, "shared CAS" + shape);
            for (unsigned block = 0; block < blocks; ++block) {
                unsigned zeros = 0;
                for (unsigned lane = 0; lane < lanes; ++lane) {
                    const uint32_t old = output.values[block * lanes + lane + 1];
                    zeros += old == 0;
                    require(old == 0 || old == replacement, "shared CAS old-value domain" + shape);
                }
                require(zeros == (replacement == 0 ? lanes : 1), "shared CAS per-block winner count" + shape);
                require(output.values[total + block + 1] == replacement,
                    "discarded shared CAS result must retain its write" + shape);
            }
        }

        std::vector<uint32_t> input(total + 2, atomic_guard), preserved(input.size());
        device_buffer in(input.size() * 4);
        auto source = in.value + 4;
        for (unsigned pattern : {0u, 1u}) {
            for (unsigned i = 0; i < total; ++i)
                input[i + 1] = pattern ? 0x80000002u : 0xfffffffdu + i * 0x7654321u;
            CONFORMANCE_CUDA(cuMemcpyHtoD(in.value, input.data(), input.size() * 4));
            for (unsigned count : {0u, 1u, total / 2 + 1, total}) {
                output.reset();
                void* args[]{&output.pointer, &source, &count};
                CONFORMANCE_CUDA(cuLaunchKernel(histogram, blocks, 1, 1, lanes, 1, 1, 32, nullptr, args, nullptr));
                output.fetch(blocks * 8 + total, "shared histogram" + shape);
                for (unsigned block = 0; block < blocks; ++block) {
                    std::array<uint32_t, 4> bins{};
                    std::array<std::vector<uint32_t>, 4> tickets;
                    for (unsigned lane = 0; lane < lanes; ++lane) {
                        const unsigned i = block * lanes + lane;
                        if (i < count) ++bins[input[i + 1] & 3];
                    }
                    for (unsigned lane = 0; lane < lanes; ++lane) {
                        const unsigned i = block * lanes + lane;
                        const unsigned bucket = i < count ? input[i + 1] & 3 : 0;
                        const uint32_t old = output.values[blocks * 8 + i + 1];
                        require(old <= bins[bucket], "histogram zero/nonzero contribution range" + shape);
                        if (i < count) tickets[bucket].push_back(old);
                    }
                    for (unsigned bucket = 0; bucket < 4; ++bucket) {
                        require(output.values[block * 8 + bucket + 1] == bins[bucket]
                            && output.values[block * 8 + bucket + 5] == bins[bucket],
                            "returned/discarded shared add state" + shape);
                        expect_atomic_tickets(tickets[bucket], "per-block histogram" + shape);
                    }
                }
            }
            CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), in.value, preserved.size() * 4));
            expect_words(preserved, input, "histogram input preservation");
        }
    }
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_atomic_operations);
}
