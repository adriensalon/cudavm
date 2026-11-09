#include "test_fixtures.hpp"
#include <algorithm>
#include <iostream>
#include <limits>

namespace cudavm {
std::vector<uint32_t> progress_oracle(std::span<const uint32_t> input, unsigned blocks, unsigned lanes, unsigned seed) {
    std::vector<uint32_t> result(input.size()), working(lanes);
    for (unsigned block = 0; block < blocks; ++block) {
        auto row = std::span(result).subspan(size_t(block) * lanes, lanes);
        for (unsigned lane = 0; lane < lanes; ++lane) row[lane] = input[size_t(block) * lanes + lane] ^ seed;
        for (unsigned round = 0; round <= ((block + seed) & 3); ++round) {
            for (unsigned lane = 0; lane < lanes; ++lane) {
                uint64_t value = row[lane];
                for (unsigned step = 0; step < (lane + round) % 4; ++step)
                    value = uint32_t((value ^ uint64_t(round * 13 + step + seed)) * 33 + lane);
                working[lane] = uint32_t(value);
            }
            for (unsigned lane = 0; lane < lanes; ++lane)
                row[lane] = uint32_t(uint64_t(working[lane]) + (working[(lane + 1) % lanes] ^ (seed + round * 17)));
        }
    }
    return result;
}
void execute_device_progress(const loaded_program& program) {
    CUdevice device{};
    CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    int multiprocessors = 0, resident_threads = 0, grid_limit = 0;
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&multiprocessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&resident_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device));
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&grid_limit, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device));
    require(multiprocessors >= 0 && resident_threads >= 0, "reported residency attributes are nonnegative");
    constexpr uint32_t guard = 0xa1b2c3d4;
    constexpr unsigned guards = 4;
    const auto kernel = program.function("progress_rounds");
    for (unsigned lanes : {13u, 33u, 64u}) {
        const uint64_t resident = uint64_t(multiprocessors) * unsigned(resident_threads);
        const auto over = std::max(uint64_t(257), resident * 2 / lanes + 3);
        require(over <= uint64_t(grid_limit) && over <= 65536 && over * lanes <= 4 * 1024 * 1024,
            "oversubscribed profile fits its explicit fixture bounds");
        for (unsigned blocks : {3u, unsigned(over)}) {
            const auto total = blocks * lanes;
            std::cout << "progress blocks=" << blocks << " lanes=" << lanes << " reported_resident_threads=" << resident << std::endl;
            std::vector<uint32_t> input(total + guards * 2, guard), expected(input), actual(input.size());
            std::vector<uint32_t> tickets(blocks + guards * 2, guard), observed_tickets(tickets.size());
            std::array<uint32_t, guards * 2 + 1> counter{}, observed_counter{};
            device_buffer source(input.size() * 4), output(input.size() * 4), ticket_buffer(tickets.size() * 4), completed(counter.size() * 4);
            auto from = source.value + guards * 4, to = output.value + guards * 4;
            auto ticket = ticket_buffer.value + guards * 4, done = completed.value + guards * 4;
            for (unsigned seed : {0u, 197u}) {
                for (unsigned i = 0; i < total; ++i) input[guards + i] = i * 0x7654321u + seed * 0x89abcdefu;
                CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
                const auto values = progress_oracle(std::span(input).subspan(guards, total), blocks, lanes, seed);
                for (unsigned count : {0u, total / 2 + 1, total}) {
                    std::fill(expected.begin(), expected.end(), guard);
                    counter.fill(guard); counter[guards] = 0;
                    CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
                    CONFORMANCE_CUDA(cuMemcpyHtoD(ticket_buffer.value, tickets.data(), tickets.size() * 4));
                    CONFORMANCE_CUDA(cuMemcpyHtoD(completed.value, counter.data(), counter.size() * 4));
                    void* args[]{&to, &from, &ticket, &done, &count, &seed};
                    CONFORMANCE_CUDA(cuLaunchKernel(kernel, blocks, 1, 1, lanes, 1, 1, 0, nullptr, args, nullptr));
                    CONFORMANCE_CUDA(cuCtxSynchronize());
                    std::copy_n(values.begin(), count, expected.begin() + guards);
                    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
                    expect_words(actual, expected, "finite divergent/barrier progress and output guards");
                    CONFORMANCE_CUDA(cuMemcpyDtoH(observed_tickets.data(), ticket_buffer.value, observed_tickets.size() * 4));
                    for (unsigned i = 0; i < guards; ++i)
                        require(observed_tickets[i] == guard && observed_tickets[guards + blocks + i] == guard, "completion ticket guards");
                    auto first = observed_tickets.begin() + guards, last = first + blocks;
                    std::sort(first, last);
                    for (unsigned i = 0; i < blocks; ++i) require(first[i] == i, "every block completes exactly once without a prescribed order");
                    counter[guards] = blocks;
                    CONFORMANCE_CUDA(cuMemcpyDtoH(observed_counter.data(), completed.value, sizeof(counter)));
                    expect_words(observed_counter, counter, "completed block count and counter guards");
                }
                CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), source.value, actual.size() * 4));
                expect_words(actual, input, "progress source/allocation preservation");
            }
        }
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_device_progress);
}
