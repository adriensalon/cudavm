#include "perf_fixtures.hpp"
#include <algorithm>
#include <array>

namespace cudavm {

uint32_t next_perf_arithmetic_word(uint32_t value) {
    const auto product = uint64_t(value) * 1664525u + 1013904223u;
    return static_cast<uint32_t>(product) ^ (value >> 13);
}

perf_result execute_arithmetic_throughput(const perf_options& options) {
    constexpr std::array<std::string_view, 4> cases{"chain_32", "chain_256", "independent_32", "independent_256"};
    perf_require(std::find(cases.begin(), cases.end(), options.case_name) != cases.end(), "unknown arithmetic throughput case");
    unsigned count = 4093, rounds = options.case_name.ends_with("_32") ? 32 : 256;
    unsigned independent = options.case_name.starts_with("independent") ? 1 : 0;
    constexpr unsigned threads = 128;
    const auto input = perf_input_words(count, 0x1943b6ef);
    std::vector<uint32_t> expected(size_t(count) * 4), reset(expected.size(), 0xa5a5a5a5);
    for (size_t index = 0; index < count; ++index) {
        uint32_t a = input[index], b = a ^ 0x9e3779b9u, c = a ^ 0x85ebca6bu, d = a ^ 0xc2b2ae35u;
        for (unsigned round = 0; round < rounds; ++round) {
            if (independent) {
                a = next_perf_arithmetic_word(a); b = next_perf_arithmetic_word(b);
                c = next_perf_arithmetic_word(c); d = next_perf_arithmetic_word(d);
            } else {
                b = next_perf_arithmetic_word(a); c = next_perf_arithmetic_word(b);
                d = next_perf_arithmetic_word(c); a = next_perf_arithmetic_word(d);
            }
        }
        expected[index * 4] = a; expected[index * 4 + 1] = b;
        expected[index * 4 + 2] = c; expected[index * 4 + 3] = d;
    }
    const auto device = select_perf_device(options.backend, require_perf_execution);
    perf_context context(device.value);
    perf_module module(perf_fixtures.ptx.data());
    const auto kernel = module.function("perf_integer_mix");
    perf_stream stream;
    perf_guarded_words source(input), destination(reset);
    CUdeviceptr from = source.data(), to = destination.data();
    void* arguments[]{&from, &to, &count, &rounds, &independent};
    const auto operation = [&] {
        PERF_CUDA(cuLaunchKernel(kernel, (count + threads - 1) / threads, 1, 1, threads, 1, 1, 0,
                                 stream.value, arguments, nullptr));
    };
    const auto verify = [&] {
        destination.expect(expected, "every final arithmetic accumulator and partial-block tail");
        source.expect(input, "arithmetic source preservation");
    };
    PERF_CUDA(cuCtxSynchronize());
    operation();
    stream.synchronize();
    verify();
    destination.reset();
    PERF_CUDA(cuCtxSynchronize());
    auto result = measure_perf(options, operation, [&] { stream.synchronize(); });
    verify();
    result.parameters = {{"api", "driver"}, {"type", "u32_modular"},
        {"dependency", independent ? "four_independent_chains" : "one_serial_chain_four_updates_per_round"},
        {"elements", std::to_string(count)}, {"rounds", std::to_string(rounds)}, {"accumulators_per_thread", "4"},
        {"requested_updates_per_thread", std::to_string(size_t(rounds) * 4)},
        {"requested_updates_per_iteration", std::to_string(size_t(count) * rounds * 4)},
        {"update", "u32(state*1664525+1013904223)^(state>>13)"},
        {"input_seed", "0x1943b6ef"}, {"distribution", "xorshift32_words"}, {"guard_bytes_per_side", "64"},
        {"grid", std::to_string((count + threads - 1) / threads) + ",1,1"}, {"block", "128,1,1"},
        {"input", "embedded_ptx_6_0_sm_50"}, {"completion", "non_blocking_stream_sync_per_batch"},
        {"state", "reused_context_module_stream_and_storage_steady_overwrite"},
        {"measurement", "completed_integer_mix_kernel_not_hardware_instruction_throughput"},
        {"device_selection", "first_filtered_endpoint_passing_independent_execution_probe"}};
    device.describe(result);
    destination.close();
    source.close();
    stream.close();
    module.close();
    context.close();
    return result;
}

}

int main(int argc, char** argv) {
    return cudavm::run_perf(argc, argv, "arithmetic_throughput", cudavm::execute_arithmetic_throughput);
}
