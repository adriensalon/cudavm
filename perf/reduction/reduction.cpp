#include "perf_fixtures.hpp"
#include <array>
#include <memory>
#include <numeric>

namespace cudavm {

perf_result execute_reduction(const perf_options& options) {
    perf_require(options.case_name == "resident_small" || options.case_name == "resident_large", "unknown reduction case");
    const unsigned count = options.case_name == "resident_small" ? 1009 : 262147;
    constexpr unsigned threads = 128;
    std::vector<std::vector<uint32_t>> expected{perf_input_words(count, 0x1943b6ef)};
    while (expected.back().size() > 1) {
        const auto& previous = expected.back();
        std::vector<uint32_t> next((previous.size() + threads - 1) / threads);
        for (size_t index = 0; index < previous.size(); ++index) next[index / threads] += previous[index];
        expected.push_back(std::move(next));
    }
    const auto total = std::accumulate(expected.front().begin(), expected.front().end(), uint64_t(0));
    perf_require(expected.back().front() == static_cast<uint32_t>(total), "staged oracle agrees with independent full-width scalar sum");
    const auto device = select_perf_device(options.backend, require_perf_execution);
    perf_context context(device.value);
    perf_module module(perf_fixtures.ptx.data());
    const auto kernel = module.function("perf_reduce_blocks");
    perf_stream stream;
    std::vector<std::unique_ptr<perf_guarded_words>> storage;
    storage.push_back(std::make_unique<perf_guarded_words>(expected.front()));
    for (size_t stage = 1; stage < expected.size(); ++stage) {
        const std::vector<uint32_t> reset(expected[stage].size(), 0xa5a5a5a5);
        storage.push_back(std::make_unique<perf_guarded_words>(reset));
    }
    const auto operation = [&] {
        for (size_t stage = 1; stage < storage.size(); ++stage) {
            auto from = storage[stage - 1]->data(), to = storage[stage]->data();
            auto elements = static_cast<unsigned>(expected[stage - 1].size());
            void* arguments[]{&from, &to, &elements};
            PERF_CUDA(cuLaunchKernel(kernel, static_cast<unsigned>(expected[stage].size()), 1, 1,
                                     threads, 1, 1, 0, stream.value, arguments, nullptr));
        }
    };
    const auto verify = [&] {
        for (size_t stage = 0; stage < storage.size(); ++stage)
            storage[stage]->expect(expected[stage], "reduction stage " + std::to_string(stage) + " complete payload and tail");
    };
    PERF_CUDA(cuCtxSynchronize());
    operation();
    stream.synchronize();
    verify();
    for (size_t stage = 1; stage < storage.size(); ++stage) storage[stage]->reset();
    PERF_CUDA(cuCtxSynchronize());
    auto result = measure_perf(options, operation, [&] { stream.synchronize(); });
    verify();
    std::string stage_sizes;
    for (const auto& stage : expected) {
        if (!stage_sizes.empty()) stage_sizes += ',';
        stage_sizes += std::to_string(stage.size());
    }
    result.parameters = {{"api", "driver"}, {"algorithm", "hierarchical_128_lane_shared_memory_tree"},
        {"type", "u32_sum_mod_2_32"}, {"elements", std::to_string(count)}, {"payload_bytes", std::to_string(size_t(count) * 4)},
        {"stage_elements", stage_sizes}, {"kernels_per_iteration", std::to_string(storage.size() - 1)},
        {"input_seed", "0x1943b6ef"}, {"distribution", "xorshift32_words"}, {"guard_bytes_per_side_per_stage", "64"},
        {"block", "128,1,1"}, {"static_shared_bytes_per_block", "512"}, {"barriers_per_block", "8"},
        {"expected_sum_u32", std::to_string(expected.back().front())}, {"input", "embedded_ptx_6_0_sm_50"},
        {"completion", "non_blocking_stream_sync_per_batch"}, {"normalization", "one_complete_multistage_reduction_per_iteration"},
        {"state", "reused_context_module_stream_and_all_stage_storage_steady_overwrite"},
        {"device_selection", "first_filtered_endpoint_passing_independent_execution_probe"}};
    device.describe(result);
    for (auto iterator = storage.rbegin(); iterator != storage.rend(); ++iterator) (*iterator)->close();
    stream.close();
    module.close();
    context.close();
    return result;
}

}

int main(int argc, char** argv) {
    return cudavm::run_perf(argc, argv, "reduction", cudavm::execute_reduction);
}
