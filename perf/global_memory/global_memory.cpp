#include "perf_fixtures.hpp"
#include <algorithm>
#include <array>
#include <numeric>

namespace cudavm {

perf_result execute_global_memory(const perf_options& options) {
    constexpr std::array<std::string_view, 6> cases{
        "contiguous_small", "contiguous_large", "strided_small", "strided_large", "indirect_small", "indirect_large"};
    perf_require(std::find(cases.begin(), cases.end(), options.case_name) != cases.end(), "unknown global memory case");
    unsigned count = options.case_name.ends_with("small") ? 1024 : 262144;
    unsigned mode = options.case_name.starts_with("contiguous") ? 0 : options.case_name.starts_with("strided") ? 1 : 2;
    constexpr unsigned threads = 128;
    const auto input = perf_input_words(count, 0x1943b6ef);
    auto random = perf_input_words(count, 0x93bd175a);
    std::vector<uint32_t> indices(count), expected(count), reset(count, 0xa5a5a5a5);
    std::iota(indices.begin(), indices.end(), 0u);
    for (size_t at = count - 1; at > 0; --at) std::swap(indices[at], indices[random[at] % (at + 1)]);
    for (uint32_t index = 0; index < count; ++index) {
        const auto address = mode == 0 ? index : mode == 1 ? (index * 17u) & (count - 1) : indices[index];
        expected[index] = input[address] ^ (index * 0x9e3779b9u + 0x85ebca6bu);
    }
    const auto device = select_perf_device(options.backend, require_perf_execution);
    perf_context context(device.value);
    perf_module module(perf_fixtures.ptx.data());
    const auto kernel = module.function("perf_global_read");
    perf_stream stream;
    perf_guarded_words source(input), lookup(indices), destination(reset);
    CUdeviceptr from = source.data(), table = lookup.data(), to = destination.data();
    void* arguments[]{&from, &table, &to, &count, &mode};
    const auto operation = [&] {
        PERF_CUDA(cuLaunchKernel(kernel, count / threads, 1, 1, threads, 1, 1, 0, stream.value, arguments, nullptr));
    };
    const auto verify = [&] {
        destination.expect(expected, "complete global read output");
        source.expect(input, "global source preservation");
        lookup.expect(indices, "indirect table preservation");
    };
    PERF_CUDA(cuCtxSynchronize());
    operation();
    stream.synchronize();
    verify();
    destination.reset();
    PERF_CUDA(cuCtxSynchronize());
    auto result = measure_perf(options, operation, [&] { stream.synchronize(); });
    verify();
    result.parameters = {{"api", "driver"}, {"pattern", mode == 0 ? "contiguous" : mode == 1 ? "stride_17_mod_count" : "fisher_yates_permutation"},
        {"elements", std::to_string(count)}, {"element_bits", "32"}, {"payload_bytes", std::to_string(size_t(count) * 4)},
        {"logical_read_write_bytes", std::to_string(size_t(count) * (mode == 2 ? 12 : 8))},
        {"guard_bytes_per_side", "64"}, {"input_seed", "0x1943b6ef"}, {"permutation_seed", "0x93bd175a"},
        {"distribution", "xorshift32_words"}, {"formula", "source[address]^(index*0x9e3779b9+0x85ebca6b)"},
        {"grid", std::to_string(count / threads) + ",1,1"}, {"block", "128,1,1"},
        {"input", "embedded_ptx_6_0_sm_50"}, {"completion", "non_blocking_stream_sync_per_batch"},
        {"state", "reused_context_module_stream_and_resident_storage_steady_overwrite"},
        {"cache_policy", "warm_reuse_no_cache_flush"},
        {"device_selection", "first_filtered_endpoint_passing_independent_execution_probe"}};
    device.describe(result);
    destination.close();
    lookup.close();
    source.close();
    stream.close();
    module.close();
    context.close();
    return result;
}

}

int main(int argc, char** argv) {
    return cudavm::run_perf(argc, argv, "global_memory", cudavm::execute_global_memory);
}
