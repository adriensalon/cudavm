#include "perf_fixtures.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

namespace cudavm {

perf_result execute_graph_replay(const perf_options& options) {
    constexpr std::array<std::string_view, 4> cases{"graph_4", "graph_16", "submissions_4", "submissions_16"};
    perf_require(std::find(cases.begin(), cases.end(), options.case_name) != cases.end(), "unknown graph replay case");
    const bool graph_route = options.case_name.starts_with("graph");
    const unsigned nodes = options.case_name.ends_with("_4") ? 4 : 16;
    constexpr auto maximum = std::numeric_limits<uint32_t>::max();
    perf_require(options.samples < maximum && options.warmup < maximum - options.samples,
                 "graph replay batch count fits the oracle");
    const auto batches = options.warmup + options.samples;
    perf_require(options.iterations <= (maximum / nodes) / batches, "graph replay step count fits the oracle");
    unsigned count = 1009;
    constexpr unsigned threads = 128;
    constexpr size_t guard = 64;
    const size_t bytes = count * sizeof(uint32_t), extent = bytes + 2 * guard;
    const auto initial = perf_input_bytes(extent, 0xfa197b3d);
    const auto device = select_perf_device(options.backend, require_perf_execution);
    perf_context context(device.value);
    perf_module module(perf_fixtures.ptx.data());
    const auto kernel = module.function("perf_graph_step");
    perf_stream stream;
    perf_device_buffer output(extent);
    CUdeviceptr pointer = output.value + guard;
    void* arguments[]{&pointer, &count};
    perf_graph graph;
    CUgraphNode previous{};
    for (unsigned index = 0; index < nodes; ++index) {
        CUDA_KERNEL_NODE_PARAMS params{};
        params.func = kernel;
        params.gridDimX = (count + threads - 1) / threads;
        params.gridDimY = params.gridDimZ = 1;
        params.blockDimX = threads;
        params.blockDimY = params.blockDimZ = 1;
        params.kernelParams = arguments;
        CUgraphNode current{};
        PERF_CUDA(cuGraphAddKernelNode(&current, graph.value, index ? &previous : nullptr, index ? 1 : 0, &params));
        previous = current;
    }
    perf_graph_exec executable(graph.value);
    const auto prepare = [&] {
        PERF_CUDA(cuMemcpyHtoD(output.value, initial.data(), extent));
        PERF_CUDA(cuCtxSynchronize());
    };
    const auto operation = [&] {
        if (graph_route) executable.launch(stream.value);
        else for (unsigned index = 0; index < nodes; ++index)
            PERF_CUDA(cuLaunchKernel(kernel, (count + threads - 1) / threads, 1, 1, threads, 1, 1, 0,
                                     stream.value, arguments, nullptr));
    };
    const auto verify = [&](uint32_t steps) {
        std::vector<unsigned char> expected = initial, actual(extent);
        for (uint32_t index = 0; index < count; ++index) {
            uint32_t value{};
            const size_t offset = guard + size_t(index) * sizeof(value);
            std::memcpy(&value, initial.data() + offset, sizeof(value));
            value += steps * (index * 7u + 3u);
            std::memcpy(expected.data() + offset, &value, sizeof(value));
        }
        PERF_CUDA(cuMemcpyDtoH(actual.data(), output.value, extent));
        perf_expect_bytes(actual, expected, "every replay step at every lane, partial block tail and both guards");
    };
    prepare();
    operation();
    stream.synchronize();
    verify(nodes);
    prepare();
    auto result = measure_perf(options, operation, [&] { stream.synchronize(); });
    verify(static_cast<uint32_t>(batches * options.iterations * nodes));
    result.parameters = {{"api", "driver"}, {"route", graph_route ? "graph" : "ordinary_submissions"},
        {"boundary", graph_route ? "cuGraphLaunch_and_batch_completion" : "cuLaunchKernel_chain_and_batch_completion"},
        {"nodes_per_iteration", std::to_string(nodes)}, {"topology", "linear_dependent_kernel_chain"},
        {"elements", std::to_string(count)}, {"bytes", std::to_string(bytes)},
        {"guard_bytes_per_side", std::to_string(guard)}, {"seed", "0xfa197b3d"},
        {"formula_per_node", "u32(state[index]+index*7+3)"}, {"input", "embedded_ptx_6_0_sm_50"},
        {"grid", std::to_string((count + threads - 1) / threads) + ",1,1"}, {"block", "128,1,1"},
        {"stream", "non_blocking"}, {"completion", "stream_sync_per_batch"},
        {"normalization", "one_complete_chain_per_iteration"},
        {"state", graph_route ? "reused_context_module_stream_allocation_and_executable_after_validation_launch"
            : "reused_context_module_stream_allocation_after_validation_launch"},
        {"device_selection", "first_filtered_endpoint_passing_independent_execution_probe"}};
    device.describe(result);
    executable.close();
    graph.close();
    output.close();
    stream.close();
    module.close();
    context.close();
    return result;
}

}

int main(int argc, char** argv) {
    return cudavm::run_perf(argc, argv, "graph_replay", cudavm::execute_graph_replay);
}
