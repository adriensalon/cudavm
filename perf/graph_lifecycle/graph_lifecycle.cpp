#include "perf_framework.hpp"
#include <algorithm>
#include <array>
#include <cstring>

namespace cudavm {

perf_result execute_graph_lifecycle(const perf_options& options) {
    constexpr std::array<std::string_view, 6> cases{
        "build_4", "build_32", "instantiate_4", "instantiate_32", "capture_4", "capture_32"};
    perf_require(std::find(cases.begin(), cases.end(), options.case_name) != cases.end(), "unknown graph lifecycle case");
    const bool capture = options.case_name.starts_with("capture");
    const bool instantiate = options.case_name.starts_with("instantiate");
    const size_t nodes = options.case_name.ends_with("_4") ? 4 : 32;
    constexpr size_t guard = 64, payload = 128, stride = payload + 2 * guard;
    const size_t extent = nodes * stride;
    const std::vector<unsigned char> reset(extent, 0xa5);
    std::vector<unsigned char> expected = reset;
    const auto fill_value = [](size_t index) { return 0x19430000u + static_cast<uint32_t>(index) * 0x010203u; };
    for (size_t index = 0; index < nodes; ++index) {
        const auto value = fill_value(index);
        for (size_t offset = index * stride + guard; offset < index * stride + guard + payload; offset += sizeof(value))
            std::memcpy(expected.data() + offset, &value, sizeof(value));
    }
    const auto device = select_perf_device(options.backend);
    perf_context context(device.value);
    perf_stream stream;
    perf_device_buffer output(extent);
    const auto prepare = [&] {
        PERF_CUDA(cuMemcpyHtoD(output.value, reset.data(), extent));
        PERF_CUDA(cuCtxSynchronize());
    };
    const auto populate = [&](CUgraph graph) {
        CUgraphNode previous{};
        for (size_t index = 0; index < nodes; ++index) {
            CUDA_MEMSET_NODE_PARAMS params{};
            params.dst = output.value + index * stride + guard;
            params.value = fill_value(index);
            params.elementSize = sizeof(uint32_t);
            params.width = payload / sizeof(uint32_t);
            params.height = 1;
            CUgraphNode current{};
            PERF_CUDA(cuGraphAddMemsetNode(&current, graph, index ? &previous : nullptr, index ? 1 : 0,
                                          &params, context.value));
            previous = current;
        }
    };
    const auto record = [&] {
        for (size_t index = 0; index < nodes; ++index)
            PERF_CUDA(cuMemsetD32Async(output.value + index * stride + guard, fill_value(index),
                                      payload / sizeof(uint32_t), stream.value));
    };
    const auto verify_bytes = [&](const std::vector<unsigned char>& wanted) {
        std::vector<unsigned char> actual(extent);
        PERF_CUDA(cuMemcpyDtoH(actual.data(), output.value, extent));
        perf_expect_bytes(actual, wanted, "every graph fill slot and its two guards");
    };
    const auto verify_graph = [&](CUgraph graph) {
        size_t node_count{}, edge_count{};
        PERF_CUDA(cuGraphGetNodes(graph, nullptr, &node_count));
        PERF_CUDA(cuGraphGetEdges(graph, nullptr, nullptr, nullptr, &edge_count));
        perf_require(node_count == nodes && edge_count == nodes - 1, "graph contains the complete linear fill chain");
        verify_bytes(reset);
        perf_graph_exec executable(graph);
        executable.launch(stream.value);
        stream.synchronize();
        verify_bytes(expected);
        executable.close();
    };
    const auto verify_representative = [&] {
        prepare();
        if (capture) {
            perf_graph graph(stream.value, record);
            verify_graph(graph.value);
            graph.close();
        } else {
            perf_graph graph;
            populate(graph.value);
            verify_graph(graph.value);
            graph.close();
        }
        prepare();
    };
    verify_representative();
    perf_result result;
    if (instantiate) {
        perf_graph graph;
        populate(graph.value);
        result = measure_perf(options, [&] {
            perf_graph_exec executable(graph.value);
            executable.close();
        });
        verify_graph(graph.value);
        graph.close();
    } else if (capture) {
        result = measure_perf(options, [&] {
            perf_graph graph(stream.value, record);
            graph.close();
        });
        verify_bytes(reset);
    } else {
        result = measure_perf(options, [&] {
            perf_graph graph;
            populate(graph.value);
            graph.close();
        });
        verify_bytes(reset);
    }
    verify_representative();
    result.parameters = {{"api", "driver"}, {"nodes", std::to_string(nodes)}, {"edges", std::to_string(nodes - 1)},
        {"topology", "linear_chain_disjoint_u32_fills"}, {"payload_bytes_per_node", std::to_string(payload)},
        {"guard_bytes_per_side_per_node", std::to_string(guard)},
        {"boundary", instantiate ? "cuGraphInstantiate_cuGraphExecDestroy" : capture
            ? "cuStreamBeginCapture_fill_submissions_cuStreamEndCapture_cuGraphDestroy"
            : "cuGraphCreate_add_memset_nodes_cuGraphDestroy"},
        {"capture_mode", capture ? "thread_local" : "not_applicable"},
        {"state", instantiate ? "new_executable_per_iteration_reused_source_graph" : "new_graph_per_iteration_reused_context_stream_allocation"},
        {"completion", "synchronous_host_graph_lifecycle_no_replay_in_interval"},
        {"verification", "node_edge_counts_complete_fill_slots_and_guards_no_execution_during_lifecycle"}};
    device.describe(result);
    output.close();
    stream.close();
    context.close();
    return result;
}

}

int main(int argc, char** argv) {
    return cudavm::run_perf(argc, argv, "graph_lifecycle", cudavm::execute_graph_lifecycle);
}
