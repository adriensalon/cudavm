#include "perf_fixtures.hpp"
#include <array>
#include <limits>

namespace cudavm {

perf_result execute_kernel_launch(const perf_options& options) {
    perf_require(options.case_name == "completed" || options.case_name == "batched", "unknown kernel launch case");
    constexpr uint32_t maximum = std::numeric_limits<uint32_t>::max();
    perf_require(options.samples < maximum && options.warmup < maximum - options.samples,
                 "launch batch count fits the oracle");
    const auto batches = options.samples + options.warmup;
    perf_require(options.iterations <= (maximum - 2) / batches, "launch count fits the exact 32-bit oracle");
    const auto device = select_perf_device(options.backend, require_perf_execution);
    perf_context context(device.value);
    perf_module module(perf_fixtures.ptx.data());
    const auto kernel = module.function("perf_increment");
    perf_stream stream;
    const std::array<uint32_t, 3> initial{0x89abcdef, 0, 0x13579bdf};
    perf_device_buffer output(sizeof(initial));
    PERF_CUDA(cuMemcpyHtoD(output.value, initial.data(), sizeof(initial)));
    CUdeviceptr pointer = output.value + sizeof(uint32_t);
    void* arguments[]{&pointer};
    const bool completed = options.case_name == "completed";
    const auto operation = [&] {
        PERF_CUDA(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, stream.value, arguments, nullptr));
        if (completed) stream.synchronize();
    };
    const auto verify = [&](uint32_t count) {
        std::array<uint32_t, 3> actual{};
        PERF_CUDA(cuMemcpyDtoH(actual.data(), output.value, sizeof(actual)));
        perf_require(actual[0] == initial[0] && actual[1] == count && actual[2] == initial[2],
                     "every launch incremented exactly once and preserved both guards");
    };
    operation();
    stream.synchronize();
    verify(1);
    auto result = measure_perf(options, operation, [&] { stream.synchronize(); });
    verify(1 + static_cast<uint32_t>(batches * options.iterations));
    result.parameters = {{"api", "driver"}, {"input", "embedded_ptx"}, {"ptx_version", "6.0"},
        {"architecture", "sm_50"}, {"grid", "1,1,1"}, {"block", "1,1,1"}, {"stream", "non_blocking"},
        {"work", "one_guarded_u32_increment_per_launch"}, {"bytes", "4"},
        {"completion", completed ? "stream_sync_per_launch_and_batch" : "stream_sync_per_batch"},
        {"device_selection", "first_filtered_endpoint_passing_independent_execution_probe"},
        {"execution_prerequisite", "ptx_6_0_sm_50_constant_store"},
        {"state", "reused_context_module_stream_allocation"}};
    device.describe(result);
    output.close();
    stream.close();
    module.close();
    context.close();
    return result;
}

}

int main(int argc, char** argv) {
    return cudavm::run_perf(argc, argv, "kernel_launch", cudavm::execute_kernel_launch);
}
