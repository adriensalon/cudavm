#include "perf_framework.hpp"

namespace cudavm {

void verify_perf_stream(unsigned flags, CUcontext expected_context) {
    perf_stream first(flags), second(flags);
    perf_require(first.value && second.value && first.value != second.value, "independent explicit stream handles");
    unsigned actual_flags{};
    CUcontext owner{};
    PERF_CUDA(cuStreamGetFlags(first.value, &actual_flags));
    PERF_CUDA(cuStreamGetCtx(first.value, &owner));
    perf_require(actual_flags == flags && owner == expected_context, "stream retains its flags and context ownership");
    PERF_CUDA(cuStreamQuery(first.value));
    PERF_CUDA(cuStreamQuery(second.value));
    first.close();
    second.synchronize();
    PERF_CUDA(cuStreamQuery(second.value));
    second.close();
}

perf_result execute_stream_lifecycle(const perf_options& options) {
    perf_require(options.case_name == "blocking" || options.case_name == "non_blocking", "unknown stream lifecycle case");
    const unsigned flags = options.case_name == "blocking" ? 0 : CU_STREAM_NON_BLOCKING;
    const auto device = select_perf_device(options.backend);
    perf_context context(device.value);
    verify_perf_stream(flags, context.value);
    const auto operation = [&] {
        perf_stream stream(flags);
        stream.close();
    };
    auto result = measure_perf(options, operation);
    verify_perf_stream(flags, context.value);
    result.parameters = {{"api", "driver"}, {"boundary", "cuStreamCreate_cuStreamDestroy"},
        {"stream_flags", std::to_string(flags)}, {"work", "idle_explicit_stream"},
        {"completion", "synchronous_lifecycle_calls"}, {"state", "reused_context_new_stream_per_iteration"}};
    device.describe(result);
    context.close();
    return result;
}

}

int main(int argc, char** argv) {
    return cudavm::run_perf(argc, argv, "stream_lifecycle", cudavm::execute_stream_lifecycle);
}
