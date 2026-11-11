#include "perf_framework.hpp"
#include <algorithm>
#include <array>
#include <cmath>

namespace cudavm {

void verify_perf_event(unsigned flags, CUstream stream) {
    perf_event event(flags);
    PERF_CUDA(cuEventQuery(event.value));
    event.record(stream);
    event.synchronize();
    PERF_CUDA(cuEventQuery(event.value));
    event.close();
}

perf_result execute_event_timing(const perf_options& options) {
    constexpr std::array<std::string_view, 5> cases{
        "create_timed", "create_disabled", "record_timed", "record_disabled", "elapsed_query"};
    perf_require(std::find(cases.begin(), cases.end(), options.case_name) != cases.end(), "unknown event timing case");
    const auto device = select_perf_device(options.backend);
    perf_context context(device.value);
    perf_stream stream;
    const unsigned flags = options.case_name.ends_with("disabled") ? CU_EVENT_DISABLE_TIMING : 0;
    perf_result result;
    std::string boundary, completion;
    if (options.case_name.starts_with("create")) {
        verify_perf_event(flags, stream.value);
        const auto operation = [&] {
            perf_event event(flags);
            event.close();
        };
        result = measure_perf(options, operation);
        verify_perf_event(flags, stream.value);
        boundary = "cuEventCreate_cuEventDestroy";
        completion = "unrecorded_event_lifecycle";
    } else if (options.case_name.starts_with("record")) {
        perf_event event(flags);
        const auto operation = [&] {
            event.record(stream.value);
            event.synchronize();
            PERF_CUDA(cuEventQuery(event.value));
        };
        operation();
        result = measure_perf(options, operation);
        PERF_CUDA(cuEventQuery(event.value));
        event.close();
        boundary = "cuEventRecord_cuEventSynchronize_cuEventQuery";
        completion = "event_sync_per_iteration";
    } else {
        perf_event start, end;
        start.record(stream.value);
        end.record(stream.value);
        end.synchronize();
        PERF_CUDA(cuEventQuery(start.value));
        PERF_CUDA(cuEventQuery(end.value));
        struct { uint32_t before = 0x1943b6ef; float value = -1; uint32_t after = 0x93bd175a; } elapsed;
        const auto operation = [&] { PERF_CUDA(cuEventElapsedTime(&elapsed.value, start.value, end.value)); };
        const auto verify = [&] {
            perf_require(elapsed.before == 0x1943b6ef && elapsed.after == 0x93bd175a, "event elapsed output guards");
            perf_require(std::isfinite(elapsed.value) && elapsed.value >= 0, "finite nonnegative elapsed milliseconds");
        };
        operation();
        verify();
        const float expected = elapsed.value;
        result = measure_perf(options, operation);
        verify();
        perf_require(elapsed.value == expected, "querying an unchanged completed timestamp pair is stable");
        end.close();
        start.close();
        boundary = "cuEventElapsedTime";
        completion = "timestamp_pair_completed_before_timing";
    }
    result.parameters = {{"api", "driver"}, {"boundary", boundary}, {"event_flags", std::to_string(flags)},
        {"stream", "non_blocking"}, {"completion", completion}, {"work", "event_api_without_payload"},
        {"state", "reused_context_and_stream"}, {"measurement", "host_api_cost_not_device_execution_time"}};
    device.describe(result);
    stream.close();
    context.close();
    return result;
}

}

int main(int argc, char** argv) {
    return cudavm::run_perf(argc, argv, "event_timing", cudavm::execute_event_timing);
}
