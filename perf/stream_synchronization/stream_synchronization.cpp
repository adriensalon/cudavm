#include "perf_framework.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

namespace cudavm {

perf_result execute_stream_synchronization(const perf_options& options) {
    constexpr std::array<std::string_view, 5> cases{
        "idle_stream", "idle_context", "submitted_stream", "submitted_context", "event_handoff"};
    perf_require(std::find(cases.begin(), cases.end(), options.case_name) != cases.end(),
                 "unknown stream synchronization case");
    constexpr auto maximum = std::numeric_limits<uint32_t>::max();
    perf_require(options.samples < maximum && options.warmup < maximum - options.samples,
                 "synchronization batch count fits the oracle");
    const auto batches = options.warmup + options.samples;
    perf_require(options.iterations <= (maximum - 1) / batches, "synchronization generation fits the oracle");
    const bool idle = options.case_name.starts_with("idle");
    const bool context_wait = options.case_name.ends_with("context");
    const bool handoff = options.case_name == "event_handoff";
    constexpr size_t bytes = 4096, guard = 64, extent = bytes + 2 * guard;
    const std::vector<unsigned char> reset(extent, 0xa5);
    const auto device = select_perf_device(options.backend);
    perf_context context(device.value);
    perf_stream producer, consumer;
    perf_event ready(CU_EVENT_DISABLE_TIMING);
    perf_device_buffer source(extent), destination(extent);
    uint32_t generation = 0;
    const auto prepare = [&] {
        PERF_CUDA(cuMemcpyHtoD(source.value, reset.data(), extent));
        PERF_CUDA(cuMemcpyHtoD(destination.value, reset.data(), extent));
        PERF_CUDA(cuCtxSynchronize());
    };
    const auto submit = [&] {
        ++generation;
        PERF_CUDA(cuMemsetD32Async(source.value + guard, generation, bytes / sizeof(uint32_t), producer.value));
        if (handoff) {
            ready.record(producer.value);
            PERF_CUDA(cuStreamWaitEvent(consumer.value, ready.value, 0));
            PERF_CUDA(cuMemcpyDtoDAsync(destination.value + guard, source.value + guard, bytes, consumer.value));
        }
    };
    const auto operation = [&] {
        if (!idle) submit();
        if (context_wait) PERF_CUDA(cuCtxSynchronize());
        else if (handoff) consumer.synchronize();
        else producer.synchronize();
    };
    const auto verify = [&] {
        std::vector<unsigned char> expected = reset, actual(extent);
        for (size_t offset = guard; offset < guard + bytes; offset += sizeof(uint32_t))
            std::memcpy(expected.data() + offset, &generation, sizeof(generation));
        PERF_CUDA(cuMemcpyDtoH(actual.data(), source.value, extent));
        perf_expect_bytes(actual, expected, "producer fill generation and both guards");
        PERF_CUDA(cuMemcpyDtoH(actual.data(), destination.value, extent));
        perf_expect_bytes(actual, handoff ? expected : reset, "consumer dependency output and both guards");
        PERF_CUDA(cuStreamQuery(producer.value));
        PERF_CUDA(cuStreamQuery(consumer.value));
        if (handoff) PERF_CUDA(cuEventQuery(ready.value));
    };
    prepare();
    if (idle) { submit(); producer.synchronize(); }
    operation();
    verify();
    auto result = measure_perf(options, operation);
    const auto expected_generation = idle ? 1u : 1u + static_cast<uint32_t>(batches * options.iterations);
    perf_require(generation == expected_generation, "every synchronization iteration submitted its declared work");
    verify();
    result.parameters = {{"api", "driver"}, {"bytes", std::to_string(bytes)},
        {"guard_bytes_per_side", std::to_string(guard)}, {"stream", "non_blocking"},
        {"boundary", idle ? (context_wait ? "cuCtxSynchronize" : "cuStreamSynchronize")
            : handoff ? "fill_record_wait_copy_consumer_sync" : context_wait ? "fill_context_sync" : "fill_stream_sync"},
        {"work", idle ? "already_completed_fill_no_submission_in_interval" : "generation_tagged_u32_fill"},
        {"completion", context_wait ? "context_sync_per_iteration" : handoff ? "consumer_stream_sync_per_iteration" : "producer_stream_sync_per_iteration"},
        {"event_flags", "disable_timing"}, {"state", "reused_context_streams_event_and_allocations"},
        {"measurement", idle ? "idle_wait_api_cost" : "submission_and_completed_wait_not_isolated_busy_wait"}};
    device.describe(result);
    destination.close();
    source.close();
    ready.close();
    consumer.close();
    producer.close();
    context.close();
    return result;
}

}

int main(int argc, char** argv) {
    return cudavm::run_perf(argc, argv, "stream_synchronization", cudavm::execute_stream_synchronization);
}
