#include "perf_framework.hpp"
#include <algorithm>
#include <array>

namespace cudavm {

perf_result execute_memory_copy(const perf_options& options) {
    constexpr std::array<std::string_view, 6> cases{"h2d_4k", "d2h_4k", "d2d_4k", "h2d_4m", "d2h_4m", "d2d_4m"};
    perf_require(std::find(cases.begin(), cases.end(), options.case_name) != cases.end(), "unknown memory copy case");
    const size_t bytes = options.case_name.ends_with("4k") ? 4096 : 4194304;
    const auto route = options.case_name.substr(0, 3);
    constexpr size_t guard = 64;
    constexpr uint32_t seed = 0x1943b6ef;
    const auto input = perf_input_bytes(bytes + guard * 2, seed);
    std::vector<unsigned char> reset(bytes + guard * 2, 0xa5), expected = reset, host_output = reset;
    std::copy_n(input.begin() + guard, bytes, expected.begin() + guard);
    const auto device = select_perf_device(options.backend);
    perf_context context(device.value);
    perf_device_buffer source(input.size()), destination(input.size());
    PERF_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size()));
    const auto prepare = [&] {
        host_output = reset;
        PERF_CUDA(cuMemcpyHtoD(destination.value, reset.data(), reset.size()));
        PERF_CUDA(cuCtxSynchronize());
    };
    const auto operation = [&] {
        if (route == "h2d") PERF_CUDA(cuMemcpyHtoD(destination.value + guard, input.data() + guard, bytes));
        else if (route == "d2h") PERF_CUDA(cuMemcpyDtoH(host_output.data() + guard, source.value + guard, bytes));
        else PERF_CUDA(cuMemcpyDtoD(destination.value + guard, source.value + guard, bytes));
    };
    const auto verify = [&] {
        std::vector<unsigned char> actual(input.size());
        PERF_CUDA(cuCtxSynchronize());
        if (route == "d2h") perf_expect_bytes(host_output, expected, "D2H payload and host guards");
        PERF_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size()));
        perf_expect_bytes(actual, route == "d2h" ? reset : expected, "destination payload and device guards");
        PERF_CUDA(cuMemcpyDtoH(actual.data(), source.value, actual.size()));
        perf_expect_bytes(actual, input, "source allocation remains unchanged");
    };
    prepare();
    operation();
    verify();
    prepare();
    auto result = measure_perf(options, operation, [] { PERF_CUDA(cuCtxSynchronize()); });
    verify();
    result.parameters = {{"api", "driver"}, {"route", std::string(route)}, {"bytes", std::to_string(bytes)},
        {"guard_bytes_per_side", std::to_string(guard)}, {"seed", "0x1943b6ef"}, {"distribution", "xorshift32_high_byte"},
        {"host_memory", route == "d2d" ? "not_in_interval" : "pageable"},
        {"completion", "synchronous_copy_api_and_context_sync_per_batch"},
        {"state", "reused_context_and_allocations_steady_overwrite"}};
    device.describe(result);
    destination.close();
    source.close();
    context.close();
    return result;
}

}

int main(int argc, char** argv) {
    return cudavm::run_perf(argc, argv, "memory_copy", cudavm::execute_memory_copy);
}
