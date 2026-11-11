#include "perf_framework.hpp"
#include <algorithm>
#include <array>
#include <cstring>

namespace cudavm {

perf_result execute_memory_fill(const perf_options& options) {
    constexpr std::array<std::string_view, 4> cases{"d8_4k", "d32_4k", "d8_4m", "d32_4m"};
    perf_require(std::find(cases.begin(), cases.end(), options.case_name) != cases.end(), "unknown memory fill case");
    const size_t bytes = options.case_name.ends_with("4k") ? 4096 : 4194304;
    const bool wide = options.case_name.starts_with("d32");
    constexpr size_t guard = 64;
    constexpr uint32_t word = 0x1943b6ef;
    std::vector<unsigned char> reset(bytes + guard * 2, 0xa5), expected = reset;
    for (size_t index = guard; index < bytes + guard; index += wide ? 4 : 1) {
        if (wide) std::memcpy(expected.data() + index, &word, sizeof(word));
        else expected[index] = 0x3a;
    }
    const auto device = select_perf_device(options.backend);
    perf_context context(device.value);
    perf_device_buffer output(reset.size());
    const auto prepare = [&] {
        PERF_CUDA(cuMemcpyHtoD(output.value, reset.data(), reset.size()));
        PERF_CUDA(cuCtxSynchronize());
    };
    const auto operation = [&] {
        if (wide) PERF_CUDA(cuMemsetD32(output.value + guard, word, bytes / sizeof(word)));
        else PERF_CUDA(cuMemsetD8(output.value + guard, 0x3a, bytes));
    };
    const auto verify = [&] {
        std::vector<unsigned char> actual(expected.size());
        PERF_CUDA(cuCtxSynchronize());
        PERF_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size()));
        perf_expect_bytes(actual, expected, "fill complete payload and both device guards");
    };
    prepare();
    operation();
    verify();
    prepare();
    auto result = measure_perf(options, operation, [] { PERF_CUDA(cuCtxSynchronize()); });
    verify();
    result.parameters = {{"api", "driver"}, {"bytes", std::to_string(bytes)}, {"element_bits", wide ? "32" : "8"},
        {"value", wide ? "0x1943b6ef" : "0x3a"}, {"guard_bytes_per_side", std::to_string(guard)},
        {"completion", "synchronous_fill_api_and_context_sync_per_batch"},
        {"state", "reused_context_and_allocation_steady_overwrite"}};
    device.describe(result);
    output.close();
    context.close();
    return result;
}

}

int main(int argc, char** argv) {
    return cudavm::run_perf(argc, argv, "memory_fill", cudavm::execute_memory_fill);
}
