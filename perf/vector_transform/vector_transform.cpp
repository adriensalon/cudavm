#include "perf_fixtures.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <memory>

namespace cudavm {

perf_result execute_vector_transform(const perf_options& options) {
    constexpr std::array<std::string_view, 5> cases{
        "resident_small", "resident_large", "pageable_small", "pageable_large", "pinned_small"};
    perf_require(std::find(cases.begin(), cases.end(), options.case_name) != cases.end(), "unknown vector transform case");
    const bool pinned = options.case_name.starts_with("pinned");
    const bool pageable = options.case_name.starts_with("pageable");
    const bool pipeline = pinned || pageable;
    unsigned count = options.case_name.ends_with("small") ? 1009 : 262147;
    constexpr unsigned threads = 128;
    constexpr size_t guard = 64;
    const size_t bytes = size_t(count) * sizeof(uint32_t), extent = bytes + guard * 2;
    const auto input = perf_input_bytes(extent, 0xfa197b3d);
    std::vector<unsigned char> reset(extent, 0xa5), expected = reset;
    for (uint32_t index = 0; index < count; ++index) {
        uint32_t value{};
        std::memcpy(&value, input.data() + guard + size_t(index) * 4, 4);
        value = value * 3u + (index ^ 0x5a17c9e3u);
        std::memcpy(expected.data() + guard + size_t(index) * 4, &value, 4);
    }
    const auto device = select_perf_device(options.backend, require_perf_execution);
    perf_context context(device.value);
    perf_module module(perf_fixtures.ptx.data());
    const auto kernel = module.function("perf_vector_transform");
    perf_stream stream;
    perf_device_buffer source(extent), destination(extent);
    std::unique_ptr<perf_host_buffer> host_input, host_output;
    std::vector<unsigned char> pageable_output;
    const unsigned char* upload = input.data();
    unsigned char* download = nullptr;
    if (pinned) {
        host_input = std::make_unique<perf_host_buffer>(extent);
        host_output = std::make_unique<perf_host_buffer>(extent);
        std::memcpy(host_input->data(), input.data(), extent);
        upload = host_input->data();
        download = host_output->data();
    } else if (pageable) {
        pageable_output.resize(extent);
        download = pageable_output.data();
    }
    CUdeviceptr from = source.value + guard, to = destination.value + guard;
    void* arguments[]{&from, &to, &count};
    const auto prepare = [&] {
        if (pipeline) std::memcpy(download, reset.data(), extent);
        PERF_CUDA(cuMemcpyHtoD(source.value, input.data(), extent));
        PERF_CUDA(cuMemcpyHtoD(destination.value, reset.data(), extent));
        PERF_CUDA(cuCtxSynchronize());
    };
    const auto operation = [&] {
        if (pinned) PERF_CUDA(cuMemcpyHtoDAsync(from, upload + guard, bytes, stream.value));
        else if (pageable) {
            PERF_CUDA(cuMemcpyHtoD(from, upload + guard, bytes));
            PERF_CUDA(cuCtxSynchronize());
        }
        PERF_CUDA(cuLaunchKernel(kernel, (count + threads - 1) / threads, 1, 1, threads, 1, 1, 0,
                                 stream.value, arguments, nullptr));
        if (pinned) PERF_CUDA(cuMemcpyDtoHAsync(download + guard, to, bytes, stream.value));
        else if (pageable) {
            stream.synchronize();
            PERF_CUDA(cuMemcpyDtoH(download + guard, to, bytes));
        }
    };
    const auto verify = [&] {
        stream.synchronize();
        std::vector<unsigned char> actual(extent);
        PERF_CUDA(cuMemcpyDtoH(actual.data(), destination.value, extent));
        perf_expect_bytes(actual, expected, "complete vector output, partial-block tail and both device guards");
        PERF_CUDA(cuMemcpyDtoH(actual.data(), source.value, extent));
        perf_expect_bytes(actual, input, "complete device source and guards preserved");
        if (pinned) {
            perf_expect_bytes({host_input->data(), extent}, input, "complete pinned host input preserved");
        }
        if (pipeline) perf_expect_bytes({download, extent}, expected, "complete host output and guards");
        perf_expect_bytes(input, perf_input_bytes(extent, 0xfa197b3d), "independent host input preservation");
    };
    prepare();
    operation();
    verify();
    prepare();
    std::function<void()> synchronize;
    if (!pageable) synchronize = [&] { stream.synchronize(); };
    auto result = measure_perf(options, operation, synchronize);
    verify();
    result.parameters = {{"api", "driver"},
        {"route", pinned ? "pinned_async_upload_kernel_download" : pageable ? "pageable_upload_kernel_download" : "resident_kernel"},
        {"elements", std::to_string(count)}, {"element_bits", "32"}, {"bytes", std::to_string(bytes)},
        {"guard_bytes_per_side", std::to_string(guard)}, {"seed", "0xfa197b3d"},
        {"distribution", "xorshift32_high_byte"}, {"formula", "u32(input*3+(index^0x5a17c9e3))"},
        {"grid", std::to_string((count + threads - 1) / threads) + ",1,1"}, {"block", "128,1,1"},
        {"host_memory", pinned ? "pinned" : pageable ? "pageable" : "not_in_interval"}, {"stream", "non_blocking"},
        {"pinned_allocation_bytes", pinned ? std::to_string(extent * 2) : "0"},
        {"completion", pageable ? "context_sync_after_upload_stream_sync_before_synchronous_download_per_iteration" : "stream_sync_per_batch"},
        {"input", "embedded_ptx_6_0_sm_50"},
        {"state", "reused_context_module_stream_and_allocations_steady_overwrite"},
        {"device_selection", "first_filtered_endpoint_passing_independent_execution_probe"}};
    device.describe(result);
    if (pinned) {
        host_output->close();
        host_input->close();
    }
    destination.close();
    source.close();
    stream.close();
    module.close();
    context.close();
    return result;
}

}

int main(int argc, char** argv) {
    return cudavm::run_perf(argc, argv, "vector_transform", cudavm::execute_vector_transform);
}
