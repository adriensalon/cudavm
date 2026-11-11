#include "perf_framework.hpp"

namespace cudavm {

void verify_perf_allocation(size_t bytes) {
    perf_device_buffer first(bytes), second(bytes);
    perf_require(first.value != 0 && second.value != 0, "nonempty allocations publish pointers");
    perf_require(first.value + bytes <= second.value || second.value + bytes <= first.value,
                 "simultaneously live allocation ranges are disjoint");
    const auto input = perf_input_bytes(bytes, 0x1943b6ef);
    const auto neighbor = perf_input_bytes(bytes, 0x93bd175a);
    PERF_CUDA(cuMemcpyHtoD(first.value, input.data(), bytes));
    PERF_CUDA(cuMemcpyHtoD(second.value, neighbor.data(), bytes));
    PERF_CUDA(cuCtxSynchronize());
    CUdeviceptr base{}; size_t extent{};
    PERF_CUDA(cuMemGetAddressRange(&base, &extent, first.value));
    perf_require(base == first.value && extent >= bytes, "allocation owns its complete requested extent");
    std::vector<unsigned char> actual(bytes);
    PERF_CUDA(cuMemcpyDtoH(actual.data(), first.value, bytes));
    perf_expect_bytes(actual, input, "fresh allocation round trip");
    first.close();
    PERF_CUDA(cuMemcpyDtoH(actual.data(), second.value, bytes));
    perf_expect_bytes(actual, neighbor, "retiring one allocation preserves the independent neighbor");
    second.close();
}

perf_result execute_device_allocation(const perf_options& options) {
    perf_require(options.case_name == "alloc_free_4k" || options.case_name == "alloc_free_4m", "unknown device allocation case");
    const size_t bytes = options.case_name.ends_with("4k") ? 4096 : 4194304;
    const auto device = select_perf_device(options.backend);
    perf_context context(device.value);
    verify_perf_allocation(bytes);
    const auto operation = [&] {
        perf_device_buffer allocation(bytes);
        allocation.close();
    };
    auto result = measure_perf(options, operation);
    verify_perf_allocation(bytes);
    result.parameters = {{"api", "driver"}, {"boundary", "cuMemAlloc_cuMemFree"},
        {"bytes", std::to_string(bytes)}, {"completion", "synchronous_lifecycle_calls"},
        {"state", "new_allocation_per_iteration_warm_context_and_allocator"},
        {"verification", "complete_extent_and_live_neighbor_isolation"}};
    device.describe(result);
    context.close();
    return result;
}

}

int main(int argc, char** argv) {
    return cudavm::run_perf(argc, argv, "device_allocation", cudavm::execute_device_allocation);
}
