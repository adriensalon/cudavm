#include "perf_framework.hpp"
#include <algorithm>
#include <array>
#include <cstring>

namespace cudavm {

void verify_perf_host_storage(unsigned char* host, size_t bytes, CUdeviceptr device, CUstream stream) {
    constexpr size_t guard = 64;
    const auto input = perf_input_bytes(bytes, 0x93bd175a);
    std::vector<unsigned char> reset(bytes, 0xa5), expected = reset, actual(bytes);
    std::copy(input.begin() + guard, input.end() - guard, expected.begin() + guard);
    std::memcpy(host, input.data(), bytes);
    unsigned flags = ~0u;
    PERF_CUDA(cuMemHostGetFlags(&flags, host));
    perf_require((flags & ~unsigned(CU_MEMHOSTALLOC_DEVICEMAP)) == 0, "default host storage flags permit implicit UVA mapping");
    PERF_CUDA(cuMemcpyHtoD(device, reset.data(), bytes));
    PERF_CUDA(cuCtxSynchronize());
    PERF_CUDA(cuMemcpyHtoDAsync(device + guard, host + guard, bytes - 2 * guard, stream));
    PERF_CUDA(cuStreamSynchronize(stream));
    PERF_CUDA(cuMemcpyDtoH(actual.data(), device, bytes));
    perf_expect_bytes(actual, expected, "complete host upload and untouched device guards");
    perf_expect_bytes({host, bytes}, input, "pinned source storage preserved");
    std::memcpy(host, reset.data(), bytes);
    PERF_CUDA(cuMemcpyDtoHAsync(host + guard, device + guard, bytes - 2 * guard, stream));
    PERF_CUDA(cuStreamSynchronize(stream));
    perf_expect_bytes({host, bytes}, expected, "complete host readback and both host guards");
}

perf_result execute_host_allocation(const perf_options& options) {
    constexpr std::array<std::string_view, 4> cases{
        "alloc_free_4k", "alloc_free_64k", "register_unregister_4k", "register_unregister_64k"};
    perf_require(std::find(cases.begin(), cases.end(), options.case_name) != cases.end(), "unknown host allocation case");
    const bool registration = options.case_name.starts_with("register");
    const size_t bytes = options.case_name.ends_with("_4k") ? 4096 : 65536;
    const auto device = select_perf_device(options.backend);
    perf_context context(device.value);
    perf_stream stream;
    perf_device_buffer storage(bytes);
    perf_result result;
    if (registration) {
        constexpr size_t alignment = 65536;
        auto backing = perf_input_bytes(bytes + 2 * alignment, 0x1943b6ef);
        const auto address = reinterpret_cast<uintptr_t>(backing.data());
        const auto offset = alignment - address % alignment;
        auto* pointer = backing.data() + offset;
        const auto verify = [&] {
            const auto before = backing;
            perf_host_registration registered(pointer, bytes);
            perf_expect_bytes(backing, before, "registration preserves backing bytes and neighbors");
            verify_perf_host_storage(pointer, bytes, storage.value, stream.value);
            const auto transferred = backing;
            registered.close();
            perf_expect_bytes(backing, transferred, "unregistration preserves caller storage");
            perf_expect_bytes({backing.data(), offset}, {before.data(), offset}, "registration prefix neighbor");
            perf_expect_bytes({pointer + bytes, backing.size() - offset - bytes},
                              {before.data() + offset + bytes, before.size() - offset - bytes}, "registration suffix neighbor");
        };
        verify();
        const auto before = backing;
        result = measure_perf(options, [&] {
            perf_host_registration registered(pointer, bytes);
            registered.close();
        });
        perf_expect_bytes(backing, before, "every timed register/unregister preserves full backing");
        verify();
    } else {
        const auto verify = [&] {
            perf_host_buffer host(bytes);
            perf_require(host.value != nullptr, "host allocation publishes storage");
            verify_perf_host_storage(host.data(), bytes, storage.value, stream.value);
            host.close();
        };
        verify();
        result = measure_perf(options, [&] {
            perf_host_buffer host(bytes);
            host.close();
        });
        verify();
    }
    result.parameters = {{"api", "driver"}, {"bytes", std::to_string(bytes)}, {"flags", "0"},
        {"boundary", registration ? "cuMemHostRegister_cuMemHostUnregister" : "cuMemHostAlloc_cuMemFreeHost"},
        {"completion", "synchronous_lifecycle_calls_no_pending_transfers"},
        {"state", registration ? "reused_prefaulted_65536_aligned_host_range" : "new_host_allocation_per_iteration_warm_allocator"},
        {"registration_alignment", registration ? "65536" : "not_applicable"},
        {"host_lock_budget_policy", "unchanged_os_limits"},
        {"cpu_touch_in_interval", "none"}, {"verification", "full_guarded_async_round_trip_before_and_after_sampling"}};
    device.describe(result);
    storage.close();
    stream.close();
    context.close();
    return result;
}

}

int main(int argc, char** argv) {
    return cudavm::run_perf(argc, argv, "host_allocation", cudavm::execute_host_allocation);
}
