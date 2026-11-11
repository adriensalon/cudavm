#include "test_fixtures.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>

namespace cudavm {
constexpr uint32_t heap_guard = 0xa1b2c3d4;
constexpr uint64_t heap_pointer_guard = 0x0123456789abcdefull;
constexpr unsigned heap_guards = 4, heap_capacity = 67;
std::vector<uint64_t> retained_heap_pointers;
unsigned retained_heap_seed = 0;

void heap_runtime(cudaError_t status) {
    require(status == cudaSuccess, std::string("Runtime heap operation: ") + cudaGetErrorString(status));
}
std::array<uint32_t, 3> heap_oracle(unsigned id, unsigned count, unsigned seed) {
    const auto owner = (id + 1) % count, bytes = 17 + (owner * 13 + seed) % 97;
    uint32_t sum = 0;
    for (unsigned i = 0; i < bytes; ++i) sum += (seed + owner * 17 + i * 7) & 255;
    return {sum, (seed + owner * 17) & 255, (seed + owner * 17 + (bytes - 1) * 7) & 255};
}
void execute_device_heap(const loaded_program& program) {
    size_t driver_limit = 0, runtime_limit = 0;
    TEST_CUDA(cuCtxGetLimit(&driver_limit, CU_LIMIT_MALLOC_HEAP_SIZE));
    heap_runtime(cudaDeviceGetLimit(&runtime_limit, cudaLimitMallocHeapSize));
    require(driver_limit == runtime_limit && driver_limit >= 1024 * 1024,
        "Driver/Runtime heap limits agree and admit this small allocation profile");
    device_buffer output((heap_capacity * 3 + heap_guards * 2) * 4);
    device_buffer table((heap_capacity + heap_guards * 2) * 8);
    auto to = output.value + heap_guards * 4, pointers = table.value + heap_guards * 8;
    std::vector<uint32_t> expected(heap_capacity * 3 + heap_guards * 2, heap_guard), actual(expected.size());
    std::vector<uint64_t> addresses(heap_capacity + heap_guards * 2, heap_pointer_guard), observed(addresses.size());
    CUstream stream{};
    TEST_CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    struct stream_guard { CUstream value; ~stream_guard() { cuStreamDestroy(value); } } cleanup{stream};
    auto allocate = program.function("heap_allocate"), consume = program.function("heap_consume");
    auto check_consumption = [&](unsigned count, unsigned seed) {
        std::fill(expected.begin(), expected.end(), heap_guard);
        TEST_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
        void* args[]{&to, &pointers, &count, &seed};
        const auto kernel = program.library_kernel("heap_consume");
        if (kernel) heap_runtime(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(3), dim3(23), args,
            0, reinterpret_cast<cudaStream_t>(stream)));
        else TEST_CUDA(cuLaunchKernel(consume, 3, 1, 1, 23, 1, 1, 0, stream, args, nullptr));
        heap_runtime(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
        for (unsigned i = 0; i < count; ++i) {
            const auto row = heap_oracle(i, count, seed);
            std::copy(row.begin(), row.end(), expected.begin() + heap_guards + i * 3);
            addresses[heap_guards + i] = 0;
        }
        TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
        expect_words(actual, expected, "device heap byte oracle and complete output guards");
        TEST_CUDA(cuMemcpyDtoH(observed.data(), table.value, observed.size() * 8));
        require(observed == addresses, "freed pointer slots, inactive slots and table guards");
    };
    // The previous module was unloaded by the runner before this library load.
    if (!retained_heap_pointers.empty()) {
        addresses = retained_heap_pointers;
        TEST_CUDA(cuMemcpyHtoD(table.value, addresses.data(), addresses.size() * 8));
        check_consumption(heap_capacity, retained_heap_seed);
        retained_heap_pointers.clear();
    }
    for (unsigned seed : {0u, 197u})
    for (unsigned count : {0u, 1u, 17u, heap_capacity}) {
        std::fill(expected.begin(), expected.end(), heap_guard);
        std::fill(addresses.begin(), addresses.end(), heap_pointer_guard);
        TEST_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
        TEST_CUDA(cuMemcpyHtoD(table.value, addresses.data(), addresses.size() * 8));
        void* args[]{&to, &pointers, &count, &seed};
        TEST_CUDA(cuLaunchKernel(allocate, 3, 1, 1, 23, 1, 1, 0, stream, args, nullptr));
        TEST_CUDA(cuStreamSynchronize(stream));
        for (unsigned i = 0; i < count; ++i) {
            expected[heap_guards + i * 3] = 1;
            expected[heap_guards + i * 3 + 1] = 0;
            expected[heap_guards + i * 3 + 2] = 17 + (i * 13 + seed) % 97;
        }
        TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
        expect_words(actual, expected, "nonzero aligned heap allocations and guards");
        TEST_CUDA(cuMemcpyDtoH(observed.data(), table.value, observed.size() * 8));
        std::vector<uint64_t> live;
        for (unsigned i = 0; i < count; ++i) {
            const auto pointer = observed[heap_guards + i];
            require(pointer && !(pointer & 15), "each device allocation is nonnull and 16-byte aligned");
            live.push_back(pointer); addresses[heap_guards + i] = pointer;
        }
        require(observed == addresses, "allocation only publishes active pointer slots");
        std::sort(live.begin(), live.end());
        require(std::adjacent_find(live.begin(), live.end()) == live.end(), "simultaneous allocations have distinct bases");
        if (count) {
            const auto pointer = observed[heap_guards];
            require(cuMemFree(pointer) == CUDA_ERROR_INVALID_VALUE, "host free cannot retire device malloc storage");
            uint32_t destination = heap_guard;
            require(cuMemcpyDtoH(&destination, pointer, 4) == CUDA_ERROR_INVALID_VALUE && destination == heap_guard,
                "host copy rejects private heap pointer without writing destination");
        }
        check_consumption(count, seed);
    }
    auto reuse = program.function("heap_reuse");
    for (unsigned rounds : {0u, 1u, 33u, 97u}) {
        unsigned seed = 251;
        std::fill(expected.begin(), expected.end(), heap_guard);
        TEST_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
        void* args[]{&to, &rounds, &seed};
        TEST_CUDA(cuLaunchKernel(reuse, 1, 1, 1, 1, 1, 1, 0, stream, args, nullptr));
        TEST_CUDA(cuStreamSynchronize(stream));
        uint32_t sum = 0;
        for (unsigned round = 0; round < rounds; ++round)
        for (unsigned i = 0; i < 19 + (round % 7) * 137; ++i) sum += (seed + round + i) & 255;
        expected[heap_guards] = sum; expected[heap_guards + 1] = 1; expected[heap_guards + 2] = 1;
        TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
        expect_words(actual, expected, "repeated allocate/free, null free, overflow failure and guards");
    }
    if (!program.library_kernel("heap_allocate")) {
        unsigned count = heap_capacity, seed = 91;
        std::fill(addresses.begin(), addresses.end(), heap_pointer_guard);
        TEST_CUDA(cuMemcpyHtoD(table.value, addresses.data(), addresses.size() * 8));
        void* args[]{&to, &pointers, &count, &seed};
        TEST_CUDA(cuLaunchKernel(allocate, 3, 1, 1, 23, 1, 1, 0, stream, args, nullptr));
        TEST_CUDA(cuStreamSynchronize(stream));
        TEST_CUDA(cuMemcpyDtoH(addresses.data(), table.value, addresses.size() * 8));
        for (unsigned i = 0; i < count; ++i) require(addresses[heap_guards + i] != 0, "retained device heap allocation succeeds");
        for (unsigned i = 0; i < heap_guards; ++i)
            require(addresses[i] == heap_pointer_guard && addresses[heap_guards + heap_capacity + i] == heap_pointer_guard,
                "retained heap pointer table guards");
        retained_heap_pointers = addresses; retained_heap_seed = seed;
    }
}
}
int main(int argc, char** argv) {
    const auto status = cudavm::run_test(argc, argv, cudavm::fixtures, cudavm::execute_device_heap);
    if (status == 0) cudavm::require(cudavm::retained_heap_pointers.empty(), "all retained device heap allocations consumed");
    return status;
}
