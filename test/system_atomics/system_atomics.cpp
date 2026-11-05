#include "fixtures.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>

namespace cudavm {
std::string_view system_profile;
struct system_page {
    uint32_t before[16];
    uint32_t words[12];
    uint32_t after[16];
};
struct system_host_page {
    system_page* value{};
    CUdeviceptr alias{};
    system_host_page() {
        void* storage{};
        CONFORMANCE_CUDA(cuMemHostAlloc(&storage, sizeof(system_page), CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE));
        value = std::construct_at(static_cast<system_page*>(storage));
        auto status = cuMemHostGetDevicePointer(&alias, value, 0);
        if (status != CUDA_SUCCESS) { std::destroy_at(value); cuMemFreeHost(value); value = nullptr; check_driver(status, "mapped system mailbox alias"); }
    }
    ~system_host_page() { if (value) { std::destroy_at(value); cuMemFreeHost(value); } }
};
struct system_stream {
    CUstream value{};
    system_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~system_stream() { if (value) cuStreamDestroy(value); }
};
// A failure while the GPU waits for another host phase must not enter SDK
// teardown or free mapped storage still in use. CTest owns this disposable process.
void system_active_require(bool condition, std::atomic_ref<uint32_t> cancel, const std::string& message) {
    if (condition) return;
    cancel.store(1, std::memory_order_release);
    std::cerr << "FAIL: " << message << std::endl;
    std::_Exit(1);
}
void system_case(const loaded_program& program, unsigned rounds, unsigned seed, bool runtime, bool rmw) {
    constexpr unsigned guards = 4;
    constexpr uint32_t guard = 0xa1b2c3d4;
    system_host_page page;
    std::fill_n(page.value->before, 16, guard); std::fill_n(page.value->after, 16, guard);
    std::fill_n(page.value->words, 12, guard); std::fill_n(page.value->words, 4, 0);
    std::atomic_ref<uint32_t> ready(page.value->words[0]), acknowledged(page.value->words[1]),
        cancel(page.value->words[2]), counter(page.value->words[3]);
    require(ready.is_lock_free() && acknowledged.is_lock_free() && counter.is_lock_free(), "CPU system mailbox atomics are lock-free");
    std::vector<uint32_t> expected(guards * 2 + rounds, guard), actual(expected.size()), cpu_tickets(rounds);
    device_buffer output(expected.size() * 4);
    CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
    auto mailbox = page.alias + offsetof(system_page, words), tickets = output.value + guards * 4;
    require(mailbox % std::atomic_ref<uint32_t>::required_alignment == 0, "mapped system word alignment");
    system_stream stream;
    unsigned enabled = rmw ? 1 : 0;
    void* args[]{&mailbox, &tickets, &rounds, &seed, &enabled};
    if (runtime) {
        auto status = cudaLaunchKernel(reinterpret_cast<const void*>(program.library_kernel("system_handoff")),
            dim3(1), dim3(1), args, 0, reinterpret_cast<cudaStream_t>(stream.value));
        require(status == cudaSuccess, std::string("Runtime system handoff launch: ") + cudaGetErrorString(status));
    } else CONFORMANCE_CUDA(cuLaunchKernel(program.function("system_handoff"), 1, 1, 1, 1, 1, 1, 0, stream.value, args, nullptr));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (unsigned round = 0; round < rounds; ++round) {
        std::array<uint32_t, 4> payload;
        for (unsigned word = 0; word < 4; ++word) {
            payload[word] = seed ^ (round * 0x9e3779b9u + word * 17u);
            page.value->words[4 + word] = payload[word];
        }
        ready.store(round + 1, std::memory_order_release);
        if (rmw) cpu_tickets[round] = counter.fetch_add(1, std::memory_order_acq_rel);
        while (acknowledged.load(std::memory_order_acquire) != round + 1) {
            system_active_require(std::chrono::steady_clock::now() < deadline, cancel, "bounded CPU/GPU system handoff completion");
            std::this_thread::yield();
        }
        for (unsigned word = 0; word < 4; ++word) {
            const auto wanted = uint32_t(uint64_t(payload[word] ^ (seed + round * 0x10001u)) + word);
            system_active_require(page.value->words[8 + word] == wanted, cancel, "system release/acquire payload at phase " + std::to_string(round));
            system_active_require(page.value->words[4 + word] == payload[word], cancel, "GPU preserves CPU-published source words");
        }
        system_active_require(counter.load(std::memory_order_acquire) == (rmw ? (round + 1) * 2 : 0), cancel,
            "exact CPU/GPU system atomic counter per phase");
    }
    CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
    require(ready.load() == rounds && acknowledged.load() == rounds && cancel.load() == 0, "system mailbox terminal generation");
    require(std::all_of(std::begin(page.value->before), std::end(page.value->before), [](auto word) { return word == guard; })
        && std::all_of(std::begin(page.value->after), std::end(page.value->after), [](auto word) { return word == guard; }), "complete mapped mailbox guards");
    if (!rounds) for (unsigned word = 4; word < 12; ++word) require(page.value->words[word] == guard, "empty handoff leaves payload untouched");
    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
    for (unsigned round = 0; round < rounds; ++round) {
        if (rmw) {
            const auto gpu = actual[guards + round], cpu = cpu_tickets[round];
            require(std::min(gpu, cpu) == round * 2 && std::max(gpu, cpu) == round * 2 + 1, "one indivisible CPU and GPU system RMW ticket per phase");
            expected[guards + round] = gpu;
        } else expected[guards + round] = round;
    }
    expect_words(actual, expected, "GPU system participant records and allocation guards");
}
void execute_system_atomics(const loaded_program& program) {
    require(system_profile == "load_store" || system_profile == "rmw", "known system atomic profile");
    CUdevice device{}; CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    int mapped = 0; CONFORMANCE_CUDA(cuDeviceGetAttribute(&mapped, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, device));
    require_capability(mapped != 0, "native mapped host memory is unavailable");
    require_capability(std::atomic_ref<uint32_t>::is_always_lock_free, "host ABI lacks lock-free 32-bit atomic references");
    const bool rmw = system_profile == "rmw";
    if (rmw) {
        int native = 0; CONFORMANCE_CUDA(cuDeviceGetAttribute(&native, CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, device));
        require_capability(native != 0, "native link does not support system RMW atomics on mapped memory");
    }
    for (unsigned rounds : {0u, 1u, 64u}) for (unsigned seed : {0u, 0x87654321u}) {
        system_case(program, rounds, seed, false, rmw);
        if (program.library_kernel("system_handoff")) system_case(program, rounds, seed, true, rmw);
    }
}
}
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: <executable> <backend> cu <load_store|rmw>\n"; return 1; }
    cudavm::system_profile = argv[3];
    return cudavm::run_conformance(3, argv, cudavm::fixtures, cudavm::execute_system_atomics, 1, CU_CTX_MAP_HOST);
}
