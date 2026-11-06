#include "fixtures.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace cudavm {
std::string_view concurrent_profile;
constexpr uint32_t concurrent_guard = 0x1937a5c9;
struct concurrent_storage {
    CUdeviceptr value{}, page{};
    size_t bytes{}, page_size{};
    std::vector<uint32_t> expected;
    explicit concurrent_storage(bool runtime) {
#ifdef _WIN32
        SYSTEM_INFO info{}; GetSystemInfo(&info); page_size = info.dwPageSize;
#else
        page_size = size_t(sysconf(_SC_PAGESIZE));
#endif
        require(page_size >= 16 && !(page_size & (page_size - 1)), "concurrent CPU page size");
        bytes = page_size * 4;
        if (runtime) {
            void* pointer{}; const auto status = cudaMallocManaged(&pointer, bytes, cudaMemAttachGlobal);
            require(status == cudaSuccess, std::string("Runtime concurrent managed allocation: ") + cudaGetErrorString(status));
            value = reinterpret_cast<uintptr_t>(pointer);
        } else CONFORMANCE_CUDA(cuMemAllocManaged(&value, bytes, CU_MEM_ATTACH_GLOBAL));
        page = (value + page_size * 2 - 1) & ~(CUdeviceptr(page_size) - 1);
        expected.assign(bytes / sizeof(uint32_t), concurrent_guard);
        std::fill_n(host(), expected.size(), concurrent_guard);
    }
    ~concurrent_storage() { if (value) cuMemFree(value); }
    uint32_t* host() const { return reinterpret_cast<uint32_t*>(uintptr_t(value)); }
    uint32_t* payload() const { return reinterpret_cast<uint32_t*>(uintptr_t(page)); }
    size_t offset() const { return size_t(page - value) / sizeof(uint32_t); }
    void verify(const char* label) const { expect_words({host(), expected.size()}, expected, label); }
};
void concurrent_require(bool condition, std::atomic_ref<uint32_t> cancel, const std::string& message) {
    if (condition) return;
    cancel.store(1, std::memory_order_release);
    std::cerr << "FAIL: " << message << std::endl;
    std::_Exit(1);
}
void concurrent_rounds(const loaded_program& program, bool runtime, unsigned rounds, unsigned seed, bool handoff) {
    concurrent_storage control(runtime), source(!runtime), destination(runtime);
    std::fill_n(control.payload(), 4, 0);
    std::fill_n(control.expected.begin() + control.offset(), 4, 0);
    std::atomic_ref<uint32_t> ready(control.payload()[0]), acknowledged(control.payload()[1]),
        cancel(control.payload()[2]), entered(control.payload()[3]);
    require(ready.is_lock_free() && acknowledged.is_lock_free() && entered.is_lock_free(), "lock-free managed control atomics");
    CUstream stream{}; CONFORMANCE_CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    struct stream_guard { CUstream value; ~stream_guard() { cuStreamDestroy(value); } } cleanup{stream};
    auto mailbox = control.page, from = source.page, to = destination.page;
    unsigned words = unsigned(source.page_size / sizeof(uint32_t)), publication = handoff ? 1 : 0;
    void* args[]{&mailbox, &from, &to, &words, &rounds, &seed, &publication};
    if (runtime && program.library_kernel("managed_participant")) {
        const auto result = cudaLaunchKernel(reinterpret_cast<const void*>(program.library_kernel("managed_participant")),
            dim3(1), dim3(1), args, 0, reinterpret_cast<cudaStream_t>(stream));
        require(result == cudaSuccess, std::string("Runtime managed participant launch: ") + cudaGetErrorString(result));
    } else CONFORMANCE_CUDA(cuLaunchKernel(program.function("managed_participant"), 1, 1, 1, 1, 1, 1, 0, stream, args, nullptr));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    auto await = [&](std::atomic_ref<uint32_t> flag, uint32_t expected, const char* label) {
        while (flag.load(std::memory_order_acquire) != expected) {
            concurrent_require(std::chrono::steady_clock::now() < deadline, cancel, label);
            std::this_thread::yield();
        }
    };
    await(entered, 1, "managed kernel entry deadline");
    for (unsigned round = 0; round < rounds; ++round) {
        for (unsigned word = 0; word < words; ++word) {
            const auto value = seed + word * 37u + round * 0x10001u;
            source.payload()[word] = value;
            source.expected[source.offset() + word] = value;
            destination.expected[destination.offset() + word] = (value ^ (seed + round * 0x9e3779b9u)) + word * 17u;
        }
        ready.store(round + 1, std::memory_order_release);
        await(acknowledged, round + 1, "managed CPU/GPU generation deadline");
        for (unsigned word = 0; word < words; ++word) {
            concurrent_require(destination.payload()[word] == destination.expected[destination.offset() + word], cancel,
                "managed published destination word " + std::to_string(word));
            concurrent_require(source.payload()[word] == source.expected[source.offset() + word], cancel, "managed CPU source preserved");
        }
    }
    CONFORMANCE_CUDA(cuStreamSynchronize(stream));
    control.expected[control.offset()] = control.expected[control.offset() + 1] = rounds;
    control.expected[control.offset() + 3] = 1;
    control.verify("managed control and complete guards");
    source.verify("managed CPU page, neighboring pages and guards");
    destination.verify("managed GPU page, neighboring pages and guards");
}
void execute_managed_concurrent_access(const loaded_program& program) {
    require(concurrent_profile == "disjoint" || concurrent_profile == "handoff", "known concurrent managed profile");
    CUdevice device{}; int managed{}, concurrent{};
    CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&managed, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, device));
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&concurrent, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, device));
    require_capability(managed && concurrent, "concurrent managed profile requires managed memory and concurrent managed access");
    require_capability(std::atomic_ref<uint32_t>::is_always_lock_free, "host ABI lacks lock-free 32-bit atomic references");
    for (bool runtime : {false, true}) for (unsigned rounds : {0u, 1u, 17u}) for (unsigned seed : {0u, 0x87654321u})
        concurrent_rounds(program, runtime, rounds, seed, concurrent_profile == "handoff");
}
}
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: <executable> <backend> cu <disjoint|handoff>\n"; return 1; }
    cudavm::concurrent_profile = argv[3];
    return cudavm::run_conformance(3, argv, cudavm::fixtures, cudavm::execute_managed_concurrent_access);
}
