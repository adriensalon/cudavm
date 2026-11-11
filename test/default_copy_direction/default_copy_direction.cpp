#include "test_framework.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>

namespace cudavm {
void check_direction_runtime(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
#define DIRECTION_RUNTIME(call) ::cudavm::check_direction_runtime((call), #call)
enum class direction_kind { pageable, pinned, registered, mapped_host, mapped_device, device, pitched, managed };
const char* direction_name(direction_kind kind) {
    switch (kind) {
    case direction_kind::pageable: return "pageable"; case direction_kind::pinned: return "pinned";
    case direction_kind::registered: return "registered"; case direction_kind::mapped_host: return "mapped_host";
    case direction_kind::mapped_device: return "mapped_device"; case direction_kind::device: return "device";
    case direction_kind::pitched: return "pitched"; case direction_kind::managed: return "managed";
    } return "unknown";
}
void* direction_pointer(CUdeviceptr value) { return reinterpret_cast<void*>(static_cast<uintptr_t>(value)); }
struct direction_stream {
    CUstream value{};
    direction_stream() { TEST_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~direction_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
};
struct direction_storage {
    direction_kind kind; bool runtime; size_t bytes = 797; CUdeviceptr address{}; void* host{};
    std::vector<unsigned char> backing, backing_expected;
    direction_storage(direction_kind selected, bool use_runtime) : kind(selected), runtime(use_runtime) {
        if (kind == direction_kind::pageable || kind == direction_kind::registered) {
            backing.resize(kind == direction_kind::registered ? 4 * 65536 : bytes, 0xeb); host = backing.data();
            if (kind == direction_kind::registered) {
                host = reinterpret_cast<void*>(((reinterpret_cast<uintptr_t>(host) + 65535) & ~uintptr_t(65535)) + 13);
                if (runtime) DIRECTION_RUNTIME(cudaHostRegister(host, bytes, cudaHostRegisterDefault)); else TEST_CUDA(cuMemHostRegister(host, bytes, 0));
            }
            address = reinterpret_cast<uintptr_t>(host);
        } else if (kind == direction_kind::pinned || kind == direction_kind::mapped_host || kind == direction_kind::mapped_device) {
            const unsigned flags = kind == direction_kind::pinned ? 0 : CU_MEMHOSTALLOC_DEVICEMAP;
            if (runtime) DIRECTION_RUNTIME(cudaHostAlloc(&host, bytes, flags)); else TEST_CUDA(cuMemHostAlloc(&host, bytes, flags));
            address = reinterpret_cast<uintptr_t>(host);
            if (kind == direction_kind::mapped_device) {
                if (runtime) { void* result{}; const auto status = cudaHostGetDevicePointer(&result, host, 0); if (status != cudaSuccess) { cudaFreeHost(host); host = nullptr; check_direction_runtime(status, "cudaHostGetDevicePointer"); } address = reinterpret_cast<uintptr_t>(result); }
                else { const auto status = cuMemHostGetDevicePointer(&address, host, 0); if (status != CUDA_SUCCESS) { cuMemFreeHost(host); host = nullptr; check_driver(status, "cuMemHostGetDevicePointer"); } }
            }
        } else if (kind == direction_kind::pitched) {
            size_t pitch{};
            if (runtime) { void* result{}; DIRECTION_RUNTIME(cudaMallocPitch(&result, &pitch, 191, 4)); address = reinterpret_cast<uintptr_t>(result); }
            else TEST_CUDA(cuMemAllocPitch(&address, &pitch, 191, 4, 16));
            bytes = pitch * 4;
        } else if (kind == direction_kind::managed) {
            if (runtime) { DIRECTION_RUNTIME(cudaMallocManaged(&host, bytes, cudaMemAttachGlobal)); address = reinterpret_cast<uintptr_t>(host); }
            else { TEST_CUDA(cuMemAllocManaged(&address, bytes, CU_MEM_ATTACH_GLOBAL)); host = direction_pointer(address); }
        } else {
            if (runtime) { void* result{}; DIRECTION_RUNTIME(cudaMalloc(&result, bytes)); address = reinterpret_cast<uintptr_t>(result); }
            else TEST_CUDA(cuMemAlloc(&address, bytes));
        }
        backing_expected = backing;
    }
    ~direction_storage() {
        cuCtxSynchronize();
        if (kind == direction_kind::registered) { if (runtime) cuMemHostUnregister(host); else cudaHostUnregister(host); }
        else if (kind == direction_kind::pinned || kind == direction_kind::mapped_host || kind == direction_kind::mapped_device) { if (runtime) cuMemFreeHost(host); else cudaFreeHost(host); }
        else if (kind != direction_kind::pageable) { if (runtime) cuMemFree(address); else cudaFree(direction_pointer(address)); }
    }
    direction_storage(const direction_storage&) = delete;
    direction_storage& operator=(const direction_storage&) = delete;
    void initialize(const std::vector<unsigned char>& data) {
        require(data.size() == bytes, "direction fixture initialization extent");
        if (host) std::memcpy(host, data.data(), bytes); else TEST_CUDA(cuMemcpyHtoD(address, data.data(), bytes));
        backing_expected = backing;
    }
    void expect(const std::vector<unsigned char>& expected, const std::string& context) const {
        std::vector<unsigned char> actual(bytes);
        if (host) std::memcpy(actual.data(), host, bytes); else TEST_CUDA(cuMemcpyDtoH(actual.data(), address, bytes));
        for (size_t i = 0; i < bytes; ++i) require(actual[i] == expected[i], context + ": byte " + std::to_string(i));
        if (!backing.empty()) {
            auto all = backing_expected; const auto offset = static_cast<unsigned char*>(host) - backing.data();
            std::copy(expected.begin(), expected.end(), all.begin() + offset);
            require(backing == all, context + ": complete caller backing and registration guards");
        }
    }
};
std::vector<unsigned char> direction_pattern(size_t bytes, unsigned seed) {
    std::vector<unsigned char> result(bytes);
    for (size_t i = 0; i < bytes; ++i) result[i] = static_cast<unsigned char>(seed + 73 * i + (i >> 4));
    return result;
}
CUresult direction_copy(bool runtime, bool asynchronous, CUdeviceptr to, CUdeviceptr from, size_t bytes, CUstream stream) {
    if (runtime) return static_cast<CUresult>(asynchronous ? cudaMemcpyAsync(direction_pointer(to), direction_pointer(from), bytes, cudaMemcpyDefault, reinterpret_cast<cudaStream_t>(stream))
                                                         : cudaMemcpy(direction_pointer(to), direction_pointer(from), bytes, cudaMemcpyDefault));
    return asynchronous ? cuMemcpyAsync(to, from, bytes, stream) : cuMemcpy(to, from, bytes);
}
void direction_pairs(std::vector<direction_kind> kinds, bool asynchronous, bool only_special = false) {
    direction_stream stream;
    for (bool runtime : {false, true}) for (auto source_kind : kinds) for (auto destination_kind : kinds) {
        if (only_special && source_kind != kinds.front() && destination_kind != kinds.front()) continue;
        direction_storage source(source_kind, !runtime), destination(destination_kind, runtime);
        const auto extent = std::min(source.bytes, destination.bytes);
        const std::array<std::array<size_t, 3>, 7> regions{{{17, 29, 0}, {17, 29, 1}, {31, 23, 7}, {19, 47, 127},
            {0, 0, extent}, {source.bytes - 97, destination.bytes - 97, 97}, {source.bytes - 1, destination.bytes - 1, 1}}};
        unsigned seed = 0;
        for (const auto& [from, to, bytes] : regions) {
            const auto input = direction_pattern(source.bytes, 19 + seed), initial = direction_pattern(destination.bytes, 137 + seed);
            source.initialize(input); destination.initialize(initial); auto expected = initial;
            std::copy_n(input.begin() + from, bytes, expected.begin() + to);
            const auto context = std::string(direction_name(source_kind)) + " -> " + direction_name(destination_kind)
                + " runtime=" + std::to_string(runtime) + " async=" + std::to_string(asynchronous) + " bytes=" + std::to_string(bytes);
            const auto status = direction_copy(runtime, asynchronous, destination.address + to, source.address + from, bytes, stream.value);
            if (status != CUDA_SUCCESS) { std::cout << context << std::endl; TEST_CUDA(status); }
            // Even synchronous device copies may be asynchronous with respect to
            // the host. Complete before directly inspecting managed/mapped bytes.
            TEST_CUDA(cuCtxSynchronize());
            destination.expect(expected, context + " destination"); source.expect(input, context + " source preservation"); seed += 59;
        }
    }
}
void direction_chain() {
    direction_stream stream; direction_storage input(direction_kind::pinned, true), output(direction_kind::pinned, false);
    direction_storage first(direction_kind::device, false), second(direction_kind::device, true);
    for (unsigned seed : {0u, 79u, 255u, 0xffffffffu}) {
        const auto source = direction_pattern(input.bytes, seed), initial_a = direction_pattern(first.bytes, 33), initial_b = direction_pattern(second.bytes, 91), initial_h = direction_pattern(output.bytes, 147);
        input.initialize(source); output.initialize(initial_h); first.initialize(initial_a); second.initialize(initial_b);
        auto a = initial_a, b = initial_b, h = initial_h;
        std::copy_n(source.begin() + 17, 613, a.begin() + 31); std::copy_n(a.begin() + 31, 613, b.begin() + 13);
        std::copy_n(b.begin() + 83, 193, a.begin() + 411); std::copy_n(a.begin() + 19, 701, h.begin() + 23);
        TEST_CUDA(direction_copy(false, true, first.address + 31, input.address + 17, 613, stream.value));
        TEST_CUDA(direction_copy(true, true, second.address + 13, first.address + 31, 613, stream.value));
        TEST_CUDA(direction_copy(false, true, first.address + 411, second.address + 83, 193, stream.value));
        TEST_CUDA(direction_copy(true, true, output.address + 23, first.address + 19, 701, stream.value));
        TEST_CUDA(cuStreamSynchronize(stream.value));
        input.expect(source, "chain input"); first.expect(a, "chain first device"); second.expect(b, "chain second device"); output.expect(h, "chain output");
    }
}
void direction_validation() {
    direction_stream stream; direction_storage source(direction_kind::device, false), destination(direction_kind::device, true);
    const auto input = direction_pattern(source.bytes, 17), initial = direction_pattern(destination.bytes, 153);
    for (bool runtime : {false, true}) for (bool asynchronous : {false, true}) {
        source.initialize(input); destination.initialize(initial);
        require(direction_copy(runtime, asynchronous, destination.address, 0, 1, stream.value) == CUDA_ERROR_INVALID_VALUE, "null nonempty source");
        require(direction_copy(runtime, asynchronous, 0, source.address, 1, stream.value) == CUDA_ERROR_INVALID_VALUE, "null nonempty destination");
        require(direction_copy(runtime, asynchronous, destination.address, source.address + source.bytes - 3, 4, stream.value) == CUDA_ERROR_INVALID_VALUE, "source overrun");
        require(direction_copy(runtime, asynchronous, destination.address + destination.bytes - 3, source.address, 4, stream.value) == CUDA_ERROR_INVALID_VALUE, "destination overrun");
        TEST_CUDA(cuStreamSynchronize(stream.value)); source.expect(input, "rejected copy source"); destination.expect(initial, "rejected copy destination");
        TEST_CUDA(direction_copy(runtime, asynchronous, destination.address + 29, source.address + 17, 631, stream.value)); TEST_CUDA(cuCtxSynchronize());
        auto expected = initial; std::copy_n(input.begin() + 17, 631, expected.begin() + 29); destination.expect(expected, "repaired copy destination"); source.expect(input, "repaired copy source");
    }
}
std::string direction_case = "all";
void execute_default_copy_direction() {
    CUdevice device{}; int unified{}; TEST_CUDA(cuCtxGetDevice(&device)); TEST_CUDA(cuDeviceGetAttribute(&unified, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device));
    require_capability(unified != 0, "default copy direction requires unified addressing");
    bool selected = false;
    const auto run = [&](const char* name, auto execute) { if (direction_case == "all" || direction_case == name) { std::cout << "case=" << name << std::endl; execute(); selected = true; } };
    run("synchronous", [] { direction_pairs({direction_kind::pageable, direction_kind::pinned, direction_kind::device, direction_kind::pitched}, false); });
    run("asynchronous", [] { direction_pairs({direction_kind::pinned, direction_kind::device, direction_kind::pitched}, true); });
    run("registered", [] { for (bool async : {false, true}) direction_pairs({direction_kind::registered, direction_kind::pinned, direction_kind::device}, async, true); });
    run("mapped", [&] {
        int mapping{}; TEST_CUDA(cuDeviceGetAttribute(&mapping, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, device)); require_capability(mapping != 0, "mapped default copies require host mapping");
        for (bool async : {false, true}) direction_pairs({direction_kind::mapped_host, direction_kind::mapped_device, direction_kind::device}, async);
    });
    run("managed", [&] {
        int managed{}; TEST_CUDA(cuDeviceGetAttribute(&managed, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, device)); require_capability(managed != 0, "managed default copies require managed memory");
        for (bool async : {false, true}) direction_pairs({direction_kind::managed, direction_kind::pinned, direction_kind::device}, async, true);
    });
    run("chain", direction_chain); run("validation", direction_validation);
    require(selected, "unknown default_copy_direction case: " + direction_case);
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::direction_case = argv[2]; --argc; }
    return cudavm::run_test(argc, argv, cudavm::execute_default_copy_direction, CU_CTX_MAP_HOST);
}
