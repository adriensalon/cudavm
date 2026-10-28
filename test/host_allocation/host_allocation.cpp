#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <utility>

namespace cudavm {
void check_host_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define HOST_RUNTIME(call) ::cudavm::check_host_runtime((call), #call)

struct host_allocator_case {
    const char* name;
    unsigned api, flags;
};
constexpr host_allocator_case host_allocators[]{
    {"cuMemAllocHost", 0, 0}, {"cudaMallocHost", 2, 0},
    {"cuMemHostAlloc/default", 1, 0}, {"cudaHostAlloc/default", 3, 0},
    {"cuMemHostAlloc/portable", 1, CU_MEMHOSTALLOC_PORTABLE},
    {"cudaHostAlloc/portable", 3, CU_MEMHOSTALLOC_PORTABLE},
    {"cuMemHostAlloc/mapped", 1, CU_MEMHOSTALLOC_DEVICEMAP},
    {"cudaHostAlloc/mapped", 3, CU_MEMHOSTALLOC_DEVICEMAP},
    {"cuMemHostAlloc/portable_mapped", 1, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP},
    {"cudaHostAlloc/portable_mapped", 3, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP}};
unsigned host_runtime_flags(unsigned flags) {
    return (flags & CU_MEMHOSTALLOC_PORTABLE ? cudaHostAllocPortable : 0)
        | (flags & CU_MEMHOSTALLOC_DEVICEMAP ? cudaHostAllocMapped : 0);
}
struct allocated_host {
    void* value{};
    size_t size;
    host_allocator_case allocator;
    allocated_host(size_t bytes, host_allocator_case kind) : size(bytes), allocator(kind) {
        if (kind.api == 0) CONFORMANCE_CUDA(cuMemAllocHost(&value, bytes));
        else if (kind.api == 1) CONFORMANCE_CUDA(cuMemHostAlloc(&value, bytes, kind.flags));
        else if (kind.api == 2) HOST_RUNTIME(cudaMallocHost(&value, bytes));
        else HOST_RUNTIME(cudaHostAlloc(&value, bytes, host_runtime_flags(kind.flags)));
        require(value != nullptr, std::string(kind.name) + ": nonempty allocation publishes a pointer");
    }
    ~allocated_host() {
        if (value) {
            if (allocator.api >= 2) cudaFreeHost(value);
            else cuMemFreeHost(value);
        }
    }
    allocated_host(const allocated_host&) = delete;
    allocated_host& operator=(const allocated_host&) = delete;
    unsigned char* data() const { return static_cast<unsigned char*>(value); }
    void destroy(bool runtime) {
        if (runtime) HOST_RUNTIME(cudaFreeHost(value));
        else CONFORMANCE_CUDA(cuMemFreeHost(value));
        value = nullptr;
    }
};
void expect_host_bytes(const allocated_host& buffer, std::span<const unsigned char> expected,
                        const std::string& operation) {
    require(buffer.size == expected.size(), operation + ": extent");
    for (size_t i = 0; i < expected.size(); ++i)
        if (buffer.data()[i] != expected[i]) throw std::runtime_error(operation + ": byte " + std::to_string(i));
}
std::vector<unsigned char> host_pattern(size_t size, unsigned seed) {
    std::vector<unsigned char> result(size);
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<unsigned char>((i * 73 + seed * 41) ^ (i >> 8));
    return result;
}
void verify_host_flags(const allocated_host& buffer) {
    CUdevice device{};
    int unified = 0;
    CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&unified, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device));
    unsigned base_flags = ~0u;
    CONFORMANCE_CUDA(cuMemHostGetFlags(&base_flags, buffer.value));
    // A native UVA allocation can additionally report implicit device mapping.
    const unsigned permitted = buffer.allocator.flags | (unified ? CU_MEMHOSTALLOC_DEVICEMAP : 0);
    require((base_flags & buffer.allocator.flags) == buffer.allocator.flags && !(base_flags & ~permitted),
        std::string(buffer.allocator.name) + ": requested flags=" + std::to_string(buffer.allocator.flags)
        + " reported=" + std::to_string(base_flags) + " unified=" + std::to_string(unified));
    for (size_t offset : {size_t(0), buffer.size / 2, buffer.size - 1}) {
        unsigned driver_flags = ~0u, runtime_flags = ~0u;
        CONFORMANCE_CUDA(cuMemHostGetFlags(&driver_flags, buffer.data() + offset));
        HOST_RUNTIME(cudaHostGetFlags(&runtime_flags, buffer.data() + offset));
        require(driver_flags == base_flags && runtime_flags == host_runtime_flags(base_flags),
            "Driver/Runtime flags agree throughout the requested allocation");
    }
}

void host_allocation_storage() {
    // Native CUDA can suballocate tiny host ranges from backing with shared
    // flag metadata. Keep each simultaneous group under one allocation policy.
    for (const auto& allocator : host_allocators) {
        std::vector<std::unique_ptr<allocated_host>> live;
        for (size_t bytes : {size_t(1), size_t(3), size_t(31), size_t(256), size_t(4093)}) {
            auto buffer = std::make_unique<allocated_host>(bytes, allocator);
            for (const auto& other : live) {
                const auto a = reinterpret_cast<uintptr_t>(buffer->value), b = reinterpret_cast<uintptr_t>(other->value);
                require(a + bytes <= b || b + other->size <= a, "live host allocation ranges are disjoint");
            }
            verify_host_flags(*buffer);
            auto expected = host_pattern(bytes, unsigned(live.size()));
            std::copy(expected.begin(), expected.end(), buffer->data());
            live.push_back(std::move(buffer));
        }
        for (size_t i = 0; i < live.size(); ++i)
            expect_host_bytes(*live[i], host_pattern(live[i]->size, unsigned(i)), "independent CPU-readable/writable host storage");
        for (size_t i = 0; i < live.size(); ++i) live[i]->destroy(i % 2 == 0);
    }
}

struct host_transfer_stream {
    CUstream value{};
    host_transfer_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~host_transfer_stream() {
        if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); }
    }
    host_transfer_stream(const host_transfer_stream&) = delete;
    host_transfer_stream& operator=(const host_transfer_stream&) = delete;
};
void host_roundtrip(const allocated_host& source, allocated_host& destination, device_buffer& device,
                    size_t source_offset, size_t device_offset, size_t destination_offset, size_t bytes,
                    bool runtime_upload, CUstream stream = nullptr) {
    auto* device_pointer = reinterpret_cast<void*>(static_cast<uintptr_t>(device.value));
    if (runtime_upload) {
        if (stream) HOST_RUNTIME(cudaMemcpyAsync(static_cast<unsigned char*>(device_pointer) + device_offset,
            source.data() + source_offset, bytes, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)));
        else HOST_RUNTIME(cudaMemcpy(static_cast<unsigned char*>(device_pointer) + device_offset,
            source.data() + source_offset, bytes, cudaMemcpyHostToDevice));
        if (stream) CONFORMANCE_CUDA(cuMemcpyDtoHAsync(destination.data() + destination_offset,
            device.value + device_offset, bytes, stream));
        else CONFORMANCE_CUDA(cuMemcpyDtoH(destination.data() + destination_offset, device.value + device_offset, bytes));
    } else {
        if (stream) CONFORMANCE_CUDA(cuMemcpyHtoDAsync(device.value + device_offset, source.data() + source_offset, bytes, stream));
        else CONFORMANCE_CUDA(cuMemcpyHtoD(device.value + device_offset, source.data() + source_offset, bytes));
        if (stream) HOST_RUNTIME(cudaMemcpyAsync(destination.data() + destination_offset,
            static_cast<unsigned char*>(device_pointer) + device_offset, bytes, cudaMemcpyDeviceToHost,
            reinterpret_cast<cudaStream_t>(stream)));
        else HOST_RUNTIME(cudaMemcpy(destination.data() + destination_offset,
            static_cast<unsigned char*>(device_pointer) + device_offset, bytes, cudaMemcpyDeviceToHost));
    }
}
void host_allocation_transfers() {
    constexpr size_t extent = 320;
    constexpr unsigned char guard = 0xd3;
    const std::array<std::array<size_t, 4>, 5> regions{{
        {16, 16, 16, 0}, {16, 16, 16, 288}, {17, 19, 23, 1}, {19, 21, 25, 137}, {303, 301, 299, 1}}};
    for (const auto& allocator : host_allocators) for (bool asynchronous : {false, true})
    for (bool runtime_upload : {false, true}) {
        allocated_host source(extent, allocator), destination(extent, host_allocators[allocator.api >= 2 ? 0 : 1]);
        device_buffer device(extent);
        host_transfer_stream stream;
        for (const auto& [source_offset, device_offset, destination_offset, bytes] : regions) {
            const auto input = host_pattern(extent, 7);
            std::copy(input.begin(), input.end(), source.data());
            std::fill_n(destination.data(), extent, guard);
            CONFORMANCE_CUDA(cuMemsetD8(device.value, guard, extent));
            host_roundtrip(source, destination, device, source_offset, device_offset, destination_offset,
                bytes, runtime_upload, asynchronous ? stream.value : nullptr);
            if (asynchronous) CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
            auto expected_host = std::vector<unsigned char>(extent, guard), expected_device = expected_host;
            std::copy_n(input.begin() + source_offset, bytes, expected_host.begin() + destination_offset);
            std::copy_n(input.begin() + source_offset, bytes, expected_device.begin() + device_offset);
            expect_host_bytes(destination, expected_host, std::string(allocator.name) + ": readback and host guards");
            expect_host_bytes(source, input, "pinned upload source preservation");
            std::vector<unsigned char> actual(extent + 32, guard);
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data() + 16, device.value, extent));
            std::vector<unsigned char> guarded_expected(extent + 32, guard);
            std::copy(expected_device.begin(), expected_device.end(), guarded_expected.begin() + 16);
            require(actual == guarded_expected, "device payload, untouched bytes and readback guards");
        }
        verify_host_flags(source);
        source.destroy(allocator.api < 2);
        destination.destroy(allocator.api >= 2);
    }
}

struct host_transfer_gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* argument) {
        auto& gate = *static_cast<host_transfer_gate*>(argument);
        std::unique_lock lock(gate.mutex);
        gate.entered = true;
        gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(15), [&] { return gate.released; })) gate.expired = true;
    }
    void await() {
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "host transfer gate entry deadline");
    }
    void release() {
        std::lock_guard lock(mutex);
        released = true;
        changed.notify_all();
    }
    ~host_transfer_gate() { release(); }
};
void host_allocation_lifetime() {
    constexpr size_t extent = 4093;
    for (const auto& allocator : host_allocators) {
        allocated_host source(extent, allocator), survivor(extent, host_allocators[allocator.api >= 2 ? 0 : 1]);
        device_buffer device(extent);
        host_transfer_stream stream;
        // On failure, release and drain the gate before its callback state dies.
        host_transfer_gate gate;
        struct gate_cleanup {
            host_transfer_gate& gate;
            CUstream stream;
            ~gate_cleanup() { gate.release(); cuStreamSynchronize(stream); }
        } cleanup{gate, stream.value};
        const auto input = host_pattern(extent, 13);
        auto expected = std::vector<unsigned char>(extent, 0xe9);
        std::copy(input.begin(), input.end(), source.data());
        std::copy(expected.begin(), expected.end(), survivor.data());
        if (allocator.api >= 2) HOST_RUNTIME(cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream.value), host_transfer_gate::call, &gate));
        else CONFORMANCE_CUDA(cuLaunchHostFunc(stream.value, host_transfer_gate::call, &gate));
        gate.await();
        host_roundtrip(source, survivor, device, 16, 19, 23, extent - 64, allocator.api >= 2, stream.value);
        require(cuStreamQuery(stream.value) == CUDA_ERROR_NOT_READY, "controlled pending use of pinned allocations");
        expect_host_bytes(survivor, expected, "readback cannot pass the closed stream gate");
        gate.release();
        CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
        { std::lock_guard lock(gate.mutex); require(!gate.expired, "host transfer gate completion deadline"); }
        std::copy_n(input.begin() + 16, extent - 64, expected.begin() + 23);
        expect_host_bytes(source, input, "pending upload retains source bytes through completion");
        expect_host_bytes(survivor, expected, "pending readback payload and guards at completion");
        source.destroy(allocator.api < 2);
        allocated_host replacement(extent, allocator);
        std::fill_n(replacement.data(), extent, 0x35);
        expect_host_bytes(survivor, expected, "surviving storage after release and replacement");
        verify_host_flags(replacement);
        replacement.destroy(allocator.api < 2);
        survivor.destroy(allocator.api >= 2);
    }
}

void host_allocation_validation() {
    require(cuMemAllocHost(nullptr, 64) == CUDA_ERROR_INVALID_VALUE, "null cuMemAllocHost output");
    require(cuMemHostAlloc(nullptr, 64, 0) == CUDA_ERROR_INVALID_VALUE, "null cuMemHostAlloc output");
    require(cudaMallocHost(nullptr, 64) == cudaErrorInvalidValue, "null cudaMallocHost output");
    require(cudaHostAlloc(nullptr, 64, 0) == cudaErrorInvalidValue, "null cudaHostAlloc output");
    for (unsigned flags : {8u, 0x80000000u, 0xffffffffu}) {
        void* output{};
        auto status = cuMemHostAlloc(&output, 64, flags);
        if (status == CUDA_SUCCESS) CONFORMANCE_CUDA(cuMemFreeHost(output));
        require(status == CUDA_ERROR_INVALID_VALUE, "invalid Driver allocation flags");
        output = nullptr;
        const auto result = cudaHostAlloc(&output, 64, flags);
        if (result == cudaSuccess) HOST_RUNTIME(cudaFreeHost(output));
        require(result == cudaErrorInvalidValue, "invalid Runtime allocation flags");
    }
    allocated_host host(64, host_allocators[0]);
    const auto expected = host_pattern(host.size, 17);
    std::copy(expected.begin(), expected.end(), host.data());
    require(cuMemHostGetFlags(nullptr, host.value) == CUDA_ERROR_INVALID_VALUE, "null Driver flags output");
    require(cudaHostGetFlags(nullptr, host.value) == cudaErrorInvalidValue, "null Runtime flags output");
    unsigned flags = 0;
    unsigned char ordinary = 0;
    for (void* pointer : {static_cast<void*>(nullptr), static_cast<void*>(&ordinary)}) {
        require(cuMemHostGetFlags(&flags, pointer) == CUDA_ERROR_INVALID_VALUE, "Driver flags require an allocated host range");
        require(cudaHostGetFlags(&flags, pointer) == cudaErrorInvalidValue, "Runtime flags require an allocated host range");
    }
    require(cuMemFreeHost(host.data() + 1) == CUDA_ERROR_INVALID_VALUE, "Driver free requires allocation base");
    require(cudaFreeHost(host.data() + 1) == cudaErrorInvalidValue, "Runtime free requires allocation base");
    verify_host_flags(host);
    expect_host_bytes(host, expected, "allocation remains usable after rejected operations");
    host.destroy(true);
}

std::string_view host_allocation_case = "all";
void execute_host_allocation() {
    require(host_allocation_case == "all" || host_allocation_case == "allocation" || host_allocation_case == "transfers"
        || host_allocation_case == "lifetime" || host_allocation_case == "validation", "unknown host_allocation case");
    const std::array<std::pair<std::string_view, void (*)()>, 4> cases{{
        {"allocation", host_allocation_storage}, {"transfers", host_allocation_transfers},
        {"lifetime", host_allocation_lifetime}, {"validation", host_allocation_validation}}};
    for (const auto& [name, execute] : cases) if (host_allocation_case == "all" || host_allocation_case == name) {
        std::cout << "host_allocation=" << name << std::endl;
        execute();
    }
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::host_allocation_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_host_allocation);
}
