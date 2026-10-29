#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <utility>

namespace cudavm {
void check_registration_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define REGISTRATION_RUNTIME(call) ::cudavm::check_registration_runtime((call), #call)

struct registered_host {
    // Separate allocations cannot share the native driver's registration pages.
    static constexpr size_t alignment = 65536, backing_size = 4 * alignment;
    std::vector<unsigned char> storage = std::vector<unsigned char>(backing_size);
    unsigned char* value;
    size_t size;
    unsigned flags;
    bool runtime, active = false;
    registered_host(size_t bytes, size_t offset, unsigned options, bool use_runtime)
        : size(bytes), flags(options), runtime(use_runtime) {
        const auto base = reinterpret_cast<uintptr_t>(storage.data());
        value = reinterpret_cast<unsigned char*>((base + alignment - 1) & ~(uintptr_t(alignment) - 1)) + offset;
        require(value + size <= storage.data() + storage.size(), "registration backing extent");
        for (size_t i = 0; i < storage.size(); ++i)
            storage[i] = static_cast<unsigned char>((i * 73 + 19) ^ (i >> 8));
        register_range();
    }
    ~registered_host() { if (active) cuMemHostUnregister(value); }
    registered_host(const registered_host&) = delete;
    registered_host& operator=(const registered_host&) = delete;
    size_t offset() const { return size_t(value - storage.data()); }
    void register_range() {
        const auto before = storage;
        if (runtime) REGISTRATION_RUNTIME(cudaHostRegister(value, size, flags));
        else CONFORMANCE_CUDA(cuMemHostRegister(value, size, flags));
        active = true;
        require(storage == before, "registration preserves caller backing and guards");
    }
    void unregister_range(bool use_runtime) {
        if (use_runtime) REGISTRATION_RUNTIME(cudaHostUnregister(value));
        else CONFORMANCE_CUDA(cuMemHostUnregister(value));
        active = false;
    }
};
void verify_registration_flags(const registered_host& host) {
    CUdevice device{};
    int unified = 0;
    CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&unified, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device));
    unsigned base = ~0u;
    CONFORMANCE_CUDA(cuMemHostGetFlags(&base, host.value));
    require((base & host.flags) == host.flags
        && !(base & ~(host.flags | (unified ? CU_MEMHOSTREGISTER_DEVICEMAP : 0))), "registration flags");
    for (size_t offset : {size_t(0), host.size / 2, host.size - 1}) {
        unsigned driver = ~0u, runtime = ~0u;
        CONFORMANCE_CUDA(cuMemHostGetFlags(&driver, host.value + offset));
        REGISTRATION_RUNTIME(cudaHostGetFlags(&runtime, host.value + offset));
        require(driver == base && runtime == base, "Driver/Runtime interior registration flags");
    }
}
void host_registration_ranges() {
    for (bool runtime : {false, true}) for (unsigned flags : {0u, 1u, 2u, 3u})
    for (const auto& [bytes, offset] : std::array<std::pair<size_t, size_t>, 5>{{
            {1, 0}, {31, 17}, {4093, 3}, {8197, 4091}, {4096, 0}}}) {
        registered_host host(bytes, offset, flags, runtime);
        const auto expected = host.storage;
        verify_registration_flags(host);
        host.unregister_range(!runtime);
        require(host.storage == expected, "unregister preserves caller backing and guards");
        for (size_t i = 0; i < bytes; ++i) host.value[i] ^= 0x5b;
        const auto changed = host.storage;
        host.runtime = !runtime;
        host.register_range();
        verify_registration_flags(host);
        host.unregister_range(runtime);
        require(host.storage == changed, "re-registration preserves caller-written bytes");
    }
}

struct registration_stream {
    CUstream value{};
    registration_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~registration_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
};
struct registration_gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* argument) {
        auto& gate = *static_cast<registration_gate*>(argument);
        std::unique_lock lock(gate.mutex);
        gate.entered = true;
        gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(15), [&] { return gate.released; })) gate.expired = true;
    }
    void await() {
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "registration gate entry deadline");
    }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
};
void registration_roundtrip(registered_host& source, registered_host& destination, CUdeviceptr device,
                            size_t source_offset, size_t device_offset, size_t destination_offset,
                            size_t bytes, bool runtime_upload, CUstream stream) {
    auto* pointer = reinterpret_cast<unsigned char*>(static_cast<uintptr_t>(device));
    const auto runtime_stream = reinterpret_cast<cudaStream_t>(stream);
    if (runtime_upload) {
        if (stream) REGISTRATION_RUNTIME(cudaMemcpyAsync(pointer + device_offset, source.value + source_offset,
            bytes, cudaMemcpyHostToDevice, runtime_stream));
        else REGISTRATION_RUNTIME(cudaMemcpy(pointer + device_offset, source.value + source_offset, bytes, cudaMemcpyHostToDevice));
        if (stream) CONFORMANCE_CUDA(cuMemcpyDtoHAsync(destination.value + destination_offset, device + device_offset, bytes, stream));
        else CONFORMANCE_CUDA(cuMemcpyDtoH(destination.value + destination_offset, device + device_offset, bytes));
    } else {
        if (stream) CONFORMANCE_CUDA(cuMemcpyHtoDAsync(device + device_offset, source.value + source_offset, bytes, stream));
        else CONFORMANCE_CUDA(cuMemcpyHtoD(device + device_offset, source.value + source_offset, bytes));
        if (stream) REGISTRATION_RUNTIME(cudaMemcpyAsync(destination.value + destination_offset, pointer + device_offset,
            bytes, cudaMemcpyDeviceToHost, runtime_stream));
        else REGISTRATION_RUNTIME(cudaMemcpy(destination.value + destination_offset, pointer + device_offset, bytes, cudaMemcpyDeviceToHost));
    }
}
void host_registration_transfers() {
    constexpr size_t extent = 8197;
    for (bool runtime : {false, true}) for (unsigned flags : {0u, 1u, 2u, 3u})
    for (bool asynchronous : {false, true}) {
        registered_host source(extent, 17, flags, runtime), destination(extent, 31, flags, !runtime);
        device_buffer device(extent);
        registration_stream stream;
        for (const auto& [src, dev, dst, bytes] : std::array<std::array<size_t, 4>, 5>{{
                {0, 0, 0, 0}, {0, 0, 0, extent}, {13, 19, 23, 1}, {17, 29, 37, 4097},
                {extent - 1, extent - 1, extent - 1, 1}}}) {
            const auto input = source.storage, initial = destination.storage;
            auto expected = initial;
            std::copy_n(source.value + src, bytes, expected.begin() + destination.offset() + dst);
            CONFORMANCE_CUDA(cuMemsetD8(device.value, 0xd3, extent));
            registration_gate gate;
            struct cleanup_gate {
                registration_gate& gate;
                CUstream stream;
                ~cleanup_gate() { gate.release(); cuStreamSynchronize(stream); }
            } cleanup{gate, stream.value};
            if (asynchronous) {
                CONFORMANCE_CUDA(cuLaunchHostFunc(stream.value, registration_gate::call, &gate));
                gate.await();
            }
            registration_roundtrip(source, destination, device.value, src, dev, dst, bytes,
                runtime, asynchronous ? stream.value : nullptr);
            if (asynchronous) {
                require(cuStreamQuery(stream.value) == CUDA_ERROR_NOT_READY, "registered transfer remains pending at gate");
                require(destination.storage == initial, "pending readback leaves destination unchanged");
                gate.release();
                CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
                require(!gate.expired, "registration transfer gate completion deadline");
            }
            require(source.storage == input, "registered upload source and backing guards");
            require(destination.storage == expected, "registered readback payload and backing guards");
            std::vector<unsigned char> actual(extent + 32, 0xe7), expected_device = actual;
            std::fill_n(expected_device.begin() + 16, extent, 0xd3);
            std::copy_n(source.value + src, bytes, expected_device.begin() + 16 + dev);
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data() + 16, device.value, extent));
            require(actual == expected_device, "registered transfer device bytes and readback guards");
        }
        source.unregister_range(!runtime);
        destination.unregister_range(runtime);
    }
}

struct registration_context {
    CUcontext original{}, value{};
    registration_context() {
        CUdevice device{};
        CONFORMANCE_CUDA(cuCtxGetCurrent(&original));
        CONFORMANCE_CUDA(cuCtxGetDevice(&device));
        CONFORMANCE_CUDA(cuCtxCreate(&value, nullptr, CU_CTX_MAP_HOST, device));
    }
    ~registration_context() { if (value) cuCtxDestroy(value); cuCtxSetCurrent(original); }
};
void host_registration_portable() {
    for (bool runtime : {false, true}) for (unsigned flags : {1u, 3u}) {
        registered_host host(4093, 17, flags, runtime);
        const auto initial = host.storage;
        {
            registration_context other;
            device_buffer device(host.size);
            CONFORMANCE_CUDA(cuMemcpyHtoD(device.value, host.value, host.size));
            std::fill_n(host.value, host.size, 0x91);
            REGISTRATION_RUNTIME(cudaMemcpy(host.value, reinterpret_cast<void*>(static_cast<uintptr_t>(device.value)),
                host.size, cudaMemcpyDeviceToHost));
            require(host.storage == initial, "portable registration roundtrip from another context");
        }
        host.unregister_range(!runtime);
        host.runtime = !runtime;
        host.register_range();
        verify_registration_flags(host);
        host.unregister_range(runtime);
        require(host.storage == initial, "portable registration backing survives context and registration lifetimes");
    }
}
void host_registration_validation() {
    registered_host host(8197, 17, 0, false);
    const auto expected = host.storage;
    for (const auto& [offset, bytes] : std::array<std::pair<size_t, size_t>, 3>{{{0, host.size}, {1, 17}, {host.size - 1, 33}}}) {
        require(cuMemHostRegister(host.value + offset, bytes, 0) == CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
            "Driver overlapping registration");
        require(cudaHostRegister(host.value + offset, bytes, 0) == cudaErrorHostMemoryAlreadyRegistered,
            "Runtime overlapping registration");
    }
    require(cuMemHostUnregister(host.value + 1) == CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED, "Driver interior unregister");
    require(cudaHostUnregister(host.value + 1) == cudaErrorHostMemoryNotRegistered, "Runtime interior unregister");
    require(cuMemFreeHost(host.value) == CUDA_ERROR_INVALID_VALUE, "Driver allocated-storage free rejects registration");
    require(cudaFreeHost(host.value) == cudaErrorInvalidValue, "Runtime allocated-storage free rejects registration");
    for (bool runtime : {false, true}) {
        auto verify_invalid = [&](void* pointer, size_t bytes, unsigned flags) {
            if (runtime) {
                const auto status = cudaHostRegister(pointer, bytes, flags);
                if (status == cudaSuccess) cudaHostUnregister(pointer);
                require(status == cudaErrorInvalidValue, "Runtime invalid registration arguments");
            } else {
                const auto status = cuMemHostRegister(pointer, bytes, flags);
                if (status == CUDA_SUCCESS) cuMemHostUnregister(pointer);
                require(status == CUDA_ERROR_INVALID_VALUE, "Driver invalid registration arguments");
            }
        };
        verify_invalid(nullptr, 64, 0);
        verify_invalid(host.storage.data(), 0, 0);
        verify_invalid(host.storage.data(), 31, 0x10);
        verify_invalid(host.storage.data(), 31, 0x80000000);
    }
    verify_registration_flags(host);
    require(host.storage == expected, "rejected registration operations preserve live backing");
    host.unregister_range(true);
    require(cuMemHostUnregister(host.value) == CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED, "Driver missing registration");
    require(cudaHostUnregister(host.value) == cudaErrorHostMemoryNotRegistered, "Runtime missing registration");
    host.value[0] ^= 0x37;
    host.register_range();
    verify_registration_flags(host);
}

std::string_view host_registration_case = "all";
void execute_host_registration() {
    const std::array<std::pair<std::string_view, void (*)()>, 4> cases{{
        {"ranges", host_registration_ranges}, {"transfers", host_registration_transfers},
        {"portable", host_registration_portable}, {"validation", host_registration_validation}}};
    bool found = false;
    for (const auto& [name, execute] : cases) if (host_registration_case == "all" || host_registration_case == name) {
        found = true;
        std::cout << "host_registration=" << name << std::endl;
        execute();
    }
    require(found, "unknown host_registration case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::host_registration_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_host_registration);
}
