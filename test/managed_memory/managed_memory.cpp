#include "fixtures.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstddef>
#include <cstring>

namespace cudavm {
void check_managed_memory_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define MANAGED_MEMORY_RUNTIME(call) ::cudavm::check_managed_memory_runtime((call), #call)
struct managed_allocation {
    static constexpr unsigned words = 256, bytes = words * sizeof(uint32_t);
    CUdeviceptr value{};
    explicit managed_allocation(bool runtime, unsigned flags) {
        if (runtime) {
            void* pointer{}; MANAGED_MEMORY_RUNTIME(cudaMallocManaged(&pointer, bytes, flags));
            value = reinterpret_cast<uintptr_t>(pointer);
        } else CONFORMANCE_CUDA(cuMemAllocManaged(&value, bytes, flags));
        require(value && value % alignof(std::max_align_t) == 0, "managed allocation address and alignment");
    }
    ~managed_allocation() { if (value) cuMemFree(value); }
    managed_allocation(const managed_allocation&) = delete;
    uint32_t* host() const { return reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(value)); }
    void initialize(std::span<const uint32_t> input) const { std::memcpy(host(), input.data(), bytes); }
    void expect(std::span<const uint32_t> expected, const std::string& context) const {
        expect_words(std::span(host(), words), expected, context);
    }
    void close(bool runtime) {
        if (runtime) MANAGED_MEMORY_RUNTIME(cudaFree(host()));
        else CONFORMANCE_CUDA(cuMemFree(value));
        value = 0;
    }
};
struct managed_memory_stream {
    CUstream value{};
    explicit managed_memory_stream(bool runtime) {
        if (runtime) {
            cudaStream_t stream{}; MANAGED_MEMORY_RUNTIME(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            value = reinterpret_cast<CUstream>(stream);
        } else CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING));
    }
    ~managed_memory_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    cudaStream_t cast() const { return reinterpret_cast<cudaStream_t>(value); }
    void attach(const managed_allocation& storage, unsigned flags, bool runtime, bool full_size) const {
        if (runtime) MANAGED_MEMORY_RUNTIME(cudaStreamAttachMemAsync(cast(), storage.host(), full_size ? storage.bytes : 0, flags));
        else CONFORMANCE_CUDA(cuStreamAttachMemAsync(value, storage.value, full_size ? storage.bytes : 0, flags));
    }
    void destroy(bool runtime) {
        if (runtime) MANAGED_MEMORY_RUNTIME(cudaStreamDestroy(cast()));
        else CONFORMANCE_CUDA(cuStreamDestroy(value));
        value = nullptr;
    }
};
struct managed_memory_event {
    CUevent value{};
    managed_memory_event() { CONFORMANCE_CUDA(cuEventCreate(&value, CU_EVENT_DISABLE_TIMING)); }
    ~managed_memory_event() { if (value) cuEventDestroy(value); }
};
void managed_memory_queries(const managed_allocation& storage) {
    for (size_t offset : {size_t(0), size_t(1), size_t(storage.bytes / 2), size_t(storage.bytes - 1)}) {
        const auto address = storage.value + offset;
        struct { uint32_t before = 0xabcdef12, value = 0, after = 0x12345678; } managed;
        CONFORMANCE_CUDA(cuPointerGetAttribute(&managed.value, CU_POINTER_ATTRIBUTE_IS_MANAGED, address));
        require(managed.before == 0xabcdef12 && managed.after == 0x12345678 && managed.value == 1, "guarded managed classification");
        struct { uint64_t before = 0x123456789abcdef0; CUdeviceptr value = 0; uint64_t after = 0xfedcba9876543210; } device;
        CONFORMANCE_CUDA(cuPointerGetAttribute(&device.value, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, address));
        require(device.before == 0x123456789abcdef0 && device.after == 0xfedcba9876543210 && device.value == address, "guarded managed device alias");
        struct { uint64_t before = 0x123456789abcdef0; void* value{}; uint64_t after = 0xfedcba9876543210; } host;
        CONFORMANCE_CUDA(cuPointerGetAttribute(&host.value, CU_POINTER_ATTRIBUTE_HOST_POINTER, address));
        require(host.before == 0x123456789abcdef0 && host.after == 0xfedcba9876543210 && reinterpret_cast<uintptr_t>(host.value) == address,
            "guarded managed host alias preserves byte offset");
        struct { uint64_t before = 0x123456789abcdef0; cudaPointerAttributes value{}; uint64_t after = 0xfedcba9876543210; } attributes;
        MANAGED_MEMORY_RUNTIME(cudaPointerGetAttributes(&attributes.value, reinterpret_cast<void*>(static_cast<uintptr_t>(address))));
        require(attributes.before == 0x123456789abcdef0 && attributes.after == 0xfedcba9876543210
            && attributes.value.type == cudaMemoryTypeManaged && reinterpret_cast<uintptr_t>(attributes.value.hostPointer) == address
            && reinterpret_cast<uintptr_t>(attributes.value.devicePointer) == address, "guarded Runtime managed classification and aliases");
    }
}
void managed_memory_launch(const loaded_program& program, CUstream stream, CUdeviceptr out, CUdeviceptr in,
    unsigned threads, unsigned count, bool runtime) {
    void* args[]{&out, &in, &count};
    if (const auto kernel = program.library_kernel("cubin_indexed"); runtime && kernel)
        MANAGED_MEMORY_RUNTIME(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(3), dim3(threads), args, 0, reinterpret_cast<cudaStream_t>(stream)));
    else CONFORMANCE_CUDA(cuLaunchKernel(program.function("cubin_indexed"), 3, 1, 1, threads, 1, 1, 0, stream, args, nullptr));
}
void managed_memory_oracle(std::vector<uint32_t>& output, unsigned to, const std::vector<uint32_t>& input,
    unsigned from, unsigned threads, unsigned count) {
    for (unsigned i = 0; i < std::min(threads * 3, count); ++i) output[to + i] = input[from + i] + i + 3;
}
void managed_memory_validation(const managed_allocation& storage, const managed_memory_stream& stream) {
    CUdeviceptr invalid{}; void* runtime{};
    require(cuMemAllocManaged(nullptr, 4, CU_MEM_ATTACH_GLOBAL) == CUDA_ERROR_INVALID_VALUE, "null Driver managed output");
    require(cudaMallocManaged(nullptr, 4, cudaMemAttachGlobal) == cudaErrorInvalidValue, "null Runtime managed output");
    require(cuMemAllocManaged(&invalid, 0, CU_MEM_ATTACH_GLOBAL) == CUDA_ERROR_INVALID_VALUE, "zero Driver managed size");
    require(cudaMallocManaged(&runtime, 0, cudaMemAttachGlobal) == cudaErrorInvalidValue, "zero Runtime managed size");
    for (unsigned flags : {0u, unsigned(CU_MEM_ATTACH_SINGLE), 3u, 0x80000000u}) {
        require(cuMemAllocManaged(&invalid, 4, flags) == CUDA_ERROR_INVALID_VALUE, "invalid Driver managed allocation flags");
        require(cudaMallocManaged(&runtime, 4, flags) == cudaErrorInvalidValue, "invalid Runtime managed allocation flags");
    }
    for (size_t bytes : {size_t(1), size_t(storage.bytes - 4), size_t(storage.bytes + 4)}) {
        require(cuStreamAttachMemAsync(stream.value, storage.value, bytes, CU_MEM_ATTACH_SINGLE) == CUDA_ERROR_INVALID_VALUE, "partial Driver attachment");
        require(cudaStreamAttachMemAsync(stream.cast(), storage.host(), bytes, cudaMemAttachSingle) == cudaErrorInvalidValue, "partial Runtime attachment");
    }
    require(cuStreamAttachMemAsync(stream.value, storage.value + 4, 0, CU_MEM_ATTACH_SINGLE) == CUDA_ERROR_INVALID_VALUE, "interior Driver attachment");
    require(cudaStreamAttachMemAsync(stream.cast(), storage.host() + 1, 0, cudaMemAttachSingle) == cudaErrorInvalidValue, "interior Runtime attachment");
    require(cuStreamAttachMemAsync(nullptr, storage.value, 0, CU_MEM_ATTACH_SINGLE) == CUDA_ERROR_INVALID_VALUE, "single Driver legacy-default attachment");
    require(cudaStreamAttachMemAsync(nullptr, storage.host(), 0, cudaMemAttachSingle) == cudaErrorInvalidValue, "single Runtime legacy-default attachment");
}
void managed_memory_handoffs(const loaded_program& program) {
    constexpr uint32_t guard = 0xd3e7a591;
    constexpr unsigned from = 7, to = 13;
    for (bool runtime : {false, true}) for (unsigned flags : {unsigned(CU_MEM_ATTACH_GLOBAL), unsigned(CU_MEM_ATTACH_HOST)}) {
        managed_allocation source(runtime, flags), destination(!runtime, flags);
        require(source.value + source.bytes <= destination.value || destination.value + destination.bytes <= source.value, "live managed allocations are disjoint");
        managed_memory_stream producer(runtime), consumer(!runtime);
        managed_memory_event ready;
        managed_memory_queries(source); managed_memory_queries(destination);
        managed_memory_validation(destination, producer);
        for (unsigned threads : {1u, 13u, 33u}) for (unsigned count : {0u, 1u, threads * 3 / 2 + 1, threads * 3, threads * 3 + 7}) {
            CONFORMANCE_CUDA(cuCtxSynchronize());
            std::vector<uint32_t> input(source.words, guard), expected(source.words, guard);
            for (unsigned i = 0; i < threads * 3; ++i) input[from + i] = i * 0x1234567u + (threads ^ (count * 0x9876543u));
            source.initialize(input); destination.initialize(expected);
            producer.attach(source, CU_MEM_ATTACH_SINGLE, !runtime, false);
            producer.attach(destination, CU_MEM_ATTACH_SINGLE, runtime, true);
            managed_memory_launch(program, producer.value, destination.value + to * 4, source.value + from * 4, threads, count, runtime);
            managed_memory_oracle(expected, to, input, from, threads, count);
            CONFORMANCE_CUDA(cuEventRecord(ready.value, producer.value));
            MANAGED_MEMORY_RUNTIME(cudaStreamWaitEvent(consumer.cast(), reinterpret_cast<cudaEvent_t>(ready.value), 0));
            consumer.attach(destination, CU_MEM_ATTACH_SINGLE, !runtime, false);
            managed_memory_launch(program, consumer.value, destination.value + to * 4, destination.value + to * 4, threads, count, !runtime);
            const auto before_second = expected;
            managed_memory_oracle(expected, to, before_second, to, threads, count);
            MANAGED_MEMORY_RUNTIME(cudaStreamSynchronize(consumer.cast()));
            CONFORMANCE_CUDA(cuCtxSynchronize());
            source.expect(input, "managed CPU observes preserved source and guards");
            destination.expect(expected, "managed CPU observes event-ordered GPU passes and guards");
            for (unsigned i = 0; i < threads * 3; ++i) expected[to + i] ^= 0x87654321u + i * 7;
            destination.initialize(expected);
            consumer.attach(destination, CU_MEM_ATTACH_GLOBAL, runtime, true);
            CONFORMANCE_CUDA(cuStreamSynchronize(consumer.value));
            managed_memory_launch(program, producer.value, destination.value + to * 4, destination.value + to * 4, threads, count, runtime);
            const auto before_third = expected;
            managed_memory_oracle(expected, to, before_third, to, threads, count);
            CONFORMANCE_CUDA(cuCtxSynchronize());
            destination.expect(expected, "managed CPU mutation publishes to a different stream after global association");
            source.expect(input, "managed independent allocation survives repeated handoffs");
        }
        source.close(!runtime); destination.close(runtime);
    }
}
void managed_memory_retirement(const loaded_program& program) {
    constexpr uint32_t guard = 0x12345678;
    for (bool runtime : {false, true}) {
        managed_allocation storage(runtime, CU_MEM_ATTACH_GLOBAL);
        std::vector<uint32_t> expected(storage.words, guard); storage.initialize(expected);
        managed_memory_stream first(runtime);
        first.attach(storage, CU_MEM_ATTACH_SINGLE, runtime, true);
        managed_memory_launch(program, first.value, storage.value + 16, storage.value + 16, 13, 29, runtime);
        managed_memory_oracle(expected, 4, expected, 4, 13, 29);
        first.destroy(!runtime);
        CONFORMANCE_CUDA(cuCtxSynchronize());
        storage.expect(expected, "managed completed bytes survive associated stream destruction");
        managed_memory_stream replacement(!runtime);
        managed_memory_launch(program, replacement.value, storage.value + 16, storage.value + 16, 13, 29, !runtime);
        managed_memory_oracle(expected, 4, expected, 4, 13, 29);
        CONFORMANCE_CUDA(cuCtxSynchronize());
        storage.expect(expected, "managed global default association restored for replacement stream");
        storage.close(!runtime);
    }
}
void managed_memory_contexts() {
    CUcontext original{}, foreign{}; CUdevice device{};
    CONFORMANCE_CUDA(cuCtxGetCurrent(&original)); CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    managed_allocation storage(false, CU_MEM_ATTACH_GLOBAL);
    std::vector<uint32_t> expected(storage.words, 0xa1b2c3d4); storage.initialize(expected);
    CONFORMANCE_CUDA(cuCtxCreate(&foreign, nullptr, 0, device));
    try {
        CONFORMANCE_CUDA(cuMemsetD32(storage.value + 16, 0x98765432, 17));
        MANAGED_MEMORY_RUNTIME(cudaMemset(storage.host() + 37, 0x5b, 23));
        CONFORMANCE_CUDA(cuCtxSynchronize());
        std::fill_n(expected.begin() + 4, 17, 0x98765432);
        std::memset(reinterpret_cast<unsigned char*>(expected.data() + 37), 0x5b, 23);
        storage.expect(expected, "foreign context updates shared managed CPU storage");
        CONFORMANCE_CUDA(cuCtxSetCurrent(original)); CONFORMANCE_CUDA(cuCtxDestroy(foreign)); foreign = nullptr;
    } catch (...) { cuCtxSetCurrent(original); if (foreign) cuCtxDestroy(foreign); throw; }
    storage.expect(expected, "owner's managed allocation survives foreign context teardown");
    storage.close(true);
}
void execute_managed_memory(const loaded_program& program) {
    CUdevice device{}; int supported = 0;
    CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, device));
    require_capability(supported != 0, "managed memory allocation requires reported managed-memory capability");
    managed_memory_handoffs(program); managed_memory_retirement(program); managed_memory_contexts();
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_managed_memory);
}
