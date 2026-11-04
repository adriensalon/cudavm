#include "fixtures.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>

namespace cudavm {
void interoperation_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define INTEROP_RUNTIME(call) ::cudavm::interoperation_runtime((call), #call)
void* interoperation_pointer(CUdeviceptr value) { return reinterpret_cast<void*>(static_cast<uintptr_t>(value)); }
struct interoperation_device {
    CUdeviceptr value{}; bool runtime;
    interoperation_device(size_t bytes, bool api) : runtime(api) {
        if (runtime) { void* pointer{}; INTEROP_RUNTIME(cudaMalloc(&pointer, bytes)); value = reinterpret_cast<uintptr_t>(pointer); }
        else CONFORMANCE_CUDA(cuMemAlloc(&value, bytes));
    }
    void close() {
        if (runtime) CONFORMANCE_CUDA(cuMemFree(value)); else INTEROP_RUNTIME(cudaFree(interoperation_pointer(value)));
        value = 0;
    }
    ~interoperation_device() { if (value) { cuCtxSynchronize(); if (runtime) cuMemFree(value); else cudaFree(interoperation_pointer(value)); } }
};
struct interoperation_host {
    void* value{}; size_t words; bool runtime;
    interoperation_host(size_t count, bool api) : words(count), runtime(api) {
        if (runtime) INTEROP_RUNTIME(cudaMallocHost(&value, words * 4)); else CONFORMANCE_CUDA(cuMemAllocHost(&value, words * 4));
    }
    uint32_t* data() const { return static_cast<uint32_t*>(value); }
    void fill(uint32_t word) const { std::fill_n(data(), words, word); }
    void close() {
        if (runtime) CONFORMANCE_CUDA(cuMemFreeHost(value)); else INTEROP_RUNTIME(cudaFreeHost(value)); value = nullptr;
    }
    ~interoperation_host() { if (value) { cuCtxSynchronize(); if (runtime) cuMemFreeHost(value); else cudaFreeHost(value); } }
};
struct interoperation_stream {
    CUstream value{}; bool runtime;
    explicit interoperation_stream(bool api) : runtime(api) {
        if (runtime) { cudaStream_t stream{}; INTEROP_RUNTIME(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); value = reinterpret_cast<CUstream>(stream); }
        else CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING));
    }
    cudaStream_t cast() const { return reinterpret_cast<cudaStream_t>(value); }
    void close() {
        if (runtime) CONFORMANCE_CUDA(cuStreamDestroy(value)); else INTEROP_RUNTIME(cudaStreamDestroy(cast())); value = nullptr;
    }
    ~interoperation_stream() { if (value) { cuStreamSynchronize(value); if (runtime) cuStreamDestroy(value); else cudaStreamDestroy(cast()); } }
};
struct interoperation_event {
    CUevent value{}; bool runtime;
    explicit interoperation_event(bool api) : runtime(api) {
        if (runtime) { cudaEvent_t event{}; INTEROP_RUNTIME(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); value = reinterpret_cast<CUevent>(event); }
        else CONFORMANCE_CUDA(cuEventCreate(&value, CU_EVENT_DISABLE_TIMING));
    }
    cudaEvent_t cast() const { return reinterpret_cast<cudaEvent_t>(value); }
    void close() {
        if (runtime) CONFORMANCE_CUDA(cuEventDestroy(value)); else INTEROP_RUNTIME(cudaEventDestroy(cast())); value = nullptr;
    }
    ~interoperation_event() { if (value) { if (runtime) cuEventDestroy(value); else cudaEventDestroy(cast()); } }
};
void interoperation_owner(const interoperation_device& allocation, CUcontext context) {
    CUcontext owner{}; CONFORMANCE_CUDA(cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, allocation.value));
    require(owner == context, "mixed allocation retains explicit context owner");
    cudaPointerAttributes attributes{}; INTEROP_RUNTIME(cudaPointerGetAttributes(&attributes, interoperation_pointer(allocation.value)));
    require(attributes.type == cudaMemoryTypeDevice && attributes.devicePointer == interoperation_pointer(allocation.value), "Runtime classifies either allocator's device pointer");
}
void interoperation_identity(const interoperation_stream& stream, CUcontext context) {
    CUcontext owner{}; unsigned long long driver_id = 0, runtime_id = 0;
    CONFORMANCE_CUDA(cuStreamGetCtx(stream.value, &owner)); CONFORMANCE_CUDA(cuStreamGetId(stream.value, &driver_id));
    INTEROP_RUNTIME(cudaStreamGetId(stream.cast(), &runtime_id));
    require(owner == context && driver_id && driver_id == runtime_id, "mixed stream APIs share context and identity");
}
void interoperation_context() {
    CUcontext original{}, separate{}; CUdevice device{}; int runtime_device = -1;
    CONFORMANCE_CUDA(cuCtxGetCurrent(&original)); CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    INTEROP_RUNTIME(cudaGetDevice(&runtime_device)); require(runtime_device == device, "Runtime adopts explicit Driver device");
    CONFORMANCE_CUDA(cuCtxCreate(&separate, nullptr, 0, device));
    try {
        interoperation_device allocation(4, true); interoperation_owner(allocation, separate);
        INTEROP_RUNTIME(cudaMemset(interoperation_pointer(allocation.value), 0x57, 4));
        uint32_t actual = 0; CONFORMANCE_CUDA(cuMemcpyDtoH(&actual, allocation.value, 4));
        require(actual == 0x57575757, "Runtime storage executes in a second Driver context"); allocation.close();
    } catch (...) { cuCtxSetCurrent(original); cuCtxDestroy(separate); throw; }
    CONFORMANCE_CUDA(cuCtxSetCurrent(original)); CONFORMANCE_CUDA(cuCtxDestroy(separate));
}
void interoperation_execute(const loaded_program& program, bool runtime_source) {
    constexpr unsigned threads = 192, device_guard = 8, host_guard = 16, words = threads + 2 * device_guard;
    constexpr uint32_t guard = 0xa1b2c3d4, host_sentinel = 0x5a6b7c8d;
    CUcontext context{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&context));
    interoperation_device source(words * 4, runtime_source), output(words * 4, !runtime_source);
    interoperation_host upload(words + host_guard * 2, runtime_source), download(words + host_guard * 2, !runtime_source), preserved(words + host_guard * 2, runtime_source);
    interoperation_stream producer(runtime_source), consumer(!runtime_source);
    interoperation_event ready(runtime_source), complete(!runtime_source);
    interoperation_owner(source, context); interoperation_owner(output, context);
    interoperation_identity(producer, context); interoperation_identity(consumer, context);
    const auto function = program.function("cubin_indexed");
    const auto kernel = program.library_kernel("cubin_indexed");
    if (kernel) {
        const char* name{}; INTEROP_RUNTIME(cudaFuncGetName(&name, reinterpret_cast<const void*>(kernel)));
        require(name && std::strcmp(name, "cubin_indexed") == 0, "Runtime resolves Driver library kernel metadata");
        for (size_t i = 0; i < 3; ++i) {
            size_t driver_offset = 0, driver_size = 0, runtime_offset = 0, runtime_size = 0;
            CONFORMANCE_CUDA(cuFuncGetParamInfo(function, i, &driver_offset, &driver_size));
            INTEROP_RUNTIME(cudaFuncGetParamInfo(reinterpret_cast<const void*>(kernel), i, &runtime_offset, &runtime_size));
            require(driver_offset == i * 8 && runtime_offset == driver_offset && driver_size == (i == 2 ? 4 : 8) && runtime_size == driver_size,
                    "Runtime and Driver share exact library-kernel parameter layout");
        }
    }
    std::vector<uint32_t> expected(words, guard), input(words), expected_host(words + host_guard * 2, host_sentinel);
    if (output.runtime) CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), words * 4));
    else INTEROP_RUNTIME(cudaMemcpy(interoperation_pointer(output.value), expected.data(), words * 4, cudaMemcpyHostToDevice));
    for (unsigned phase = 0; phase < 2; ++phase) {
        for (unsigned i = 0; i < words; ++i) input[i] = 0xfedcba98u + i * 0x1234567u + phase * 0x9876543u;
        upload.fill(host_sentinel); download.fill(host_sentinel); preserved.fill(host_sentinel);
        std::copy(input.begin(), input.end(), upload.data() + host_guard);
        if (source.runtime) CONFORMANCE_CUDA(cuMemcpyHtoDAsync(source.value, upload.data() + host_guard, words * 4, producer.value));
        else INTEROP_RUNTIME(cudaMemcpyAsync(interoperation_pointer(source.value), upload.data() + host_guard, words * 4, cudaMemcpyHostToDevice, producer.cast()));
        CUdeviceptr destination = output.value + device_guard * 4, from = source.value + device_guard * 4;
        unsigned count = phase ? threads / 2 + 3 : threads;
        void* arguments[]{&destination, &from, &count};
        if (kernel && phase) INTEROP_RUNTIME(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(2, 3, 2), dim3(4, 2, 2), arguments, 0, producer.cast()));
        else CONFORMANCE_CUDA(cuLaunchKernel(function, 2, 3, 2, 4, 2, 2, 0, producer.value, arguments, nullptr));
        if (ready.runtime) CONFORMANCE_CUDA(cuEventRecord(ready.value, producer.value));
        else INTEROP_RUNTIME(cudaEventRecord(ready.cast(), producer.cast()));
        if (consumer.runtime) CONFORMANCE_CUDA(cuStreamWaitEvent(consumer.value, ready.value, 0));
        else INTEROP_RUNTIME(cudaStreamWaitEvent(consumer.cast(), ready.cast(), 0));
        if (output.runtime) CONFORMANCE_CUDA(cuMemcpyDtoHAsync(download.data() + host_guard, output.value, words * 4, consumer.value));
        else INTEROP_RUNTIME(cudaMemcpyAsync(download.data() + host_guard, interoperation_pointer(output.value), words * 4, cudaMemcpyDeviceToHost, consumer.cast()));
        if (source.runtime) CONFORMANCE_CUDA(cuMemcpyDtoHAsync(preserved.data() + host_guard, source.value, words * 4, consumer.value));
        else INTEROP_RUNTIME(cudaMemcpyAsync(preserved.data() + host_guard, interoperation_pointer(source.value), words * 4, cudaMemcpyDeviceToHost, consumer.cast()));
        if (complete.runtime) { CONFORMANCE_CUDA(cuEventRecord(complete.value, consumer.value)); CONFORMANCE_CUDA(cuEventSynchronize(complete.value)); }
        else { INTEROP_RUNTIME(cudaEventRecord(complete.cast(), consumer.cast())); INTEROP_RUNTIME(cudaEventSynchronize(complete.cast())); }
        CONFORMANCE_CUDA(cuEventQuery(ready.value)); INTEROP_RUNTIME(cudaEventQuery(ready.cast()));
        CONFORMANCE_CUDA(cuEventQuery(complete.value)); INTEROP_RUNTIME(cudaEventQuery(complete.cast()));
        for (unsigned i = 0; i < count; ++i) expected[device_guard + i] = input[device_guard + i] + i + 12;
        std::fill(expected_host.begin(), expected_host.end(), host_sentinel); std::copy(expected.begin(), expected.end(), expected_host.begin() + host_guard);
        expect_words({download.data(), download.words}, expected_host, "mixed allocation, stream, event and launch result with complete guards");
        std::copy(input.begin(), input.end(), expected_host.begin() + host_guard);
        expect_words({preserved.data(), preserved.words}, expected_host, "mixed API source preservation and pinned readback guards");
        expect_words({upload.data(), upload.words}, expected_host, "mixed API upload source preservation and guards");
    }
    CONFORMANCE_CUDA(cuCtxSynchronize());
    complete.close(); ready.close(); consumer.close(); producer.close(); preserved.close(); download.close(); upload.close(); output.close(); source.close();
    CUcontext after{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&after)); require(after == context, "mixed APIs retain explicit current context");
}
void execute_runtime_driver_interoperation(const loaded_program& program) {
    interoperation_context(); for (bool runtime_source : {false, true}) interoperation_execute(program, runtime_source);
}
}
int main(int argc, char** argv) { return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_runtime_driver_interoperation); }
