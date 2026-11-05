#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>

namespace cudavm {
std::string_view batch_case;
void batch_runtime(cudaError_t status) {
    require(status == cudaSuccess, std::string("batch Runtime: ") + cudaGetErrorString(status));
}
struct batch_stream {
    CUstream value{};
    batch_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~batch_stream() { cuStreamSynchronize(value); cuStreamDestroy(value); }
};
struct batch_host {
    unsigned char* value{};
    explicit batch_host(size_t size) { CONFORMANCE_CUDA(cuMemAllocHost(reinterpret_cast<void**>(&value), size)); }
    ~batch_host() { cuMemFreeHost(value); }
};
struct batch_gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false, expired = false;
    CUstream stream;
    explicit batch_gate(CUstream value) : stream(value) {
        CONFORMANCE_CUDA(cuLaunchHostFunc(stream, call, this));
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "batch gate entry deadline");
    }
    ~batch_gate() { release(); cuStreamSynchronize(stream); }
    static void CUDA_CB call(void* argument) {
        auto& gate = *static_cast<batch_gate*>(argument);
        std::unique_lock lock(gate.mutex);
        gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(15), [&] { return gate.released; })) gate.expired = true;
    }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
    void finish() {
        release(); CONFORMANCE_CUDA(cuStreamSynchronize(stream));
        require(!expired, "batch gate completion deadline");
    }
};
void batch_expect(std::span<const unsigned char> actual, std::span<const unsigned char> expected, const char* label) {
    require(actual.size() == expected.size(), std::string(label) + " size");
    for (size_t i = 0; i < actual.size(); ++i)
        require(actual[i] == expected[i], std::string(label) + " byte " + std::to_string(i));
}
CUdeviceptr batch_address(const void* pointer) { return reinterpret_cast<uintptr_t>(pointer); }
CUresult batch_submit(bool runtime, CUdeviceptr* to, CUdeviceptr* from, size_t* sizes, size_t count,
                       CUmemcpyAttributes* attributes, size_t* indices, size_t attribute_count, CUstream stream,
                       bool single = false) {
    if (!runtime) {
        if (single) return cuMemcpyWithAttributesAsync(to[0], from[0], sizes[0], attributes, stream);
        return cuMemcpyBatchAsync(to, from, sizes, count, attributes, indices, attribute_count, stream);
    }
    std::vector<void*> destinations(count);
    std::vector<const void*> sources(count);
    for (size_t i = 0; i < count; ++i) {
        destinations[i] = reinterpret_cast<void*>(uintptr_t(to ? to[i] : 0));
        sources[i] = reinterpret_cast<const void*>(uintptr_t(from ? from[i] : 0));
    }
    std::vector<cudaMemcpyAttributes> values(attribute_count);
    for (size_t i = 0; attributes && i < attribute_count; ++i) {
        values[i].srcAccessOrder = static_cast<cudaMemcpySrcAccessOrder>(attributes[i].srcAccessOrder);
        values[i].flags = attributes[i].flags;
    }
    const auto lane = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t result;
    if (single) result = cudaMemcpyWithAttributesAsync(destinations[0], sources[0], sizes[0], attributes ? values.data() : nullptr, lane);
    else result = cudaMemcpyBatchAsync(to ? destinations.data() : nullptr, from ? sources.data() : nullptr,
        sizes, count, attributes ? values.data() : nullptr, indices, attribute_count, lane);
    if (result == cudaSuccess) return CUDA_SUCCESS;
    require(result == cudaErrorInvalidValue, std::string("unexpected Runtime batch error: ") + cudaGetErrorString(result));
    require(cudaGetLastError() == result && cudaGetLastError() == cudaSuccess, "Runtime batch error clears");
    return CUDA_ERROR_INVALID_VALUE;
}
CUmemcpyAttributes batch_attributes(CUmemcpySrcAccessOrder order = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM) {
    CUmemcpyAttributes value{}; value.srcAccessOrder = order; return value;
}
void batch_transfers() {
    constexpr size_t extent = 1024;
    batch_host host_source(extent), host_destination(extent);
    device_buffer source(extent), destination(extent);
    batch_stream stream;
    std::vector<unsigned char> input(extent), initial(extent), expected(extent), actual(extent);
    for (size_t i = 0; i < extent; ++i) { input[i] = static_cast<unsigned char>((i * 73 + 17) ^ (i >> 3)); initial[i] = static_cast<unsigned char>(i * 13 + 0x91); }
    const std::array<size_t, 4> offsets{17, 259, 521, 769}, source_offsets{5, 277, 513, 801};
    for (bool runtime : {false, true}) for (unsigned direction = 0; direction < 3; ++direction) {
        std::copy(input.begin(), input.end(), host_source.value);
        std::copy(initial.begin(), initial.end(), host_destination.value);
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), extent));
        CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, initial.data(), extent));
        std::array<CUdeviceptr, 4> to{}, from{};
        std::array<size_t, 4> sizes{1, 13, 127, 31};
        std::array<CUmemcpyAttributes, 2> attributes{batch_attributes(), batch_attributes()};
        attributes[1].flags = CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE;
        std::array<size_t, 2> indices{0, 2};
        expected = initial;
        for (size_t i = 0; i < sizes.size(); ++i) {
            from[i] = (direction == 0 ? batch_address(host_source.value) : source.value) + source_offsets[i];
            to[i] = (direction == 2 ? batch_address(host_destination.value) : destination.value) + offsets[i];
            std::copy_n(input.begin() + source_offsets[i], sizes[i], expected.begin() + offsets[i]);
        }
        CONFORMANCE_CUDA(batch_submit(runtime, to.data(), from.data(), sizes.data(), sizes.size(), attributes.data(), indices.data(), attributes.size(), stream.value));
        CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, extent));
        batch_expect(actual, direction == 2 ? initial : expected, "batch device destination and guards");
        batch_expect({host_destination.value, extent}, direction == 2 ? expected : initial, "batch host destination and guards");
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), source.value, extent));
        batch_expect(actual, input, "batch device source preservation");
        batch_expect({host_source.value, extent}, input, "batch host source preservation");
    }
}
void batch_ordering() {
    constexpr size_t extent = 257;
    for (bool runtime : {false, true}) {
        batch_host input(extent), output(extent);
        device_buffer source(extent), destination(extent);
        batch_stream stream;
        std::vector<unsigned char> expected(extent, 0xb7);
        for (size_t i = 0; i < extent; ++i) { input.value[i] = static_cast<unsigned char>(i * 37 + 51); output.value[i] = 0xb7; }
        CONFORMANCE_CUDA(cuMemsetD8(source.value, 0x45, extent));
        CONFORMANCE_CUDA(cuMemsetD8(destination.value, 0xb7, extent));
        CUevent event{}; CONFORMANCE_CUDA(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
        struct event_cleanup { CUevent value; ~event_cleanup() { cuEventDestroy(value); } } cleanup{event};
        batch_gate gate(stream.value);
        CONFORMANCE_CUDA(cuMemcpyHtoDAsync(source.value, input.value, extent, stream.value));
        std::array<CUdeviceptr, 2> to{destination.value + 17, destination.value + 157};
        std::array<CUdeviceptr, 2> from{source.value + 3, source.value + 173};
        std::array<size_t, 2> sizes{97, 67}, indices{0, 1};
        std::array<CUmemcpyAttributes, 2> attributes{batch_attributes(), batch_attributes()};
        std::copy_n(input.value + 3, 97, expected.begin() + 17);
        std::copy_n(input.value + 173, 67, expected.begin() + 157);
        CONFORMANCE_CUDA(batch_submit(runtime, to.data(), from.data(), sizes.data(), 2, attributes.data(), indices.data(), 2, stream.value));
        to.fill(0); from.fill(0); sizes.fill(0); indices.fill(~size_t(0)); attributes.fill({});
        batch_runtime(cudaMemcpyAsync(output.value, reinterpret_cast<void*>(uintptr_t(destination.value)), extent, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream.value)));
        CONFORMANCE_CUDA(cuEventRecord(event, stream.value));
        require(cuEventQuery(event) == CUDA_ERROR_NOT_READY, "batch event pending behind producer gate");
        require(std::all_of(output.value, output.value + extent, [](auto value) { return value == 0xb7; }), "batch consumer remains pending");
        gate.finish(); CONFORMANCE_CUDA(cuEventSynchronize(event));
        batch_expect({output.value, extent}, expected, "stream-order batch and owned descriptors");
    }
}
void batch_snapshot() {
    constexpr size_t extent = 173;
    for (bool runtime : {false, true}) for (bool single : {false, true})
    for (auto order : {CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL, CU_MEMCPY_SRC_ACCESS_ORDER_ANY}) {
        std::vector<unsigned char> input(extent);
        for (size_t i = 0; i < extent; ++i) input[i] = static_cast<unsigned char>(i * 53 + 37);
        const auto original = input;
        std::vector<unsigned char> expected(extent, 0xd9), actual(extent);
        std::copy_n(input.begin() + 11, 113, expected.begin() + 23);
        device_buffer destination(extent); batch_stream stream;
        CONFORMANCE_CUDA(cuMemsetD8(destination.value, 0xd9, extent));
        batch_gate gate(stream.value);
        CUdeviceptr to = destination.value + 23, from = batch_address(input.data() + 11);
        size_t bytes = 113, index = 0;
        auto attribute = batch_attributes(order);
        CONFORMANCE_CUDA(batch_submit(runtime, &to, &from, &bytes, 1, &attribute, &index, 1, stream.value, single));
        if (order == CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL) std::fill(input.begin(), input.end(), 0x6e);
        require(cuStreamQuery(stream.value) == CUDA_ERROR_NOT_READY, "attribute copy destination remains stream ordered");
        gate.finish();
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, extent));
        batch_expect(actual, expected, "attribute copy source access lifetime and guards");
        if (order == CU_MEMCPY_SRC_ACCESS_ORDER_ANY) batch_expect(input, original, "ANY source preservation");
    }
}
void batch_validation() {
    device_buffer source(64), destination(64); batch_stream stream;
    CONFORMANCE_CUDA(cuMemsetD8(source.value, 0x73, 64));
    for (bool runtime : {false, true}) for (unsigned invalid = 0; invalid < 13; ++invalid) {
        std::array<CUdeviceptr, 2> to{destination.value, destination.value + 32}, from{source.value, source.value + 32};
        std::array<size_t, 2> sizes{16, 16}, indices{0, 1};
        std::array<CUmemcpyAttributes, 2> attributes{batch_attributes(), batch_attributes()};
        auto* to_ptr = to.data(); auto* from_ptr = from.data(); auto* size_ptr = sizes.data();
        auto* attribute_ptr = attributes.data(); auto* index_ptr = indices.data();
        size_t count = 2, attribute_count = 2; CUstream lane = stream.value;
        switch (invalid) {
        case 0: to_ptr = nullptr; break;
        case 1: from_ptr = nullptr; break;
        case 2: size_ptr = nullptr; break;
        case 3: attribute_ptr = nullptr; break;
        case 4: index_ptr = nullptr; break;
        case 5: attribute_count = 0; break;
        case 6: indices[0] = 1; break;
        case 7: indices[1] = 0; break;
        case 8: indices[1] = count; break;
        case 9: attributes[0].srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_INVALID; break;
        case 10: attributes[0].flags = 0x80000000u; break;
        case 11: lane = nullptr; break;
        case 12: lane = CU_STREAM_LEGACY; break;
        }
        require(batch_submit(runtime, to_ptr, from_ptr, size_ptr, count, attribute_ptr, index_ptr, attribute_count, lane) == CUDA_ERROR_INVALID_VALUE,
                "batch invalid argument " + std::to_string(invalid));
        CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
        CONFORMANCE_CUDA(cuMemsetD8(destination.value, 0xa9, 64));
        auto repaired = batch_attributes(); size_t start = 0;
        CONFORMANCE_CUDA(batch_submit(runtime, to.data(), from.data(), sizes.data(), 2, &repaired, &start, 1, stream.value));
        CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
        std::array<unsigned char, 64> actual{}, expected{}; expected.fill(0xa9);
        std::fill_n(expected.begin(), 16, 0x73); std::fill_n(expected.begin() + 32, 16, 0x73);
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size()));
        batch_expect(actual, expected, "repaired batch after invalid arguments");
    }
}
void batch_single_streams() {
    constexpr size_t extent = 193;
    for (bool runtime : {false, true}) for (auto stream : {CUstream(nullptr), CU_STREAM_LEGACY, CU_STREAM_PER_THREAD}) {
        device_buffer source(extent), destination(extent); batch_host output(extent);
        std::vector<unsigned char> input(extent), expected(extent, 0xc7);
        for (size_t i = 0; i < extent; ++i) input[i] = static_cast<unsigned char>(i * 71 + 43);
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), extent));
        CONFORMANCE_CUDA(cuMemsetD8(destination.value, 0xc7, extent));
        CUdeviceptr to = destination.value + 23, from = source.value + 7; size_t bytes = 129, index = 0;
        auto attribute = batch_attributes();
        CONFORMANCE_CUDA(batch_submit(runtime, &to, &from, &bytes, 1, &attribute, &index, 1, stream, true));
        CONFORMANCE_CUDA(cuMemcpyDtoHAsync(output.value, destination.value, extent, stream));
        CONFORMANCE_CUDA(cuStreamSynchronize(stream));
        std::copy_n(input.begin() + 7, bytes, expected.begin() + 23);
        batch_expect({output.value, extent}, expected, "single attribute copy default/legacy/per-thread stream");
    }
}
void execute_memory_copy_batch() {
    if (batch_case == "transfers") batch_transfers();
    else if (batch_case == "ordering") batch_ordering();
    else if (batch_case == "snapshot") batch_snapshot();
    else if (batch_case == "validation") batch_validation();
    else if (batch_case == "single_streams") batch_single_streams();
    else throw std::runtime_error("unknown memory_copy_batch case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::batch_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_memory_copy_batch);
}
