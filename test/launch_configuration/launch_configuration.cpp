#include "test_fixtures.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>

namespace cudavm {
struct launch_configuration_gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false, expired = false;
    static void CUDA_CB hold(void* pointer) {
        auto& gate = *static_cast<launch_configuration_gate*>(pointer);
        std::unique_lock lock(gate.mutex); gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(15), [&] { return gate.released; })) gate.expired = true;
    }
    void await() {
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "launch stream reaches snapshot gate");
    }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
    ~launch_configuration_gate() { release(); cuCtxSynchronize(); }
};
struct configured_stream {
    CUstream value{};
    configured_stream() { TEST_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~configured_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
};
void configured_runtime(cudaError_t status) {
    require(status == cudaSuccess, std::string("Runtime extended launch: ") + cudaGetErrorString(status));
}
void configured_launch(CUfunction function, CUkernel kernel, bool runtime, bool packed, bool ignored_pointer) {
    constexpr unsigned total = 192, blocks = 12, guards = 4, words = total + guards * 2 + 7;
    constexpr uint32_t guard = 0xa1b2c3d4;
    std::vector<uint32_t> input(words, guard), expected(words, guard), actual(words);
    for (unsigned i = 0; i < total; ++i) input[guards + i] = 0xfffffff0u + i * 0x1020305u;
    device_buffer source(words * 4), output(words * 4);
    TEST_CUDA(cuMemcpyHtoD(source.value, input.data(), words * 4));
    TEST_CUDA(cuMemcpyHtoD(output.value, expected.data(), words * 4));
    configured_stream stream;
    launch_configuration_gate gate;
    auto from = source.value + guards * 4, to = output.value + guards * 4;
    unsigned count = total - 5;
    void* args[]{&to, &from, &count};
    std::array<unsigned char, 20> packet;
    std::memcpy(packet.data(), &to, 8); std::memcpy(packet.data() + 8, &from, 8); std::memcpy(packet.data() + 16, &count, 4);
    size_t packet_bytes = packet.size();
    void* extra[]{CU_LAUNCH_PARAM_BUFFER_POINTER, packet.data(), CU_LAUNCH_PARAM_BUFFER_SIZE, &packet_bytes, CU_LAUNCH_PARAM_END};
    CUlaunchAttribute attributes[4]{};
    attributes[0].id = CU_LAUNCH_ATTRIBUTE_IGNORE;
    attributes[1].id = attributes[2].id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
    attributes[1].value.cooperative = 1; attributes[2].value.cooperative = 0;
    attributes[3].id = CU_LAUNCH_ATTRIBUTE_PRIORITY; attributes[3].value.priority = -1;
    CUlaunchConfig config{2, 3, 2, 4, 2, 2, 0, stream.value, attributes, 4};
    cudaLaunchAttribute runtime_attributes[4]{};
    runtime_attributes[0].id = cudaLaunchAttributeIgnore;
    runtime_attributes[1].id = runtime_attributes[2].id = cudaLaunchAttributeCooperative;
    runtime_attributes[1].val.cooperative = 1; runtime_attributes[2].val.cooperative = 0;
    runtime_attributes[3].id = cudaLaunchAttributePriority; runtime_attributes[3].val.priority = -1;
    cudaLaunchConfig_t runtime_config{};
    runtime_config.gridDim = dim3(2, 3, 2); runtime_config.blockDim = dim3(4, 2, 2);
    runtime_config.stream = reinterpret_cast<cudaStream_t>(stream.value);
    runtime_config.attrs = runtime_attributes; runtime_config.numAttrs = 4;
    if (ignored_pointer) {
        config.numAttrs = 0; config.attrs = reinterpret_cast<CUlaunchAttribute*>(uintptr_t(1));
        runtime_config.numAttrs = 0; runtime_config.attrs = reinterpret_cast<cudaLaunchAttribute*>(uintptr_t(1));
    }
    TEST_CUDA(cuLaunchHostFunc(stream.value, launch_configuration_gate::hold, &gate)); gate.await();
    if (runtime) configured_runtime(cudaLaunchKernelExC(&runtime_config, reinterpret_cast<const void*>(kernel), args));
    else TEST_CUDA(cuLaunchKernelEx(&config, function, packed ? nullptr : args, packed ? extra : nullptr));
    require(cuStreamQuery(stream.value) == CUDA_ERROR_NOT_READY, "accepted extended launch remains pending behind gate");
    config.gridDimX = config.blockDimY = 0; config.numAttrs = 1; config.attrs = attributes;
    runtime_config.gridDim.x = runtime_config.blockDim.y = 0; runtime_config.numAttrs = 1; runtime_config.attrs = runtime_attributes;
    attributes[0].id = static_cast<CUlaunchAttributeID>(999);
    runtime_attributes[0].id = static_cast<cudaLaunchAttributeID>(999);
    count = 0; to = from = 0; packet.fill(0); packet_bytes = 0; args[0] = args[1] = args[2] = nullptr;
    gate.release(); TEST_CUDA(cuStreamSynchronize(stream.value));
    require(!gate.expired, "snapshot gate released before deadline");
    for (unsigned i = 0; i < total - 5; ++i) expected[guards + i] = input[guards + i] + i + blocks;
    TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, words * 4));
    expect_words(actual, expected, "extended launch preserves accepted geometry, attributes and arguments");
    TEST_CUDA(cuMemcpyDtoH(actual.data(), source.value, words * 4));
    expect_words(actual, input, "extended launch source and guard preservation");
}
void configured_validation(CUfunction function, CUkernel kernel) {
    constexpr uint32_t guard = 0xa1b2c3d4;
    const std::array<uint32_t, 6> expected{guard, guard, guard, guard, guard, guard};
    std::array<uint32_t, 6> actual{};
    device_buffer source(sizeof(expected)), output(sizeof(expected));
    TEST_CUDA(cuMemcpyHtoD(source.value, expected.data(), sizeof(expected)));
    TEST_CUDA(cuMemcpyHtoD(output.value, expected.data(), sizeof(expected)));
    auto from = source.value + 4, to = output.value + 4; unsigned count = 1;
    void* args[]{&to, &from, &count};
    CUlaunchConfig config{1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, 0};
    auto rejected = [&](CUresult status) {
        require(status == CUDA_ERROR_INVALID_VALUE, "invalid extended descriptor rejected with exact Driver status");
        TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, sizeof(actual)));
        expect_words(actual, expected, "invalid extended launch preserves all output");
    };
    rejected(cuLaunchKernelEx(nullptr, function, args, nullptr));
    config.numAttrs = 1; rejected(cuLaunchKernelEx(&config, function, args, nullptr)); config.numAttrs = 0;
    for (unsigned axis = 0; axis < 3; ++axis) {
        unsigned* grids[]{&config.gridDimX, &config.gridDimY, &config.gridDimZ};
        unsigned* blocks[]{&config.blockDimX, &config.blockDimY, &config.blockDimZ};
        *grids[axis] = 0; rejected(cuLaunchKernelEx(&config, function, args, nullptr)); *grids[axis] = 1;
        *blocks[axis] = 0; rejected(cuLaunchKernelEx(&config, function, args, nullptr)); *blocks[axis] = 1;
    }
    int maximum = 0; TEST_CUDA(cuFuncGetAttribute(&maximum, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function));
    config.blockDimX = unsigned(maximum) + 1; rejected(cuLaunchKernelEx(&config, function, args, nullptr)); config.blockDimX = 1;
    config.sharedMemBytes = std::numeric_limits<unsigned>::max(); rejected(cuLaunchKernelEx(&config, function, args, nullptr)); config.sharedMemBytes = 0;
    rejected(cuLaunchKernelEx(&config, function, nullptr, nullptr));
    CUlaunchAttribute attribute{}; config.numAttrs = 1; config.attrs = &attribute;
    attribute.id = static_cast<CUlaunchAttributeID>(999); rejected(cuLaunchKernelEx(&config, function, args, nullptr));
    attribute.id = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN; attribute.value.memSyncDomain = static_cast<CUlaunchMemSyncDomain>(2);
    rejected(cuLaunchKernelEx(&config, function, args, nullptr));
    if (kernel) {
        cudaLaunchConfig_t runtime{}; runtime.gridDim = runtime.blockDim = dim3(1);
        const void* token = reinterpret_cast<const void*>(kernel);
        require(cudaLaunchKernelExC(nullptr, token, args) == cudaErrorInvalidValue && cudaGetLastError() == cudaErrorInvalidValue, "Runtime null descriptor validation");
        runtime.blockDim.x = 0;
        require(cudaLaunchKernelExC(&runtime, token, args) == cudaErrorInvalidConfiguration && cudaGetLastError() == cudaErrorInvalidConfiguration, "Runtime invalid geometry translation");
        runtime.blockDim.x = 1; runtime.numAttrs = 1;
        require(cudaLaunchKernelExC(&runtime, token, args) == cudaErrorInvalidValue && cudaGetLastError() == cudaErrorInvalidValue, "Runtime missing attributes validation");
        TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, sizeof(actual)));
        expect_words(actual, expected, "rejected Runtime descriptors preserve all output");
    }
    config.numAttrs = 0; config.attrs = nullptr;
    TEST_CUDA(cuLaunchKernelEx(&config, function, args, nullptr));
    TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, sizeof(actual)));
    auto repaired = expected; repaired[1] = guard + 1;
    expect_words(actual, repaired, "legal extended launch after rejected configurations");
}
void execute_launch_configuration(const loaded_program& program) {
    const auto function = program.function("cubin_indexed");
    const auto kernel = program.library_kernel("cubin_indexed");
    for (bool ignored : {false, true}) {
        configured_launch(function, kernel, false, false, ignored);
        configured_launch(function, kernel, false, true, ignored);
        if (kernel) configured_launch(function, kernel, true, false, ignored);
    }
    configured_validation(function, kernel);
}
}
int main(int argc, char** argv) {
    return cudavm::run_test(argc, argv, cudavm::fixtures, cudavm::execute_launch_configuration);
}
