#include "test_framework.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <exception>
#include <thread>
#include <utility>

namespace cudavm {
void discovery_runtime(cudaError_t result, const char* operation) {
    require(result == cudaSuccess, std::string(operation) + ": " + cudaGetErrorString(result));
}
#define DISCOVERY_RUNTIME(call) ::cudavm::discovery_runtime((call), #call)
template<class T> struct discovery_output {
    std::array<uint64_t, 2> before{0xd37ca619e80542bfull, 0x12f9a4306b7ed85cull};
    T value{};
    std::array<unsigned char, 16> after;
    discovery_output() { after.fill(0xd3); }
    T get() const {
        require(before == std::array<uint64_t, 2>{0xd37ca619e80542bfull, 0x12f9a4306b7ed85cull}
            && std::all_of(after.begin(), after.end(), [](auto byte) { return byte == 0xd3; }), "device query writes only its output extent");
        return value;
    }
};
CUcontext discovery_current() {
    discovery_output<CUcontext> output;
    CONFORMANCE_CUDA(cuCtxGetCurrent(&output.value));
    return output.get();
}
int discovery_count() {
    discovery_output<int> driver, runtime;
    CONFORMANCE_CUDA(cuDeviceGetCount(&driver.value));
    DISCOVERY_RUNTIME(cudaGetDeviceCount(&runtime.value));
    require(driver.get() > 0 && runtime.get() == driver.get(), "Driver and Runtime enumerate the same nonempty inventory");
    return driver.get();
}
void discovery_enumeration() {
    const auto context = discovery_current();
    const auto count = discovery_count();
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        CONFORMANCE_CUDA(cuInit(0));
        require(discovery_count() == count, "repeated initialization preserves device count");
        for (int ordinal = 0; ordinal < count; ++ordinal) {
            discovery_output<CUdevice> output;
            CONFORMANCE_CUDA(cuDeviceGet(&output.value, ordinal));
            require(output.get() == ordinal, "dense logical device ordinals round-trip");
        }
    }
    require(discovery_current() == context, "enumeration preserves current context");
}
void discovery_versions() {
    discovery_output<int> driver, runtime_driver, runtime, repeated;
    CONFORMANCE_CUDA(cuDriverGetVersion(&driver.value));
    DISCOVERY_RUNTIME(cudaDriverGetVersion(&runtime_driver.value));
    DISCOVERY_RUNTIME(cudaRuntimeGetVersion(&runtime.value));
    CONFORMANCE_CUDA(cuDriverGetVersion(&repeated.value));
    require(driver.get() >= 1000 && driver.get() % 10 == 0 && repeated.get() == driver.get()
        && runtime_driver.get() == driver.get(), "stable CUDA-encoded Driver version agrees across facades");
    require(runtime.get() == CUDART_VERSION, "Runtime reports its pinned public version");
}
std::string discovery_name(CUdevice device, size_t length) {
    std::vector<char> bytes(length + 32, char(0x6b));
    CONFORMANCE_CUDA(cuDeviceGetName(bytes.data() + 16, static_cast<int>(length), device));
    for (size_t i = 0; i < bytes.size(); ++i)
        if (i < 16 || i >= 16 + length) require(bytes[i] == char(0x6b), "device name respects caller buffer length");
    const auto end = std::find(bytes.begin() + 16, bytes.begin() + 16 + length, '\0');
    require(end != bytes.begin() + 16 + length, "device name terminates within caller buffer");
    return {bytes.begin() + 16, end};
}
void discovery_names() {
    for (int device = 0, count = discovery_count(); device < count; ++device) {
        const auto full = discovery_name(device, 256);
        require(!full.empty() && full == discovery_name(device, 256), "stable nonempty device name");
        for (const auto length : {size_t{1}, size_t{2}, size_t{7}, full.size(), full.size() + 1}) {
            const auto short_name = discovery_name(device, length);
            require(short_name.size() < length, "short device name fits requested extent");
            if (length == 1) require(short_name.empty(), "one-byte device name contains only its terminator");
            if (length > full.size()) require(short_name == full, "exact full-name buffer retains complete name");
        }
    }
}
int discovery_attribute(CUdevice device, CUdevice_attribute driver_attribute, cudaDeviceAttr runtime_attribute) {
    discovery_output<int> driver, runtime;
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&driver.value, driver_attribute, device));
    DISCOVERY_RUNTIME(cudaDeviceGetAttribute(&runtime.value, runtime_attribute, device));
    require(driver.get() == runtime.get(), "Driver and Runtime attribute values agree");
    return driver.get();
}
void discovery_properties() {
    for (int device = 0, count = discovery_count(); device < count; ++device) {
        discovery_output<cudaDeviceProp> output;
        DISCOVERY_RUNTIME(cudaGetDeviceProperties(&output.value, device));
        const auto properties = output.get();
        const auto name_end = std::find(std::begin(properties.name), std::end(properties.name), '\0');
        require(name_end != std::end(properties.name) && std::string(properties.name) == discovery_name(device, 256), "aggregate and individual device names agree");
        discovery_output<size_t> memory;
        CONFORMANCE_CUDA(cuDeviceTotalMem(&memory.value, device));
        require(memory.get() > 0 && properties.totalGlobalMem == memory.get(), "aggregate and individual total memory agree");
        require(properties.maxThreadsPerBlock > 0 && properties.warpSize == 32, "nonzero dispatch size and CUDA logical warp");
        require(properties.maxThreadsPerBlock == discovery_attribute(device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, cudaDevAttrMaxThreadsPerBlock), "aggregate block limit");
        require(properties.warpSize == discovery_attribute(device, CU_DEVICE_ATTRIBUTE_WARP_SIZE, cudaDevAttrWarpSize), "aggregate warp size");
        const std::array<CUdevice_attribute, 3> block{CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z};
        const std::array<cudaDeviceAttr, 3> runtime_block{cudaDevAttrMaxBlockDimX, cudaDevAttrMaxBlockDimY, cudaDevAttrMaxBlockDimZ};
        const std::array<CUdevice_attribute, 3> grid{CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z};
        const std::array<cudaDeviceAttr, 3> runtime_grid{cudaDevAttrMaxGridDimX, cudaDevAttrMaxGridDimY, cudaDevAttrMaxGridDimZ};
        for (size_t axis = 0; axis < 3; ++axis) {
            require(properties.maxThreadsDim[axis] > 0 && properties.maxThreadsDim[axis] == discovery_attribute(device, block[axis], runtime_block[axis]), "aggregate per-axis block limits");
            require(properties.maxGridSize[axis] > 0 && properties.maxGridSize[axis] == discovery_attribute(device, grid[axis], runtime_grid[axis]), "aggregate per-axis grid limits");
        }
        discovery_output<int> major, minor;
        CONFORMANCE_CUDA(cuDeviceComputeCapability(&major.value, &minor.value, device));
        require(major.get() == properties.major && minor.get() == properties.minor, "legacy compute capability agrees with aggregate");
    }
}
void discovery_validation() {
    const auto count = discovery_count();
    discovery_output<CUdevice> device;
    require(cuInit(1) == CUDA_ERROR_INVALID_VALUE, "initialization rejects nonzero flags");
    require(cuDeviceGetCount(nullptr) == CUDA_ERROR_INVALID_VALUE && cudaGetDeviceCount(nullptr) == cudaErrorInvalidValue, "null count outputs");
    require(cuDeviceGet(nullptr, 0) == CUDA_ERROR_INVALID_VALUE, "null ordinal output");
    for (int invalid : {-1, count}) {
        require(cuDeviceGet(&device.value, invalid) == CUDA_ERROR_INVALID_DEVICE, "invalid logical ordinal");
        char name[32]{}; size_t memory{}; int attribute{}; cudaDeviceProp properties{};
        require(cuDeviceGetName(name, sizeof(name), invalid) == CUDA_ERROR_INVALID_DEVICE, "invalid name device");
        require(cuDeviceTotalMem(&memory, invalid) == CUDA_ERROR_INVALID_DEVICE, "invalid memory device");
        require(cuDeviceGetAttribute(&attribute, CU_DEVICE_ATTRIBUTE_WARP_SIZE, invalid) == CUDA_ERROR_INVALID_DEVICE, "invalid attribute device");
        require(cudaGetDeviceProperties(&properties, invalid) == cudaErrorInvalidDevice, "invalid Runtime properties device");
        require(cudaDeviceGetAttribute(&attribute, cudaDevAttrWarpSize, invalid) == cudaErrorInvalidDevice, "invalid Runtime attribute device");
    }
    device.get();
    char name[8]{};
    require(cuDeviceGetName(nullptr, 8, 0) == CUDA_ERROR_INVALID_VALUE && cuDeviceGetName(name, 0, 0) == CUDA_ERROR_INVALID_VALUE
        && cuDeviceGetName(name, -1, 0) == CUDA_ERROR_INVALID_VALUE, "invalid name buffer arguments");
    require(cuDeviceTotalMem(nullptr, 0) == CUDA_ERROR_INVALID_VALUE && cudaGetDeviceProperties(nullptr, 0) == cudaErrorInvalidValue, "null property outputs");
    require(cuDeviceGetAttribute(nullptr, CU_DEVICE_ATTRIBUTE_WARP_SIZE, 0) == CUDA_ERROR_INVALID_VALUE
        && cudaDeviceGetAttribute(nullptr, cudaDevAttrWarpSize, 0) == cudaErrorInvalidValue, "null attribute outputs");
    require(cuDriverGetVersion(nullptr) == CUDA_ERROR_INVALID_VALUE && cudaDriverGetVersion(nullptr) == cudaErrorInvalidValue
        && cudaRuntimeGetVersion(nullptr) == cudaErrorInvalidValue, "null version outputs");
    require(discovery_count() == count, "rejected discovery queries preserve inventory");
    require(cudaGetLastError() == cudaErrorInvalidValue, "clear expected Runtime validation error");
}
void discovery_state() {
    const auto parent = discovery_current();
    std::exception_ptr failure;
    std::jthread worker([&] {
        try {
            require(discovery_current() == nullptr, "fresh query worker starts unbound");
            const auto count = discovery_count();
            for (int device = 0; device < count; ++device) {
                unsigned flags{}; int active = -1;
                CONFORMANCE_CUDA(cuDevicePrimaryCtxGetState(device, &flags, &active));
                require(active == 0, "query profile starts without active primary context");
            }
            discovery_enumeration(); discovery_versions(); discovery_names(); discovery_properties();
            require(discovery_current() == nullptr, "queries leave fresh worker unbound");
            for (int device = 0; device < count; ++device) {
                unsigned flags{}; int active = -1;
                CONFORMANCE_CUDA(cuDevicePrimaryCtxGetState(device, &flags, &active));
                require(active == 0, "queries do not activate primary contexts");
            }
        } catch (...) { failure = std::current_exception(); }
    });
    worker.join();
    require(discovery_current() == parent, "query worker preserves parent current context");
    if (failure) std::rethrow_exception(failure);
}
std::string_view device_discovery_case = "all";
void execute_device_discovery() {
    const auto original = discovery_current();
    const std::array<std::pair<std::string_view, void (*)()>, 6> cases{{
        {"enumeration", discovery_enumeration}, {"versions", discovery_versions}, {"names", discovery_names},
        {"properties", discovery_properties}, {"validation", discovery_validation}, {"state", discovery_state}}};
    bool selected = false;
    for (const auto& [name, execute] : cases) if (device_discovery_case == "all" || device_discovery_case == name) {
        selected = true; execute(); require(discovery_current() == original, "discovery case preserves current context");
    }
    require(selected, "unknown device_discovery case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::device_discovery_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_device_discovery);
}
