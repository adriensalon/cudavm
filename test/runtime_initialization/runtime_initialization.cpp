#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <exception>
#include <iostream>
#include <thread>
#include <utility>

namespace cudavm {
void check_initialization_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define INITIALIZATION_RUNTIME(call) ::cudavm::check_initialization_runtime((call), #call)
CUcontext initialization_current() {
    CUcontext current{};
    CONFORMANCE_CUDA(cuCtxGetCurrent(&current));
    return current;
}
void initialization_primary_state(int device, unsigned flags, bool active) {
    unsigned actual_flags = ~0u;
    int actual_active = -1;
    CONFORMANCE_CUDA(cuDevicePrimaryCtxGetState(device, &actual_flags, &actual_active));
    require(actual_flags == flags && actual_active == int(active), "Runtime primary flags/active state");
}
void initialization_fresh_thread(void (*execute)(int)) {
    CUdevice device{};
    CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    require(device == 0, "implicit default-device profile requires CUDAVM_BACKENDS filtering to logical device zero");
    const auto parent = initialization_current();
    std::exception_ptr error;
    std::jthread worker([&] {
        struct cleanup {
            bool active = true;
            ~cleanup() { if (active) { cudaDeviceReset(); cuCtxSetCurrent(nullptr); } }
        } on_failure;
        try {
            require(initialization_current() == nullptr, "fresh worker has no Driver current context");
            initialization_primary_state(device, 0, false);
            execute(device);
            INITIALIZATION_RUNTIME(cudaDeviceReset());
            CONFORMANCE_CUDA(cuCtxSetCurrent(nullptr));
            require(initialization_current() == nullptr, "initialization cleanup leaves worker unbound");
            initialization_primary_state(device, 0, false);
            require(cuDevicePrimaryCtxRelease(device) == CUDA_ERROR_INVALID_CONTEXT, "Runtime cleanup balances repeated initialization retains");
            on_failure.active = false;
        } catch (...) { error = std::current_exception(); }
    });
    worker.join();
    require(initialization_current() == parent, "worker Runtime initialization preserves parent explicit context");
    if (error) std::rethrow_exception(error);
}
void initialization_queries(int device) {
    int count = 0, selected = -1;
    INITIALIZATION_RUNTIME(cudaGetDeviceCount(&count));
    INITIALIZATION_RUNTIME(cudaGetDevice(&selected));
    require(count > 0 && selected == device, "initial Runtime device selection");
    cudaDeviceProp properties{};
    INITIALIZATION_RUNTIME(cudaGetDeviceProperties(&properties, selected));
    int runtime_warp = 0, driver_warp = 0;
    INITIALIZATION_RUNTIME(cudaDeviceGetAttribute(&runtime_warp, cudaDevAttrWarpSize, selected));
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&driver_warp, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device));
    require(properties.name[0] && properties.warpSize == driver_warp && runtime_warp == driver_warp, "query-only device properties agree");
    require(initialization_current() == nullptr, "query-only Runtime calls do not bind a context");
    initialization_primary_state(device, 0, false);
}
struct initialization_driver_retain {
    int device;
    CUcontext value{};
    explicit initialization_driver_retain(int ordinal) : device(ordinal) { CONFORMANCE_CUDA(cuDevicePrimaryCtxRetain(&value, device)); }
    ~initialization_driver_retain() { if (value) cuDevicePrimaryCtxRelease(device); }
    void release() { CONFORMANCE_CUDA(cuDevicePrimaryCtxRelease(device)); value = nullptr; }
};
void initialization_selection(int device) {
    CUcontext first{};
    for (unsigned repeat = 0; repeat < 4; ++repeat) {
        INITIALIZATION_RUNTIME(cudaSetDevice(device));
        const auto current = initialization_current();
        require(current != nullptr && (!first || first == current), "repeated Runtime selection binds the same primary");
        first = current;
        CUdevice driver_device{};
        int runtime_device = -1;
        CONFORMANCE_CUDA(cuCtxGetDevice(&driver_device));
        INITIALIZATION_RUNTIME(cudaGetDevice(&runtime_device));
        require(driver_device == device && runtime_device == device, "Runtime/Driver bound selection agrees");
        initialization_driver_retain retained(device);
        require(retained.value == current, "Runtime selection binds the Driver primary context");
        retained.release();
        initialization_primary_state(device, 0, true);
    }
}
void initialization_allocation_bytes(int device, unsigned flags) {
    CUcontext primary{};
    for (size_t size : {size_t(1), size_t(257), size_t(4093)}) {
        void* pointer{};
        INITIALIZATION_RUNTIME(cudaMalloc(&pointer, size));
        struct release_storage { void* pointer; ~release_storage() { if (pointer) cudaFree(pointer); } } cleanup{pointer};
        const auto current = initialization_current();
        require(current != nullptr && (!primary || primary == current), "implicit allocations retain one bound primary");
        primary = current;
        initialization_primary_state(device, flags, true);
        CUcontext owner{};
        CONFORMANCE_CUDA(cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, reinterpret_cast<uintptr_t>(pointer)));
        require(owner == current, "Runtime allocation belongs to implicitly bound primary");
        INITIALIZATION_RUNTIME(cudaMemset(pointer, 0x5b, size));
        std::vector<unsigned char> actual(size + 32, 0xd3);
        INITIALIZATION_RUNTIME(cudaMemcpy(actual.data() + 16, pointer, size, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < actual.size(); ++i)
            require(actual[i] == (i >= 16 && i < size + 16 ? 0x5b : 0xd3), "implicit allocation payload and readback guards");
        CONFORMANCE_CUDA(cuMemFree(reinterpret_cast<uintptr_t>(pointer))); cleanup.pointer = nullptr;
    }
}
void initialization_allocation(int device) { initialization_allocation_bytes(device, 0); }
void initialization_init_device(int device) {
    INITIALIZATION_RUNTIME(cudaInitDevice(device, cudaDeviceScheduleYield, cudaInitDeviceFlagsAreValid));
    initialization_primary_state(device, CU_CTX_SCHED_YIELD, true);
    require(initialization_current() == nullptr, "cudaInitDevice activates without binding");
    int selected = -1;
    INITIALIZATION_RUNTIME(cudaGetDevice(&selected));
    require(selected == device && initialization_current() == nullptr, "cudaInitDevice preserves initial Runtime selection and null current context");
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        INITIALIZATION_RUNTIME(cudaInitDevice(device, 0x80000000u, 0));
        initialization_primary_state(device, CU_CTX_SCHED_YIELD, true);
        require(initialization_current() == nullptr, "repeated initialization remains unbound and ignores invalid deviceFlags without flags-valid bit");
    }
    initialization_allocation_bytes(device, CU_CTX_SCHED_YIELD);
}
void initialization_validation(int device) {
    require(cudaGetDevice(nullptr) == cudaErrorInvalidValue, "null initial Runtime device output");
    require(cudaGetDeviceCount(nullptr) == cudaErrorInvalidValue, "null initial Runtime count output");
    require(cudaSetDevice(-1) == cudaErrorInvalidDevice, "invalid Runtime device selection");
    require(cudaInitDevice(-1, 0, 0) == cudaErrorInvalidDevice, "invalid Runtime initialization device");
    require(cudaInitDevice(device, 3, cudaInitDeviceFlagsAreValid) == cudaErrorInvalidValue, "invalid explicit initialization scheduling mode");
    require(initialization_current() == nullptr, "invalid initialization leaves worker unbound");
    initialization_primary_state(device, 0, false);
}
std::string_view runtime_initialization_case = "all";
void execute_runtime_initialization() {
    const std::array<std::pair<std::string_view, void (*)(int)>, 5> cases{{
        {"queries", initialization_queries}, {"selection", initialization_selection}, {"allocation", initialization_allocation},
        {"init_device", initialization_init_device}, {"validation", initialization_validation}}};
    bool found = false;
    for (const auto& [name, execute] : cases) if (runtime_initialization_case == "all" || runtime_initialization_case == name) {
        found = true; std::cout << "runtime_initialization=" << name << std::endl; initialization_fresh_thread(execute);
    }
    require(found, "unknown runtime_initialization case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::runtime_initialization_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_runtime_initialization);
}
