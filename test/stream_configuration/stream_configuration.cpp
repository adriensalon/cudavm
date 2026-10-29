#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <iostream>
#include <limits>
#include <set>
#include <utility>

namespace cudavm {
void check_configuration_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define CONFIGURATION_RUNTIME(call) ::cudavm::check_configuration_runtime((call), #call)
struct configured_stream {
    CUstream value{};
    bool runtime;
    configured_stream(bool use_runtime, unsigned flags = CU_STREAM_NON_BLOCKING, int priority = 0) : runtime(use_runtime) {
        if (runtime) {
            cudaStream_t stream{};
            CONFIGURATION_RUNTIME(cudaStreamCreateWithPriority(&stream, flags, priority));
            value = reinterpret_cast<CUstream>(stream);
        } else CONFORMANCE_CUDA(cuStreamCreateWithPriority(&value, flags, priority));
    }
    ~configured_stream() { if (value) cuStreamDestroy(value); }
    cudaStream_t runtime_value() const { return reinterpret_cast<cudaStream_t>(value); }
    void destroy() {
        if (runtime) CONFORMANCE_CUDA(cuStreamDestroy(value));
        else CONFIGURATION_RUNTIME(cudaStreamDestroy(runtime_value()));
        value = nullptr;
    }
};
unsigned long long configuration_identity(const configured_stream& stream) {
    unsigned long long driver = 0, runtime = 0;
    CONFORMANCE_CUDA(cuStreamGetId(stream.value, &driver));
    CONFIGURATION_RUNTIME(cudaStreamGetId(stream.runtime_value(), &runtime));
    require(driver != 0 && driver == runtime, "Driver/Runtime stable stream identity");
    return driver;
}
void configuration_owner(const configured_stream& stream, CUcontext owner, CUdevice device) {
    CUcontext driver{}, extended{}, current{};
    CUgreenCtx green{};
    CUdevice driver_device{};
    int runtime_device = -1;
    CONFORMANCE_CUDA(cuStreamGetCtx(stream.value, &driver));
    CONFORMANCE_CUDA(cuStreamGetCtx_v2(stream.value, &extended, &green));
    CONFORMANCE_CUDA(cuStreamGetDevice(stream.value, &driver_device));
    CONFIGURATION_RUNTIME(cudaStreamGetDevice(stream.runtime_value(), &runtime_device));
    CONFORMANCE_CUDA(cuCtxGetCurrent(&current));
    require(driver == owner && extended == owner && !green && driver_device == device && runtime_device == device,
        "stream context/device ownership and regular context kind");
    require(current == owner, "stream queries preserve current context");
}
void stream_configuration_priority() {
    int least = 0, greatest = 0, runtime_least = 0, runtime_greatest = 0;
    CONFORMANCE_CUDA(cuCtxGetStreamPriorityRange(&least, &greatest));
    CONFIGURATION_RUNTIME(cudaDeviceGetStreamPriorityRange(&runtime_least, &runtime_greatest));
    require(least >= greatest && least == runtime_least && greatest == runtime_greatest, "shared numerical priority range");
    int bound = 0;
    CONFORMANCE_CUDA(cuCtxGetStreamPriorityRange(&bound, nullptr)); require(bound == least, "optional Driver greatest output");
    CONFORMANCE_CUDA(cuCtxGetStreamPriorityRange(nullptr, &bound)); require(bound == greatest, "optional Driver least output");
    CONFIGURATION_RUNTIME(cudaDeviceGetStreamPriorityRange(&bound, nullptr)); require(bound == least, "optional Runtime greatest output");
    CONFIGURATION_RUNTIME(cudaDeviceGetStreamPriorityRange(nullptr, &bound)); require(bound == greatest, "optional Runtime least output");
    for (bool runtime : {false, true}) for (unsigned flags : {0u, unsigned(CU_STREAM_NON_BLOCKING)})
    for (int priority : {0, least, greatest, std::numeric_limits<int>::min(), std::numeric_limits<int>::max()}) {
        configured_stream stream(runtime, flags, priority);
        int driver_priority = 0, runtime_priority = 0;
        unsigned driver_flags = ~0u, runtime_flags = ~0u;
        CONFORMANCE_CUDA(cuStreamGetPriority(stream.value, &driver_priority));
        CONFIGURATION_RUNTIME(cudaStreamGetPriority(stream.runtime_value(), &runtime_priority));
        CONFORMANCE_CUDA(cuStreamGetFlags(stream.value, &driver_flags));
        CONFIGURATION_RUNTIME(cudaStreamGetFlags(stream.runtime_value(), &runtime_flags));
        require(driver_priority == std::clamp(priority, greatest, least) && runtime_priority == driver_priority,
            "requested priority clamps at reported bounds without integer overflow");
        require(driver_flags == flags && runtime_flags == flags, "requested stream flags");
        stream.destroy();
    }
}
void stream_configuration_identity() {
    CUcontext owner{};
    CUdevice device{};
    CONFORMANCE_CUDA(cuCtxGetCurrent(&owner)); CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    std::set<unsigned long long> identities;
    for (unsigned generation = 0; generation < 4; ++generation) for (bool runtime : {false, true}) {
        configured_stream stream(runtime);
        const auto id = configuration_identity(stream);
        require(identities.insert(id).second, "stream IDs are unique across retired lifetimes");
        for (unsigned query = 0; query < 4; ++query) {
            require(configuration_identity(stream) == id, "repeated stream ID is stable");
            configuration_owner(stream, owner, device);
        }
        stream.destroy();
    }
}
void configuration_set_policy(const configured_stream& stream, CUsynchronizationPolicy policy, bool runtime) {
    if (runtime) {
        cudaStreamAttrValue value{}; value.syncPolicy = static_cast<cudaSynchronizationPolicy>(policy);
        CONFIGURATION_RUNTIME(cudaStreamSetAttribute(stream.runtime_value(), cudaStreamAttributeSynchronizationPolicy, &value));
    } else {
        CUstreamAttrValue value{}; value.syncPolicy = policy;
        CONFORMANCE_CUDA(cuStreamSetAttribute(stream.value, CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY, &value));
    }
}
void configuration_expect_policy(const configured_stream& stream, CUsynchronizationPolicy expected) {
    CUstreamAttrValue driver{}; cudaStreamAttrValue runtime{};
    CONFORMANCE_CUDA(cuStreamGetAttribute(stream.value, CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY, &driver));
    CONFIGURATION_RUNTIME(cudaStreamGetAttribute(stream.runtime_value(), cudaStreamAttributeSynchronizationPolicy, &runtime));
    require(driver.syncPolicy == expected && unsigned(runtime.syncPolicy) == unsigned(expected), "stream synchronization policy roundtrip");
}
void configuration_set_window(const configured_stream& stream, CUdeviceptr address, float ratio, bool runtime) {
    auto* base = reinterpret_cast<void*>(static_cast<uintptr_t>(address));
    if (runtime) {
        cudaStreamAttrValue value{};
        value.accessPolicyWindow = {base, 0, ratio, cudaAccessPropertyPersisting, cudaAccessPropertyStreaming};
        CONFIGURATION_RUNTIME(cudaStreamSetAttribute(stream.runtime_value(), cudaStreamAttributeAccessPolicyWindow, &value));
    } else {
        CUstreamAttrValue value{};
        value.accessPolicyWindow = {base, 0, ratio, CU_ACCESS_PROPERTY_PERSISTING, CU_ACCESS_PROPERTY_STREAMING};
        CONFORMANCE_CUDA(cuStreamSetAttribute(stream.value, CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &value));
    }
}
void configuration_expect_window(const configured_stream& stream, CUdeviceptr address, float ratio) {
    CUstreamAttrValue driver{}; cudaStreamAttrValue runtime{};
    CONFORMANCE_CUDA(cuStreamGetAttribute(stream.value, CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &driver));
    CONFIGURATION_RUNTIME(cudaStreamGetAttribute(stream.runtime_value(), cudaStreamAttributeAccessPolicyWindow, &runtime));
    const auto& d = driver.accessPolicyWindow;
    const auto& r = runtime.accessPolicyWindow;
    require(reinterpret_cast<uintptr_t>(d.base_ptr) == address && !d.num_bytes && d.hitRatio == ratio
        && d.hitProp == CU_ACCESS_PROPERTY_PERSISTING && d.missProp == CU_ACCESS_PROPERTY_STREAMING, "Driver empty access-policy fields");
    require(r.base_ptr == d.base_ptr && r.num_bytes == d.num_bytes && r.hitRatio == d.hitRatio
        && r.hitProp == cudaAccessPropertyPersisting && r.missProp == cudaAccessPropertyStreaming, "Runtime empty access-policy fields");
}
void stream_configuration_attributes() {
    device_buffer allocation(257);
    for (bool runtime : {false, true}) {
        configured_stream stream(runtime);
        for (auto policy : {CU_SYNC_POLICY_AUTO, CU_SYNC_POLICY_SPIN, CU_SYNC_POLICY_YIELD, CU_SYNC_POLICY_BLOCKING_SYNC}) {
            configuration_set_policy(stream, policy, runtime); configuration_expect_policy(stream, policy);
        }
        for (float ratio : {0.0f, 0.25f, 1.0f}) {
            configuration_set_window(stream, allocation.value + 17, ratio, !runtime);
            configuration_expect_window(stream, allocation.value + 17, ratio);
        }
        stream.destroy();
    }
}
void stream_configuration_copy() {
    device_buffer allocation(257);
    CUcontext owner{}; CUdevice device{};
    CONFORMANCE_CUDA(cuCtxGetCurrent(&owner)); CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    for (bool runtime : {false, true}) {
        configured_stream source(runtime, 0), destination(!runtime, CU_STREAM_NON_BLOCKING);
        const auto identity = configuration_identity(destination);
        configuration_set_policy(source, CU_SYNC_POLICY_YIELD, runtime);
        configuration_set_window(source, allocation.value + 17, 0.25f, !runtime);
        auto copy = [&] {
            if (runtime) CONFIGURATION_RUNTIME(cudaStreamCopyAttributes(destination.runtime_value(), source.runtime_value()));
            else CONFORMANCE_CUDA(cuStreamCopyAttributes(destination.value, source.value));
        };
        copy();
        configuration_expect_policy(destination, CU_SYNC_POLICY_YIELD);
        configuration_expect_window(destination, allocation.value + 17, 0.25f);
        configuration_set_policy(source, CU_SYNC_POLICY_SPIN, !runtime);
        configuration_set_window(source, allocation.value + 31, 1.0f, runtime);
        configuration_expect_policy(destination, CU_SYNC_POLICY_YIELD);
        configuration_expect_window(destination, allocation.value + 17, 0.25f);
        copy();
        configuration_expect_policy(destination, CU_SYNC_POLICY_SPIN);
        configuration_expect_window(destination, allocation.value + 31, 1.0f);
        CONFORMANCE_CUDA(cuStreamCopyAttributes(destination.value, destination.value));
        CONFIGURATION_RUNTIME(cudaStreamCopyAttributes(destination.runtime_value(), destination.runtime_value()));
        require(configuration_identity(destination) == identity, "attribute copies retain stream ID");
        unsigned flags = 0;
        CONFORMANCE_CUDA(cuStreamGetFlags(destination.value, &flags));
        require(flags == CU_STREAM_NON_BLOCKING, "attribute copies retain destination creation flags");
        configuration_owner(destination, owner, device);
        source.destroy(); destination.destroy();
    }
}
void stream_configuration_validation() {
    configured_stream stream(false);
    require(cuStreamGetFlags(stream.value, nullptr) == CUDA_ERROR_INVALID_VALUE, "null Driver flags output");
    require(cudaStreamGetFlags(stream.runtime_value(), nullptr) == cudaErrorInvalidValue, "null Runtime flags output");
    require(cuStreamGetPriority(stream.value, nullptr) == CUDA_ERROR_INVALID_VALUE, "null Driver priority output");
    require(cudaStreamGetPriority(stream.runtime_value(), nullptr) == cudaErrorInvalidValue, "null Runtime priority output");
    require(cuStreamGetId(stream.value, nullptr) == CUDA_ERROR_INVALID_VALUE, "null Driver ID output");
    require(cudaStreamGetId(stream.runtime_value(), nullptr) == cudaErrorInvalidValue, "null Runtime ID output");
    require(cuStreamGetDevice(stream.value, nullptr) == CUDA_ERROR_INVALID_VALUE, "null Driver device output");
    require(cudaStreamGetDevice(stream.runtime_value(), nullptr) == cudaErrorInvalidValue, "null Runtime device output");
    require(cuStreamGetCtx(stream.value, nullptr) == CUDA_ERROR_INVALID_VALUE, "null stream context output");
    configuration_set_policy(stream, CU_SYNC_POLICY_YIELD, false);
    CUstreamAttrValue driver{}; cudaStreamAttrValue runtime{};
    require(cuStreamGetAttribute(stream.value, CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY, nullptr) == CUDA_ERROR_INVALID_VALUE, "null Driver attribute output");
    require(cudaStreamGetAttribute(stream.runtime_value(), cudaStreamAttributeSynchronizationPolicy, nullptr) == cudaErrorInvalidValue, "null Runtime attribute output");
    require(cuStreamSetAttribute(stream.value, CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY, nullptr) == CUDA_ERROR_INVALID_VALUE, "null Driver attribute input");
    require(cudaStreamSetAttribute(stream.runtime_value(), cudaStreamAttributeSynchronizationPolicy, nullptr) == cudaErrorInvalidValue, "null Runtime attribute input");
    require(cuStreamGetAttribute(stream.value, static_cast<CUstreamAttrID>(0x7fffffff), &driver) == CUDA_ERROR_INVALID_VALUE, "invalid Driver attribute ID");
    require(cudaStreamGetAttribute(stream.runtime_value(), static_cast<cudaStreamAttrID>(0x7fffffff), &runtime) == cudaErrorInvalidValue, "invalid Runtime attribute ID");
    for (unsigned policy : {0u, 5u}) {
        driver.syncPolicy = static_cast<CUsynchronizationPolicy>(policy);
        runtime.syncPolicy = static_cast<cudaSynchronizationPolicy>(policy);
        require(cuStreamSetAttribute(stream.value, CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY, &driver) == CUDA_ERROR_INVALID_VALUE, "invalid Driver synchronization policy");
        require(cudaStreamSetAttribute(stream.runtime_value(), cudaStreamAttributeSynchronizationPolicy, &runtime) == cudaErrorInvalidValue, "invalid Runtime synchronization policy");
        configuration_expect_policy(stream, CU_SYNC_POLICY_YIELD);
    }
    stream.destroy();
}
std::string_view stream_configuration_case = "all";
void execute_stream_configuration() {
    const std::array<std::pair<std::string_view, void (*)()>, 5> cases{{
        {"priority", stream_configuration_priority}, {"identity", stream_configuration_identity},
        {"attributes", stream_configuration_attributes}, {"copy", stream_configuration_copy}, {"validation", stream_configuration_validation}}};
    bool found = false;
    for (const auto& [name, execute] : cases) if (stream_configuration_case == "all" || stream_configuration_case == name) {
        found = true; std::cout << "stream_configuration=" << name << std::endl; execute();
    }
    require(found, "unknown stream_configuration case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::stream_configuration_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_stream_configuration);
}
