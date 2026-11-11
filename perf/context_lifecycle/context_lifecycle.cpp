#include "perf_framework.hpp"

namespace cudavm {

void check_perf_explicit_context(CUdevice device) {
    CUcontext previous{}, current{};
    CUdevice selected{};
    PERF_CUDA(cuCtxGetCurrent(&previous));
    perf_context context(device);
    PERF_CUDA(cuCtxGetCurrent(&current));
    PERF_CUDA(cuCtxGetDevice(&selected));
    perf_require(current == context.value && selected == device, "new context is current on the selected endpoint");
    context.close();
    PERF_CUDA(cuCtxGetCurrent(&current));
    perf_require(current == previous, "context retirement restores the previous current context");
}

perf_result execute_context_lifecycle(const perf_options& options) {
    perf_require(options.case_name == "explicit" || options.case_name == "primary_retain", "unknown context lifecycle case");
    const auto device = select_perf_device(options.backend);
    perf_result result;
    if (options.case_name == "explicit") {
        check_perf_explicit_context(device.value);
        const auto operation = [&] {
            CUcontext context{};
            PERF_CUDA(cuCtxCreate(&context, nullptr, 0, device.value));
            PERF_CUDA(cuCtxDestroy(context));
        };
        result = measure_perf(options, operation);
        check_perf_explicit_context(device.value);
        result.parameters = {{"boundary", "cuCtxCreate_cuCtxDestroy"}, {"state", "fresh_explicit_context_warm_process"}};
    } else {
        CUcontext previous{}, current{}, retained{};
        unsigned flags{}; int active{};
        PERF_CUDA(cuDevicePrimaryCtxGetState(device.value, &flags, &active));
        perf_require(!active, "primary measurement starts without another owner");
        PERF_CUDA(cuCtxGetCurrent(&previous));
        perf_primary_context anchor(device.value);
        const auto operation = [&] {
            PERF_CUDA(cuDevicePrimaryCtxRetain(&retained, device.value));
            PERF_CUDA(cuDevicePrimaryCtxRelease(device.value));
        };
        const auto verify = [&] {
            PERF_CUDA(cuDevicePrimaryCtxGetState(device.value, &flags, &active));
            PERF_CUDA(cuCtxGetCurrent(&current));
            perf_require(active && retained == anchor.value, "primary retain reuses the live anchored context");
            perf_require(current == previous, "primary ownership does not change the current context");
        };
        operation();
        verify();
        result = measure_perf(options, operation);
        verify();
        anchor.close();
        PERF_CUDA(cuDevicePrimaryCtxGetState(device.value, &flags, &active));
        perf_require(!active, "all timed primary retains were balanced before the anchor was released");
        result.parameters = {{"boundary", "cuDevicePrimaryCtxRetain_cuDevicePrimaryCtxRelease"},
            {"state", "one_anchor_reference_held"}};
    }
    result.parameters.emplace_back("context_flags", "0");
    result.parameters.emplace_back("api", "driver");
    result.parameters.emplace_back("completion", "synchronous_lifecycle_calls");
    device.describe(result);
    return result;
}

}

int main(int argc, char** argv) {
    return cudavm::run_perf(argc, argv, "context_lifecycle", cudavm::execute_context_lifecycle);
}
