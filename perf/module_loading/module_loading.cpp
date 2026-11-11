#include "perf_fixtures.hpp"
#include <array>

namespace cudavm {

perf_result execute_module_loading(const perf_options& options) {
    perf_require(options.case_name == "module" || options.case_name == "library", "unknown module loading case");
    const bool library = options.case_name == "library";
    const auto device = select_perf_device(options.backend, require_perf_execution);
    perf_context context(device.value);
    perf_stream stream;
    const std::array<uint32_t, 3> initial{0x93bd175a, 0, 0x1943b6ef};
    perf_device_buffer output(sizeof(initial));
    CUdeviceptr pointer = output.value + sizeof(uint32_t);
    uint32_t input = 0x8b2f394d;
    void* arguments[]{&pointer, &input};
    const auto use_function = [&](CUfunction function, bool verify) {
        if (!function) throw std::runtime_error("loader resolves the named function");
        if (!verify) return;
        PERF_CUDA(cuMemcpyHtoD(output.value, initial.data(), sizeof(initial)));
        PERF_CUDA(cuCtxSynchronize());
        PERF_CUDA(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, stream.value, arguments, nullptr));
        stream.synchronize();
        std::array<uint32_t, 3> actual{};
        PERF_CUDA(cuMemcpyDtoH(actual.data(), output.value, sizeof(actual)));
        const uint32_t expected = (input ^ 0x5a17c9e3u) + 17u;
        perf_require(actual[0] == initial[0] && actual[1] == expected && actual[2] == initial[2],
                     "freshly loaded function executes the independent ABI/value oracle and preserves guards");
    };
    const auto transaction = [&](bool verify) {
        if (library) {
            perf_library owner(perf_fixtures.ptx.data());
            use_function(owner.function("perf_loaded_value"), verify);
            owner.close();
        } else {
            perf_module owner(perf_fixtures.ptx.data());
            use_function(owner.function("perf_loaded_value"), verify);
            owner.close();
        }
    };
    transaction(true);
    auto result = measure_perf(options, [&] { transaction(false); });
    transaction(true);
    result.parameters = {{"api", "driver"}, {"loader", library ? "library" : "module"},
        {"boundary", library ? "load_get_kernel_get_function_unload" : "load_get_function_unload"},
        {"input", "embedded_ptx"}, {"input_bytes", std::to_string(perf_fixtures.ptx.size())},
        {"ptx_version", "6.0"}, {"architecture", "sm_50"}, {"completion", "loader_calls_without_dispatch"},
        {"state", "reused_context_and_input_new_owner_per_iteration_warm_process"},
        {"verification", "one_guarded_u32_function_with_runtime_parameter_before_and_after_sampling"},
        {"device_selection", "first_filtered_endpoint_passing_independent_execution_probe"}};
    device.describe(result);
    output.close();
    stream.close();
    context.close();
    return result;
}

}

int main(int argc, char** argv) {
    return cudavm::run_perf(argc, argv, "module_loading", cudavm::execute_module_loading);
}
