#include "test_fixtures.hpp"
#include <iostream>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <sstream>
namespace cudavm {
struct printf_capture {
    explicit printf_capture(const std::string& label) { std::cout << "BEGIN " << label << std::endl; }
    void finish() { std::cout << "END" << std::endl; }
};
void printf_runtime(cudaError_t status) {
    require(status == cudaSuccess, std::string("Runtime printf operation: ") + cudaGetErrorString(status));
}
void printf_lines(const std::string& text, std::vector<std::string> expected) {
    std::istringstream stream(text);
    std::vector<std::string> actual;
    for (std::string line; std::getline(stream, line);) actual.push_back(line);
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    require(actual == expected, "exact printf lines including duplicate counts; got: " + text);
}
void execute_device_printf(const loaded_program& program) {
    constexpr uint32_t guard = 0xa1b2c3d4;
    constexpr unsigned guards = 4, count = 17, capacity = 24;
    static bool configured = false;
    if (!configured) {
        constexpr size_t requested = 2 * 1024 * 1024;
        TEST_CUDA(cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, requested));
        size_t effective = 0;
        printf_runtime(cudaDeviceGetLimit(&effective, cudaLimitPrintfFifoSize));
        require(effective >= requested, "effective printf FIFO covers requested bytes");
        configured = true;
    }
    std::vector<uint32_t> input(capacity + guards * 2, guard), expected(input), actual(input.size());
    for (unsigned i = 0; i < count; ++i) input[guards + i] = 0xabcdu + i;
    device_buffer source(input.size() * 4), output(input.size() * 4);
    TEST_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
    auto from = source.value + guards * 4, to = output.value + guards * 4;
    CUstream stream{};
    TEST_CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    struct stream_guard { CUstream value; ~stream_guard() { cuStreamDestroy(value); } } cleanup{stream};
    auto records = program.function("printf_records");
    auto active = count;
    void* args[]{&to, &from, &active};
    for (unsigned mode = 0; mode < 4; ++mode) {
        std::fill(expected.begin(), expected.end(), guard);
        TEST_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
        for (unsigned i = 0; i < count; ++i) expected[guards + i] = 6;
        {
            printf_capture capture("records." + std::to_string(mode));
            if (mode == 3) {
                TEST_CUDA(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
                TEST_CUDA(cuLaunchKernel(records, 2, 1, 1, 13, 1, 1, 0, stream, args, nullptr));
                CUgraph graph{}; CUgraphExec executable{};
                TEST_CUDA(cuStreamEndCapture(stream, &graph));
                TEST_CUDA(cuGraphInstantiateWithFlags(&executable, graph, 0));
                TEST_CUDA(cuGraphDestroy(graph));
                struct graph_guard { CUgraphExec value; ~graph_guard() { cuGraphExecDestroy(value); } } release{executable};
                active = 0; // Captured parameters retain the original extent.
                for (unsigned repeat = 0; repeat < 2; ++repeat) {
                    TEST_CUDA(cuGraphLaunch(executable, stream));
                    printf_runtime(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
                }
                active = count;
            } else {
                const auto kernel = program.library_kernel("printf_records");
                if (mode == 1 && kernel)
                    printf_runtime(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(2), dim3(13), args, 0,
                        reinterpret_cast<cudaStream_t>(stream)));
                else TEST_CUDA(cuLaunchKernel(records, 2, 1, 1, 13, 1, 1, 0, stream, args, nullptr));
                if (mode == 0) TEST_CUDA(cuStreamSynchronize(stream));
                else if (mode == 1) printf_runtime(cudaDeviceSynchronize());
                else {
                    CUevent event{};
                    TEST_CUDA(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
                    const auto record_status = cuEventRecord(event, stream);
                    const auto sync_status = record_status == CUDA_SUCCESS ? cuEventSynchronize(event) : record_status;
                    cuEventDestroy(event);
                    TEST_CUDA(sync_status);
                }
            }
            // Observe output at the chosen completion point before a copy or a
            // second synchronization could conceal a missing output flush.
            capture.finish();
        }
        TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
        expect_words(actual, expected, "printf argument counts and inactive/guard preservation");
    }
    const char* formats[]{"mutable %u|%s\n", "updated %08x|%s\n"};
    device_buffer format(64);
    for (unsigned phase = 0; phase < 2; ++phase) {
        const auto text = std::string(formats[phase]);
        std::array<char, 64> bytes{};
        std::copy(text.begin(), text.end(), bytes.begin());
        TEST_CUDA(cuMemcpyHtoD(format.value, bytes.data(), bytes.size()));
        std::fill(expected.begin(), expected.end(), guard);
        TEST_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
        unsigned value = phase ? 9 : 7;
        void* format_args[]{&to, &format.value, &value};
        {
            printf_capture capture("formats." + std::to_string(phase));
            TEST_CUDA(cuLaunchKernel(program.function("printf_formats"), 1, 1, 1, 1, 1, 1, 0, stream, format_args, nullptr));
            TEST_CUDA(cuStreamSynchronize(stream));
            capture.finish();
        }
        expected[guards] = 2; expected[guards + 1] = 6; expected[guards + 2] = 0; expected[guards + 3] = ~0u;
        TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
        expect_words(actual, expected, "mutable printf formats, promotion and null/literal returns");
        std::array<char, 64> preserved{};
        TEST_CUDA(cuMemcpyDtoH(preserved.data(), format.value, preserved.size()));
        require(preserved == bytes, "format allocation remains byte-exact");
    }
    std::fill(expected.begin(), expected.end(), guard);
    TEST_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
    {
        printf_capture capture("arguments");
        void* argument_args[]{&to};
        TEST_CUDA(cuLaunchKernel(program.function("printf_arguments"), 1, 1, 1, 1, 1, 1, 0, stream, argument_args, nullptr));
        TEST_CUDA(cuCtxSynchronize());
        capture.finish();
    }
    expected[guards] = 32;
    TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
    expect_words(actual, expected, "printf 32-argument boundary and guards");
    TEST_CUDA(cuMemcpyDtoH(actual.data(), source.value, actual.size() * 4));
    expect_words(actual, input, "printf input preservation");
    size_t before = 0, after = 0;
    TEST_CUDA(cuCtxGetLimit(&before, CU_LIMIT_PRINTF_FIFO_SIZE));
    require(cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, before * 2) == CUDA_ERROR_INVALID_VALUE,
        "printf FIFO cannot be changed after output launch");
    printf_runtime(cudaDeviceGetLimit(&after, cudaLimitPrintfFifoSize));
    require(before == after, "rejected FIFO change preserves the effective limit");
}
std::vector<std::string> printf_expected(const std::string& label) {
    if (label.starts_with("records.")) {
        require(label.size() == 9 && label.back() >= '0' && label.back() <= '3', "record phase label");
        std::vector<std::string> lines;
        for (unsigned repeat = 0; repeat < (label.back() == '3' ? 2u : 1u); ++repeat)
        for (unsigned i = 0; i < 17; ++i) {
            std::ostringstream line;
            line << "record " << i << '|' << -int(i) - 7 << '|' << std::hex << std::setw(8) << std::setfill('0')
                << 0xabcdu + i << "|-1234567890123|1.25|ok" << char('A' + i % 26) << "|%";
            lines.push_back(line.str());
        }
        return lines;
    }
    if (label == "formats.0" || label == "formats.1") return {
        label == "formats.0" ? "mutable 7|captured" : "updated 00000009|captured",
        "promoted -19|123|Q|3.50|-23|1234", "literal %"};
    require(label == "arguments", "known output phase");
    std::string line = "arguments";
    for (unsigned i = 0; i < 32; ++i) line += " " + std::to_string(i);
    return {line};
}
void verify_printf_output(std::string text) {
    text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
    std::istringstream stream(text);
    std::string label, segment;
    std::vector<std::string> phases;
    for (std::string line; std::getline(stream, line);) {
        if (line.starts_with("BEGIN ")) {
            require(label.empty(), "output phases do not overlap");
            label = line.substr(6); segment.clear();
        } else if (line == "END") {
            require(!label.empty(), "output phase has a begin marker");
            printf_lines(segment, printf_expected(label));
            phases.push_back(label); label.clear();
        } else if (!label.empty()) segment += line + '\n';
        else require(!line.starts_with("record ") && !line.starts_with("promoted ")
            && !line.starts_with("literal ") && !line.starts_with("mutable ")
            && !line.starts_with("updated ") && !line.starts_with("arguments "),
            "device output must arrive before its completion marker");
    }
    const std::vector<std::string> one{"records.0", "records.1", "records.2", "records.3", "formats.0", "formats.1", "arguments"};
    auto expected = one; expected.insert(expected.end(), one.begin(), one.end());
    require(label.empty() && phases == expected, "both loaders complete every output phase exactly once");
}
}
int main(int argc, char** argv) {
    if (argc == 4 && std::string_view(argv[3]) == "--worker")
        return cudavm::run_test(argc - 1, argv, cudavm::fixtures, cudavm::execute_device_printf);
    try {
        cudavm::require(argc == 3 && std::string_view(argv[2]) == "cu", "usage: <executable> <backend> cu");
        const std::array<std::string_view, 5> backends{"cuda", "vulkan", "directx", "opencl", "opengl"};
        cudavm::require(std::find(backends.begin(), backends.end(), argv[1]) != backends.end(), "known backend");
        const auto child = cudavm::run_test_child(argv[0], argv[1], argv[2]);
        std::cout << child.output;
        if (child.status != 0) return child.status;
        cudavm::verify_printf_output(child.output);
        std::cout << "PASS: captured device output at every completion point\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
