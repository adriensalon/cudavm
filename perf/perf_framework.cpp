#include "perf_framework.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace cudavm {
namespace {

size_t parse_count(std::string_view text) {
    size_t value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    perf_require(error == std::errc{} && end == text.data() + text.size(),
                 "invalid count: " + std::string(text));
    return value;
}

void write_json_string(std::string_view value) {
    constexpr char digits[] = "0123456789abcdef";
    std::cout << '"';
    for (unsigned char character : value) {
        if (character == '"' || character == '\\') std::cout << '\\' << char(character);
        else if (character < 0x20)
            std::cout << "\\u00" << digits[character >> 4] << digits[character & 15];
        else std::cout << char(character);
    }
    std::cout << '"';
}

void report_perf(const perf_options& options, const perf_result& result) {
    perf_require(result.milliseconds.size() == options.samples, "incorrect performance sample count");
    perf_require(options.backend == "host" || !result.device.empty(), "GPU results require a device name");
    for (double sample : result.milliseconds)
        perf_require(std::isfinite(sample) && sample >= 0, "invalid performance sample");
    auto sorted = result.milliseconds;
    std::sort(sorted.begin(), sorted.end());
    const auto middle = sorted.size() / 2;
    const double median = sorted.size() % 2 ? sorted[middle] : sorted[middle - 1] / 2 + sorted[middle] / 2;
    std::cout << "{\"slice\":";
    write_json_string(options.slice);
    std::cout << ",\"backend\":";
    write_json_string(options.backend);
    std::cout << ",\"case\":";
    write_json_string(options.case_name);
    std::cout << ",\"device\":";
    write_json_string(result.device);
    std::cout << ",\"clock\":\"steady_clock\",\"unit\":\"ms/iteration\",\"warmup\":" << options.warmup
              << ",\"iterations\":" << options.iterations << ",\"samples\":[" << std::setprecision(17);
    for (size_t index = 0; index < result.milliseconds.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << result.milliseconds[index];
    }
    std::cout << "],\"minimum\":" << sorted.front() << ",\"median\":" << median << "}\n";
}

}

void perf_require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

perf_result measure_perf(const perf_options& options, const std::function<void()>& operation,
                         const std::function<void()>& synchronize) {
    perf_require(options.samples > 0 && options.iterations > 0, "samples and iterations must be positive");
    perf_require(bool(operation), "missing performance operation");
    perf_result result;
    result.milliseconds.reserve(options.samples);
    const auto batch = [&] {
        for (size_t iteration = 0; iteration < options.iterations; ++iteration) operation();
        if (synchronize) synchronize();
    };
    for (size_t warmup = 0; warmup < options.warmup; ++warmup) {
        if (synchronize) synchronize();
        batch();
    }
    for (size_t sample = 0; sample < options.samples; ++sample) {
        if (synchronize) synchronize();
        const auto start = std::chrono::steady_clock::now();
        batch();
        const auto end = std::chrono::steady_clock::now();
        result.milliseconds.push_back(std::chrono::duration<double, std::milli>(end - start).count() / options.iterations);
    }
    return result;
}

int run_perf(int argc, char** argv, std::string_view slice,
             perf_result (*execute)(const perf_options&)) {
    try {
        perf_require(argc >= 3, "usage: <executable> <backend|host> <case> [--warmup N] [--samples N] [--iterations N]");
        perf_options options{slice, argv[1], argv[2]};
        constexpr std::array<std::string_view, 6> backends{"host", "cuda", "vulkan", "directx", "opencl", "opengl"};
        perf_require(std::find(backends.begin(), backends.end(), options.backend) != backends.end(), "unknown backend");
        for (int argument = 3; argument < argc; argument += 2) {
            perf_require(argument + 1 < argc, "performance option requires a count");
            const std::string_view name = argv[argument];
            const auto value = parse_count(argv[argument + 1]);
            if (name == "--warmup") options.warmup = value;
            else if (name == "--samples") options.samples = value;
            else if (name == "--iterations") options.iterations = value;
            else throw std::runtime_error("unknown performance option: " + std::string(name));
        }
        perf_require(options.samples > 0 && options.iterations > 0, "samples and iterations must be positive");
        perf_require(execute != nullptr, "missing performance slice");
        report_perf(options, execute(options));
        return 0;
    } catch (const perf_unavailable& error) {
        std::cerr << "SKIP: " << error.what() << '\n';
        return 77;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

}
