#pragma once

#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cudavm {

struct perf_options {
    std::string_view slice, backend, case_name;
    size_t warmup = 3;
    size_t samples = 10;
    size_t iterations = 1;
};

struct perf_fixture_data {
    std::string_view source, ptx, nvvm;
    std::span<const unsigned char> cubin;
};

struct perf_result {
    std::string device;
    // Host wall time in milliseconds per operation, retaining every sample.
    std::vector<double> milliseconds;
};

struct perf_unavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void perf_require(bool condition, const std::string& message);

// The slice owns setup and correctness checks outside this call. Async work
// requires a completion callback: it runs before and after every batch, with
// the trailing completion included in the measured interval.
perf_result measure_perf(const perf_options& options, const std::function<void()>& operation,
                         const std::function<void()>& synchronize = {});

// A slice returns only after checking its result. The runner then writes one
// JSON record; exceptions produce failure (1) or explicit unavailability (77).
int run_perf(int argc, char** argv, std::string_view slice,
             perf_result (*execute)(const perf_options&));

}
