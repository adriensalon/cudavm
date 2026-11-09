#pragma once
#include "test_framework.hpp"

namespace cudavm {
inline std::vector<std::string> symbol_sources(std::string_view profile) {
    if (profile == "weak") {
        const std::string header = R"cuda(
extern "C" __device__ unsigned symbol_function(unsigned);
extern "C" __device__ unsigned symbol_peer(unsigned);
extern "C" __device__ unsigned symbol_value;
extern "C" __constant__ unsigned symbol_bias;
)cuda";
        return {header + R"cuda(
extern "C" {
__device__ __attribute__((weak)) unsigned symbol_function(unsigned value) { return value * 3u + 5u; }
__device__ __attribute__((weak)) unsigned symbol_value = 17;
__constant__ __attribute__((weak)) unsigned symbol_bias = 29;
__device__ unsigned symbol_peer(unsigned value) { return symbol_function(value) + symbol_value + symbol_bias; }
}
)cuda", header + R"cuda(
extern "C" {
__device__ unsigned symbol_function(unsigned value) { return value * 7u + 11u; }
__device__ unsigned symbol_value = 71;
__constant__ unsigned symbol_bias = 89;
}
)cuda", header + R"cuda(
extern "C" __global__ void symbol_results(unsigned* output, const unsigned* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned value = input[i]; unsigned* row = output + i * 8;
    row[0] = symbol_function(value); row[1] = symbol_peer(value ^ i);
    row[2] = symbol_value; row[3] = symbol_bias;
    row[4] = symbol_function(value + i); row[5] = symbol_peer(i);
    row[6] = value ^ i; row[7] = symbol_value + symbol_bias;
}
)cuda"};
    }
    if (profile == "comdat") {
        const std::string header = R"cuda(
extern "C" { inline __device__ unsigned symbol_counter = 17; }
using symbol_callback = unsigned (*)(unsigned);
template<unsigned Bias> __device__ __noinline__ unsigned symbol_step(unsigned value) {
    const unsigned old = symbol_counter; symbol_counter += value + Bias; return old;
}
extern "C" __device__ unsigned symbol_left(unsigned);
extern "C" __device__ symbol_callback symbol_left_callback();
)cuda";
        return {header + R"cuda(
extern "C" __device__ unsigned symbol_left(unsigned value) { return symbol_step<7>(value); }
extern "C" __device__ symbol_callback symbol_left_callback() { return &symbol_step<7>; }
)cuda", header + R"cuda(
extern "C" __device__ unsigned symbol_right(unsigned value) { return symbol_step<7>(value + 3u); }
extern "C" __global__ void symbol_results(unsigned* output, const unsigned* input, unsigned count) {
    if (blockIdx.x || threadIdx.x) return;
    const auto callback = symbol_left_callback();
    for (unsigned i = 0; i < count; ++i) {
        const unsigned value = input[i]; unsigned* row = output + i * 8;
        row[0] = symbol_left(value); row[1] = symbol_right(value ^ i);
        row[2] = symbol_counter; row[3] = symbol_step<7>(i);
        row[4] = symbol_counter; row[5] = callback == &symbol_step<7>;
        row[6] = callback(i + 5u); row[7] = symbol_counter;
    }
}
)cuda"};
    }
    if (profile == "aliases") {
        const std::string header = R"cuda(
using symbol_callback = unsigned (*)(unsigned);
extern "C" __device__ unsigned symbol_target(unsigned);
extern "C" __device__ unsigned symbol_alias(unsigned);
extern "C" __device__ unsigned symbol_chain(unsigned);
extern "C" __device__ symbol_callback symbol_alias_pointer;
)cuda";
        return {R"cuda(
using symbol_callback = unsigned (*)(unsigned);
extern "C" {
__device__ __noinline__ unsigned symbol_target(unsigned value) { return value * 13u + 17u; }
__device__ unsigned symbol_alias(unsigned) __attribute__((alias("symbol_target")));
__device__ unsigned symbol_chain(unsigned) __attribute__((alias("symbol_alias")));
__device__ symbol_callback symbol_alias_pointer = &symbol_alias;
}
)cuda", header + R"cuda(
extern "C" __global__ void symbol_results(unsigned* output, const unsigned* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned value = input[i]; unsigned* row = output + i * 8;
    row[0] = symbol_target(value); row[1] = symbol_alias(value ^ i);
    row[2] = symbol_chain(value + i); row[3] = symbol_alias(value + i * 3u);
    row[4] = symbol_alias_pointer(value ^ 0x193759bdu); row[5] = symbol_chain(i * 3u);
    row[6] = value ^ i; row[7] = symbol_alias_pointer != nullptr;
}
)cuda"};
    }
    require(profile == "relocations", "unknown device-symbol source profile");
    const std::string header = R"cuda(
extern "C" __device__ unsigned symbol_values[4];
extern "C" __device__ unsigned* symbol_pointer;
extern "C" __device__ unsigned long long symbol_address;
)cuda";
    return {R"cuda(
extern "C" __device__ unsigned symbol_values[4] = {3, 17, 29, 43};
)cuda", header + R"cuda(
extern "C" {
__device__ unsigned* symbol_pointer = symbol_values + 2;
__device__ unsigned long long symbol_address = (unsigned long long)(symbol_values + 1);
}
)cuda", header + R"cuda(
extern "C" __global__ void symbol_results(unsigned* output, const unsigned* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned value = input[i]; unsigned* row = output + i * 8;
    const auto pointer_bits = (unsigned long long)symbol_pointer;
    const auto address_bits = symbol_address;
    row[0] = *symbol_pointer + value; row[1] = *(unsigned*)address_bits ^ value;
    row[2] = symbol_values[0] + i; row[3] = symbol_values[3] ^ i;
    row[4] = unsigned(pointer_bits); row[5] = unsigned(pointer_bits >> 32);
    row[6] = unsigned(address_bits); row[7] = unsigned(address_bits >> 32);
}
)cuda"};
}
}
