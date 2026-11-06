#include "conformance.hpp"
#include "fixtures.hpp"
#include <cmath>
#include <iostream>

namespace cudavm {
bool fetch_half;
struct fetch_format {
    CUarray_format format;
    cudaChannelFormatKind kind;
    unsigned bits;
    const char* kernel;
};
constexpr fetch_format fetch_formats[]{
    {CU_AD_FORMAT_SIGNED_INT8, cudaChannelFormatKindSigned, 8, "fetch_s8"},
    {CU_AD_FORMAT_UNSIGNED_INT8, cudaChannelFormatKindUnsigned, 8, "fetch_u8"},
    {CU_AD_FORMAT_SIGNED_INT16, cudaChannelFormatKindSigned, 16, "fetch_s16"},
    {CU_AD_FORMAT_UNSIGNED_INT16, cudaChannelFormatKindUnsigned, 16, "fetch_u16"},
    {CU_AD_FORMAT_SIGNED_INT32, cudaChannelFormatKindSigned, 32, "fetch_s32"},
    {CU_AD_FORMAT_UNSIGNED_INT32, cudaChannelFormatKindUnsigned, 32, "fetch_u32"},
    {CU_AD_FORMAT_HALF, cudaChannelFormatKindFloat, 16, "fetch_float"},
    {CU_AD_FORMAT_FLOAT, cudaChannelFormatKindFloat, 32, "fetch_float"}};
uint32_t fetch_component(unsigned index, const fetch_format& format) {
    if (format.kind == cudaChannelFormatKindFloat) {
        if (fetch_half) {
            index %= 0xf800;
            return index < 0x7c00 ? index : index + 0x400;
        }
        constexpr uint16_t half[]{0, 0x8000, 0x3c00, 0xbc00, 0x4000, 0x3800, 0xc200, 0x7bff};
        constexpr float values[]{0.0f, -0.0f, 1.0f, -1.0f, 2.0f, 0.5f, -3.0f, 65504.0f};
        return format.bits == 16 ? half[index % 8] : std::bit_cast<uint32_t>(values[index % 8]);
    }
    const auto mask = UINT32_MAX >> (32 - format.bits);
    const std::array<uint32_t, 10> edges{0, 1, mask, mask - 1, mask / 2, mask / 2 + 1, mask / 2 + 2, 2, 3, mask - 2};
    return index % 3 == 0 ? edges[(index / 3) % edges.size()] : (index * 197 + 29) & mask;
}
uint32_t fetch_oracle(uint32_t raw, const fetch_format& format, bool normalized) {
    if (format.kind == cudaChannelFormatKindFloat) {
        if (format.bits == 32) return raw;
        const unsigned exponent = (raw >> 10) & 31, fraction = raw & 1023;
        require(exponent != 31, "finite half oracle domain");
        double value = exponent ? std::ldexp(double(1024 + fraction), int(exponent) - 25) : std::ldexp(double(fraction), -24);
        if (raw & 0x8000) value = -value;
        return std::bit_cast<uint32_t>(float(value));
    }
    const bool signed_format = format.kind == cudaChannelFormatKindSigned;
    const uint32_t mask = UINT32_MAX >> (32 - format.bits);
    const uint32_t extended = signed_format && (raw & (1u << (format.bits - 1))) ? raw | ~mask : raw;
    if (!normalized) return extended;
    const double numerator = signed_format ? double(std::bit_cast<int32_t>(extended)) : double(raw);
    const double denominator = signed_format ? double(mask >> 1) : double(mask);
    return std::bit_cast<uint32_t>(float(std::max(-1.0, numerator / denominator)));
}
bool fetch_close(uint32_t actual, uint32_t expected, bool normalized) {
    if (!normalized || (expected & 0x7fffffff) == 0 || (expected & 0x7fffffff) == 0x3f800000) return actual == expected;
    if ((actual ^ expected) & 0x80000000) return false;
    return actual > expected ? actual - expected <= 2 : expected - actual <= 2;
}
bool fetch_normalized;
void execute_texture_fetch(const loaded_program& program) {
    const unsigned width = fetch_half ? 248 : 19, height = fetch_half ? 256 : 7, count = width * height;
    constexpr unsigned guards = 16;
    for (const auto& format : fetch_formats) {
        if (fetch_half && format.format != CU_AD_FORMAT_HALF) continue;
        if (fetch_normalized && (format.kind == cudaChannelFormatKindFloat || format.bits == 32)) continue;
        for (unsigned channels : {1u, 2u, 4u}) for (auto resource : {CU_RESOURCE_TYPE_LINEAR, CU_RESOURCE_TYPE_PITCH2D, CU_RESOURCE_TYPE_ARRAY})
        for (bool runtime : {false, true}) {
            try {
                image_test_storage storage(resource, format.format, format.kind, format.bits, channels, width, height, runtime);
                for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) for (unsigned c = 0; c < channels; ++c) {
                    auto raw = fetch_component((y * width + x) * channels + c, format);
                    for (unsigned byte = 0; byte < format.bits / 8; ++byte)
                        storage.bytes[y * storage.pitch + (x * channels + c) * (format.bits / 8) + byte] = static_cast<unsigned char>(raw >> (byte * 8));
                }
                storage.upload(); image_test_texture texture(storage, runtime, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_POINT, false, fetch_normalized);
                std::vector<uint32_t> actual(guards + count * 4 + guards, 0xa7b96351);
                device_buffer output(actual.size() * 4); CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, actual.data(), actual.size() * 4));
                auto pointer = output.value + guards * 4; unsigned columns = width, size = count, linear = resource == CU_RESOURCE_TYPE_LINEAR;
                void* args[]{&texture.value, &pointer, &columns, &size, &channels, &linear};
                CONFORMANCE_CUDA(cuLaunchKernel(program.function(fetch_normalized ? "fetch_float" : format.kernel),
                    (count + 63) / 64, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr));
                CONFORMANCE_CUDA(cuCtxSynchronize()); CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
                for (unsigned i = 0; i < actual.size(); ++i) {
                    uint32_t expected = 0xa7b96351;
                    if (i >= guards && i < guards + count * 4) {
                        auto sample = (i - guards) / 4, c = (i - guards) % 4;
                        const auto index = (sample * 37 + 11) % count;
                        expected = c < channels ? fetch_oracle(fetch_component(index * channels + c, format), format, fetch_normalized) : 0;
                    }
                    if (!fetch_close(actual[i], expected, fetch_normalized && i >= guards && i < guards + count * 4))
                        throw std::runtime_error("result slot " + std::to_string(i) + " got=" + std::to_string(actual[i]) + " expected=" + std::to_string(expected));
                }
                storage.verify();
            } catch (const std::exception& error) {
                throw std::runtime_error("texture format=" + std::to_string(format.format) + " channels=" + std::to_string(channels)
                    + " resource=" + std::to_string(resource) + (runtime ? " Runtime: " : " Driver: ") + error.what());
            }
        }
    }
}
}
int main(int argc, char** argv) {
    if (argc != 4) return 1;
    const std::string_view profile = argv[3];
    if (profile != "element" && profile != "normalized" && profile != "half") return 1;
    cudavm::fetch_normalized = profile == "normalized";
    cudavm::fetch_half = profile == "half";
    return cudavm::run_conformance(argc - 1, argv, cudavm::fixtures, cudavm::execute_texture_fetch);
}
