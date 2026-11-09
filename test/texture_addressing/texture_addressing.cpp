#include "test_framework.hpp"
#include "test_fixtures.hpp"
#include <cmath>

namespace cudavm {
bool addressing_normalized;
float addressing_pixel(unsigned x, unsigned y, unsigned component) {
    return float(int(x * 13 + y * 71 + component * 257) - 300) * 0.25f;
}
int addressing_axis(float coordinate, unsigned extent, CUaddress_mode mode) {
    const auto bits = std::bit_cast<uint32_t>(coordinate);
    if ((bits & 0x7f800000) == 0) coordinate = 0;
    int64_t index = int64_t(std::floor(double(coordinate) * (addressing_normalized ? extent : 1)));
    if (!addressing_normalized && (mode == CU_TR_ADDRESS_MODE_WRAP || mode == CU_TR_ADDRESS_MODE_MIRROR)) mode = CU_TR_ADDRESS_MODE_CLAMP;
    if (mode == CU_TR_ADDRESS_MODE_BORDER) return index < 0 || index >= extent ? -1 : int(index);
    if (mode == CU_TR_ADDRESS_MODE_CLAMP) return int(std::clamp(index, int64_t(0), int64_t(extent - 1)));
    const int64_t period = mode == CU_TR_ADDRESS_MODE_MIRROR ? extent * 2 : extent;
    index %= period; if (index < 0) index += period;
    if (mode == CU_TR_ADDRESS_MODE_MIRROR && index >= extent) index = period - 1 - index;
    return int(index);
}
void execute_texture_addressing(const loaded_program& program) {
    constexpr unsigned width = 8, height = 4, guards = 16;
    const std::array<float, 23> values{-65536, -2, -1.5f, -1, -0.5f, -0.25f, -0.125f, -0.0625f,
        -std::numeric_limits<float>::min(), -std::numeric_limits<float>::denorm_min(), -0.0f, 0,
        std::numeric_limits<float>::denorm_min(), 0.0625f, 0.125f, 0.25f, 0.5f, 0.9375f, 1, 1.125f, 1.5f, 2, 65536};
    std::vector<float> coordinates(guards, std::bit_cast<float>(0x4a716395));
    for (auto y : values) for (auto x : values) { coordinates.push_back(x); coordinates.push_back(y); }
    coordinates.resize(coordinates.size() + guards, std::bit_cast<float>(0x4a716395));
    unsigned count = unsigned(values.size() * values.size());
    device_buffer input(coordinates.size() * 4); CONFORMANCE_CUDA(cuMemcpyHtoD(input.value, coordinates.data(), coordinates.size() * 4));
    for (auto resource : {CU_RESOURCE_TYPE_ARRAY, CU_RESOURCE_TYPE_PITCH2D}) for (bool runtime : {false, true})
    for (auto mode : {CU_TR_ADDRESS_MODE_WRAP, CU_TR_ADDRESS_MODE_MIRROR, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_BORDER}) {
        image_test_storage storage(resource, CU_AD_FORMAT_FLOAT, cudaChannelFormatKindFloat, 32, 4, width, height, runtime);
        for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) for (unsigned c = 0; c < 4; ++c) {
            float value = addressing_pixel(x, y, c); std::memcpy(storage.bytes.data() + y * storage.pitch + (x * 4 + c) * 4, &value, 4);
        }
        storage.upload(); const std::array<float, 4> border{23, -17, 31, -5};
        image_test_texture texture(storage, runtime, mode, CU_TR_FILTER_MODE_POINT, addressing_normalized, false, border);
        std::vector<uint32_t> output(guards + count * 4 + guards, 0xa7b96351);
        device_buffer destination(output.size() * 4); CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, output.data(), output.size() * 4));
        auto pointer = destination.value + guards * 4, positions = input.value + guards * 4;
        void* args[]{&texture.value, &pointer, &positions, &count};
        CONFORMANCE_CUDA(cuLaunchKernel(program.function("address_texture"), (count + 63) / 64, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr));
        CONFORMANCE_CUDA(cuCtxSynchronize()); CONFORMANCE_CUDA(cuMemcpyDtoH(output.data(), destination.value, output.size() * 4));
        for (unsigned i = 0; i < output.size(); ++i) {
            uint32_t expected = 0xa7b96351;
            if (i >= guards && i < guards + count * 4) {
                auto sample = (i - guards) / 4, component = (i - guards) % 4;
                auto x = addressing_axis(coordinates[guards + sample * 2], width, mode);
                auto y = addressing_axis(coordinates[guards + sample * 2 + 1], height, mode);
                expected = std::bit_cast<uint32_t>(x < 0 || y < 0 ? border[component] : addressing_pixel(unsigned(x), unsigned(y), component));
            }
            require(output[i] == expected, "texture addressing mode=" + std::to_string(mode) + " resource=" + std::to_string(resource)
                + " slot=" + std::to_string(i) + " got=" + std::to_string(output[i]) + " expected=" + std::to_string(expected));
        }
        storage.verify();
    }
    std::vector<float> preserved(coordinates.size()); CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), input.value, preserved.size() * 4));
    require(std::memcmp(preserved.data(), coordinates.data(), coordinates.size() * 4) == 0, "preserve coordinate bits and both input guards");
}
}
int main(int argc, char** argv) {
    if (argc != 4) return 1;
    const std::string_view profile = argv[3];
    if (profile != "normalized" && profile != "unnormalized") return 1;
    cudavm::addressing_normalized = profile == "normalized";
    return cudavm::run_conformance(argc - 1, argv, cudavm::fixtures, cudavm::execute_texture_addressing);
}
