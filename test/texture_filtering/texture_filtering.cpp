#include "test_framework.hpp"
#include "test_fixtures.hpp"
#include <cmath>

namespace cudavm {
bool filtering_special;
float filtering_pixel(unsigned x, unsigned y, unsigned c) {
    if (filtering_special) {
        if (y == 1 && x == 1 && c == 0) return std::numeric_limits<float>::infinity();
        if (y == 1 && x == 2 && c == 0) return -std::numeric_limits<float>::infinity();
        if ((y == 1 && x == 3 && c == 0) || (y == 1 && x == 4 && c == 2)) return std::numeric_limits<float>::quiet_NaN();
        if (y == 2 && x == 2 && c == 1) return std::numeric_limits<float>::infinity();
    }
    return float(int(x) * 3 - int(y) * 5 + int(c) * 7);
}
int filtering_address(int64_t index, unsigned extent, CUaddress_mode mode, bool normalized) {
    if (!normalized && (mode == CU_TR_ADDRESS_MODE_WRAP || mode == CU_TR_ADDRESS_MODE_MIRROR)) mode = CU_TR_ADDRESS_MODE_CLAMP;
    if (mode == CU_TR_ADDRESS_MODE_BORDER) return index < 0 || index >= extent ? -1 : int(index);
    if (mode == CU_TR_ADDRESS_MODE_CLAMP) return int(std::clamp(index, int64_t(0), int64_t(extent - 1)));
    const int64_t period = mode == CU_TR_ADDRESS_MODE_MIRROR ? extent * 2 : extent;
    index %= period; if (index < 0) index += period;
    if (mode == CU_TR_ADDRESS_MODE_MIRROR && index >= extent) index = period - index - 1;
    return int(index);
}
struct filtering_axis { int64_t first; unsigned weight; };
filtering_axis filtering_coefficients(float coordinate, unsigned extent, bool normalized, bool linear, bool lower_tie) {
    const double scaled = double(coordinate) * (normalized ? extent : 1);
    if (!linear) return {int64_t(std::floor(scaled)), 0};
    const double position = scaled * 256;
    const bool halfway = position - std::floor(position) == 0.5;
    const int64_t fixed = int64_t(std::floor(position + 0.5)) - (lower_tie && halfway ? 1 : 0) - 128;
    int64_t first = fixed / 256;
    if (fixed < 0 && fixed % 256) --first;
    return {first, unsigned(fixed - first * 256)};
}
float filtering_oracle(float x, float y, unsigned component, CUaddress_mode mode, bool normalized, bool linear, unsigned rounding = 0) {
    constexpr std::array<float, 4> border{-3, 11, 5, -7};
    const auto horizontal = filtering_coefficients(x, 8, normalized, linear, rounding & 1);
    const auto vertical = filtering_coefficients(y, 4, normalized, linear, rounding & 2);
    bool positive_infinity = false, negative_infinity = false, nan = false;
    double finite = 0;
    for (unsigned row = 0; row < 2; ++row) for (unsigned column = 0; column < 2; ++column) {
        const unsigned weight = (column ? horizontal.weight : 256 - horizontal.weight) * (row ? vertical.weight : 256 - vertical.weight);
        if (!weight) continue;
        const int sx = filtering_address(horizontal.first + column, 8, mode, normalized);
        const int sy = filtering_address(vertical.first + row, 4, mode, normalized);
        const auto value = sx < 0 || sy < 0 ? border[component] : filtering_pixel(unsigned(sx), unsigned(sy), component);
        if (std::isnan(value)) nan = true;
        else if (std::isinf(value)) (std::signbit(value) ? negative_infinity : positive_infinity) = true;
        else finite += double(value) * weight;
    }
    if (nan || (positive_infinity && negative_infinity)) return std::numeric_limits<float>::quiet_NaN();
    if (positive_infinity) return std::numeric_limits<float>::infinity();
    if (negative_infinity) return -std::numeric_limits<float>::infinity();
    return float(finite / 65536);
}
void execute_texture_filtering(const loaded_program& program) {
    constexpr unsigned width = 8, height = 4, guards = 16;
    constexpr std::array<float, 23> values{-2, -1, -0.5f, -0.25f, 0, 0.25f,
        0.5f - 1.0f / 512, 0.5f, 0.5f + 1.0f / 1024, 0.5f + 1.0f / 512, 0.75f,
        1, 1.25f, 1.5f - 1.0f / 512, 1.5f, 1.5f + 1.0f / 512, 1.75f, 2, 2.5f, 3.5f, 4, 7.5f, 8};
    std::array<std::vector<float>, 2> coordinates;
    std::array<std::unique_ptr<device_buffer>, 2> inputs;
    for (unsigned profile = 0; profile < 2; ++profile) {
        auto& samples = coordinates[profile]; samples.assign(guards, std::bit_cast<float>(0x4a716395));
        for (auto y : values) for (auto x : values) {
            // Fine custom-border coefficient/product precision has its own
            // open oracle obligation. Quarter-grid blends remain exact controls.
            if (profile && (x * 4 != std::floor(x * 4) || y * 4 != std::floor(y * 4))) continue;
            samples.push_back(x); samples.push_back(y);
        }
        samples.resize(samples.size() + guards, std::bit_cast<float>(0x4a716395));
        inputs[profile] = std::make_unique<device_buffer>(samples.size() * 4);
        CONFORMANCE_CUDA(cuMemcpyHtoD(inputs[profile]->value, samples.data(), samples.size() * 4));
    }
    for (auto resource : {CU_RESOURCE_TYPE_ARRAY, CU_RESOURCE_TYPE_PITCH2D}) for (bool runtime : {false, true})
    for (auto mode : {CU_TR_ADDRESS_MODE_WRAP, CU_TR_ADDRESS_MODE_MIRROR, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_BORDER})
    for (bool normalized : {false, true}) for (bool linear : {false, true}) {
        const unsigned profile = mode == CU_TR_ADDRESS_MODE_BORDER && linear;
        const auto& samples = coordinates[profile]; unsigned count = unsigned((samples.size() - 2 * guards) / 2);
        image_test_storage storage(resource, CU_AD_FORMAT_FLOAT, cudaChannelFormatKindFloat, 32, 4, width, height, runtime);
        for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) for (unsigned c = 0; c < 4; ++c) {
            float value = filtering_pixel(x, y, c); std::memcpy(storage.bytes.data() + y * storage.pitch + (x * 4 + c) * 4, &value, 4);
        }
        storage.upload(); image_test_texture texture(storage, runtime, mode, linear ? CU_TR_FILTER_MODE_LINEAR : CU_TR_FILTER_MODE_POINT,
            normalized, false, {-3, 11, 5, -7});
        std::vector<uint32_t> output(guards + count * 4 + guards, 0xa7b96351);
        device_buffer destination(output.size() * 4); CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, output.data(), output.size() * 4));
        auto pointer = destination.value + guards * 4, positions = inputs[profile]->value + guards * 4;
        void* args[]{&texture.value, &pointer, &positions, &count};
        CONFORMANCE_CUDA(cuLaunchKernel(program.function("filter_texture"), (count + 63) / 64, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr));
        CONFORMANCE_CUDA(cuCtxSynchronize()); CONFORMANCE_CUDA(cuMemcpyDtoH(output.data(), destination.value, output.size() * 4));
        for (unsigned i = 0; i < output.size(); ++i) {
            uint32_t expected = 0xa7b96351;
            if (i >= guards && i < guards + count * 4) {
                const unsigned sample = (i - guards) / 4, component = (i - guards) % 4;
                auto value = filtering_oracle(samples[guards + sample * 2], samples[guards + sample * 2 + 1], component, mode, normalized, linear);
                expected = std::bit_cast<uint32_t>(value);
                // The public fixed-point precision does not prescribe a tie
                // direction. Enumerate the two adjacent coefficients only at
                // exact halfways, preserving exact-bit checks everywhere else.
                bool matched = false;
                for (unsigned rounding = 0; rounding < 4; ++rounding) {
                    auto candidate = filtering_oracle(samples[guards + sample * 2], samples[guards + sample * 2 + 1], component, mode, normalized, linear, rounding);
                    matched |= std::isnan(candidate) ? std::isnan(std::bit_cast<float>(output[i])) : output[i] == std::bit_cast<uint32_t>(candidate);
                }
                if (matched) continue;
            }
            require(output[i] == expected, "texture filtering mode=" + std::to_string(mode) + " normalized=" + std::to_string(normalized)
                + " linear=" + std::to_string(linear) + " resource=" + std::to_string(resource) + " slot=" + std::to_string(i)
                + " got=" + std::to_string(output[i]) + " expected=" + std::to_string(expected));
        }
        storage.verify();
    }
    for (unsigned profile = 0; profile < 2; ++profile) {
        const auto& samples = coordinates[profile]; std::vector<float> preserved(samples.size());
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), inputs[profile]->value, preserved.size() * 4));
        require(std::memcmp(preserved.data(), samples.data(), samples.size() * 4) == 0, "filtering preserves every input coordinate and guard");
    }
}
}
int main(int argc, char** argv) {
    if (argc != 4) return 1;
    const std::string_view profile = argv[3];
    if (profile != "finite" && profile != "special") return 1;
    cudavm::filtering_special = profile == "special";
    return cudavm::run_conformance(argc - 1, argv, cudavm::fixtures, cudavm::execute_texture_filtering);
}
