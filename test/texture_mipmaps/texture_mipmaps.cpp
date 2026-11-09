#include "test_framework.hpp"
#include "test_fixtures.hpp"
#include <cmath>

namespace cudavm {
std::string_view mip_texture_case;
float mip_texture_pixel(unsigned level, unsigned x, unsigned y, unsigned component) {
    return float(level * 16 + x + y * 4 + component * 64);
}
struct mip_texture_storage {
    CUmipmappedArray value{};
    explicit mip_texture_storage(bool runtime) {
        if (runtime) {
            cudaMipmappedArray_t result{}; cudaChannelFormatDesc channel{32, 32, 32, 32, cudaChannelFormatKindFloat};
            image_runtime(cudaMallocMipmappedArray(&result, &channel, {8, 8, 0}, 4, 0)); value = reinterpret_cast<CUmipmappedArray>(result);
        } else {
            CUDA_ARRAY3D_DESCRIPTOR descriptor{8, 8, 0, CU_AD_FORMAT_FLOAT, 4, 0};
            CONFORMANCE_CUDA(cuMipmappedArrayCreate(&value, &descriptor, 4));
        }
        for (unsigned level = 0; level < 4; ++level) {
            const unsigned width = 8 >> level; std::vector<float> pixels(width * width * 4);
            for (unsigned y = 0; y < width; ++y) for (unsigned x = 0; x < width; ++x) for (unsigned c = 0; c < 4; ++c)
                pixels[(y * width + x) * 4 + c] = mip_texture_pixel(level, x, y, c);
            CUDA_MEMCPY2D copy{}; copy.srcMemoryType = CU_MEMORYTYPE_HOST; copy.srcHost = pixels.data();
            copy.srcPitch = copy.WidthInBytes = width * 16; copy.Height = width; copy.dstMemoryType = CU_MEMORYTYPE_ARRAY; copy.dstArray = array(level);
            CONFORMANCE_CUDA(cuMemcpy2D(&copy));
        }
    }
    ~mip_texture_storage() { if (value) cuMipmappedArrayDestroy(value); }
    CUarray array(unsigned level) const { CUarray result{}; CONFORMANCE_CUDA(cuMipmappedArrayGetLevel(&result, value, level)); return result; }
    void verify() const {
        for (unsigned level = 0; level < 4; ++level) {
            const unsigned width = 8 >> level, guard = 16;
            std::vector<uint32_t> actual(guard + width * width * 4 + guard, 0xa7b96351);
            image_runtime(cudaMemcpy2DFromArray(actual.data() + guard, width * 16, reinterpret_cast<cudaArray_t>(array(level)),
                0, 0, width * 16, width, cudaMemcpyDeviceToHost));
            for (unsigned i = 0; i < actual.size(); ++i) {
                uint32_t expected = 0xa7b96351;
                if (i >= guard && i < actual.size() - guard) {
                    const auto texel = (i - guard) / 4;
                    expected = std::bit_cast<uint32_t>(mip_texture_pixel(level, texel % width, texel / width, (i - guard) % 4));
                }
                require(actual[i] == expected, "mipmap level backing bytes and readback guards");
            }
        }
    }
};
struct mip_texture_object {
    CUtexObject value{};
    mip_texture_object(const mip_texture_storage& storage, bool runtime, bool linear, bool clamps) {
        CUDA_RESOURCE_DESC resource{}; resource.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY; resource.res.mipmap.hMipmappedArray = storage.value;
        CUDA_TEXTURE_DESC sampler{}; sampler.flags = CU_TRSF_NORMALIZED_COORDINATES;
        for (auto& mode : sampler.addressMode) mode = CU_TR_ADDRESS_MODE_CLAMP;
        sampler.filterMode = CU_TR_FILTER_MODE_POINT; sampler.mipmapFilterMode = linear ? CU_TR_FILTER_MODE_LINEAR : CU_TR_FILTER_MODE_POINT;
        sampler.minMipmapLevelClamp = clamps ? 1.0f : 0; sampler.maxMipmapLevelClamp = clamps ? 2.0f : 3.0f; sampler.maxAnisotropy = 1;
        if (mip_texture_case == "validation") {
            auto invalid = sampler; invalid.flags = 0; CUtexObject rejected{};
            require(cuTexObjectCreate(&rejected, &resource, &invalid, nullptr) == CUDA_ERROR_INVALID_VALUE, "mipmapped resource requires normalized coordinates");
        }
        if (runtime) {
            cudaResourceDesc r{}; r.resType = cudaResourceTypeMipmappedArray; r.res.mipmap.mipmap = reinterpret_cast<cudaMipmappedArray_t>(storage.value);
            cudaTextureDesc t{}; for (auto& mode : t.addressMode) mode = cudaAddressModeClamp;
            t.normalizedCoords = 1; t.readMode = cudaReadModeElementType; t.maxAnisotropy = 1;
            t.mipmapFilterMode = linear ? cudaFilterModeLinear : cudaFilterModePoint;
            t.minMipmapLevelClamp = sampler.minMipmapLevelClamp; t.maxMipmapLevelClamp = sampler.maxMipmapLevelClamp;
            if (mip_texture_case == "validation") {
                auto invalid = t; invalid.normalizedCoords = 0; cudaTextureObject_t rejected{};
                require(cudaCreateTextureObject(&rejected, &r, &invalid, nullptr) == cudaErrorInvalidValue, "Runtime requires normalized mipmap coordinates");
            }
            image_runtime(cudaCreateTextureObject(&value, &r, &t, nullptr));
        } else CONFORMANCE_CUDA(cuTexObjectCreate(&value, &resource, &sampler, nullptr));
        CUDA_RESOURCE_DESC driver{}; cudaResourceDesc other{};
        CONFORMANCE_CUDA(cuTexObjectGetResourceDesc(&driver, value)); image_runtime(cudaGetTextureObjectResourceDesc(&other, value));
        require(driver.resType == CU_RESOURCE_TYPE_MIPMAPPED_ARRAY && driver.res.mipmap.hMipmappedArray == storage.value
            && other.resType == cudaResourceTypeMipmappedArray && other.res.mipmap.mipmap == reinterpret_cast<cudaMipmappedArray_t>(storage.value), "public mipmap resource identity");
        CUDA_TEXTURE_DESC queried{}; cudaTextureDesc runtime_sampler{};
        CONFORMANCE_CUDA(cuTexObjectGetTextureDesc(&queried, value)); image_runtime(cudaGetTextureObjectTextureDesc(&runtime_sampler, value));
        require(queried.mipmapFilterMode == sampler.mipmapFilterMode && queried.minMipmapLevelClamp == sampler.minMipmapLevelClamp
            && queried.maxMipmapLevelClamp == sampler.maxMipmapLevelClamp && runtime_sampler.minMipmapLevelClamp == sampler.minMipmapLevelClamp
            && runtime_sampler.maxMipmapLevelClamp == sampler.maxMipmapLevelClamp, "mipmap sampler queries preserve filter and clamps");
    }
    ~mip_texture_object() { if (value) cuTexObjectDestroy(value); }
    void close() { CONFORMANCE_CUDA(cuTexObjectDestroy(value)); value = 0; }
};
void execute_texture_mipmaps(const loaded_program& program) {
    const bool gradients = mip_texture_case == "gradients", clamps = mip_texture_case == "clamps";
    constexpr unsigned guards = 16;
    constexpr std::array<float, 14> lods{-4, -1, 0, 0.25f, 0.75f, 1, 1.25f, 1.75f, 2, 2.25f, 2.75f, 3, 4, 8};
    for (bool runtime : {false, true}) for (bool linear : {false, true}) {
        if (mip_texture_case == "point" && linear) continue;
        if (mip_texture_case == "linear" && !linear) continue;
        mip_texture_storage storage(runtime); mip_texture_object texture(storage, runtime, linear, clamps);
        std::vector<float> records(guards, std::bit_cast<float>(0x4a716395));
        std::vector<float> expected;
        for (unsigned i = 0; i < lods.size(); ++i) {
            if (gradients && (lods[i] < 0 || lods[i] > 3 || std::floor(lods[i]) != lods[i])) continue;
            for (float y : {0.125f, 0.375f, 0.625f, 0.875f}) for (float x : {0.125f, 0.375f, 0.625f, 0.875f}) {
                records.insert(records.end(), {x, y, gradients ? std::ldexp(1.0f, int(lods[i]) - 3) : lods[i], 0});
                const float lod = std::clamp(lods[i], clamps ? 1.0f : 0, clamps ? 2.0f : 3.0f);
                const unsigned low = linear ? unsigned(std::floor(lod)) : unsigned(std::floor(lod + 0.5f));
                const unsigned high = std::min(low + 1, 3u); const float weight = linear ? lod - low : 0;
                for (unsigned c = 0; c < 4; ++c) {
                    const auto a = mip_texture_pixel(low, unsigned(x * (8 >> low)), unsigned(y * (8 >> low)), c);
                    const auto b = mip_texture_pixel(high, unsigned(x * (8 >> high)), unsigned(y * (8 >> high)), c);
                    expected.push_back(a * (1 - weight) + b * weight);
                }
            }
        }
        records.resize(records.size() + guards, std::bit_cast<float>(0x4a716395));
        unsigned count = unsigned(expected.size() / 4), use_gradients = gradients;
        device_buffer input(records.size() * 4), output((expected.size() + 2 * guards) * 4);
        CONFORMANCE_CUDA(cuMemcpyHtoD(input.value, records.data(), records.size() * 4));
        std::vector<uint32_t> actual(expected.size() + 2 * guards, 0xa7b96351);
        CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, actual.data(), actual.size() * 4));
        auto destination = output.value + guards * 4, samples = input.value + guards * 4;
        void* args[]{&texture.value, &destination, &samples, &count, &use_gradients};
        CONFORMANCE_CUDA(cuLaunchKernel(program.function("sample_mipmap"), (count + 63) / 64, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr));
        if (mip_texture_case == "lifetime") texture.close(); else CONFORMANCE_CUDA(cuCtxSynchronize());
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
        for (unsigned i = 0; i < actual.size(); ++i) {
            const auto wanted = i < guards || i >= guards + expected.size() ? 0xa7b96351 : std::bit_cast<uint32_t>(expected[i - guards]);
            require(actual[i] == wanted, "mipmap LOD/filter/gradient result slot=" + std::to_string(i)
                + " got=" + std::to_string(actual[i]) + " expected=" + std::to_string(wanted));
        }
        std::vector<float> preserved(records.size()); CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), input.value, preserved.size() * 4));
        require(std::memcmp(records.data(), preserved.data(), records.size() * 4) == 0, "mipmap sample records and guards preserved"); storage.verify();
    }
}
}
int main(int argc, char** argv) {
    if (argc != 4) return 1;
    cudavm::mip_texture_case = argv[3];
    if (cudavm::mip_texture_case != "point" && cudavm::mip_texture_case != "linear" && cudavm::mip_texture_case != "gradients"
        && cudavm::mip_texture_case != "clamps" && cudavm::mip_texture_case != "lifetime" && cudavm::mip_texture_case != "validation") return 1;
    return cudavm::run_conformance(argc - 1, argv, cudavm::fixtures, cudavm::execute_texture_mipmaps);
}
