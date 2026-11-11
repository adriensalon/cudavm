#include "test_framework.hpp"
#include "test_fixtures.hpp"

namespace cudavm {
bool surface_boundaries;
struct surface_access_format { const char* kernel; CUarray_format format; unsigned bits, channels, bytes; };
constexpr surface_access_format surface_access_formats[]{
    {"access_8_1", CU_AD_FORMAT_UNSIGNED_INT8, 8, 1, 1}, {"access_8_2", CU_AD_FORMAT_UNSIGNED_INT8, 8, 2, 2},
    {"access_8_4", CU_AD_FORMAT_UNSIGNED_INT8, 8, 4, 4}, {"access_16_1", CU_AD_FORMAT_UNSIGNED_INT16, 16, 1, 2},
    {"access_16_2", CU_AD_FORMAT_UNSIGNED_INT16, 16, 2, 4}, {"access_16_4", CU_AD_FORMAT_UNSIGNED_INT16, 16, 4, 8},
    {"access_32_1", CU_AD_FORMAT_UNSIGNED_INT32, 32, 1, 4}, {"access_32_2", CU_AD_FORMAT_UNSIGNED_INT32, 32, 2, 8},
    {"access_32_4", CU_AD_FORMAT_UNSIGNED_INT32, 32, 4, 16}, {"access_64_1", CU_AD_FORMAT_UNSIGNED_INT32, 32, 2, 8},
    {"access_64_2", CU_AD_FORMAT_UNSIGNED_INT32, 32, 4, 16}};
struct surface_access_object {
    CUsurfObject value{};
    surface_access_object(const image_test_storage& storage, bool runtime) {
        if (runtime) {
            auto resource = storage.runtime_descriptor(); image_runtime(cudaCreateSurfaceObject(&value, &resource));
        } else {
            auto resource = storage.driver_descriptor(); TEST_CUDA(cuSurfObjectCreate(&value, &resource));
        }
    }
    ~surface_access_object() { if (value) cuSurfObjectDestroy(value); }
};
unsigned char surface_access_byte(unsigned record, unsigned byte) {
    return static_cast<unsigned char>((record * 73 + byte * 37 + 19) ^ (record >> 2));
}
void execute_surface_access(const loaded_program& program) {
    constexpr unsigned width = 11, height = 3, count = width * height, guards = 16;
    for (const auto& format : surface_access_formats) for (bool runtime : {false, true}) {
        image_test_storage storage(CU_RESOURCE_TYPE_ARRAY, format.format, cudaChannelFormatKindUnsigned,
            format.bits, format.channels, width, height, runtime, CUDA_ARRAY3D_SURFACE_LDST);
        storage.upload(); surface_access_object surface(storage, runtime);
        std::vector<uint32_t> input(guards + (count + 2) * 4 + guards, 0xa7b96351);
        for (unsigned record = 0; record < count + 2; ++record) {
            std::array<unsigned char, 16> raw{};
            for (unsigned byte = 0; byte < format.bytes; ++byte) raw[byte] = surface_access_byte(record, byte);
            for (unsigned word = 0; word < 4; ++word)
                input[guards + record * 4 + word] = uint32_t(raw[word * 4]) | (uint32_t(raw[word * 4 + 1]) << 8)
                    | (uint32_t(raw[word * 4 + 2]) << 16) | (uint32_t(raw[word * 4 + 3]) << 24);
        }
        device_buffer source(input.size() * 4), destination((guards + (count + 4) * 4 + guards) * 4);
        TEST_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        auto in = source.value + guards * 4, out = destination.value + guards * 4;
        auto launch = [&](unsigned phase) {
            void* args[]{&surface.value, &in, &out, &phase};
            TEST_CUDA(cuLaunchKernel(program.function(format.kernel), 1, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr));
            TEST_CUDA(cuCtxSynchronize());
        };
        launch(0);
        for (bool replaced : {false, true}) {
            if (replaced && !surface_boundaries) break;
            if (replaced) launch(2);
            std::vector<uint32_t> actual(guards + (count + 4) * 4 + guards, 0xa7b96351);
            TEST_CUDA(cuMemcpyHtoD(destination.value, actual.data(), actual.size() * 4)); launch(1);
            TEST_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
            for (unsigned i = 0; i < actual.size(); ++i) {
                uint32_t expected = 0xa7b96351;
                if (i >= guards && i < actual.size() - guards) {
                    const unsigned slot = (i - guards) / 4, word = (i - guards) % 4;
                    unsigned record = slot;
                    if (slot == count || slot == count + 2) expected = 0;
                    else {
                        if (slot == count + 1) record = 0;
                        if (slot == count + 3) record = count - 1;
                        if (replaced && !record) record = count;
                        if (replaced && record == count - 1) record = count + 1;
                        expected = input[guards + record * 4 + word];
                    }
                }
                require(actual[i] == expected, std::string(format.kernel) + " surface result slot=" + std::to_string(i));
            }
            for (unsigned record = 0; record < count; ++record) for (unsigned byte = 0; byte < format.bytes; ++byte) {
                const unsigned original = replaced && record == 0 ? count : replaced && record == count - 1 ? count + 1 : record;
                storage.bytes[record * format.bytes + byte] = surface_access_byte(original, byte);
            }
            storage.verify();
        }
        std::vector<uint32_t> preserved(input.size()); TEST_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
        expect_words(preserved, input, "surface access preserves input bytes and both guards");
    }
}
}
int main(int argc, char** argv) {
    if (argc != 4) return 1;
    const std::string_view profile = argv[3]; if (profile != "vectors" && profile != "boundaries") return 1;
    cudavm::surface_boundaries = profile == "boundaries";
    return cudavm::run_test(argc - 1, argv, cudavm::fixtures, cudavm::execute_surface_access);
}
