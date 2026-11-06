#include "conformance.hpp"
#include "fixtures.hpp"

namespace cudavm {
std::string_view coherence_case;
uint32_t coherence_word(unsigned record, unsigned component, unsigned seed) {
    return std::bit_cast<uint32_t>(float(seed * 257 + (record % 8) * 7 + (record / 8) * 13 + component * 31));
}
void coherence_expected(image_test_storage& storage, unsigned seed, bool upload) {
    for (unsigned record = 0; record < 32; ++record) for (unsigned c = 0; c < 4; ++c) {
        auto word = coherence_word(record, c, seed); std::memcpy(storage.bytes.data() + (record * 4 + c) * 4, &word, 4);
    }
    if (upload) storage.upload();
}
struct coherence_stream {
    CUstream value{};
    explicit coherence_stream(bool runtime) {
        if (runtime) { cudaStream_t stream{}; image_runtime(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); value = reinterpret_cast<CUstream>(stream); }
        else CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING));
    }
    ~coherence_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    void wait() { CONFORMANCE_CUDA(cuStreamSynchronize(value)); }
};
struct coherence_surface {
    CUsurfObject value{};
    coherence_surface(const image_test_storage& storage, bool runtime) {
        if (runtime) { auto resource = storage.runtime_descriptor(); image_runtime(cudaCreateSurfaceObject(&value, &resource)); }
        else { auto resource = storage.driver_descriptor(); CONFORMANCE_CUDA(cuSurfObjectCreate(&value, &resource)); }
    }
    ~coherence_surface() { if (value) cuSurfObjectDestroy(value); }
};
struct coherence_result {
    static constexpr unsigned guards = 16, words = 32 * 4;
    std::vector<uint32_t> actual = std::vector<uint32_t>(guards + words + guards, 0xa7b96351);
    device_buffer device{actual.size() * 4};
    coherence_result() { reset(); }
    void reset() { std::fill(actual.begin(), actual.end(), 0xa7b96351); CONFORMANCE_CUDA(cuMemcpyHtoD(device.value, actual.data(), actual.size() * 4)); }
    void sample(CUfunction kernel, CUtexObject object, CUstream stream) {
        auto pointer = device.value + guards * 4; void* args[]{&object, &pointer};
        CONFORMANCE_CUDA(cuLaunchKernel(kernel, 1, 1, 1, 64, 1, 1, 0, stream, args, nullptr));
    }
    void verify(unsigned seed) {
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), device.value, actual.size() * 4));
        for (unsigned i = 0; i < actual.size(); ++i) {
            const auto expected = i < guards || i >= guards + words ? 0xa7b96351 : coherence_word((i - guards) / 4, (i - guards) % 4, seed);
            require(actual[i] == expected, "image coherence payload/guard word=" + std::to_string(i));
        }
    }
};
struct coherence_upload {
    static constexpr unsigned pitch = 8 * 16 + 32, bytes = 32 + pitch * 4;
    unsigned char* value{};
    std::vector<unsigned char> expected = std::vector<unsigned char>(bytes, 0xa7);
    explicit coherence_upload(unsigned seed) {
        void* allocation{}; CONFORMANCE_CUDA(cuMemAllocHost(&allocation, bytes)); value = static_cast<unsigned char*>(allocation);
        for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 8; ++x) for (unsigned c = 0; c < 4; ++c) {
            auto word = coherence_word(y * 8 + x, c, seed); std::memcpy(expected.data() + 16 + y * pitch + (x * 4 + c) * 4, &word, 4);
        }
        std::copy(expected.begin(), expected.end(), value);
    }
    ~coherence_upload() { if (value) cuMemFreeHost(value); }
    void submit(CUarray array, CUstream stream, bool runtime) {
        if (runtime) image_runtime(cudaMemcpy2DToArrayAsync(reinterpret_cast<cudaArray_t>(array), 0, 0, value + 16, pitch,
            8 * 16, 4, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)));
        else {
            CUDA_MEMCPY2D copy{}; copy.srcMemoryType = CU_MEMORYTYPE_HOST; copy.srcHost = value + 16; copy.srcPitch = pitch;
            copy.dstMemoryType = CU_MEMORYTYPE_ARRAY; copy.dstArray = array; copy.WidthInBytes = 8 * 16; copy.Height = 4;
            CONFORMANCE_CUDA(cuMemcpy2DAsync(&copy, stream));
        }
    }
    void verify() const { require(std::equal(expected.begin(), expected.end(), value), "coherence upload source, row padding and outer guards"); }
};
void execute_image_coherence(const loaded_program& program) {
    const auto sample_kernel = program.function("coherence_sample"), store_kernel = program.function("coherence_store");
    for (bool runtime : {false, true}) {
        image_test_storage storage(CU_RESOURCE_TYPE_ARRAY, CU_AD_FORMAT_FLOAT, cudaChannelFormatKindFloat, 32, 4, 8, 4, runtime, CUDA_ARRAY3D_SURFACE_LDST);
        image_test_storage control(CU_RESOURCE_TYPE_ARRAY, CU_AD_FORMAT_FLOAT, cudaChannelFormatKindFloat, 32, 4, 8, 4, !runtime, CUDA_ARRAY3D_SURFACE_LDST);
        coherence_expected(storage, 1, true); coherence_expected(control, 7, true);
        image_test_texture first(storage, runtime), second(storage, !runtime), independent(control, runtime);
        coherence_surface surface(storage, !runtime); coherence_stream writer(runtime), reader(!runtime);
        coherence_result first_result, second_result, control_result;
        coherence_upload upload(13);
        first_result.sample(sample_kernel, first.value, writer.value); control_result.sample(sample_kernel, independent.value, writer.value);
        writer.wait(); first_result.verify(1); control_result.verify(7); storage.verify(); control.verify();
        for (unsigned seed : {13u, 19u}) {
            if (seed == 19 && coherence_case != "surfaces") break;
            first_result.reset(); second_result.reset(); control_result.reset();
            CUevent ready{};
            if (coherence_case == "events") CONFORMANCE_CUDA(cuEventCreate(&ready, CU_EVENT_DISABLE_TIMING));
            if (coherence_case == "uploads") upload.submit(storage.array, writer.value, runtime);
            else {
                void* args[]{&surface.value, &seed};
                CONFORMANCE_CUDA(cuLaunchKernel(store_kernel, 1, 1, 1, 64, 1, 1, 0, writer.value, args, nullptr));
            }
            CUstream reads = writer.value;
            if (ready) {
                CONFORMANCE_CUDA(cuEventRecord(ready, writer.value));
                image_runtime(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(reader.value), reinterpret_cast<cudaEvent_t>(ready), 0));
                CONFORMANCE_CUDA(cuEventDestroy(ready)); reads = reader.value;
            }
            if (seed == 19) {
                second_result.sample(sample_kernel, second.value, reads); first_result.sample(sample_kernel, first.value, reads);
            } else {
                first_result.sample(sample_kernel, first.value, reads);
                if (coherence_case == "lifetime") {
                    image_runtime(cudaDestroyTextureObject(first.value)); first.value = 0;
                }
                second_result.sample(sample_kernel, second.value, reads);
            }
            control_result.sample(sample_kernel, independent.value, reads);
            reader.wait(); writer.wait();
            first_result.verify(seed); second_result.verify(seed); control_result.verify(7);
            coherence_expected(storage, seed, false); storage.verify(); control.verify(); upload.verify();
        }
    }
}
}
int main(int argc, char** argv) {
    if (argc != 4) return 1;
    cudavm::coherence_case = argv[3];
    if (cudavm::coherence_case != "uploads" && cudavm::coherence_case != "surfaces" && cudavm::coherence_case != "events" && cudavm::coherence_case != "lifetime") return 1;
    return cudavm::run_conformance(argc - 1, argv, cudavm::fixtures, cudavm::execute_image_coherence);
}
