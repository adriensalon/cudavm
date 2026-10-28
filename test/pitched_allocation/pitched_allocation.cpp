#include "conformance.hpp"
#include <algorithm>
#include <limits>

namespace cudavm {
struct pitched_storage {
    CUdeviceptr value{};
    size_t pitch{};
    pitched_storage(size_t width, size_t height, unsigned element) {
        CONFORMANCE_CUDA(cuMemAllocPitch(&value, &pitch, width, height, element));
    }
    ~pitched_storage() { if (value) cuMemFree(value); }
};
void execute_pitched_allocation() {
    constexpr unsigned char guard = 0xb7;
    for (unsigned element : {4u, 8u, 16u}) for (size_t width : {size_t(1), size_t(3), size_t(7), size_t(16), size_t(257), size_t(4093)})
    for (size_t payload_rows : {size_t(1), size_t(2), size_t(5)}) {
        const size_t rows = payload_rows + 2;
        pitched_storage storage(width, rows, element);
        require(storage.value && storage.value % 16 == 0, "pitched scalar/vector base alignment");
        require(storage.pitch >= width && storage.pitch % element == 0, "pitch contains byte width and aligns element hint");
        require(storage.pitch <= std::numeric_limits<size_t>::max() / rows, "pitched allocation extent does not overflow");
        const size_t volume = storage.pitch * rows;
        CUdeviceptr base{};
        size_t extent = 0;
        CONFORMANCE_CUDA(cuMemGetAddressRange(&base, &extent, storage.value));
        require(base == storage.value && extent >= volume, "pitched allocation contains every requested row");
        std::vector<unsigned char> initial(volume + 32, guard), expected(initial), actual(initial);
        CONFORMANCE_CUDA(cuMemcpyHtoD(storage.value, initial.data() + 16, volume));
        for (unsigned phase = 0; phase < 2; ++phase) {
            const size_t column = phase && width > 7 ? 3 : 0;
            const size_t count = phase && width > 7 ? width - 6 : width;
            for (size_t row = 1; row <= payload_rows; ++row) {
                if (phase && row % 2 == 0) continue;
                std::vector<unsigned char> payload(count);
                for (size_t x = 0; x < count; ++x) payload[x] = static_cast<unsigned char>(element * 7 + row * 31 + (x + column) * 13 + phase * 57);
                const auto preserved = payload;
                CONFORMANCE_CUDA(cuMemcpyHtoD(storage.value + row * storage.pitch + column, payload.data(), count));
                require(payload == preserved, "pitched row upload source preservation");
                std::copy(payload.begin(), payload.end(), expected.begin() + 16 + row * storage.pitch + column);
            }
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data() + 16, storage.value, volume));
            const auto context = "pitched allocation width=" + std::to_string(width)
                + " rows=" + std::to_string(payload_rows) + " hint=" + std::to_string(element)
                + " pitch=" + std::to_string(storage.pitch) + " phase=" + std::to_string(phase);
            for (size_t i = 0; i < actual.size(); ++i)
                if (actual[i] != expected[i]) throw std::runtime_error(context + ": byte " + std::to_string(i));
            for (size_t row = 0; row < rows; ++row) {
                CUdeviceptr row_base{};
                size_t row_extent = 0;
                CONFORMANCE_CUDA(cuMemGetAddressRange(&row_base, &row_extent, storage.value + row * storage.pitch + width - 1));
                require(row_base == base && row_extent == extent, context + ": row address identity");
            }
        }
        CONFORMANCE_CUDA(cuMemFree(storage.value)); storage.value = 0;
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::execute_pitched_allocation);
}
