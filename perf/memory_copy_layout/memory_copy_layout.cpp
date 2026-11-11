#include "perf_framework.hpp"
#include <algorithm>
#include <array>
#include <memory>

namespace cudavm {

perf_result execute_memory_copy_layout(const perf_options& options) {
    constexpr std::array<std::string_view, 6> cases{
        "linear_64k", "linear_4m", "pitched_2d_64k", "pitched_2d_4m", "pitched_3d_64k", "pitched_3d_4m"};
    perf_require(std::find(cases.begin(), cases.end(), options.case_name) != cases.end(), "unknown memory copy layout case");
    const bool large = options.case_name.ends_with("4m");
    const bool linear = options.case_name.starts_with("linear");
    const bool volume = options.case_name.starts_with("pitched_3d");
    constexpr size_t guard = 64;
    const size_t bytes = large ? 4194304 : 65536;
    const size_t width = linear ? bytes : volume ? (large ? 65536 : 1024) : (large ? 262144 : 4096);
    const size_t height = linear ? 1 : volume ? 8 : 16;
    const size_t depth = volume ? 8 : 1;
    const size_t source_rows = height + (linear ? 0 : 3), destination_rows = height + (linear ? 0 : 5);
    const size_t slices = depth + (volume ? 2 : 0);
    const size_t source_x = linear ? 0 : 96, destination_x = linear ? 0 : 128;
    const size_t source_y = linear ? 0 : 1, destination_y = linear ? 0 : 2, first_z = volume ? 1 : 0;
    const auto device = select_perf_device(options.backend);
    perf_context context(device.value);
    perf_stream stream;
    size_t source_pitch = width, destination_pitch = width;
    std::unique_ptr<perf_device_buffer> source, destination;
    if (linear) {
        source = std::make_unique<perf_device_buffer>(bytes + 2 * guard);
        destination = std::make_unique<perf_device_buffer>(bytes + 2 * guard);
    } else {
        source = std::make_unique<perf_device_buffer>(width + source_x + guard, source_rows * slices + 1, source_pitch);
        destination = std::make_unique<perf_device_buffer>(width + 1024, destination_rows * slices + 1, destination_pitch);
    }
    const size_t source_extent = linear ? bytes + 2 * guard : source_pitch * (source_rows * slices + 1);
    const size_t destination_extent = linear ? bytes + 2 * guard : destination_pitch * (destination_rows * slices + 1);
    const size_t base_offset = linear ? guard : 0;
    const auto input = perf_input_bytes(source_extent, 0x1943b6ef);
    const std::vector<unsigned char> reset(destination_extent, 0xa5);
    auto expected = reset;
    for (size_t z = 0; z < depth; ++z) for (size_t y = 0; y < height; ++y) {
        const size_t from = base_offset + ((z + first_z) * source_rows + y + source_y) * source_pitch + source_x;
        const size_t to = base_offset + ((z + first_z) * destination_rows + y + destination_y) * destination_pitch + destination_x;
        std::copy_n(input.begin() + from, width, expected.begin() + to);
    }
    perf_require(width * height * depth == bytes, "layout cases preserve the declared useful byte count");
    CUDA_MEMCPY2D plane{};
    plane.srcMemoryType = plane.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    plane.srcDevice = source->value; plane.dstDevice = destination->value;
    plane.srcPitch = source_pitch; plane.dstPitch = destination_pitch;
    plane.srcXInBytes = source_x; plane.dstXInBytes = destination_x;
    plane.srcY = source_y; plane.dstY = destination_y;
    plane.WidthInBytes = width; plane.Height = height;
    CUDA_MEMCPY3D box{};
    box.srcMemoryType = box.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    box.srcDevice = source->value; box.dstDevice = destination->value;
    box.srcPitch = source_pitch; box.dstPitch = destination_pitch;
    box.srcHeight = source_rows; box.dstHeight = destination_rows;
    box.srcXInBytes = source_x; box.dstXInBytes = destination_x;
    box.srcY = source_y; box.dstY = destination_y;
    box.srcZ = box.dstZ = first_z;
    box.WidthInBytes = width; box.Height = height; box.Depth = depth;
    const auto prepare = [&] {
        PERF_CUDA(cuMemcpyHtoD(source->value, input.data(), input.size()));
        PERF_CUDA(cuMemcpyHtoD(destination->value, reset.data(), reset.size()));
        PERF_CUDA(cuCtxSynchronize());
    };
    const auto operation = [&] {
        if (linear) PERF_CUDA(cuMemcpyDtoDAsync(destination->value + guard, source->value + guard, bytes, stream.value));
        else if (volume) PERF_CUDA(cuMemcpy3DAsync(&box, stream.value));
        else PERF_CUDA(cuMemcpy2DAsync(&plane, stream.value));
    };
    const auto verify = [&] {
        std::vector<unsigned char> actual_destination(reset.size()), actual_source(input.size());
        PERF_CUDA(cuMemcpyDtoH(actual_destination.data(), destination->value, actual_destination.size()));
        perf_expect_bytes(actual_destination, expected, "every layout row, slice, padding byte and allocation guard");
        PERF_CUDA(cuMemcpyDtoH(actual_source.data(), source->value, actual_source.size()));
        perf_expect_bytes(actual_source, input, "complete layout source and padding preserved");
    };
    prepare();
    operation();
    stream.synchronize();
    verify();
    prepare();
    auto result = measure_perf(options, operation, [&] { stream.synchronize(); });
    verify();
    result.parameters = {{"api", "driver"}, {"route", linear ? "linear_d2d_async" : volume ? "pitched_3d_d2d_async" : "pitched_2d_d2d_async"},
        {"useful_bytes", std::to_string(bytes)}, {"width_bytes", std::to_string(width)},
        {"height_rows", std::to_string(height)}, {"depth_slices", std::to_string(depth)},
        {"source_pitch_bytes", std::to_string(source_pitch)}, {"destination_pitch_bytes", std::to_string(destination_pitch)},
        {"source_slice_rows", std::to_string(source_rows)}, {"destination_slice_rows", std::to_string(destination_rows)},
        {"source_origin", std::to_string(source_x) + "," + std::to_string(source_y) + "," + std::to_string(first_z)},
        {"destination_origin", std::to_string(destination_x) + "," + std::to_string(destination_y) + "," + std::to_string(first_z)},
        {"source_storage_bytes", std::to_string(input.size())}, {"destination_storage_bytes", std::to_string(reset.size())},
        {"guard_bytes_minimum_per_side", std::to_string(guard)}, {"seed", "0x1943b6ef"},
        {"allocation", linear ? "cuMemAlloc" : "cuMemAllocPitch_native_pitch_with_padding_rows"},
        {"completion", "non_blocking_stream_sync_per_batch"}, {"state", "reused_context_stream_and_storage_steady_overwrite"},
        {"normalization", "one_complete_logical_copy_per_iteration"}};
    device.describe(result);
    destination->close();
    source->close();
    stream.close();
    context.close();
    return result;
}

}

int main(int argc, char** argv) {
    return cudavm::run_perf(argc, argv, "memory_copy_layout", cudavm::execute_memory_copy_layout);
}
