#include "conformance.hpp"
#include <algorithm>
#include <memory>

namespace cudavm {
struct allocation_record {
    std::unique_ptr<device_buffer> storage;
    std::vector<unsigned char> expected;
};
void verify_allocation(const allocation_record& allocation, const std::string& context) {
    constexpr unsigned char guard = 0xd7;
    std::vector<unsigned char> actual(allocation.expected.size() + 16, guard);
    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data() + 8, allocation.storage->value, allocation.expected.size()));
    require(std::equal(actual.begin(), actual.begin() + 8, actual.end() - 8), context + ": host output guards");
    require(std::all_of(actual.begin(), actual.begin() + 8, [](auto value) { return value == guard; }), context + ": prefix guard");
    for (size_t i = 0; i < allocation.expected.size(); ++i)
        require(actual[i + 8] == allocation.expected[i], context + ": byte " + std::to_string(i));
    CUdeviceptr base{};
    size_t extent = 0;
    CONFORMANCE_CUDA(cuMemGetAddressRange(&base, &extent, allocation.storage->value));
    require(base == allocation.storage->value && extent >= allocation.expected.size(), context + ": base and allocation extent");
    for (size_t offset : {size_t(0), allocation.expected.size() / 2, allocation.expected.size() - 1}) {
        CUdeviceptr interior_base{};
        size_t interior_extent = 0;
        CONFORMANCE_CUDA(cuMemGetAddressRange(&interior_base, &interior_extent, allocation.storage->value + offset));
        require(interior_base == base && interior_extent == extent, context + ": interior allocation identity");
        CONFORMANCE_CUDA(cuMemGetAddressRange(nullptr, &interior_extent, allocation.storage->value + offset));
        require(interior_extent == extent, context + ": optional base output");
        CONFORMANCE_CUDA(cuMemGetAddressRange(&interior_base, nullptr, allocation.storage->value + offset));
        require(interior_base == base, context + ": optional extent output");
        CONFORMANCE_CUDA(cuMemGetAddressRange(nullptr, nullptr, allocation.storage->value + offset));
    }
}
void execute_device_allocation() {
    CUdeviceptr zero{};
    require(cuMemAlloc(&zero, 0) == CUDA_ERROR_INVALID_VALUE, "zero-size Driver allocation status");
    // Failure output values and fresh allocation contents are unspecified.
    for (unsigned generation = 0; generation < 3; ++generation) {
        std::vector<allocation_record> allocations;
        for (size_t size : {size_t(1), size_t(3), size_t(7), size_t(16), size_t(31),
                            size_t(64), size_t(255), size_t(256), size_t(257), size_t(4093)}) {
            allocation_record allocation{std::make_unique<device_buffer>(size), std::vector<unsigned char>(size)};
            require(allocation.storage->value && allocation.storage->value % 16 == 0, "scalar/vector allocation alignment");
            for (size_t i = 0; i < size; ++i) allocation.expected[i] = static_cast<unsigned char>(i * 37 + size * 13 + generation * 71);
            const auto preserved = allocation.expected;
            CONFORMANCE_CUDA(cuMemcpyHtoD(allocation.storage->value, allocation.expected.data(), size));
            require(preserved == allocation.expected, "allocation upload source preservation");
            for (const auto& other : allocations) {
                const auto a = allocation.storage->value, b = other.storage->value;
                require(a + size <= b || b + other.expected.size() <= a, "live allocations have disjoint requested ranges");
            }
            allocations.push_back(std::move(allocation));
        }
        for (unsigned phase = 0; phase < 3; ++phase) {
            for (size_t index = 0; index < allocations.size(); ++index) {
                auto& allocation = allocations[index];
                const size_t offset = phase == 0 ? 0 : phase == 1 ? allocation.expected.size() / 2 : allocation.expected.size() - 1;
                const unsigned char value = static_cast<unsigned char>(0x83 + index * 7 + phase * 31);
                CONFORMANCE_CUDA(cuMemcpyHtoD(allocation.storage->value + offset, &value, 1));
                allocation.expected[offset] = value;
            }
            for (size_t index = 0; index < allocations.size(); ++index)
                verify_allocation(allocations[index], "allocation phase=" + std::to_string(phase) + " index=" + std::to_string(index));
        }
        // Retire alternate owners, then allocate and initialize replacements.
        // Surviving allocations must retain their complete byte contents.
        for (size_t index = 0; index < allocations.size(); index += 2) {
            CONFORMANCE_CUDA(cuMemFree(allocations[index].storage->value));
            allocations[index].storage->value = 0;
        }
        for (size_t index = 0; index < allocations.size(); index += 2) {
            auto& allocation = allocations[index];
            allocation.storage = std::make_unique<device_buffer>(allocation.expected.size());
            std::fill(allocation.expected.begin(), allocation.expected.end(), static_cast<unsigned char>(generation + index + 19));
            CONFORMANCE_CUDA(cuMemcpyHtoD(allocation.storage->value, allocation.expected.data(), allocation.expected.size()));
        }
        for (size_t index = 0; index < allocations.size(); ++index)
            verify_allocation(allocations[index], "replacement allocation index=" + std::to_string(index));
        for (auto& allocation : allocations) {
            CONFORMANCE_CUDA(cuMemFree(allocation.storage->value));
            allocation.storage->value = 0;
        }
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::execute_device_allocation);
}
