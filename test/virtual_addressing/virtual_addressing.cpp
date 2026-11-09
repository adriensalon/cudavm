#include "test_framework.hpp"
#include <algorithm>
#include <limits>

namespace cudavm {
std::string_view virtual_case;
CUmemAllocationProp virtual_properties() {
    CUmemAllocationProp properties{};
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    CONFORMANCE_CUDA(cuCtxGetDevice(&properties.location.id));
    return properties;
}
size_t virtual_granularity(CUmemAllocationGranularity_flags option) {
    struct { uint64_t before = 0x19375928abcdef01; size_t value = 0; uint64_t after = 0xfedcba0129183745; } output;
    const auto properties = virtual_properties();
    CONFORMANCE_CUDA(cuMemGetAllocationGranularity(&output.value, &properties, option));
    require(output.before == 0x19375928abcdef01 && output.after == 0xfedcba0129183745, "allocation granularity exact output width");
    require(output.value > 0 && !(output.value & (output.value - 1)), "allocation granularity is a positive power of two");
    return output.value;
}
struct virtual_range {
    CUdeviceptr value{};
    size_t size;
    virtual_range(size_t bytes, size_t alignment = 0, CUdeviceptr hint = 0) : size(bytes) {
        struct { CUdeviceptr before = 0x123456789abcdef0, value = 0, after = 0xfedcba9876543210; } output;
        CONFORMANCE_CUDA(cuMemAddressReserve(&output.value, size, alignment, hint, 0));
        require(output.before == 0x123456789abcdef0 && output.after == 0xfedcba9876543210, "reservation exact output width");
        value = output.value;
        require(value && size <= std::numeric_limits<CUdeviceptr>::max() - value, "reservation nonoverflowing extent");
        require(!alignment || value % alignment == 0, "reservation requested alignment");
    }
    ~virtual_range() { if (value) cuMemAddressFree(value, size); }
    void close() { CONFORMANCE_CUDA(cuMemAddressFree(value, size)); value = 0; }
};
void virtual_granularities() {
    const auto minimum = virtual_granularity(CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    const auto recommended = virtual_granularity(CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
    require(recommended >= minimum && recommended % minimum == 0, "recommended granularity is compatible with minimum");
    for (unsigned i = 0; i < 3; ++i) {
        require(virtual_granularity(CU_MEM_ALLOC_GRANULARITY_MINIMUM) == minimum, "stable minimum granularity");
        require(virtual_granularity(CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) == recommended, "stable recommended granularity");
        virtual_range range(minimum * (i + 1), minimum);
        range.close();
    }
}
void virtual_reservations() {
    const auto minimum = virtual_granularity(CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    require(minimum <= std::numeric_limits<size_t>::max() / 16, "bounded virtual test size");
    for (size_t alignment : {size_t(0), minimum, minimum * 2, minimum * 8}) {
        virtual_range first(minimum * 3, alignment), second(minimum * 5, alignment), third(minimum, alignment);
        const std::array<const virtual_range*, 3> ranges{&first, &second, &third};
        for (size_t i = 0; i < ranges.size(); ++i) for (size_t j = i + 1; j < ranges.size(); ++j)
            require(ranges[i]->value + ranges[i]->size <= ranges[j]->value || ranges[j]->value + ranges[j]->size <= ranges[i]->value,
                    "simultaneously live reservations are disjoint");
        const auto hint = second.value; second.close();
        virtual_range replacement(minimum * 5, alignment, hint);
        require(replacement.value + replacement.size <= first.value || first.value + first.size <= replacement.value, "replacement leaves first reservation disjoint");
        require(replacement.value + replacement.size <= third.value || third.value + third.size <= replacement.value, "replacement leaves third reservation disjoint");
        third.close(); first.close(); replacement.close();
    }
}
void virtual_validation() {
    const auto minimum = virtual_granularity(CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    auto properties = virtual_properties(); size_t granularity = 0; CUdeviceptr output = 0;
    require(cuMemGetAllocationGranularity(nullptr, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_ERROR_INVALID_VALUE, "missing granularity output");
    require(cuMemGetAllocationGranularity(&granularity, nullptr, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_ERROR_INVALID_VALUE, "missing granularity properties");
    require(cuMemGetAllocationGranularity(&granularity, &properties, static_cast<CUmemAllocationGranularity_flags>(7)) == CUDA_ERROR_INVALID_VALUE, "invalid granularity option");
    require(cuMemAddressReserve(nullptr, minimum, 0, 0, 0) == CUDA_ERROR_INVALID_VALUE, "missing reservation output");
    require(cuMemAddressReserve(&output, 0, 0, 0, 0) == CUDA_ERROR_INVALID_VALUE, "empty reservation");
    require(cuMemAddressReserve(&output, minimum + 1, 0, 0, 0) == CUDA_ERROR_INVALID_VALUE, "non-page-multiple reservation size");
    require(cuMemAddressReserve(&output, minimum, 3, 0, 0) == CUDA_ERROR_INVALID_VALUE, "non-power-of-two reservation alignment");
    require(cuMemAddressReserve(&output, minimum, 0, 0, 1) == CUDA_ERROR_INVALID_VALUE, "reserved reservation flags");
    virtual_range repaired(minimum * 3, minimum);
    require(cuMemAddressFree(repaired.value, minimum * 2) == CUDA_ERROR_INVALID_VALUE, "wrong reservation free size");
    require(cuMemAddressFree(repaired.value + minimum, minimum * 2) == CUDA_ERROR_INVALID_VALUE, "interior reservation free");
    require(virtual_granularity(CU_MEM_ALLOC_GRANULARITY_MINIMUM) == minimum, "granularity after rejected inputs");
    repaired.close(); virtual_range replacement(minimum); replacement.close();
}
void execute_virtual_addressing() {
    CUdevice device{}; int supported = 0;
    CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));
    require_capability(supported != 0, "virtual addressing requires advertised virtual-memory management");
    if (virtual_case == "granularity") virtual_granularities();
    else if (virtual_case == "reservations") virtual_reservations();
    else if (virtual_case == "validation") virtual_validation();
    else throw std::runtime_error("unknown virtual_addressing case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::virtual_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_virtual_addressing);
}
