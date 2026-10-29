#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <set>
#include <utility>

namespace cudavm {
void check_pointer_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define POINTER_RUNTIME(call) ::cudavm::check_pointer_runtime((call), #call)

template<class T> struct pointer_output {
    alignas(T) std::array<unsigned char, sizeof(T) + 32> bytes;
    pointer_output() { bytes.fill(0xe7); }
    void* data() { return bytes.data() + 16; }
    T get() const {
        for (size_t i = 0; i < bytes.size(); ++i)
            if (i < 16 || i >= 16 + sizeof(T)) require(bytes[i] == 0xe7, "pointer attribute output byte guards");
        T value{};
        std::memcpy(&value, bytes.data() + 16, sizeof(T));
        return value;
    }
};
template<class T> T pointer_query(CUpointer_attribute attribute, CUdeviceptr address) {
    pointer_output<T> output;
    CONFORMANCE_CUDA(cuPointerGetAttribute(output.data(), attribute, address));
    return output.get();
}
cudaPointerAttributes pointer_runtime_query(CUdeviceptr address) {
    pointer_output<cudaPointerAttributes> output;
    POINTER_RUNTIME(cudaPointerGetAttributes(static_cast<cudaPointerAttributes*>(output.data()),
        reinterpret_cast<void*>(static_cast<uintptr_t>(address))));
    return output.get();
}
struct pointer_snapshot {
    CUcontext owner;
    unsigned type;
    CUdeviceptr device;
    void* host;
    unsigned long long id;
    unsigned managed;
    int ordinal;
    unsigned mapped;
    bool operator==(const pointer_snapshot&) const = default;
};
pointer_snapshot pointer_bulk_query(CUdeviceptr address) {
    pointer_output<CUcontext> owner;
    pointer_output<unsigned> type, managed, mapped;
    pointer_output<CUdeviceptr> device;
    pointer_output<void*> host;
    pointer_output<unsigned long long> id;
    pointer_output<int> ordinal;
    CUpointer_attribute attributes[]{CU_POINTER_ATTRIBUTE_CONTEXT, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
        CU_POINTER_ATTRIBUTE_DEVICE_POINTER, CU_POINTER_ATTRIBUTE_HOST_POINTER, CU_POINTER_ATTRIBUTE_BUFFER_ID,
        CU_POINTER_ATTRIBUTE_IS_MANAGED, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, CU_POINTER_ATTRIBUTE_MAPPED};
    void* outputs[]{owner.data(), type.data(), device.data(), host.data(), id.data(), managed.data(), ordinal.data(), mapped.data()};
    CONFORMANCE_CUDA(cuPointerGetAttributes(unsigned(std::size(attributes)), attributes, outputs, address));
    return {owner.get(), type.get(), device.get(), host.get(), id.get(), managed.get(), ordinal.get(), mapped.get()};
}
void pointer_scalar_agreement(CUdeviceptr address, const pointer_snapshot& expected) {
    require(pointer_query<CUcontext>(CU_POINTER_ATTRIBUTE_CONTEXT, address) == expected.owner, "scalar pointer context");
    require(pointer_query<unsigned>(CU_POINTER_ATTRIBUTE_MEMORY_TYPE, address) == expected.type, "scalar memory type");
    require(pointer_query<CUdeviceptr>(CU_POINTER_ATTRIBUTE_DEVICE_POINTER, address) == expected.device, "scalar device alias");
    require(pointer_query<unsigned long long>(CU_POINTER_ATTRIBUTE_BUFFER_ID, address) == expected.id, "scalar buffer identity");
    require(pointer_query<unsigned>(CU_POINTER_ATTRIBUTE_IS_MANAGED, address) == expected.managed, "scalar managed state");
    require(pointer_query<int>(CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, address) == expected.ordinal, "scalar device ordinal");
    require(pointer_query<unsigned>(CU_POINTER_ATTRIBUTE_MAPPED, address) == expected.mapped, "scalar mapped state");
    if (expected.host) require(pointer_query<void*>(CU_POINTER_ATTRIBUTE_HOST_POINTER, address) == expected.host, "scalar host alias");
    else {
        void* unused{};
        require(cuPointerGetAttribute(&unused, CU_POINTER_ATTRIBUTE_HOST_POINTER, address) == CUDA_ERROR_INVALID_VALUE,
            "device-only scalar host alias is unavailable");
    }
}
struct pointer_allocation {
    unsigned kind;
    CUdeviceptr address{};
    void* host{};
    size_t size;
    CUcontext owner{};
    CUdevice device{};
    std::vector<unsigned char> backing, expected;
    pointer_allocation(unsigned form, size_t bytes, bool portable = false) : kind(form), size(bytes) {
        CONFORMANCE_CUDA(cuCtxGetCurrent(&owner));
        CONFORMANCE_CUDA(cuCtxGetDevice(&device));
        if (kind == 0) CONFORMANCE_CUDA(cuMemAlloc(&address, bytes));
        else if (kind == 1) {
            void* pointer{};
            POINTER_RUNTIME(cudaMalloc(&pointer, bytes));
            address = reinterpret_cast<uintptr_t>(pointer);
        } else if (kind == 2 || kind == 3) {
            size_t pitch = 0;
            if (kind == 2) CONFORMANCE_CUDA(cuMemAllocPitch(&address, &pitch, bytes, 3, 4));
            else {
                void* pointer{};
                POINTER_RUNTIME(cudaMallocPitch(&pointer, &pitch, bytes, 3));
                address = reinterpret_cast<uintptr_t>(pointer);
            }
            size = pitch * 3;
        } else {
            const unsigned flags = CU_MEMHOSTALLOC_DEVICEMAP | (portable ? CU_MEMHOSTALLOC_PORTABLE : 0);
            if (kind == 4) CONFORMANCE_CUDA(cuMemHostAlloc(&host, bytes, flags));
            else if (kind == 5) POINTER_RUNTIME(cudaHostAlloc(&host, bytes, flags));
            else {
                constexpr uintptr_t alignment = 65536;
                backing.resize(alignment * 3, 0xa9);
                const auto start = reinterpret_cast<uintptr_t>(backing.data());
                host = reinterpret_cast<void*>(((start + alignment - 1) & ~(alignment - 1)) + 17);
                if (kind == 6) CONFORMANCE_CUDA(cuMemHostRegister(host, bytes, flags));
                else POINTER_RUNTIME(cudaHostRegister(host, bytes, flags));
            }
            CONFORMANCE_CUDA(cuMemHostGetDevicePointer(&address, host, 0));
        }
        expected.resize(size);
        for (size_t i = 0; i < size; ++i) expected[i] = static_cast<unsigned char>((i * 73 + kind * 31) ^ (i >> 8));
        if (host) std::memcpy(host, expected.data(), size);
        else CONFORMANCE_CUDA(cuMemcpyHtoD(address, expected.data(), size));
    }
    ~pointer_allocation() { if (host) { if (kind < 6) cuMemFreeHost(host); else cuMemHostUnregister(host); } else if (address) cuMemFree(address); }
    pointer_allocation(const pointer_allocation&) = delete;
    pointer_allocation& operator=(const pointer_allocation&) = delete;
    void release() {
        if (kind == 4) POINTER_RUNTIME(cudaFreeHost(host));
        else if (kind == 5) CONFORMANCE_CUDA(cuMemFreeHost(host));
        else if (kind == 6) POINTER_RUNTIME(cudaHostUnregister(host));
        else if (kind == 7) CONFORMANCE_CUDA(cuMemHostUnregister(host));
        else if (kind % 2) CONFORMANCE_CUDA(cuMemFree(address));
        else POINTER_RUNTIME(cudaFree(reinterpret_cast<void*>(static_cast<uintptr_t>(address))));
        host = nullptr; address = 0;
    }
    void verify_bytes() const {
        std::vector<unsigned char> actual(size);
        if (host) std::memcpy(actual.data(), host, size);
        else CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), address, size));
        require(actual == expected, "pointer queries preserve allocation bytes");
        if (!backing.empty()) {
            const auto offset = static_cast<const unsigned char*>(host) - backing.data();
            for (size_t i = 0; i < backing.size(); ++i)
                if ((i < size_t(offset) || i >= size_t(offset) + size) && backing[i] != 0xa9)
                    throw std::runtime_error("pointer queries preserve registered backing guards");
        }
    }
};
void verify_pointer_allocation(const pointer_allocation& allocation) {
    const auto id = pointer_query<unsigned long long>(CU_POINTER_ATTRIBUTE_BUFFER_ID, allocation.address);
    require(id != 0, "allocation has a nonzero buffer ID");
    for (size_t offset : {size_t(0), allocation.size / 2, allocation.size - 1}) {
        const auto device = allocation.address + offset;
        const auto host = allocation.host ? static_cast<unsigned char*>(allocation.host) + offset : nullptr;
        const pointer_snapshot expected{allocation.owner, unsigned(allocation.host ? CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_DEVICE),
            device, host, id, 0, allocation.device, 1};
        for (unsigned view = 0; view < (host ? 2u : 1u); ++view) {
            const auto address = view ? CUdeviceptr(reinterpret_cast<uintptr_t>(host)) : device;
            require(pointer_bulk_query(address) == expected, "bulk pointer identity, kind, ownership and aliases");
            pointer_scalar_agreement(address, expected);
            const auto runtime = pointer_runtime_query(address);
            require(runtime.type == (host ? cudaMemoryTypeHost : cudaMemoryTypeDevice)
                && runtime.device == allocation.device && runtime.hostPointer == host
                && reinterpret_cast<uintptr_t>(runtime.devicePointer) == device,
                "Runtime classification, owner ordinal and interior aliases");
            require(runtime.localityDomainOrdinal == -1, "nonlocalized allocation locality domain");
            for (auto reserved : runtime.reserved) require(reserved == 0, "Runtime reserved pointer fields");
        }
        if (!allocation.host) {
            CUdeviceptr base{};
            size_t extent = 0;
            CONFORMANCE_CUDA(cuMemGetAddressRange(&base, &extent, device));
            require(base == allocation.address && extent >= allocation.size, "queried allocation range contains requested bytes");
            require(pointer_query<CUdeviceptr>(CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, device) == base, "pointer range base");
            require(pointer_query<size_t>(CU_POINTER_ATTRIBUTE_RANGE_SIZE, device) == extent, "pointer range extent");
            pointer_output<CUdeviceptr> bulk_base;
            pointer_output<size_t> bulk_size;
            CUpointer_attribute attributes[]{CU_POINTER_ATTRIBUTE_RANGE_SIZE, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR};
            void* outputs[]{bulk_size.data(), bulk_base.data()};
            CONFORMANCE_CUDA(cuPointerGetAttributes(2, attributes, outputs, device));
            require(bulk_base.get() == base && bulk_size.get() == extent, "bulk reordered range attributes");
        }
    }
    allocation.verify_bytes();
}
void pointer_attributes_device() {
    std::set<unsigned long long> identities;
    for (unsigned kind = 0; kind < 4; ++kind) for (size_t size : {size_t(1), size_t(3), size_t(257), size_t(8193)}) {
        pointer_allocation allocation(kind, size);
        verify_pointer_allocation(allocation);
        require(identities.insert(pointer_query<unsigned long long>(CU_POINTER_ATTRIBUTE_BUFFER_ID, allocation.address)).second,
            "distinct allocation lifetimes have unique buffer IDs");
        allocation.release();
    }
}
void pointer_attributes_host() {
    for (unsigned kind = 4; kind < 8; ++kind) for (bool portable : {false, true})
    for (size_t size : {size_t(1), size_t(31), size_t(4093)}) {
        pointer_allocation allocation(kind, size, portable);
        verify_pointer_allocation(allocation);
        allocation.release();
    }
}
struct pointer_query_context {
    CUcontext original{}, value{};
    pointer_query_context() {
        CUdevice device{};
        CONFORMANCE_CUDA(cuCtxGetCurrent(&original));
        CONFORMANCE_CUDA(cuCtxGetDevice(&device));
        CONFORMANCE_CUDA(cuCtxCreate(&value, nullptr, CU_CTX_MAP_HOST, device));
    }
    ~pointer_query_context() { if (value) cuCtxDestroy(value); cuCtxSetCurrent(original); }
};
void pointer_attributes_ownership() {
    for (unsigned kind : {0u, 1u, 4u, 5u, 6u, 7u}) {
        pointer_allocation allocation(kind, 257, true);
        {
            pointer_query_context other;
            for (size_t offset : {size_t(0), size_t(127), size_t(256)}) {
                require(pointer_query<CUcontext>(CU_POINTER_ATTRIBUTE_CONTEXT, allocation.address + offset) == allocation.owner,
                    "foreign-current-context query retains allocation owner");
                require(pointer_query<int>(CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, allocation.address + offset) == allocation.device,
                    "foreign-current-context query retains allocation device");
                CUcontext current{};
                CONFORMANCE_CUDA(cuCtxGetCurrent(&current));
                require(current == other.value, "pointer introspection preserves current context");
            }
        }
        verify_pointer_allocation(allocation);
        allocation.release();
    }
}
void expect_unregistered_pointer(void* pointer) {
    const auto address = CUdeviceptr(reinterpret_cast<uintptr_t>(pointer));
    require(pointer_runtime_query(address).type == cudaMemoryTypeUnregistered, "Runtime unregistered host classification");
    require(pointer_bulk_query(address) == pointer_snapshot{}, "unregistered bulk query uses null defaults");
    unsigned type = 0;
    require(cuPointerGetAttribute(&type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, address) == CUDA_ERROR_INVALID_VALUE,
        "unregistered scalar pointer query status");
}
void pointer_attributes_lifetime() {
    std::set<unsigned long long> ids;
    for (unsigned round = 0; round < 4; ++round) {
        pointer_allocation allocation(round % 2, 257);
        require(ids.insert(pointer_query<unsigned long long>(CU_POINTER_ATTRIBUTE_BUFFER_ID, allocation.address)).second,
            "retired allocation buffer ID is never reused");
        verify_pointer_allocation(allocation);
        allocation.release();
    }
    std::vector<unsigned char> backing(3 * 65536, 0x39);
    const auto start = reinterpret_cast<uintptr_t>(backing.data());
    auto* host = reinterpret_cast<unsigned char*>((start + 65535) & ~uintptr_t(65535)) + 17;
    for (unsigned round = 0; round < 4; ++round) {
        expect_unregistered_pointer(host);
        POINTER_RUNTIME(cudaHostRegister(host, 4093, cudaHostRegisterMapped));
        struct cleanup_registration { void* pointer; ~cleanup_registration() { if (pointer) cuMemHostUnregister(pointer); } } cleanup{host};
        const auto attributes = pointer_runtime_query(reinterpret_cast<uintptr_t>(host + 31));
        require(attributes.type == cudaMemoryTypeHost && attributes.hostPointer == host + 31, "registration changes pointer classification");
        CONFORMANCE_CUDA(cuMemHostUnregister(host)); cleanup.pointer = nullptr;
        expect_unregistered_pointer(host + 31);
        CONFORMANCE_CUDA(cuMemHostRegister(host, 4093, CU_MEMHOSTREGISTER_DEVICEMAP)); cleanup.pointer = host;
        require(pointer_runtime_query(reinterpret_cast<uintptr_t>(host)).type == cudaMemoryTypeHost, "opposite API re-registration classification");
        POINTER_RUNTIME(cudaHostUnregister(host)); cleanup.pointer = nullptr;
        expect_unregistered_pointer(host);
        require(std::all_of(backing.begin(), backing.end(), [](auto byte) { return byte == 0x39; }), "registration classification lifetime preserves caller bytes");
    }
}
void pointer_attributes_validation() {
    pointer_allocation allocation(0, 257);
    auto attribute = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    unsigned value = 0;
    void* output = &value;
    require(cuPointerGetAttribute(nullptr, attribute, allocation.address) == CUDA_ERROR_INVALID_VALUE, "null scalar attribute output");
    require(cudaPointerGetAttributes(nullptr, reinterpret_cast<void*>(static_cast<uintptr_t>(allocation.address))) == cudaErrorInvalidValue,
        "null Runtime attributes output");
    require(cuPointerGetAttributes(1, nullptr, &output, allocation.address) == CUDA_ERROR_INVALID_VALUE, "missing bulk attribute array");
    require(cuPointerGetAttributes(1, &attribute, nullptr, allocation.address) == CUDA_ERROR_INVALID_VALUE, "missing bulk output array");
    output = nullptr;
    require(cuPointerGetAttributes(1, &attribute, &output, allocation.address) == CUDA_ERROR_INVALID_VALUE, "null bulk attribute element");
    for (auto invalid : {static_cast<CUpointer_attribute>(0), static_cast<CUpointer_attribute>(0x7fffffff)}) {
        output = &value;
        require(cuPointerGetAttribute(&value, invalid, allocation.address) == CUDA_ERROR_INVALID_VALUE, "invalid scalar attribute enum");
        require(cuPointerGetAttributes(1, &invalid, &output, allocation.address) == CUDA_ERROR_INVALID_VALUE, "invalid bulk attribute enum");
    }
    require(pointer_bulk_query(0) == pointer_snapshot{}, "null pointer bulk defaults");
    require(cuPointerGetAttribute(&value, attribute, 0) == CUDA_ERROR_INVALID_VALUE, "null pointer scalar error");
    std::array<unsigned char, 31> ordinary{};
    expect_unregistered_pointer(ordinary.data() + 3);
    verify_pointer_allocation(allocation);
    allocation.release();
}
std::string_view pointer_attributes_case = "all";
void execute_pointer_attributes() {
    const std::array<std::pair<std::string_view, void (*)()>, 5> cases{{
        {"device", pointer_attributes_device}, {"host", pointer_attributes_host}, {"ownership", pointer_attributes_ownership},
        {"lifetime", pointer_attributes_lifetime}, {"validation", pointer_attributes_validation}}};
    bool found = false;
    for (const auto& [name, execute] : cases) if (pointer_attributes_case == "all" || pointer_attributes_case == name) {
        found = true;
        std::cout << "pointer_attributes=" << name << std::endl;
        execute();
    }
    require(found, "unknown pointer_attributes case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::pointer_attributes_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_pointer_attributes);
}
