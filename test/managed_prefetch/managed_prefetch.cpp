#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <cudaTypedefs.h>
#include <algorithm>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace cudavm {
std::string_view prefetch_case;
void prefetch_runtime(cudaError_t status) {
    require(status == cudaSuccess, std::string("prefetch Runtime: ") + cudaGetErrorString(status));
}
CUmemLocation prefetch_location(bool device = false) {
    CUmemLocation location{};
    location.type = device ? CU_MEM_LOCATION_TYPE_DEVICE : CU_MEM_LOCATION_TYPE_HOST;
    if (device) CONFORMANCE_CUDA(cuCtxGetDevice(&location.id));
    else location.id = 12345; // Ignored for the host location type.
    return location;
}
struct prefetch_storage {
    CUdeviceptr value{}, page{};
    size_t page_size{}, bytes{};
    std::vector<unsigned char> expected;
    explicit prefetch_storage(bool runtime) {
#ifdef _WIN32
        SYSTEM_INFO info{}; GetSystemInfo(&info); page_size = info.dwPageSize;
#else
        page_size = size_t(sysconf(_SC_PAGESIZE));
#endif
        require(page_size && !(page_size & (page_size - 1)), "prefetch CPU page size");
        bytes = page_size * 6;
        if (runtime) { void* pointer{}; prefetch_runtime(cudaMallocManaged(&pointer, bytes, cudaMemAttachGlobal)); value = reinterpret_cast<uintptr_t>(pointer); }
        else CONFORMANCE_CUDA(cuMemAllocManaged(&value, bytes, CU_MEM_ATTACH_GLOBAL));
        page = (value + page_size - 1) & ~(CUdeviceptr(page_size) - 1);
        expected.resize(bytes);
        for (size_t i = 0; i < bytes; ++i) expected[i] = static_cast<unsigned char>((i * 37 + 51) ^ (i >> 7));
        std::copy(expected.begin(), expected.end(), host());
    }
    ~prefetch_storage() { if (value) cuMemFree(value); }
    unsigned char* host() const { return reinterpret_cast<unsigned char*>(uintptr_t(value)); }
    void verify() const { require(std::equal(expected.begin(), expected.end(), host()), "prefetch full managed bytes and guards"); }
};
struct prefetch_stream {
    CUstream value{};
    prefetch_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~prefetch_stream() { cuStreamSynchronize(value); cuStreamDestroy(value); }
};
CUresult prefetch_submit(bool runtime, CUdeviceptr pointer, size_t bytes, CUmemLocation location, CUstream stream, unsigned flags = 0) {
    if (!runtime) return cuMemPrefetchAsync(pointer, bytes, location, flags, stream);
    cudaMemLocation target{}; target.type = static_cast<cudaMemLocationType>(location.type); target.id = location.id;
    const auto result = cudaMemPrefetchAsync(reinterpret_cast<void*>(uintptr_t(pointer)), bytes, target, flags, reinterpret_cast<cudaStream_t>(stream));
    if (result == cudaSuccess) return CUDA_SUCCESS;
    require(result == cudaErrorInvalidValue || result == cudaErrorInvalidDevice, "prefetch documented Runtime validation error");
    require(cudaGetLastError() == result && cudaGetLastError() == cudaSuccess, "prefetch Runtime error clears");
    return result == cudaErrorInvalidValue ? CUDA_ERROR_INVALID_VALUE : CUDA_ERROR_INVALID_DEVICE;
}
int prefetch_query(CUdeviceptr pointer, size_t bytes, bool runtime) {
    struct { int before = 0x19375928, value = 99, after = 0x29183745; } output;
    if (runtime) prefetch_runtime(cudaMemRangeGetAttribute(&output.value, sizeof(int), cudaMemRangeAttributeLastPrefetchLocation, reinterpret_cast<void*>(uintptr_t(pointer)), bytes));
    else CONFORMANCE_CUDA(cuMemRangeGetAttribute(&output.value, sizeof(int), CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION, pointer, bytes));
    require(output.before == 0x19375928 && output.after == 0x29183745, "prefetch query exact-width guards");
    return output.value;
}
void prefetch_discovery() {
    using modern = PFN_cuMemPrefetchAsync_v12020;
    using legacy = PFN_cuMemPrefetchAsync_v8000;
    for (auto flags : {CU_GET_PROC_ADDRESS_LEGACY_STREAM, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM}) {
        for (int version : {8000, 12020}) {
            void* address{}; CUdriverProcAddressQueryResult status{};
            CONFORMANCE_CUDA(cuGetProcAddress("cuMemPrefetchAsync", &address, version, flags, &status));
            require(address && status == CU_GET_PROC_ADDRESS_SUCCESS, "prefetch ABI discovery");
            const auto result = version == 8000 ? reinterpret_cast<legacy>(address)(0, 1, CU_DEVICE_CPU, nullptr) :
                reinterpret_cast<modern>(address)(0, 1, {}, 0, nullptr);
            require(result == CUDA_ERROR_NOT_INITIALIZED, "discovered prefetch does not initialize Driver");
            void* runtime{}; cudaDriverEntryPointQueryResult query{};
            prefetch_runtime(cudaGetDriverEntryPointByVersion("cuMemPrefetchAsync", &runtime, version, flags, &query));
            require(runtime == address && query == cudaDriverEntryPointSuccess, "Runtime discovers matching Driver prefetch ABI");
        }
    }
    void* missing{}; CUdriverProcAddressQueryResult status{};
    CONFORMANCE_CUDA(cuGetProcAddress("cuMemPrefetchAsync", &missing, 7999, 0, &status));
    require(!missing && status == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "prefetch introduction version");
}
void prefetch_ordering(bool dependencies) {
    for (bool runtime : {false, true}) {
        prefetch_storage storage(runtime), independent(!runtime);
        prefetch_stream producer, consumer;
        device_buffer snapshot(storage.bytes);
        CUevent event{}; CONFORMANCE_CUDA(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
        struct event_guard { CUevent value; ~event_guard() { cuEventDestroy(value); } } cleanup{event};
        for (unsigned phase = 0; phase < 3; ++phase) {
            CONFORMANCE_CUDA(prefetch_submit(runtime, storage.value, storage.bytes, prefetch_location(true), producer.value));
            const size_t offset = 17 + phase * 71, count = storage.bytes - offset - 29;
            const auto pattern = static_cast<unsigned char>(phase * 37 + 19);
            if (runtime) prefetch_runtime(cudaMemsetAsync(storage.host() + offset, pattern, count, reinterpret_cast<cudaStream_t>(producer.value)));
            else CONFORMANCE_CUDA(cuMemsetD8Async(storage.value + offset, pattern, count, producer.value));
            auto stream = producer.value;
            if (dependencies) {
                prefetch_runtime(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event), reinterpret_cast<cudaStream_t>(producer.value)));
                CONFORMANCE_CUDA(cuStreamWaitEvent(consumer.value, event, 0)); stream = consumer.value;
            }
            CONFORMANCE_CUDA(cuMemcpyDtoDAsync(snapshot.value, storage.value, storage.bytes, stream));
            CONFORMANCE_CUDA(prefetch_submit(!runtime, storage.value, storage.bytes, prefetch_location(), stream));
            prefetch_runtime(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
            std::fill_n(storage.expected.begin() + offset, count, pattern);
            storage.verify(); independent.verify();
            std::vector<unsigned char> actual(storage.bytes);
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), snapshot.value, actual.size()));
            require(actual == storage.expected, "prefetch event/stream ordering and complete snapshot");
        }
        for (auto flags : {CU_GET_PROC_ADDRESS_LEGACY_STREAM, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM}) {
            for (int version : {8000, 12020}) {
                void* address{}; CONFORMANCE_CUDA(cuGetProcAddress("cuMemPrefetchAsync", &address, version, flags, nullptr));
                for (CUstream stream : {CUstream(nullptr), flags == CU_GET_PROC_ADDRESS_LEGACY_STREAM ? CU_STREAM_LEGACY : CU_STREAM_PER_THREAD}) {
                    if (version == 8000) CONFORMANCE_CUDA(reinterpret_cast<PFN_cuMemPrefetchAsync_v8000>(address)(storage.value, storage.bytes, CU_DEVICE_CPU, stream));
                    else CONFORMANCE_CUDA(reinterpret_cast<PFN_cuMemPrefetchAsync_v12020>(address)(storage.value, storage.bytes, prefetch_location(), 0, stream));
                    CONFORMANCE_CUDA(cuStreamSynchronize(stream ? stream : flags == CU_GET_PROC_ADDRESS_LEGACY_STREAM ? CU_STREAM_LEGACY : CU_STREAM_PER_THREAD));
                    storage.verify();
                }
            }
        }
    }
}
void prefetch_ranges() {
    for (bool runtime : {false, true}) {
        prefetch_storage storage(runtime), independent(!runtime); prefetch_stream stream;
        const auto page = storage.page, size = storage.page_size;
        CONFORMANCE_CUDA(prefetch_submit(runtime, page + size / 2, 1, prefetch_location(true), stream.value));
        CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
        const int device = prefetch_location(true).id;
        require(prefetch_query(page, size, !runtime) == device, "interior prefetch rounds to its CPU page");
        require(prefetch_query(page + 2 * size, size, runtime) == CU_DEVICE_INVALID, "prefetch preserves distant page metadata");
        require(prefetch_query(page, 3 * size, runtime) == CU_DEVICE_INVALID, "mixed page destinations have no common last prefetch");
        require(prefetch_query(independent.page, size, !runtime) == CU_DEVICE_INVALID, "prefetch independent allocation metadata");
        CONFORMANCE_CUDA(prefetch_submit(!runtime, page + size - 1, 2, prefetch_location(), stream.value));
        CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
        require(prefetch_query(page, 2 * size, runtime) == CU_DEVICE_CPU, "cross-page host prefetch rounds both endpoints");
        storage.verify(); independent.verify();
    }
}
void prefetch_validation(bool concurrent) {
    prefetch_storage storage(false); prefetch_stream stream; device_buffer ordinary(64);
    for (bool runtime : {false, true}) {
        auto location = prefetch_location();
        require(prefetch_submit(runtime, storage.value, storage.bytes, location, stream.value, 1) == CUDA_ERROR_INVALID_VALUE, "reserved prefetch flags");
        auto invalid = location; invalid.type = CU_MEM_LOCATION_TYPE_INVALID;
        require(prefetch_submit(runtime, storage.value, storage.bytes, invalid, stream.value) == CUDA_ERROR_INVALID_VALUE, "invalid prefetch location type");
        invalid = prefetch_location(true); invalid.id = -2;
        require(prefetch_submit(runtime, storage.value, storage.bytes, invalid, stream.value) == CUDA_ERROR_INVALID_DEVICE, "invalid prefetch device");
        for (auto pointer : {CUdeviceptr(0), ordinary.value})
            require(prefetch_submit(runtime, pointer, 1, location, stream.value) == CUDA_ERROR_INVALID_VALUE, "prefetch requires managed storage");
        require(prefetch_submit(runtime, storage.value + storage.bytes - 1, 2, location, stream.value) == CUDA_ERROR_INVALID_VALUE, "prefetch range extent");
        if (concurrent) CONFORMANCE_CUDA(prefetch_submit(runtime, storage.value, storage.bytes, location, stream.value));
        CONFORMANCE_CUDA(cuMemsetD8Async(storage.value + 7, 0x73, 13, stream.value));
        CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
        std::fill_n(storage.expected.begin() + 7, 13, 0x73); storage.verify();
    }
}
void execute_managed_prefetch() {
    CUdevice device{}; int managed{}, concurrent{};
    CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&managed, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, device));
    require_capability(managed != 0, "prefetch requires managed memory");
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&concurrent, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, device));
    if (prefetch_case == "validation") { prefetch_validation(concurrent != 0); return; }
    require_capability(concurrent != 0, "successful prefetch requires concurrent managed access on the stream device");
    if (prefetch_case == "ordering") prefetch_ordering(false);
    else if (prefetch_case == "dependencies") prefetch_ordering(true);
    else if (prefetch_case == "ranges") prefetch_ranges();
    else throw std::runtime_error("unknown managed_prefetch case");
}
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "discovery") return cudavm::run_host_conformance(argc, argv, cudavm::prefetch_discovery);
    if (argc == 3) { cudavm::prefetch_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_managed_prefetch);
}
