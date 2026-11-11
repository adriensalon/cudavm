#include "test_framework.hpp"
#include <iostream>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>

namespace cudavm {
void sparse_runtime(cudaError_t status) {
    require(status == cudaSuccess, std::string("sparse Runtime: ") + cudaGetErrorString(status));
}
template<class T> struct sparse_output {
    uint64_t before = 0x123456789abcdef0;
    T value{};
    uint64_t after = 0xfedcba9876543210;
    void verify() const {
        require(before == 0x123456789abcdef0 && after == 0xfedcba9876543210, "sparse query output guards");
    }
};
struct sparse_resource {
    CUarray array{};
    CUmipmappedArray mipmap{};
    unsigned levels;
    sparse_resource(bool runtime = false, unsigned count = 1, unsigned depth = 0, unsigned extra_flags = 0)
        : levels(count) {
        CUDA_ARRAY3D_DESCRIPTOR descriptor{1024, 1024, depth, CU_AD_FORMAT_UNSIGNED_INT32, 1, CUDA_ARRAY3D_SPARSE | extra_flags};
        cudaChannelFormatDesc channel{32, 0, 0, 0, cudaChannelFormatKindUnsigned};
        unsigned flags = cudaArraySparse | (extra_flags & CUDA_ARRAY3D_LAYERED ? cudaArrayLayered : 0);
        if (count == 1) {
            if (runtime) {
                cudaArray_t output{};
                sparse_runtime(cudaMalloc3DArray(&output, &channel, {1024, 1024, depth}, flags));
                array = reinterpret_cast<CUarray>(output);
            } else TEST_CUDA(cuArray3DCreate(&array, &descriptor));
        } else if (runtime) {
            cudaMipmappedArray_t output{};
            sparse_runtime(cudaMallocMipmappedArray(&output, &channel, {1024, 1024, depth}, count, flags));
            mipmap = reinterpret_cast<CUmipmappedArray>(output);
        } else TEST_CUDA(cuMipmappedArrayCreate(&mipmap, &descriptor, count));
    }
    ~sparse_resource() { if (array) cuArrayDestroy(array); if (mipmap) cuMipmappedArrayDestroy(mipmap); }
    sparse_resource(const sparse_resource&) = delete;
    void close() {
        if (array) { TEST_CUDA(cuArrayDestroy(array)); array = nullptr; }
        if (mipmap) { TEST_CUDA(cuMipmappedArrayDestroy(mipmap)); mipmap = nullptr; }
    }
    CUarray level(unsigned index) const {
        if (!mipmap) return array;
        CUarray result{}; TEST_CUDA(cuMipmappedArrayGetLevel(&result, mipmap, index)); return result;
    }
    CUDA_ARRAY_SPARSE_PROPERTIES properties() const {
        sparse_output<CUDA_ARRAY_SPARSE_PROPERTIES> driver, again;
        sparse_output<cudaArraySparseProperties> runtime;
        if (mipmap) {
            TEST_CUDA(cuMipmappedArrayGetSparseProperties(&driver.value, mipmap));
            TEST_CUDA(cuMipmappedArrayGetSparseProperties(&again.value, mipmap));
            sparse_runtime(cudaMipmappedArrayGetSparseProperties(&runtime.value, reinterpret_cast<cudaMipmappedArray_t>(mipmap)));
        } else {
            TEST_CUDA(cuArrayGetSparseProperties(&driver.value, array));
            TEST_CUDA(cuArrayGetSparseProperties(&again.value, array));
            sparse_runtime(cudaArrayGetSparseProperties(&runtime.value, reinterpret_cast<cudaArray_t>(array)));
        }
        driver.verify(); again.verify(); runtime.verify();
        const auto& p = driver.value; const auto& r = runtime.value; const auto& a = again.value;
        require(p.tileExtent.width && p.tileExtent.height && p.tileExtent.depth, "nonzero sparse tile extents");
        require(p.tileExtent.width == a.tileExtent.width && p.tileExtent.height == a.tileExtent.height
            && p.tileExtent.depth == a.tileExtent.depth && p.miptailFirstLevel == a.miptailFirstLevel
            && p.miptailSize == a.miptailSize && p.flags == a.flags, "stable sparse properties");
        require(p.tileExtent.width == r.tileExtent.width && p.tileExtent.height == r.tileExtent.height
            && p.tileExtent.depth == r.tileExtent.depth && p.miptailFirstLevel == r.miptailFirstLevel
            && p.miptailSize == r.miptailSize && p.flags == r.flags, "Driver/Runtime sparse properties agree");
        require(!(p.flags & ~CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL), "known sparse layout flags");
        if (mipmap) require(p.miptailFirstLevel <= levels, "mip-tail boundary within chain");
        return p;
    }
    CUarrayMapInfo tile(unsigned x, CUmemGenericAllocationHandle physical, unsigned width, unsigned height) const {
        CUdevice device{}; TEST_CUDA(cuCtxGetDevice(&device));
        require(device >= 0 && device < 32, "bounded sparse device mask");
        CUarrayMapInfo result{};
        result.resourceType = mipmap ? CU_RESOURCE_TYPE_MIPMAPPED_ARRAY : CU_RESOURCE_TYPE_ARRAY;
        if (mipmap) result.resource.mipmap = mipmap; else result.resource.array = array;
        result.subresourceType = CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL;
        result.subresource.sparseLevel.offsetX = x;
        result.subresource.sparseLevel.extentWidth = width;
        result.subresource.sparseLevel.extentHeight = height;
        result.subresource.sparseLevel.extentDepth = 1;
        result.memOperationType = physical ? CU_MEM_OPERATION_TYPE_MAP : CU_MEM_OPERATION_TYPE_UNMAP;
        result.memHandleType = CU_MEM_HANDLE_TYPE_GENERIC; result.memHandle.memHandle = physical;
        result.deviceBitMask = 1u << device;
        return result;
    }
};
struct sparse_stream {
    CUstream value{};
    sparse_stream() { TEST_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~sparse_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    void wait() { TEST_CUDA(cuStreamSynchronize(value)); }
};
struct sparse_host {
    unsigned char* value{};
    size_t pitch, height;
    std::vector<unsigned char> expected;
    sparse_host(unsigned width, unsigned rows, unsigned seed, bool upload, unsigned origin_x = 0)
        : pitch(size_t(width) * 4 + 28), height(rows), expected(32 + pitch * rows, 0xa7) {
        void* output{}; TEST_CUDA(cuMemAllocHost(&output, expected.size()));
        value = static_cast<unsigned char*>(output);
        for (size_t y = 0; y < rows; ++y) for (size_t x = 0; x < size_t(width) * 4; ++x)
            expected[16 + y * pitch + x] = static_cast<unsigned char>(((x + origin_x * 4) * 37 + y * 71 + seed * 43) ^ ((x + origin_x * 4) >> 7));
        if (upload) std::copy(expected.begin(), expected.end(), value);
        else std::fill_n(value, expected.size(), 0xa7);
    }
    ~sparse_host() { if (value) cuMemFreeHost(value); }
    void verify() const {
        for (size_t i = 0; i < expected.size(); ++i)
            if (value[i] != expected[i]) throw std::runtime_error("sparse payload/padding/guard byte " + std::to_string(i));
    }
};
void sparse_copy(CUarray array, unsigned x, unsigned width, sparse_host& host, bool upload, bool runtime, CUstream stream) {
    if (runtime) {
        if (upload) sparse_runtime(cudaMemcpy2DToArrayAsync(reinterpret_cast<cudaArray_t>(array), size_t(x) * 4, 0,
            host.value + 16, host.pitch, size_t(width) * 4, host.height, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)));
        else sparse_runtime(cudaMemcpy2DFromArrayAsync(host.value + 16, host.pitch, reinterpret_cast<cudaArray_t>(array), size_t(x) * 4, 0,
            size_t(width) * 4, host.height, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream)));
    } else {
        CUDA_MEMCPY2D p{};
        if (upload) {
            p.srcMemoryType = CU_MEMORYTYPE_HOST; p.srcHost = host.value + 16; p.srcPitch = host.pitch;
            p.dstMemoryType = CU_MEMORYTYPE_ARRAY; p.dstArray = array; p.dstXInBytes = size_t(x) * 4;
        } else {
            p.srcMemoryType = CU_MEMORYTYPE_ARRAY; p.srcArray = array; p.srcXInBytes = size_t(x) * 4;
            p.dstMemoryType = CU_MEMORYTYPE_HOST; p.dstHost = host.value + 16; p.dstPitch = host.pitch;
        }
        p.WidthInBytes = size_t(width) * 4; p.Height = host.height;
        TEST_CUDA(cuMemcpy2DAsync(&p, stream));
    }
}
void sparse_require() {
    CUdevice device{}; int supported{};
    TEST_CUDA(cuCtxGetDevice(&device));
    TEST_CUDA(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED, device));
    require_capability(supported != 0, "sparse arrays require advertised sparse CUDA array support");
    int runtime{}; sparse_runtime(cudaDeviceGetAttribute(&runtime, cudaDevAttrSparseCudaArraySupported, device));
    require(runtime == supported, "Driver/Runtime sparse capability agreement");
}
CUmemAllocationProp sparse_pool_properties() {
    auto result = virtual_memory_properties(); result.allocFlags.usage = CU_MEM_CREATE_USAGE_TILE_POOL; return result;
}
void sparse_properties() {
    sparse_require();
    for (bool runtime : {false, true}) for (unsigned levels : {1u, 11u})
    for (unsigned shape = 0; shape < 3; ++shape) {
        sparse_resource resource(runtime, levels, shape == 0 ? 0 : shape == 1 ? 3 : 16,
            shape == 1 ? CUDA_ARRAY3D_LAYERED : 0);
        resource.properties();
        if (resource.mipmap) {
            CUDA_ARRAY_SPARSE_PROPERTIES output{};
            require(cuArrayGetSparseProperties(&output, resource.level(0)) == CUDA_ERROR_INVALID_VALUE,
                "query sparse layout from the whole mipmap instead of a level");
            require(cudaArrayGetSparseProperties(nullptr, reinterpret_cast<cudaArray_t>(resource.level(0))) == cudaErrorInvalidValue,
                "Runtime rejects null sparse properties output");
        }
    }
}
void sparse_tiles(bool remap, bool lifetime) {
    sparse_require();
    for (bool runtime : {false, true}) {
        sparse_resource resource(runtime); const auto p = resource.properties();
        const auto width = p.tileExtent.width, height = p.tileExtent.height;
        require(width <= 512 && height <= 1024 && p.tileExtent.depth == 1, "two 2D tiles fit the fixture");
        auto properties = sparse_pool_properties(); const auto granularity = virtual_memory_granularity(properties);
        const size_t tile_bytes = size_t(width) * height * 4;
        const auto pool_bytes = (tile_bytes + granularity - 1) / granularity * granularity;
        virtual_physical_memory first(pool_bytes, properties), second(pool_bytes, properties), replacement(pool_bytes, properties);
        sparse_stream upload, download;
        std::array mappings{resource.tile(0, first.value, width, height), resource.tile(width, second.value, width, height)};
        TEST_CUDA(cuMemMapArrayAsync(mappings.data(), unsigned(mappings.size()), upload.value));
        mappings = {}; // Accepted descriptors are owned by the API.
        upload.wait();
        if (lifetime) first.close();
        sparse_host left(width, height, 1, true), right(width, height, 7, true);
        sparse_host left_read(width, height, remap ? 13 : 1, false), right_read(width, height, 7, false);
        sparse_copy(resource.array, 0, width, left, true, runtime, upload.value);
        sparse_copy(resource.array, width, width, right, true, !runtime, upload.value);
        if (remap) {
            auto unmap = resource.tile(0, 0, width, height);
            TEST_CUDA(cuMemMapArrayAsync(&unmap, 1, upload.value));
            auto map = resource.tile(0, replacement.value, width, height);
            TEST_CUDA(cuMemMapArrayAsync(&map, 1, upload.value));
            sparse_host changed(width, height, 13, true);
            sparse_copy(resource.array, 0, width, changed, true, !runtime, upload.value);
            upload.wait(); changed.verify();
        }
        CUevent ready{}; TEST_CUDA(cuEventCreate(&ready, CU_EVENT_DISABLE_TIMING));
        TEST_CUDA(cuEventRecord(ready, upload.value));
        TEST_CUDA(cuStreamWaitEvent(download.value, ready, 0));
        TEST_CUDA(cuEventDestroy(ready));
        sparse_copy(resource.array, 0, width, left_read, false, !runtime, download.value);
        sparse_copy(resource.array, width, width, right_read, false, runtime, download.value);
        if (lifetime) resource.close(); else download.wait();
        left.verify(); right.verify(); left_read.verify(); right_read.verify();
        if (!lifetime) {
            std::array unmaps{resource.tile(0, 0, width, height), resource.tile(width, 0, width, height)};
            TEST_CUDA(cuMemMapArrayAsync(unmaps.data(), unsigned(unmaps.size()), download.value)); download.wait();
        }
    }
}
void sparse_miptail() {
    sparse_require();
    for (bool runtime : {false, true}) {
        sparse_resource resource(runtime, 11); auto p = resource.properties();
        require(p.miptailFirstLevel < 11 && p.miptailSize, "fixture has a nonempty mip tail");
        auto properties = sparse_pool_properties(); const auto granularity = virtual_memory_granularity(properties);
        virtual_physical_memory physical((size_t(p.miptailSize) + granularity - 1) / granularity * granularity, properties);
        sparse_stream stream;
        auto map = resource.tile(0, physical.value, 1, 1);
        map.subresourceType = CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL; map.subresource = {};
        map.subresource.miptail.size = p.miptailSize;
        TEST_CUDA(cuMemMapArrayAsync(&map, 1, stream.value)); stream.wait();
        for (unsigned level = p.miptailFirstLevel; level < 11; ++level) {
            const auto width = std::max(1u, 1024u >> level), height = width;
            sparse_host source(width, height, 31 + level, true), output(width, height, 31 + level, false);
            sparse_copy(resource.level(level), 0, width, source, true, runtime, stream.value);
            sparse_copy(resource.level(level), 0, width, output, false, !runtime, stream.value);
            stream.wait(); source.verify(); output.verify();
        }
        map.memOperationType = CU_MEM_OPERATION_TYPE_UNMAP; map.memHandle.memHandle = 0;
        TEST_CUDA(cuMemMapArrayAsync(&map, 1, stream.value)); stream.wait();
    }
}
void sparse_subregions() {
    sparse_require(); sparse_resource resource; auto p = resource.properties(); sparse_stream stream;
    auto properties = sparse_pool_properties(); const auto granularity = virtual_memory_granularity(properties);
    const auto width = p.tileExtent.width, height = p.tileExtent.height;
    const size_t tile_bytes = size_t(width) * height * 4;
    require(width <= 512, "two sparse tiles fit subregion fixture");
    virtual_physical_memory physical((2 * tile_bytes + granularity - 1) / granularity * granularity, properties);
    virtual_physical_memory replacement((tile_bytes + granularity - 1) / granularity * granularity, properties);
    auto whole = resource.tile(0, physical.value, width * 2, height);
    TEST_CUDA(cuMemMapArrayAsync(&whole, 1, stream.value)); stream.wait();
    physical.close();
    sparse_host source(width * 2, height, 19, true), right(width, height, 19, false, width);
    sparse_copy(resource.array, 0, width * 2, source, true, false, stream.value);
    auto left = resource.tile(0, 0, width, height);
    TEST_CUDA(cuMemMapArrayAsync(&left, 1, stream.value));
    sparse_copy(resource.array, width, width, right, false, true, stream.value);
    auto fresh = resource.tile(0, replacement.value, width, height);
    TEST_CUDA(cuMemMapArrayAsync(&fresh, 1, stream.value));
    sparse_host changed(width, height, 37, true), output(width, height, 37, false);
    sparse_copy(resource.array, 0, width, changed, true, true, stream.value);
    sparse_copy(resource.array, 0, width, output, false, false, stream.value);
    stream.wait(); source.verify(); right.verify(); changed.verify(); output.verify();
    auto right_unmap = resource.tile(width, 0, width, height);
    TEST_CUDA(cuMemMapArrayAsync(&right_unmap, 1, stream.value));
    TEST_CUDA(cuMemMapArrayAsync(&left, 1, stream.value)); stream.wait();
}
struct sparse_callback_gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false, returned = false, expired = false;
    static void CUDA_CB callback(void* pointer) {
        auto& gate = *static_cast<sparse_callback_gate*>(pointer);
        std::unique_lock lock(gate.mutex); gate.entered = true; gate.changed.notify_all();
        gate.changed.wait(lock, [&] { return gate.released; });
    }
};
void sparse_ordering() {
    sparse_require(); sparse_resource resource; auto p = resource.properties(); sparse_stream stream;
    auto properties = sparse_pool_properties(); const auto granularity = virtual_memory_granularity(properties);
    const auto width = p.tileExtent.width, height = p.tileExtent.height;
    virtual_physical_memory physical(std::max(granularity, size_t(width) * height * 4), properties);
    sparse_callback_gate gate;
    TEST_CUDA(cuLaunchHostFunc(stream.value, sparse_callback_gate::callback, &gate));
    {
        std::unique_lock lock(gate.mutex);
        if (!gate.changed.wait_for(lock, std::chrono::seconds(5), [&] { return gate.entered; })) {
            gate.released = true; lock.unlock(); gate.changed.notify_all(); stream.wait();
            require(false, "bounded sparse callback entry");
        }
    }
    auto map = resource.tile(0, physical.value, width, height);
    // A watchdog releases the participant even if the API incorrectly waits for
    // preceding work. Completion speed is not the oracle: callback dependency is.
    std::thread watchdog([&] {
        std::unique_lock lock(gate.mutex);
        if (!gate.changed.wait_for(lock, std::chrono::seconds(5), [&] { return gate.returned; })) gate.expired = true;
        gate.released = true; lock.unlock(); gate.changed.notify_all();
    });
    const auto result = cuMemMapArrayAsync(&map, 1, stream.value);
    { std::lock_guard lock(gate.mutex); gate.returned = true; }
    gate.changed.notify_all(); watchdog.join();
    TEST_CUDA(result); stream.wait();
    require(!gate.expired, "mapping returns before a preceding callback can complete");
    sparse_host source(width, height, 61, true), output(width, height, 61, false);
    sparse_copy(resource.array, 0, width, source, true, false, stream.value);
    sparse_copy(resource.array, 0, width, output, false, true, stream.value);
    stream.wait(); source.verify(); output.verify();
    map.memOperationType = CU_MEM_OPERATION_TYPE_UNMAP; map.memHandle.memHandle = 0;
    TEST_CUDA(cuMemMapArrayAsync(&map, 1, stream.value)); stream.wait();
}
void sparse_validation() {
    sparse_require(); sparse_resource resource; auto p = resource.properties(); sparse_stream stream;
    auto properties = sparse_pool_properties(); const auto granularity = virtual_memory_granularity(properties);
    virtual_physical_memory physical(std::max(granularity, size_t(p.tileExtent.width) * p.tileExtent.height * 4), properties);
    auto valid = resource.tile(0, physical.value, p.tileExtent.width, p.tileExtent.height);
    require(cuMemMapArrayAsync(nullptr, 1, stream.value) == CUDA_ERROR_INVALID_VALUE, "null mapping array");
    require(cuArrayGetSparseProperties(nullptr, resource.array) == CUDA_ERROR_INVALID_VALUE, "null sparse output");
    for (unsigned field = 0; field < 10; ++field) {
        auto bad = valid;
        switch (field) {
        case 0: bad.flags = 1; break;
        case 1: bad.reserved[1] = 1; break;
        case 2: bad.deviceBitMask = 0; break;
        case 3: bad.memHandleType = static_cast<CUmemHandleType>(99); break;
        case 4: bad.subresource.sparseLevel.offsetX = 1; break;
        case 5: bad.subresource.sparseLevel.extentWidth = 0; break;
        case 6: bad.subresource.sparseLevel.level = 1; break;
        case 7: bad.subresource.sparseLevel.layer = 1; break;
        case 8: bad.offset = 1; break;
        case 9: bad.memOperationType = CU_MEM_OPERATION_TYPE_UNMAP; break;
        }
        require(cuMemMapArrayAsync(&bad, 1, stream.value) == CUDA_ERROR_INVALID_VALUE, "reject malformed sparse mapping " + std::to_string(field));
        auto repair = valid; TEST_CUDA(cuMemMapArrayAsync(&repair, 1, stream.value)); stream.wait();
        sparse_host source(p.tileExtent.width, p.tileExtent.height, 51 + field, true), output(p.tileExtent.width, p.tileExtent.height, 51 + field, false);
        sparse_copy(resource.array, 0, p.tileExtent.width, source, true, false, stream.value);
        sparse_copy(resource.array, 0, p.tileExtent.width, output, false, true, stream.value);
        repair.memOperationType = CU_MEM_OPERATION_TYPE_UNMAP; repair.memHandle.memHandle = 0;
        TEST_CUDA(cuMemMapArrayAsync(&repair, 1, stream.value)); stream.wait(); source.verify(); output.verify();
    }
    auto ordinary_properties = virtual_memory_properties();
    auto ordinary_granularity = virtual_memory_granularity(ordinary_properties);
    virtual_physical_memory ordinary(std::max(ordinary_granularity, physical.bytes), ordinary_properties);
    auto bad = valid; bad.memHandle.memHandle = ordinary.value;
    require(cuMemMapArrayAsync(&bad, 1, stream.value) == CUDA_ERROR_INVALID_VALUE, "backing must have tile-pool usage");
    CUDA_ARRAY3D_DESCRIPTOR descriptor{16, 16, 0, CU_AD_FORMAT_UNSIGNED_INT32, 1, 0};
    CUarray dense{}; TEST_CUDA(cuArray3DCreate(&dense, &descriptor));
    CUDA_ARRAY_SPARSE_PROPERTIES output{};
    require(cuArrayGetSparseProperties(&output, dense) == CUDA_ERROR_INVALID_VALUE, "ordinary array has no sparse layout");
    bad = valid; bad.resource.array = dense;
    require(cuMemMapArrayAsync(&bad, 1, stream.value) == CUDA_ERROR_INVALID_VALUE, "ordinary array cannot be mapped");
    TEST_CUDA(cuArrayDestroy(dense));
    sparse_resource mipmap(false, 11);
    require(cuMipmappedArrayGetSparseProperties(nullptr, mipmap.mipmap) == CUDA_ERROR_INVALID_VALUE, "null whole-mipmap output");
    require(cudaMipmappedArrayGetSparseProperties(nullptr, reinterpret_cast<cudaMipmappedArray_t>(mipmap.mipmap)) == cudaErrorInvalidValue,
        "null Runtime whole-mipmap output");
}
void sparse_discovery() {
    std::array<void*, 2> mapping_entries{};
    for (const char* name : {"cuArrayGetSparseProperties", "cuMipmappedArrayGetSparseProperties", "cuMemMapArrayAsync"})
    for (auto flags : {CU_GET_PROC_ADDRESS_LEGACY_STREAM, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM}) {
        void* driver{}; CUdriverProcAddressQueryResult query{};
        TEST_CUDA(cuGetProcAddress(name, &driver, 11009, flags, &query));
        require(!driver && query == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "sparse API introduction boundary");
        TEST_CUDA(cuGetProcAddress(name, &driver, 11010, flags, &query));
        require(driver && query == CU_GET_PROC_ADDRESS_SUCCESS, "versioned sparse Driver discovery");
        void* runtime{}; cudaDriverEntryPointQueryResult runtime_query{};
        sparse_runtime(cudaGetDriverEntryPointByVersion(name, &runtime, 11010, flags, &runtime_query));
        require(runtime == driver && runtime_query == cudaDriverEntryPointSuccess, "Runtime discovers the same sparse entry");
        if (std::string_view(name) == "cuMemMapArrayAsync")
            mapping_entries[flags == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM] = driver;
    }
    require(mapping_entries[0] != mapping_entries[1], "legacy and PTDS mapping entries differ");
    using map_function = CUresult (CUDAAPI*)(CUarrayMapInfo*, unsigned, CUstream);
    for (auto entry : mapping_entries)
        require(reinterpret_cast<map_function>(entry)(nullptr, 1, nullptr) == CUDA_ERROR_NOT_INITIALIZED, "cold mapping call preserves initialization boundary");
}
void sparse_execute(const std::string& name) {
    if (name == "properties") sparse_properties();
    else if (name == "tiles") sparse_tiles(false, false);
    else if (name == "remap") sparse_tiles(true, false);
    else if (name == "lifetime") sparse_tiles(false, true);
    else if (name == "miptail") sparse_miptail();
    else if (name == "subregions") sparse_subregions();
    else if (name == "ordering") sparse_ordering();
    else if (name == "validation") sparse_validation();
    else throw std::runtime_error("unknown sparse arrays case");
}
std::string sparse_case;
void sparse_run() { sparse_execute(sparse_case); }
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "discovery") return cudavm::run_host_test(argc, argv, cudavm::sparse_discovery);
    if (argc == 3) { cudavm::sparse_case = argv[2]; --argc; }
    return cudavm::run_test(argc, argv, cudavm::sparse_run);
}
