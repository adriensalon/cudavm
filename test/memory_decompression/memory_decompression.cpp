#include "test_framework.hpp"
#include <cudaTypedefs.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <type_traits>
#include "decompression_fixtures.hpp"
#include "decompression_reference.hpp"

namespace cudavm {
std::string_view decompression_case;
constexpr size_t decompression_guard = 256;
constexpr uint64_t decompression_before = 0x19375928abcdef01, decompression_after = 0xfedcba0129183745;
CUresult decompression_ptds(CUmemDecompressParams*, size_t, unsigned, size_t*, CUstream);
void decompression_runtime(cudaError_t result) { require(result == cudaSuccess, std::string("decompression Runtime: ") + cudaGetErrorString(result)); }
void* decompression_pointer(CUdeviceptr pointer) { return reinterpret_cast<void*>(uintptr_t(pointer)); }
std::vector<unsigned char> decompression_expected(const decompression_fixture& fixture) {
    std::vector<unsigned char> result(fixture.output_size);
    for (size_t i = 0; i < result.size(); ++i) {
        if (!fixture.mode) result[i] = static_cast<unsigned char>((i * 73 + fixture.seed * 37) ^ (i >> 7));
        else if (fixture.mode == 1) result[i] = static_cast<unsigned char>((i % 32 * 17 + fixture.seed * 29) ^ (i % 32 >> 2));
        else result[i] = static_cast<unsigned char>(fixture.seed);
    }
    return result;
}
void decompression_expect(std::span<const unsigned char> actual, std::span<const unsigned char> expected, const char* label) {
    require(actual.size() == expected.size(), std::string(label) + " byte extent");
    for (size_t i = 0; i < actual.size(); ++i) require(actual[i] == expected[i], std::string(label) + " byte " + std::to_string(i));
}
void decompression_validate_fixtures() {
    for (const auto& fixture : decompression_fixtures) {
        decompression_expect(decompression_reference(fixture.algorithm, fixture.data), decompression_expected(fixture), fixture.name);
        bool rejected{}; try { (void)decompression_reference(fixture.algorithm, fixture.data.first(fixture.data.size() - 1)); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, std::string(fixture.name) + " truncated fixture rejected by the independent reference");
    }
}
struct decompression_capabilities {
    int mask{}, maximum{};
    decompression_capabilities() {
        CUdevice device{}; TEST_CUDA(cuCtxGetDevice(&device));
        for (auto attribute : {CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK, CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH}) {
            struct { int before = 0x19375928, value{}, after = 0x6edcba01; } output;
            TEST_CUDA(cuDeviceGetAttribute(&output.value, attribute, device));
            require(output.before == 0x19375928 && output.after == 0x6edcba01 && output.value >= 0, "decompression capability exact output width");
            (attribute == CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK ? mask : maximum) = output.value;
        }
        require(!mask || maximum > 0, "advertised decompression algorithms have a positive operation limit");
    }
    void need(unsigned algorithms) const { require_capability((unsigned(mask) & algorithms) == algorithms && maximum >= 8191, "profile requires advertised hardware decompression algorithms and its fixture extent"); }
};
bool decompression_capable(CUdeviceptr pointer, bool bulk = false) {
    struct { unsigned char before = 0x37; bool value{}; unsigned char after = 0xa9; } output;
    if (bulk) {
        CUpointer_attribute attribute = CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE; void* destination = &output.value;
        TEST_CUDA(cuPointerGetAttributes(1, &attribute, &destination, pointer));
    } else TEST_CUDA(cuPointerGetAttribute(&output.value, CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE, pointer));
    require(output.before == 0x37 && output.after == 0xa9, "decompression boolean attribute exact output width"); return output.value;
}
struct decompression_buffer {
    CUdeviceptr value{}; size_t bytes{}; bool runtime{}, owns = true;
    explicit decompression_buffer(size_t size, bool api = false, CUdeviceptr external = 0) : bytes(size), runtime(api), owns(!external) {
        if (external) value = external;
        else if (runtime) { void* pointer{}; decompression_runtime(cudaMalloc(&pointer, bytes)); value = reinterpret_cast<uintptr_t>(pointer); }
        else TEST_CUDA(cuMemAlloc(&value, bytes));
    }
    ~decompression_buffer() { if (value && owns) cuMemFree(value); }
};
struct decompression_stream {
    CUstream value{};
    decompression_stream() { TEST_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~decompression_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
};
struct decompression_host {
    unsigned char* value{}; size_t bytes{};
    explicit decompression_host(size_t size) : bytes(size) { TEST_CUDA(cuMemHostAlloc(reinterpret_cast<void**>(&value), bytes, CU_MEMHOSTALLOC_PORTABLE)); }
    ~decompression_host() { if (value) cuMemFreeHost(value); }
};
struct decompression_event {
    CUevent value{}; decompression_event() { TEST_CUDA(cuEventCreate(&value, CU_EVENT_DISABLE_TIMING)); }
    ~decompression_event() { if (value) cuEventDestroy(value); }
};
struct decompression_gate {
    CUcontext owner{}; CUstream stream{}; std::mutex mutex; std::condition_variable changed; bool entered{}, released{}, expired{};
    static void CUDA_CB callback(void* argument) {
        auto& gate = *static_cast<decompression_gate*>(argument); std::unique_lock lock(gate.mutex); gate.entered = true; gate.changed.notify_all();
        gate.expired = !gate.changed.wait_for(lock, std::chrono::seconds(15), [&] { return gate.released; });
    }
    explicit decompression_gate(CUstream value) : stream(value) {
        TEST_CUDA(cuCtxGetCurrent(&owner)); TEST_CUDA(cuLaunchHostFunc(stream, callback, this));
        std::unique_lock lock(mutex); const auto started = changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; });
        if (!started) { lock.unlock(); release(); cuStreamSynchronize(stream); throw std::runtime_error("decompression gate entry deadline"); }
    }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
    void verify() { std::lock_guard lock(mutex); require(!expired, "decompression asynchronous API returns before the gate deadline"); }
    ~decompression_gate() {
        release(); if (owner) { CUcontext previous{}; cuCtxGetCurrent(&previous); if (cuCtxSetCurrent(owner) == CUDA_SUCCESS) cuCtxSynchronize(); cuCtxSetCurrent(previous); }
    }
};
struct decompression_job {
    const decompression_fixture& fixture;
    decompression_buffer source, destination, count;
    std::vector<unsigned char> source_bytes, initial_destination, initial_count, expected_destination, expected_count;
    explicit decompression_job(const decompression_fixture& sample, bool runtime = false, std::array<CUdeviceptr, 3> external = {})
        : fixture(sample), source(sample.data.size() + 2 * decompression_guard, runtime, external[0]),
          destination(sample.output_size + 2 * decompression_guard, !runtime, external[1]), count(64, runtime, external[2]),
          source_bytes(virtual_memory_pattern(source.bytes, 43)), initial_destination(virtual_memory_pattern(destination.bytes, 47)),
          initial_count(virtual_memory_pattern(count.bytes, 53)), expected_destination(initial_destination), expected_count(initial_count) {
        std::copy(sample.data.begin(), sample.data.end(), source_bytes.begin() + decompression_guard);
        const auto expected = decompression_expected(sample); std::copy(expected.begin(), expected.end(), expected_destination.begin() + decompression_guard);
        const auto size = static_cast<cuuint32_t>(sample.output_size); std::memcpy(expected_count.data() + 16, &size, sizeof(size));
        virtual_memory_write(source.value, source_bytes, runtime); virtual_memory_write(destination.value, initial_destination, !runtime); virtual_memory_write(count.value, initial_count, runtime);
    }
    CUmemDecompressParams record(size_t hint = 0) const {
        CUmemDecompressParams value{}; value.srcNumBytes = fixture.data.size(); value.dstNumBytes = hint;
        value.src = decompression_pointer(source.value + decompression_guard); value.dst = decompression_pointer(destination.value + decompression_guard);
        value.dstActBytes = reinterpret_cast<cuuint32_t*>(uintptr_t(count.value + 16)); value.algo = fixture.algorithm; return value;
    }
    void verify(bool initial = false, bool source_alive = true) const {
        if (source_alive) virtual_memory_expect(source.value, source_bytes, "decompression preserves source and all source guards", source.runtime);
        virtual_memory_expect(destination.value, initial ? initial_destination : expected_destination, "decompression destination exact payload and all guards", destination.runtime);
        virtual_memory_expect(count.value, initial ? initial_count : expected_count, "decompression writes exactly one 32-bit actual byte count", count.runtime);
    }
};
PFN_cuMemBatchDecompressAsync_v12060 decompression_procedure(bool runtime, bool ptds = false) {
    void* address{}; const auto flags = ptds ? CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM : CU_GET_PROC_ADDRESS_LEGACY_STREAM;
    if (runtime) {
        cudaDriverEntryPointQueryResult status{}; decompression_runtime(cudaGetDriverEntryPointByVersion("cuMemBatchDecompressAsync", &address, 12060, flags, &status));
        require(status == cudaDriverEntryPointSuccess && address, "Runtime Driver-entry lookup for decompression");
    } else TEST_CUDA(cuGetProcAddress("cuMemBatchDecompressAsync", &address, 12060, flags, nullptr));
    return reinterpret_cast<PFN_cuMemBatchDecompressAsync_v12060>(address);
}
void decompression_submit(std::span<CUmemDecompressParams> records, CUstream stream, bool runtime = false) {
    struct { size_t before = size_t(decompression_before), index{}, after = size_t(decompression_after); } output;
    if (runtime) TEST_CUDA(decompression_procedure(true)(records.data(), records.size(), 0, &output.index, stream));
    else TEST_CUDA(cuMemBatchDecompressAsync(records.data(), records.size(), 0, &output.index, stream));
    require(output.before == size_t(decompression_before) && output.after == size_t(decompression_after), "decompression error-index exact output width");
}
void decompression_metadata() {
    const decompression_capabilities capabilities;
    for (bool runtime : {false, true}) {
        decompression_buffer allocation(8197, runtime); const auto expected = virtual_memory_pattern(allocation.bytes, 67); virtual_memory_write(allocation.value, expected, runtime);
        for (bool bulk : {false, true}) for (size_t offset : {size_t(0), size_t(37), allocation.bytes - 1})
            require(decompression_capable(allocation.value + offset, bulk) == (capabilities.mask != 0), "ordinary allocation decompression capability agrees with device support");
        virtual_memory_expect(allocation.value, expected, "decompression metadata preserves complete initialized bytes", !runtime);
    }
    CUdevice device{}; CUmemoryPool pool{}; TEST_CUDA(cuCtxGetDevice(&device)); TEST_CUDA(cuDeviceGetDefaultMemPool(&pool, device));
    struct { int before = 0x19375928, value = -1, after = 0x6edcba01; } enabled;
    TEST_CUDA(cuMemPoolGetAttribute(pool, CU_MEMPOOL_ATTR_HW_DECOMPRESS_ENABLED, &enabled.value));
    require(enabled.before == 0x19375928 && enabled.after == 0x6edcba01 && enabled.value == 0, "ordinary default pool hardware-decompression usage and exact width");
}
void decompression_rejected(CUmemDecompressParams* records, size_t count, unsigned flags, CUstream stream, CUresult expected, size_t index) {
    struct { size_t before = size_t(decompression_before), value = 123456, after = size_t(decompression_after); } output;
    require(cuMemBatchDecompressAsync(records, count, flags, &output.value, stream) == expected, "decompression validation result");
    require(output.before == size_t(decompression_before) && output.after == size_t(decompression_after) && output.value == index, "decompression exact general or operation error index");
}
void decompression_validation() {
    decompression_job job(decompression_fixtures[0]); decompression_stream stream; const auto good = job.record(); const auto general = std::numeric_limits<size_t>::max();
    std::array<CUmemDecompressParams, 3> records{good, good, good};
    decompression_rejected(records.data(), records.size(), 1, stream.value, CUDA_ERROR_INVALID_VALUE, general);
    decompression_rejected(nullptr, 1, 0, stream.value, CUDA_ERROR_INVALID_VALUE, general);
    decompression_rejected(records.data(), general, 0, stream.value, CUDA_ERROR_INVALID_VALUE, general);
    decompression_rejected(records.data(), 1, 0, reinterpret_cast<CUstream>(uintptr_t(0x12345)), CUDA_ERROR_INVALID_HANDLE, general);
    auto reject = [&](size_t index) { decompression_rejected(records.data(), records.size(), 0, stream.value, CUDA_ERROR_INVALID_VALUE, index); job.verify(true); records.fill(good); };
    records[2].padding[19] = 1; reject(2); records[1].algo = CU_MEM_DECOMPRESS_UNSUPPORTED; reject(1);
    records[0].algo = static_cast<CUmemDecompressAlgorithm>(8); reject(0); records[2].algo = static_cast<CUmemDecompressAlgorithm>(3); reject(2);
    records[0].src = nullptr; reject(0); records[1].srcNumBytes = job.source.bytes; reject(1);
    records[1].dst = decompression_pointer(job.destination.value + job.destination.bytes); reject(1);
    records[1].dstActBytes = reinterpret_cast<cuuint32_t*>(uintptr_t(job.count.value + 17)); reject(1); records[2].dstActBytes = nullptr; reject(2);
    require(cuMemBatchDecompressAsync(records.data(), records.size(), 1, nullptr, stream.value) == CUDA_ERROR_INVALID_VALUE, "optional error-index output on failure");
    job.verify(true);
    const auto mask = decompression_capabilities{}.mask;
    if (mask & good.algo) { std::array repaired{good}; decompression_submit(repaired, stream.value, true); TEST_CUDA(cuStreamSynchronize(stream.value)); job.verify(); }
    else {
        auto aliased = good;
        require(cuMemBatchDecompressAsync(&aliased, 1, 0, &aliased.srcNumBytes, stream.value) == CUDA_ERROR_NOT_SUPPORTED,
                "an error-index output alias does not overwrite descriptors before validation"); job.verify(true);
    }
}
void decompression_unsupported() {
    const decompression_capabilities capabilities; require_capability(!capabilities.mask, "absence profile needs no advertised decompression algorithms");
    decompression_job job(decompression_fixtures[0]); decompression_stream stream; auto record = job.record();
    decompression_rejected(&record, 1, 0, stream.value, CUDA_ERROR_NOT_SUPPORTED, std::numeric_limits<size_t>::max()); job.verify(true);
    CUmemPoolProps props{}; CUdevice device{}; TEST_CUDA(cuCtxGetDevice(&device)); props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    props.location = {CU_MEM_LOCATION_TYPE_DEVICE, device}; props.usage = CU_MEM_POOL_CREATE_USAGE_HW_DECOMPRESS; CUmemoryPool pool{};
    require(cuMemPoolCreate(&pool, &props) == CUDA_ERROR_NOT_SUPPORTED, "hardware pool usage does not fabricate unsupported decompression");
}
void decompression_formats(CUmemDecompressAlgorithm algorithm) {
    decompression_capabilities{}.need(unsigned(algorithm)); unsigned position{};
    for (const auto& fixture : decompression_fixtures) if (fixture.algorithm == algorithm) {
        for (size_t hint : {size_t(0), size_t(1), fixture.output_size}) {
            decompression_job job(fixture, position % 2 != 0); decompression_stream stream; std::array records{job.record(hint)};
            decompression_submit(records, position % 2 ? stream.value : nullptr, position % 2 != 0);
            if (position % 2) TEST_CUDA(cuStreamSynchronize(stream.value)); else decompression_runtime(cudaDeviceSynchronize());
            job.verify(); ++position;
        }
    }
}
const decompression_fixture& decompression_available_fixture() {
    const decompression_capabilities capabilities; require_capability(capabilities.mask != 0 && capabilities.maximum >= 8191, "profile requires a hardware decompression algorithm and the fixture extent");
    for (const auto& fixture : decompression_fixtures) if (capabilities.mask & fixture.algorithm) return fixture;
    throw std::runtime_error("advertised hardware decompression has no pinned algorithm fixture");
}
void decompression_batch() {
    (void)decompression_available_fixture(); const auto mask = decompression_capabilities{}.mask;
    std::vector<std::unique_ptr<decompression_job>> jobs; std::vector<CUmemDecompressParams> records; decompression_stream stream;
    for (const auto& fixture : decompression_fixtures) if (mask & fixture.algorithm) {
        jobs.push_back(std::make_unique<decompression_job>(fixture, jobs.size() % 2 != 0)); records.push_back(jobs.back()->record(jobs.size() % 2 ? 0 : fixture.output_size));
    }
    decompression_submit(records, stream.value, true); std::memset(records.data(), 0xcd, records.size() * sizeof(records[0]));
    TEST_CUDA(cuStreamSynchronize(stream.value)); for (const auto& job : jobs) job->verify();
}
void decompression_ordering(bool ptds) {
    decompression_job job(decompression_available_fixture(), true); decompression_stream explicit_producer, consumer; const auto producer = ptds ? CU_STREAM_PER_THREAD : explicit_producer.value;
    decompression_host input(job.source.bytes), output(job.destination.bytes), count(job.count.bytes);
    std::copy(job.source_bytes.begin(), job.source_bytes.end(), input.value); TEST_CUDA(cuMemsetD8(job.source.value, 0x5d, job.source.bytes));
    decompression_event ready, completed; decompression_gate gate(producer);
    TEST_CUDA(cuMemcpyHtoDAsync(job.source.value, input.value, input.bytes, producer));
    std::array records{job.record(1)}; size_t index{};
    if (ptds) TEST_CUDA(decompression_ptds(records.data(), records.size(), 0, &index, nullptr)); else decompression_submit(records, producer, true);
    std::memset(records.data(), 0xd7, sizeof(records));
    TEST_CUDA(cuEventRecord(ready.value, producer)); TEST_CUDA(cuStreamWaitEvent(consumer.value, ready.value, 0));
    TEST_CUDA(cuMemcpyDtoHAsync(output.value, job.destination.value, output.bytes, consumer.value));
    TEST_CUDA(cuMemcpyDtoHAsync(count.value, job.count.value, count.bytes, consumer.value)); TEST_CUDA(cuEventRecord(completed.value, consumer.value));
    require(cuEventQuery(ready.value) == CUDA_ERROR_NOT_READY && cuEventQuery(completed.value) == CUDA_ERROR_NOT_READY, "decompression and event-ordered consumer remain pending behind input gate");
    if (ptds) {
        decompression_buffer independent(17); CUcontext owner{}; TEST_CUDA(cuCtxGetCurrent(&owner)); std::atomic<bool> done{}; CUresult status = CUDA_ERROR_UNKNOWN;
        std::thread worker([&] { cuCtxSetCurrent(owner); status = cuMemsetD8Async(independent.value, 0x6d, independent.bytes, CU_STREAM_PER_THREAD); if (status == CUDA_SUCCESS) status = cuStreamSynchronize(CU_STREAM_PER_THREAD); done.store(true, std::memory_order_release); });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
        const auto independent_finished = done.load(std::memory_order_acquire); if (!independent_finished) gate.release(); worker.join(); TEST_CUDA(status);
        require(independent_finished, "another host thread's per-thread default progresses while decompression is gated");
    }
    gate.release(); TEST_CUDA(cuStreamSynchronize(consumer.value)); gate.verify(); job.verify();
    decompression_expect({output.value, output.bytes}, job.expected_destination, "event consumer sees complete decompressed bytes and guards");
    decompression_expect({count.value, count.bytes}, job.expected_count, "event consumer sees exact 32-bit actual count and guards");
}
struct decompression_context {
    CUcontext value{}, previous{};
    decompression_context() {
        TEST_CUDA(cuCtxGetCurrent(&previous)); CUdevice device{}; TEST_CUDA(cuCtxGetDevice(&device));
        TEST_CUDA(cuCtxCreate(&value, nullptr, 0, device)); CUcontext popped{}; TEST_CUDA(cuCtxPopCurrent(&popped));
        require(popped == value, "decompression context creation stack"); TEST_CUDA(cuCtxSetCurrent(value));
    }
    ~decompression_context() { if (value) cuCtxDestroy(value); cuCtxSetCurrent(previous); }
};
void decompression_retirement() {
    const auto& fixture = decompression_available_fixture();
    for (bool destroy_context : {false, true}) {
        // Portable host storage belongs to the original context and survives
        // the dedicated importing/execution context's retirement.
        decompression_host output(fixture.output_size + 2 * decompression_guard), count(64);
        decompression_context context; decompression_job job(fixture, true); decompression_stream stream;
        decompression_gate gate(stream.value); std::array records{job.record()}; decompression_submit(records, stream.value, true);
        if (destroy_context) {
            TEST_CUDA(cuMemcpyDtoHAsync(output.value, job.destination.value, output.bytes, stream.value));
            TEST_CUDA(cuMemcpyDtoHAsync(count.value, job.count.value, count.bytes, stream.value));
        }
        std::atomic<bool> started{}, done{}; CUresult status = CUDA_ERROR_UNKNOWN;
        std::thread retire([&] {
            cuCtxSetCurrent(context.value); started.store(true, std::memory_order_release);
            status = destroy_context ? cuCtxDestroy(context.value) : cuMemFree(job.source.value); done.store(true, std::memory_order_release);
        });
        while (!started.load(std::memory_order_acquire)) std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); const bool blocked = !done.load(std::memory_order_acquire);
        int version{}; const auto query = cuDriverGetVersion(&version); gate.release(); retire.join(); TEST_CUDA(status); TEST_CUDA(query);
        gate.verify(); require(blocked && version > 0, "decompression retirement waits for accepted work while unrelated queries progress");
        job.source.value = 0;
        if (destroy_context) {
            gate.owner = nullptr; stream.value = nullptr; context.value = nullptr; job.destination.value = job.count.value = 0;
            TEST_CUDA(cuCtxSetCurrent(context.previous));
            decompression_expect({output.value, output.bytes}, job.expected_destination, "retired context completes accepted decompression and consumer output");
            decompression_expect({count.value, count.bytes}, job.expected_count, "retired context completes exact actual count");
        } else job.verify(false, false);
    }
}
size_t decompression_aligned(size_t bytes) { return (bytes + 255) / 256 * 256; }
std::array<CUdeviceptr, 3> decompression_views(CUdeviceptr base, const decompression_fixture& fixture) {
    const auto destination = base + decompression_aligned(fixture.data.size() + 2 * decompression_guard);
    return {base, destination, destination + decompression_aligned(fixture.output_size + 2 * decompression_guard)};
}
void decompression_allocation_probe(CUdeviceptr base, const decompression_fixture& fixture) {
    auto addresses = decompression_views(base, fixture); decompression_job job(fixture, true, addresses);
    for (auto address : addresses) require(decompression_capable(address) && decompression_capable(address, true), "explicit hardware usage produces capable native backing");
    decompression_stream stream; std::array records{job.record(fixture.output_size)}; decompression_submit(records, stream.value, true);
    TEST_CUDA(cuStreamSynchronize(stream.value)); job.verify();
}
void decompression_allocators() {
    const auto& fixture = decompression_available_fixture();
    const auto requested = size_t(decompression_views(0, fixture)[2]) + 64;
    auto properties = virtual_memory_properties(); properties.allocFlags.usage = CU_MEM_CREATE_USAGE_HW_DECOMPRESS;
    const auto granularity = virtual_memory_granularity(properties), bytes = (requested + granularity - 1) / granularity * granularity;
    {
        virtual_physical_memory physical(bytes, properties); virtual_memory_reservation reservation(bytes);
        virtual_memory_mapping mapping(reservation.value, bytes, physical.value); mapping.grant(properties.location);
        CUmemAllocationProp actual{}; TEST_CUDA(cuMemGetAllocationPropertiesFromHandle(&actual, physical.value));
        require((actual.allocFlags.usage & CU_MEM_CREATE_USAGE_HW_DECOMPRESS) != 0, "VMM hardware-decompression usage survives creation");
        decompression_allocation_probe(mapping.value, fixture);
    }
    for (bool runtime : {false, true}) {
        CUdevice device{}; TEST_CUDA(cuCtxGetDevice(&device)); CUmemoryPool pool{};
        if (runtime) {
            cudaMemPoolProps props{}; props.allocType = cudaMemAllocationTypePinned; props.location = {cudaMemLocationTypeDevice, device}; props.usage = cudaMemPoolCreateUsageHwDecompress;
            cudaMemPool_t result{}; decompression_runtime(cudaMemPoolCreate(&result, &props)); pool = reinterpret_cast<CUmemoryPool>(result);
        } else {
            CUmemPoolProps props{}; props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED; props.location = {CU_MEM_LOCATION_TYPE_DEVICE, device}; props.usage = CU_MEM_POOL_CREATE_USAGE_HW_DECOMPRESS;
            TEST_CUDA(cuMemPoolCreate(&pool, &props));
        }
        struct pool_owner { CUmemoryPool value; ~pool_owner() { if (value) cuMemPoolDestroy(value); } } owner{pool};
        struct { int before = 0x19375928, value{}, after = 0x6edcba01; } enabled;
        TEST_CUDA(cuMemPoolGetAttribute(pool, CU_MEMPOOL_ATTR_HW_DECOMPRESS_ENABLED, &enabled.value));
        require(enabled.before == 0x19375928 && enabled.after == 0x6edcba01 && enabled.value == 1, "native hardware pool attribute and exact width");
        decompression_stream stream; CUdeviceptr pointer{};
        if (runtime) TEST_CUDA(cuMemAllocFromPoolAsync(&pointer, requested, pool, stream.value));
        else {
            void* result{}; decompression_runtime(cudaMallocFromPoolAsync(&result, requested, reinterpret_cast<cudaMemPool_t>(pool), reinterpret_cast<cudaStream_t>(stream.value)));
            pointer = reinterpret_cast<uintptr_t>(result);
        }
        struct allocation_owner { CUdeviceptr value; ~allocation_owner() { if (value) cuMemFree(value); } } allocation{pointer};
        TEST_CUDA(cuStreamSynchronize(stream.value)); decompression_allocation_probe(pointer, fixture);
        TEST_CUDA(cuMemFreeAsync(pointer, stream.value)); TEST_CUDA(cuStreamSynchronize(stream.value)); allocation.value = 0;
        if (runtime) TEST_CUDA(cuMemPoolDestroy(pool)); else decompression_runtime(cudaMemPoolDestroy(reinterpret_cast<cudaMemPool_t>(pool)));
        owner.value = nullptr;
    }
}
void decompression_discovery() {
    static_assert(std::is_same_v<decltype(&cuMemBatchDecompressAsync), PFN_cuMemBatchDecompressAsync_v12060>);
    void* legacy{}, *per_thread{};
    for (unsigned flags : {unsigned(CU_GET_PROC_ADDRESS_LEGACY_STREAM), unsigned(CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)}) {
        void* address{}; CUdriverProcAddressQueryResult status{};
        TEST_CUDA(cuGetProcAddress("cuMemBatchDecompressAsync", &address, 12059, flags, &status)); require(!address && status == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "decompression introduction version boundary");
        TEST_CUDA(cuGetProcAddress("cuMemBatchDecompressAsync", &address, 12060, flags, &status)); require(address && status == CU_GET_PROC_ADDRESS_SUCCESS, "decompression procedure introduction");
        (flags == CU_GET_PROC_ADDRESS_LEGACY_STREAM ? legacy : per_thread) = address;
    }
    require(legacy != per_thread && decompression_procedure(false) == reinterpret_cast<PFN_cuMemBatchDecompressAsync_v12060>(legacy) &&
            decompression_procedure(true, true) == reinterpret_cast<PFN_cuMemBatchDecompressAsync_v12060>(per_thread), "decompression default-stream procedure selection through both lookup APIs");
    size_t index{}; require(decompression_procedure(false)(nullptr, 1, 0, &index, nullptr) == CUDA_ERROR_NOT_INITIALIZED && index == std::numeric_limits<size_t>::max(), "cold decompression entry and general index");
    require(decompression_ptds(nullptr, 1, 0, &index, nullptr) == CUDA_ERROR_NOT_INITIALIZED, "cold header-selected per-thread decompression entry");
}
void execute_memory_decompression() {
    if (decompression_case == "metadata") decompression_metadata();
    else if (decompression_case == "validation") decompression_validation();
    else if (decompression_case == "unsupported") decompression_unsupported();
    else if (decompression_case == "deflate") decompression_formats(CU_MEM_DECOMPRESS_ALGORITHM_DEFLATE);
    else if (decompression_case == "snappy") decompression_formats(CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY);
    else if (decompression_case == "lz4") decompression_formats(CU_MEM_DECOMPRESS_ALGORITHM_LZ4);
    else if (decompression_case == "batch") decompression_batch();
    else if (decompression_case == "ordering") decompression_ordering(false);
    else if (decompression_case == "ptds") decompression_ordering(true);
    else if (decompression_case == "retirement") decompression_retirement();
    else if (decompression_case == "allocators") decompression_allocators();
    else throw std::runtime_error("unknown memory_decompression case");
}
}
int main(int argc, char** argv) {
    if (argc == 2 && (std::string_view(argv[1]) == "discovery" || std::string_view(argv[1]) == "fixtures"))
        return cudavm::run_host_test(argc, argv, std::string_view(argv[1]) == "discovery" ? cudavm::decompression_discovery : cudavm::decompression_validate_fixtures);
    if (argc == 3) { cudavm::decompression_case = argv[2]; --argc; }
    return cudavm::run_test(argc, argv, cudavm::execute_memory_decompression);
}
