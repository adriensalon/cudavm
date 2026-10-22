#include "dynamic_library.hpp"
#include "device_resources.hpp"
#include "../cudavm/dxil_artifact.hpp"
#if defined(_WIN32)
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#if defined(CUDAVM_DXC_LIBRARY)
#include <dxc/dxcapi.h>
#endif
namespace cudavm {
namespace {
using Microsoft::WRL::ComPtr;
void check(HRESULT value, const char* operation) {
    if (SUCCEEDED(value)) return;
    throw failure(value == E_OUTOFMEMORY ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN, std::string(operation) + " failed: " + std::to_string(static_cast<unsigned long>(value)));
}
struct directx_api {
    dynamic_library dxgi{{"dxgi.dll"}}, d3d{{"d3d12.dll"}};
    decltype(&CreateDXGIFactory1) factory = dxgi.get<decltype(factory)>("CreateDXGIFactory1");
    decltype(&D3D12CreateDevice) create = d3d.get<decltype(create)>("D3D12CreateDevice");
};
struct directx_device {
    std::shared_ptr<directx_api> api;
    std::shared_ptr<ID3D12Device> resource_owner;
    ComPtr<ID3D12Device> value;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE event = nullptr;
    uint64_t epoch = 0;
    bool native_dxil = false;
    bool native_double = false;
    directx_device(std::shared_ptr<directx_api> input, std::shared_ptr<ID3D12Device> device)
        : api(std::move(input)), resource_owner(std::move(device)), value(resource_owner.get()) {
        D3D12_COMMAND_QUEUE_DESC description{};
        description.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        check(value->CreateCommandQueue(&description, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
        check(value->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "CreateCommandAllocator");
        check(value->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "CreateCommandList");
        check(list->Close(), "Close");
        check(value->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
        event = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        require(event != nullptr, CUDA_ERROR_UNKNOWN, "DirectX fence event creation failed");
#if defined(CUDAVM_DXC_LIBRARY) && defined(CUDAVM_DXIL_LIBRARY)
        D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_0};
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 features{};
        D3D12_FEATURE_DATA_D3D12_OPTIONS base{};
        native_double = SUCCEEDED(value->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS,&base,sizeof(base))) && base.DoublePrecisionFloatShaderOps;
        native_dxil = SUCCEEDED(value->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL,&model,sizeof(model))) && model.HighestShaderModel>=D3D_SHADER_MODEL_6_0
            && SUCCEEDED(value->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1,&features,sizeof(features))) && features.Int64ShaderOps;
#endif
    }
    ~directx_device() { if (event) CloseHandle(event); }
    ComPtr<ID3D12Resource> resource(size_t bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state) {
        D3D12_HEAP_PROPERTIES properties{};
        properties.Type = heap;
        properties.CreationNodeMask = properties.VisibleNodeMask = 1;
        D3D12_RESOURCE_DESC description{};
        description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        description.Width = bytes;
        description.Height = 1;
        description.DepthOrArraySize = description.MipLevels = 1;
        description.SampleDesc.Count = 1;
        description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (heap == D3D12_HEAP_TYPE_DEFAULT) description.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ComPtr<ID3D12Resource> result;
        check(value->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &description, state, nullptr, IID_PPV_ARGS(&result)), "CreateCommittedResource");
        return result;
    }
    void begin() { check(allocator->Reset(), "Reset allocator"); check(list->Reset(allocator.Get(), nullptr), "Reset command list"); }
    void barrier(ID3D12Resource* resource, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = from;
        barrier.Transition.StateAfter = to;
        list->ResourceBarrier(1, &barrier);
    }
    void wait() {
        check(queue->Signal(fence.Get(), ++epoch), "Signal fence");
        if (fence->GetCompletedValue() < epoch) {
            check(fence->SetEventOnCompletion(epoch, event), "SetEventOnCompletion");
            require(WaitForSingleObject(event, 45000) == WAIT_OBJECT_0, CUDA_ERROR_UNKNOWN, "DirectX transfer completion timed out");
        }
        check(value->GetDeviceRemovedReason(), "DirectX device status");
    }
    void submit() {
        check(list->Close(), "Close command list");
        ID3D12CommandList* commands[] = {list.Get()};
        queue->ExecuteCommandLists(1, commands);
        wait();
    }
};
struct directx_buffer : buffer {
    std::shared_ptr<directx_device> owner;
    ComPtr<ID3D12Resource> value;
    size_t base_offset = 0;
    D3D12_RESOURCE_STATES resting_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    std::shared_ptr<external_memory> imported;
    directx_buffer(std::shared_ptr<directx_device> input, size_t bytes, ComPtr<ID3D12Resource> resource, size_t offset, std::shared_ptr<external_memory> memory)
        : buffer(bytes), owner(std::move(input)), value(std::move(resource)), base_offset(offset), resting_state(D3D12_RESOURCE_STATE_COMMON), imported(std::move(memory)) {}
    directx_buffer(std::shared_ptr<directx_device> input, size_t bytes) : buffer(bytes), owner(std::move(input)), value(owner->resource(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) {}
    size_t graph_backing_size() const override {
        auto description = value->GetDesc();
        auto info = owner->value->GetResourceAllocationInfo(0, 1, &description);
        require(info.SizeInBytes != std::numeric_limits<uint64_t>::max(), CUDA_ERROR_UNKNOWN, "invalid DirectX graph backing size");
        return static_cast<size_t>(info.SizeInBytes);
    }
    size_t physical_size() const override { return imported ? 0 : graph_backing_size(); }
};
struct directx_task : device_task {
    std::shared_ptr<directx_device> owner;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    uint64_t epoch = 0;
    ComPtr<ID3D12QueryHeap> timestamp_heap;
    ComPtr<ID3D12Resource> timestamp_result;
    std::shared_ptr<device_timestamp> timestamp;
    uint64_t timestamp_frequency = 0;
    std::shared_ptr<compute_profile> marker_profile,dispatch_profile;
    bool measured = false;
    explicit directx_task(std::shared_ptr<directx_device> input) : owner(std::move(input)) {
        check(owner->value->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "CreateCommandAllocator");
        check(owner->value->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "CreateCommandList");
    }
    void barrier(directx_buffer& buffer, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {buffer.value.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to}; list->ResourceBarrier(1, &barrier);
    }
    void submit() {
        check(list->Close(), "Close command list"); ID3D12CommandList* commands[] = {list.Get()};
        owner->queue->ExecuteCommandLists(1, commands);
        epoch = ++owner->epoch;
        try { check(owner->queue->Signal(owner->fence.Get(), epoch), "Signal fence"); }
        catch (...) { owner->wait(); throw; }
    }
    bool query() override {
        check(owner->value->GetDeviceRemovedReason(), "DirectX device status");
        if(owner->fence->GetCompletedValue()<epoch) return false;
        if(timestamp) {
            void* pointer{};D3D12_RANGE read{0,sizeof(uint64_t)};
            check(timestamp_result->Map(0,&read,&pointer),"Map timestamp result");std::memcpy(&timestamp->ticks,pointer,sizeof(uint64_t));
            D3D12_RANGE empty{0,0};timestamp_result->Unmap(0,&empty);timestamp->nanoseconds_per_tick=1e9/double(timestamp_frequency);
            if(marker_profile) marker_profile->snapshot(*timestamp);
        }
        if(dispatch_profile && !measured) {
            void* pointer{};D3D12_RANGE read{0,2*sizeof(uint64_t)};uint64_t ticks[2]{};
            check(timestamp_result->Map(0,&read,&pointer),"Map dispatch timestamps");std::memcpy(ticks,pointer,sizeof(ticks));
            D3D12_RANGE empty{0,0};timestamp_result->Unmap(0,&empty);
            dispatch_profile->nanoseconds.fetch_add(uint64_t(double(ticks[1]-ticks[0])*1e9/double(timestamp_frequency)));measured=true;
        }
        return true;
    }
    ~directx_task() {
        if (epoch && owner->fence->GetCompletedValue() < epoch) {
            auto event = CreateEventA(nullptr, FALSE, FALSE, nullptr);
            if (event) { if (SUCCEEDED(owner->fence->SetEventOnCompletion(epoch, event))) WaitForSingleObject(event, INFINITE); CloseHandle(event); }
        }
    }
};
template<class T, class Desc> ComPtr<T> open_shared(directx_device& owner, const Desc& d) {
    HANDLE handle = static_cast<HANDLE>(d.handle.win32.handle);
    bool named = !handle;
    if (named) check(owner.value->OpenSharedHandleByName(static_cast<LPCWSTR>(d.handle.win32.name), GENERIC_ALL, &handle), "OpenSharedHandleByName");
    ComPtr<T> result;
    auto status = owner.value->OpenSharedHandle(handle, IID_PPV_ARGS(&result));
    if (named) CloseHandle(handle);
    require(SUCCEEDED(status), CUDA_ERROR_INVALID_HANDLE, "DirectX external handle is incompatible with the selected device");
    return result;
}
struct directx_memory : external_memory, std::enable_shared_from_this<directx_memory> {
    std::shared_ptr<directx_device> owner;
    ComPtr<ID3D12Resource> resource;
    ComPtr<ID3D12Heap> heap;
    directx_memory(std::shared_ptr<directx_device> input, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC& d) : owner(std::move(input)) {
        if (d.type == CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE) {
            resource = open_shared<ID3D12Resource>(*owner, d);
            auto desc = resource->GetDesc();
            require(desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER, CUDA_ERROR_NOT_SUPPORTED, "external DirectX images are not implemented");
            auto allocation = owner->value->GetResourceAllocationInfo(0, 1, &desc);
            require(d.size == allocation.SizeInBytes, CUDA_ERROR_INVALID_VALUE, "DirectX allocation size does not match the descriptor");
        } else {
            heap = open_shared<ID3D12Heap>(*owner, d);
            require(d.size == heap->GetDesc().SizeInBytes, CUDA_ERROR_INVALID_VALUE, "DirectX heap size does not match the descriptor");
        }
    }
    std::unique_ptr<buffer> map(const CUDA_EXTERNAL_MEMORY_BUFFER_DESC& d) override {
        auto result = resource;
        size_t base = size_t(d.offset);
        if (result) {
            auto width = result->GetDesc().Width;
            require(d.offset <= width && d.size <= width - d.offset, CUDA_ERROR_INVALID_VALUE, "external buffer range exceeds the DirectX resource");
        } else {
            D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; desc.Width = d.size;
            desc.Height = 1; desc.DepthOrArraySize = desc.MipLevels = 1; desc.SampleDesc.Count = 1;
            desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            check(owner->value->CreatePlacedResource(heap.Get(), d.offset, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&result)), "CreatePlacedResource for imported heap");
            base = 0;
        }
        return std::make_unique<directx_buffer>(owner, size_t(d.size), std::move(result), base, shared_from_this());
    }
};
struct directx_semaphore : external_semaphore {
    std::shared_ptr<directx_device> owner;
    ComPtr<ID3D12Fence> value;
    directx_semaphore(std::shared_ptr<directx_device> input, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC& d)
        : owner(std::move(input)), value(open_shared<ID3D12Fence>(*owner, d)) {}
    void wait(uint64_t epoch) override { check(owner->queue->Wait(value.Get(), epoch), "Wait external DirectX fence"); }
    void signal(uint64_t epoch) override { check(owner->queue->Signal(value.Get(), epoch), "Signal external DirectX fence"); }
};
struct directx_pipeline : compute_pipeline {
    std::shared_ptr<directx_device> owner;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> state;
    explicit directx_pipeline(std::shared_ptr<directx_device> device) : owner(std::move(device)) {}
};
struct directx_session : session {
    std::shared_ptr<directx_device> owner;
    explicit directx_session(std::shared_ptr<directx_device> input) : owner(std::move(input)) {}
    interop_capabilities interop() override { return {handle_bit(CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP) | handle_bit(CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE), handle_bit(CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE)}; }
    std::shared_ptr<external_memory> import_memory(const CUDA_EXTERNAL_MEMORY_HANDLE_DESC& d, const cudavmExternalMemoryMetadata* = nullptr) override {
        require(interop().memory & handle_bit(d.type), CUDA_ERROR_NOT_SUPPORTED, "DirectX external memory handle type is unavailable");
        return std::make_shared<directx_memory>(owner, d);
    }
    std::shared_ptr<external_semaphore> import_semaphore(const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC& d) override {
        require(interop().semaphore & handle_bit(d.type), CUDA_ERROR_NOT_SUPPORTED, "DirectX external semaphore handle type is unavailable");
        return std::make_shared<directx_semaphore>(owner, d);
    }
    std::unique_ptr<buffer> allocate(size_t bytes) override { return std::make_unique<directx_buffer>(owner, bytes); }
    device_work enqueue_copy(buffer& to, size_t destination, buffer& from, size_t source, size_t bytes, native_queue* queue) override {
        if (&to.storage() != &to || &from.storage() != &from) return enqueue_copy(to.storage(), destination + to.storage_offset(), from.storage(), source + from.storage_offset(), bytes, queue);
        if (to.host_address() || from.host_address()) return session::enqueue_copy(to, destination, from, source, bytes, queue);
        auto& a = static_cast<directx_buffer&>(to); auto& b = static_cast<directx_buffer&>(from);
        if (a.value == b.value) {
            std::shared_ptr<buffer> temporary = allocate(bytes);
            auto tasks = std::make_shared<device_group>(); tasks->tasks.reserve(2);
            auto retained = std::make_shared<device_continuation>(tasks, [temporary] {});
            tasks->tasks.push_back(enqueue_copy(*temporary, 0, from, source, bytes, queue));
            tasks->tasks.push_back(enqueue_copy(to, destination, *temporary, 0, bytes, queue)); return retained;
        }
        auto task = std::make_shared<directx_task>(owner);
        task->barrier(b, b.resting_state, D3D12_RESOURCE_STATE_COPY_SOURCE); task->barrier(a, a.resting_state, D3D12_RESOURCE_STATE_COPY_DEST);
        task->list->CopyBufferRegion(a.value.Get(), destination + a.base_offset, b.value.Get(), source + b.base_offset, bytes);
        task->barrier(b, D3D12_RESOURCE_STATE_COPY_SOURCE, b.resting_state); task->barrier(a, D3D12_RESOURCE_STATE_COPY_DEST, a.resting_state);
        task->submit(); return task;
    }
    device_work enqueue_write(buffer& value, size_t offset, std::span<const std::byte> bytes, native_queue* queue) override {
        if (&value.storage() != &value) return enqueue_write(value.storage(), offset + value.storage_offset(), bytes, queue);
        if (value.host_address()) return session::enqueue_write(value, offset, bytes, queue);
        auto upload = owner->resource(bytes.size(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* pointer = nullptr; D3D12_RANGE empty{0, 0}; check(upload->Map(0, &empty, &pointer), "Map upload buffer");
        std::memcpy(pointer, bytes.data(), bytes.size()); D3D12_RANGE written{0, bytes.size()}; upload->Unmap(0, &written);
        auto continuation = std::make_shared<device_continuation>(device_work{}, [upload] {});
        auto task = std::make_shared<directx_task>(owner); auto& target = static_cast<directx_buffer&>(value);
        task->barrier(target, target.resting_state, D3D12_RESOURCE_STATE_COPY_DEST);
        task->list->CopyBufferRegion(target.value.Get(), offset + target.base_offset, upload.Get(), 0, bytes.size());
        task->barrier(target, D3D12_RESOURCE_STATE_COPY_DEST, target.resting_state);
        task->submit(); continuation->task = task; return continuation;
    }
    device_work enqueue_read(buffer& value, size_t offset, std::span<std::byte> bytes, native_queue* queue) override {
        if (&value.storage() != &value) return enqueue_read(value.storage(), offset + value.storage_offset(), bytes, queue);
        if (value.host_address()) return session::enqueue_read(value, offset, bytes, queue);
        auto download = owner->resource(bytes.size(), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        auto continuation = std::make_shared<device_continuation>(device_work{}, [download, bytes] {
            void* pointer = nullptr; D3D12_RANGE range{0, bytes.size()}; check(download->Map(0, &range, &pointer), "Map readback buffer");
            std::memcpy(bytes.data(), pointer, bytes.size()); D3D12_RANGE empty{0, 0}; download->Unmap(0, &empty);
        });
        auto task = std::make_shared<directx_task>(owner); auto& target = static_cast<directx_buffer&>(value);
        task->barrier(target, target.resting_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        task->list->CopyBufferRegion(download.Get(), 0, target.value.Get(), offset + target.base_offset, bytes.size());
        task->barrier(target, D3D12_RESOURCE_STATE_COPY_SOURCE, target.resting_state);
        task->submit(); continuation->task = task; return continuation;
    }
    device_work enqueue_fill(buffer& value, size_t offset, std::span<const std::byte> pattern, size_t bytes, native_queue* queue) override {
        if (!bytes) return {};
        if (&value.storage() != &value) return enqueue_fill(value.storage(), offset + value.storage_offset(), pattern, bytes, queue);
        if (value.host_address()) return session::enqueue_fill(value, offset, pattern, bytes, queue);
        auto& target = static_cast<directx_buffer&>(value);
        if ((((offset + target.base_offset) | bytes) & 3) || bytes / 4 > std::numeric_limits<UINT>::max() ||
            !(target.value->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
            return session::enqueue_fill(value, offset, pattern, bytes, queue);
        auto word = repeated_fill_word(pattern);
        D3D12_DESCRIPTOR_HEAP_DESC heap{}; heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; heap.NumDescriptors = 1;
        ComPtr<ID3D12DescriptorHeap> cpu, gpu;
        check(owner->value->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&cpu)), "CreateDescriptorHeap fill CPU");
        heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        check(owner->value->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&gpu)), "CreateDescriptorHeap fill GPU");
        D3D12_UNORDERED_ACCESS_VIEW_DESC view{}; view.Format = DXGI_FORMAT_R32_UINT; view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        view.Buffer.FirstElement = (offset + target.base_offset) / 4; view.Buffer.NumElements = UINT(bytes / 4);
        owner->value->CreateUnorderedAccessView(target.value.Get(), nullptr, &view, cpu->GetCPUDescriptorHandleForHeapStart());
        owner->value->CreateUnorderedAccessView(target.value.Get(), nullptr, &view, gpu->GetCPUDescriptorHandleForHeapStart());
        auto retained = std::make_shared<device_continuation>(device_work{}, [cpu, gpu] {});
        auto task = std::make_shared<directx_task>(owner);
        if (target.resting_state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            task->barrier(target, target.resting_state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; barrier.UAV.pResource = target.value.Get();
        task->list->ResourceBarrier(1, &barrier);
        ID3D12DescriptorHeap* heaps[] = {gpu.Get()}; task->list->SetDescriptorHeaps(1, heaps);
        UINT values[] = {word, word, word, word};
        task->list->ClearUnorderedAccessViewUint(gpu->GetGPUDescriptorHandleForHeapStart(), cpu->GetCPUDescriptorHandleForHeapStart(), target.value.Get(), values, 0, nullptr);
        task->list->ResourceBarrier(1, &barrier);
        if (target.resting_state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            task->barrier(target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, target.resting_state);
        task->submit(); retained->task = task; return retained;
    }
    void fill_device(buffer& value, size_t offset, std::span<const std::byte> pattern, size_t bytes) override {
        wait_device(enqueue_fill(value, offset, pattern, bytes, nullptr));
    }
    void write_device(buffer& value, size_t offset, std::span<const std::byte> bytes) override {
        if (&value.storage() != &value) { write_device(value.storage(), offset + value.storage_offset(), bytes); return; }
        auto upload = owner->resource(bytes.size(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped = nullptr;
        D3D12_RANGE empty{0, 0};
        check(upload->Map(0, &empty, &mapped), "Map upload buffer");
        std::memcpy(mapped, bytes.data(), bytes.size());
        D3D12_RANGE written{0, bytes.size()};
        upload->Unmap(0, &written);
        auto& target = static_cast<directx_buffer&>(value);
        auto destination = target.value.Get();
        owner->begin();
        owner->barrier(destination, target.resting_state, D3D12_RESOURCE_STATE_COPY_DEST);
        owner->list->CopyBufferRegion(destination, offset + target.base_offset, upload.Get(), 0, bytes.size());
        owner->barrier(destination, D3D12_RESOURCE_STATE_COPY_DEST, target.resting_state);
        owner->submit();
    }
    void read_device(buffer& value, size_t offset, std::span<std::byte> bytes) override {
        if (&value.storage() != &value) { read_device(value.storage(), offset + value.storage_offset(), bytes); return; }
        auto download = owner->resource(bytes.size(), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        auto& target = static_cast<directx_buffer&>(value);
        auto source = target.value.Get();
        owner->begin();
        owner->barrier(source, target.resting_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        owner->list->CopyBufferRegion(download.Get(), 0, source, offset + target.base_offset, bytes.size());
        owner->barrier(source, D3D12_RESOURCE_STATE_COPY_SOURCE, target.resting_state);
        owner->submit();
        D3D12_RANGE range{0, bytes.size()};
        void* mapped = nullptr;
        check(download->Map(0, &range, &mapped), "Map readback buffer");
        std::memcpy(bytes.data(), mapped, bytes.size());
        D3D12_RANGE empty{0, 0};
        download->Unmap(0, &empty);
    }
    void copy_device(buffer& to, size_t to_offset, buffer& from, size_t from_offset, size_t bytes) override {
        if (&to.storage() != &to || &from.storage() != &from) { copy_device(to.storage(), to_offset + to.storage_offset(), from.storage(), from_offset + from.storage_offset(), bytes); return; }
        if (static_cast<directx_buffer&>(to).value == static_cast<directx_buffer&>(from).value) {
            auto temporary = allocate(bytes);
            copy(*temporary, 0, from, from_offset, bytes);
            copy(to, to_offset, *temporary, 0, bytes);
            return;
        }
        auto& source_buffer = static_cast<directx_buffer&>(from); auto& destination_buffer = static_cast<directx_buffer&>(to);
        auto source = source_buffer.value.Get(); auto destination = destination_buffer.value.Get();
        owner->begin();
        owner->barrier(source, source_buffer.resting_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        owner->barrier(destination, destination_buffer.resting_state, D3D12_RESOURCE_STATE_COPY_DEST);
        owner->list->CopyBufferRegion(destination, to_offset + destination_buffer.base_offset, source, from_offset + source_buffer.base_offset, bytes);
        owner->barrier(source, D3D12_RESOURCE_STATE_COPY_SOURCE, source_buffer.resting_state);
        owner->barrier(destination, D3D12_RESOURCE_STATE_COPY_DEST, destination_buffer.resting_state);
        owner->submit();
    }
    void synchronize() override { owner->wait(); }
    cudavmBackend kind() const override { return cudavmBackendDirectX; }
    device_work enqueue_timestamp(std::shared_ptr<device_timestamp> timestamp,native_queue*) override {
        auto task=std::make_shared<directx_task>(owner);task->timestamp=std::move(timestamp);task->marker_profile=profile;
        check(owner->queue->GetTimestampFrequency(&task->timestamp_frequency),"GetTimestampFrequency");
        require(task->timestamp_frequency!=0,CUDA_ERROR_NOT_SUPPORTED,"DirectX timestamp frequency is unavailable");
        D3D12_QUERY_HEAP_DESC heap{};heap.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP;heap.Count=1;
        check(owner->value->CreateQueryHeap(&heap,IID_PPV_ARGS(&task->timestamp_heap)),"CreateQueryHeap timestamp");
        task->timestamp_result=owner->resource(sizeof(uint64_t),D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        task->list->EndQuery(task->timestamp_heap.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0);
        task->list->ResolveQueryData(task->timestamp_heap.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,1,task->timestamp_result.Get(),0);
        task->submit();return task;
    }
    unsigned direct_binding_limit() const override { return 6; }
    bool can_copy_direct(const buffer& value) const override {
        auto native = dynamic_cast<const directx_buffer*>(&value.storage());
        return native && native->owner->value == owner->value && !native->imported;
    }
    bool can_bind_direct(const buffer& value) const override {
        auto native=dynamic_cast<const directx_buffer*>(&value.storage());
        return native && native->owner->value==owner->value && !native->imported && value.size%4==0 && value.storage_offset()%4==0 && value.size<=UINT32_MAX && value.storage_offset()<=UINT32_MAX-value.size;
    }
    std::unique_ptr<compute_pipeline> compile_binary(const void* code_data,size_t code_size,unsigned direct_buffers=0) {
        auto compiled=std::make_unique<directx_pipeline>(owner); compiled->direct_buffers=direct_buffers;
        auto serialize=owner->api->d3d.get<decltype(&D3D12SerializeRootSignature)>("D3D12SerializeRootSignature");
        ComPtr<ID3DBlob> errors;
        std::vector<D3D12_ROOT_PARAMETER> slots(2+direct_buffers);slots[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;slots[0].Descriptor.ShaderRegister=0;
        slots[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV;slots[1].Descriptor.ShaderRegister=0;
        for(unsigned i=0;i<direct_buffers;++i) {slots[i+2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;slots[i+2].Descriptor.ShaderRegister=i+1;}
        D3D12_ROOT_SIGNATURE_DESC description{};description.NumParameters=UINT(slots.size());description.pParameters=slots.data();
        ComPtr<ID3DBlob> signature;check(serialize(&description,D3D_ROOT_SIGNATURE_VERSION_1,&signature,&errors),"D3D12SerializeRootSignature");
        check(owner->value->CreateRootSignature(0,signature->GetBufferPointer(),signature->GetBufferSize(),IID_PPV_ARGS(&compiled->root)),"CreateRootSignature");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};pipeline.pRootSignature=compiled->root.Get();pipeline.CS={code_data,code_size};
        check(owner->value->CreateComputePipelineState(&pipeline,IID_PPV_ARGS(&compiled->state)),"CreateComputePipelineState");
        return compiled;
    }
    bool supports_native_dxil(std::string_view code) const override {
        if(!owner->native_dxil) return false;
        try {
            auto flags=cudavm::inspect_dxil_program(code).features;
            uint64_t supported=(1ull<<1)|(1ull<<2)|(1ull<<15);
            if(owner->native_double) supported|=1ull|(1ull<<5);
            return !(flags & ~supported);
        } catch(const std::exception&) { return false; }
    }
    unsigned native_block_limit() const override { return D3D12_CS_THREAD_GROUP_MAX_THREADS_PER_GROUP; }
    shader_resource_limits shader_limits() const override {
        return {D3D12_CS_THREAD_GROUP_MAX_THREADS_PER_GROUP,
            {D3D12_CS_THREAD_GROUP_MAX_X, D3D12_CS_THREAD_GROUP_MAX_Y, D3D12_CS_THREAD_GROUP_MAX_Z},
            {D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION, D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION, D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION},
            D3D12_CS_TGSM_REGISTER_COUNT * 4};
    }
    void validate_native_dxil_shape(const launch_shape& shape) const override {
        uint64_t threads=1;
        for(unsigned axis=0;axis<3;++axis) {
            require(shape.grid[axis] && shape.grid[axis]<=D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION,CUDA_ERROR_INVALID_VALUE,"native DXIL grid exceeds DirectX dispatch limits");
            require(shape.block[axis] && shape.block[axis]<=1024,CUDA_ERROR_INVALID_VALUE,"native DXIL block exceeds DirectX dimensions"); threads*=shape.block[axis];
        }
        require(threads<=D3D12_CS_THREAD_GROUP_MAX_THREADS_PER_GROUP && shape.block[2]<=64,CUDA_ERROR_INVALID_VALUE,"native DXIL block exceeds DirectX invocation limits");
    }
    std::unique_ptr<compute_pipeline> compile_native_dxil(std::string_view bytes,const launch_shape& shape) override {
        require(supports_native_dxil(bytes),CUDA_ERROR_NOT_SUPPORTED,"DirectX native DXIL requires shader model 6.0 and Int64ShaderOps");
        validate_native_dxil_shape(shape);
#if defined(CUDAVM_DXC_LIBRARY) && defined(CUDAVM_DXIL_LIBRARY)
        static dynamic_library compiler({"dxcompiler.dll",CUDAVM_DXC_LIBRARY},true,CUDA_ERROR_JIT_COMPILER_NOT_FOUND);
        static dynamic_library validator({"dxil.dll",CUDAVM_DXIL_LIBRARY},true,CUDA_ERROR_JIT_COMPILER_NOT_FOUND);
        auto create_utils=compiler.get<decltype(&DxcCreateInstance)>("DxcCreateInstance");
        auto create_validator=validator.get<decltype(&DxcCreateInstance)>("DxcCreateInstance");
        ComPtr<IDxcUtils> utilities; ComPtr<IDxcValidator> validation;
        check(create_utils(CLSID_DxcUtils,IID_PPV_ARGS(&utilities)),"Create DXIL utilities");
        check(create_validator(CLSID_DxcValidator,IID_PPV_ARGS(&validation)),"Create DXIL validator");
        ComPtr<IDxcBlobEncoding> input; check(utilities->CreateBlob(bytes.data(),UINT(bytes.size()),DXC_CP_ACP,&input),"Create native DXIL blob");
        ComPtr<IDxcOperationResult> result; check(validation->Validate(input.Get(),DxcValidatorFlags_InPlaceEdit,&result),"Validate native DXIL");
        HRESULT status; check(result->GetStatus(&status),"Get native DXIL validation status");
        if(FAILED(status)) { ComPtr<IDxcBlobEncoding> errors; result->GetErrorBuffer(&errors); throw failure(CUDA_ERROR_INVALID_IMAGE,"Native DXIL validation failed: " +
            (errors?std::string(static_cast<const char*>(errors->GetBufferPointer()),errors->GetBufferSize()):std::to_string(status))); }
        ComPtr<IDxcBlob> code; check(result->GetResult(&code),"Get signed native DXIL");
        return compile_binary(code->GetBufferPointer(),code->GetBufferSize());
#else
        throw failure(CUDA_ERROR_NOT_SUPPORTED,"DXIL validation is unavailable in this build");
#endif
    }
    void dispatch(compute_pipeline& pipeline,const launch_shape& shape,buffer& memory,buffer& parameters) override {
        wait_device(enqueue_dispatch(pipeline, shape, memory, parameters, nullptr));
    }
    device_work enqueue_dispatch(compute_pipeline& pipeline,const launch_shape& shape,buffer& memory,buffer& parameters, native_queue*) override {
        return enqueue_dispatch_bound(pipeline,shape,memory,parameters,{},nullptr);
    }
    device_work enqueue_dispatch_bound(compute_pipeline& pipeline,const launch_shape& shape,buffer& memory,buffer& parameters,std::span<buffer* const> bindings,native_queue*) override {
        require(bindings.size()==pipeline.direct_buffers,CUDA_ERROR_INVALID_VALUE,"DirectX direct binding count mismatch");
        for(auto resource:bindings)require(can_bind_direct(*resource),CUDA_ERROR_INVALID_VALUE,"incompatible DirectX direct buffer");
        auto task = std::make_shared<directx_task>(owner);
        auto& compiled=static_cast<directx_pipeline&>(pipeline);
        auto mem=static_cast<directx_buffer&>(memory).value.Get(),args=static_cast<directx_buffer&>(parameters).value.Get();
        task->barrier(static_cast<directx_buffer&>(parameters),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        task->list->SetPipelineState(compiled.state.Get());task->list->SetComputeRootSignature(compiled.root.Get());
        task->list->SetComputeRootUnorderedAccessView(0,mem->GetGPUVirtualAddress());task->list->SetComputeRootShaderResourceView(1,args->GetGPUVirtualAddress());
        for(unsigned i=0;i<bindings.size();++i)task->list->SetComputeRootUnorderedAccessView(i+2,static_cast<directx_buffer&>(bindings[i]->storage()).value->GetGPUVirtualAddress());
        if(profile->enabled) {
            task->dispatch_profile=profile;check(owner->queue->GetTimestampFrequency(&task->timestamp_frequency),"GetTimestampFrequency dispatch");
            require(task->timestamp_frequency!=0,CUDA_ERROR_NOT_SUPPORTED,"DirectX dispatch timestamp frequency unavailable");
            D3D12_QUERY_HEAP_DESC query{};query.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP;query.Count=2;
            check(owner->value->CreateQueryHeap(&query,IID_PPV_ARGS(&task->timestamp_heap)),"CreateQueryHeap dispatch");
            task->timestamp_result=owner->resource(2*sizeof(uint64_t),D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
            task->list->EndQuery(task->timestamp_heap.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0);
        }
        task->list->Dispatch(shape.grid[0],shape.grid[1],shape.grid[2]);
        if(task->dispatch_profile) {
            task->list->EndQuery(task->timestamp_heap.Get(),D3D12_QUERY_TYPE_TIMESTAMP,1);
            task->list->ResolveQueryData(task->timestamp_heap.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,2,task->timestamp_result.Get(),0);
        }
        D3D12_RESOURCE_BARRIER uav{};uav.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;uav.UAV.pResource=nullptr;task->list->ResourceBarrier(1,&uav);
        task->barrier(static_cast<directx_buffer&>(parameters),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);task->submit(); return task;
    }
};
struct directx_backend : shader_backend {
    shader_resource_limits shader_limits(int) override {
        return {D3D12_CS_THREAD_GROUP_MAX_THREADS_PER_GROUP,
            {D3D12_CS_THREAD_GROUP_MAX_X, D3D12_CS_THREAD_GROUP_MAX_Y, D3D12_CS_THREAD_GROUP_MAX_Z},
            {D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION, D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION, D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION},
            D3D12_CS_TGSM_REGISTER_COUNT * 4};
    }
    std::shared_ptr<directx_api> api = std::make_shared<directx_api>();
    std::vector<ComPtr<IDXGIAdapter1>> devices;
    device_luid_info device_luid(int ordinal) override {
        DXGI_ADAPTER_DESC1 info{}; check(devices.at(ordinal)->GetDesc1(&info), "GetDesc1");
        device_luid_info result; static_assert(sizeof(info.AdapterLuid) == sizeof(result.bytes));
        std::memcpy(result.bytes.data(), &info.AdapterLuid, result.bytes.size());
        result.node_mask = 1; // This transport creates resources on node zero.
        return result;
    }
    device_resources<ID3D12Device> resources;
    std::vector<device_info> discover() override {
        ComPtr<IDXGIFactory1> factory;
        check(api->factory(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
        devices.clear();
        std::vector<device_info> result;
        for (UINT ordinal = 0;; ++ordinal) {
            ComPtr<IDXGIAdapter1> adapter;
            HRESULT status = factory->EnumAdapters1(ordinal, &adapter);
            if (status == DXGI_ERROR_NOT_FOUND) break;
            check(status, "EnumAdapters1");
            DXGI_ADAPTER_DESC1 description{};
            check(adapter->GetDesc1(&description), "GetDesc1");
            if (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (FAILED(api->create(adapter.Get(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr))) continue;
            int size = WideCharToMultiByte(CP_UTF8, 0, description.Description, -1, nullptr, 0, nullptr, nullptr);
            require(size > 0, CUDA_ERROR_UNKNOWN, "DirectX adapter name conversion failed");
            std::vector<char> name(size);
            WideCharToMultiByte(CP_UTF8, 0, description.Description, -1, name.data(), size, nullptr, nullptr);
            devices.push_back(adapter);
            result.push_back({name.data(), description.DedicatedVideoMemory});
        }
        return result;
    }
    std::unique_ptr<session> open(int ordinal) override {
        auto adapter = devices.at(ordinal);
        auto device = resources.get(ordinal, [&] {
            ID3D12Device* value = nullptr;
            check(api->create(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&value)), "D3D12CreateDevice");
            return std::shared_ptr<ID3D12Device>(value, [](ID3D12Device* p) { p->Release(); });
        });
        return std::make_unique<directx_session>(std::make_shared<directx_device>(api, std::move(device)));
    }
};
}
std::unique_ptr<backend> make_directx_backend() { return std::make_unique<directx_backend>(); }
}
#else
namespace cudavm {
std::unique_ptr<backend> make_directx_backend() { throw failure(CUDA_ERROR_NOT_SUPPORTED, "DirectX requires Windows"); }
}
#endif
