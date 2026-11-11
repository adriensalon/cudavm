#pragma once
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace cudavm {
struct graph_external_fences {
    template<class T> using pointer = Microsoft::WRL::ComPtr<T>;
    HMODULE dxgi{}, directx{};
    pointer<ID3D12Device> device;
    pointer<ID3D12CommandQueue> producer, consumer;
    pointer<ID3D12Fence> input, output, gate, completion;
    HANDLE input_handle{}, output_handle{}, completed{};
    std::mutex mutex; std::condition_variable changed;
    bool finished = false; std::atomic<bool> expired = false;
    std::thread watchdog;
    uint64_t epoch{};
    static void check(HRESULT value) { require(SUCCEEDED(value), "independent D3D12 fence operation: " + std::to_string(value)); }
    graph_external_fences() {
        dxgi = LoadLibraryExW(L"dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        directx = LoadLibraryExW(L"d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        require_capability(dxgi && directx, "independent D3D12 runtime unavailable");
        auto factory_function = reinterpret_cast<decltype(&CreateDXGIFactory1)>(GetProcAddress(dxgi, "CreateDXGIFactory1"));
        auto create = reinterpret_cast<decltype(&D3D12CreateDevice)>(GetProcAddress(directx, "D3D12CreateDevice"));
        require(factory_function && create, "independent D3D12 entry points");
        CUdevice selected{}; TEST_CUDA(cuCtxGetDevice(&selected)); char luid[8]{}; unsigned mask{};
        TEST_CUDA(cuDeviceGetLuid(luid, &mask, selected));
        pointer<IDXGIFactory1> factory; check(factory_function(IID_PPV_ARGS(&factory)));
        for (UINT i = 0; ; ++i) {
            pointer<IDXGIAdapter1> adapter; auto status = factory->EnumAdapters1(i, &adapter);
            if (status == DXGI_ERROR_NOT_FOUND) break;
            check(status); DXGI_ADAPTER_DESC1 desc{}; check(adapter->GetDesc1(&desc));
            if (!std::memcmp(&desc.AdapterLuid, luid, sizeof(luid))) { check(create(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))); break; }
        }
        require_capability(bool(device), "no independent D3D12 adapter matches selected device LUID");
        D3D12_COMMAND_QUEUE_DESC description{}; description.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        check(device->CreateCommandQueue(&description, IID_PPV_ARGS(&producer)));
        check(device->CreateCommandQueue(&description, IID_PPV_ARGS(&consumer)));
        check(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&input)));
        check(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&output)));
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate)));
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&completion)));
        check(device->CreateSharedHandle(input.Get(), nullptr, GENERIC_ALL, nullptr, &input_handle));
        check(device->CreateSharedHandle(output.Get(), nullptr, GENERIC_ALL, nullptr, &output_handle));
        completed = CreateEventW(nullptr, FALSE, FALSE, nullptr); require(completed, "native fence completion event");
        watchdog = std::thread([this] {
            std::unique_lock lock(mutex);
            if (!changed.wait_for(lock, std::chrono::seconds(20), [&] { return finished; })) {
                expired = true; gate->Signal(UINT64_MAX); input->Signal(UINT64_MAX); output->Signal(UINT64_MAX);
            }
        });
    }
    ~graph_external_fences() {
        { std::lock_guard lock(mutex); finished = true; } changed.notify_all();
        if (watchdog.joinable()) watchdog.join();
        if (input_handle) CloseHandle(input_handle); if (output_handle) CloseHandle(output_handle); if (completed) CloseHandle(completed);
        // COM objects must retire before unloading their runtime modules.
        producer.Reset(); consumer.Reset(); input.Reset(); output.Reset(); gate.Reset(); completion.Reset(); device.Reset();
        if (directx) FreeLibrary(directx); if (dxgi) FreeLibrary(dxgi);
    }
    void prepare(uint64_t incoming, uint64_t outgoing) {
        ++epoch; check(producer->Wait(gate.Get(), epoch)); check(producer->Signal(input.Get(), incoming));
        check(consumer->Wait(output.Get(), outgoing)); check(consumer->Signal(completion.Get(), epoch));
    }
    void release() { check(gate->Signal(epoch)); }
    void await() {
        check(completion->SetEventOnCompletion(epoch, completed));
        require(WaitForSingleObject(completed, 5000) == WAIT_OBJECT_0, "bounded independent D3D12 consumer completion");
        require(!expired, "external handoff completes before watchdog release");
    }
};
}
#endif
