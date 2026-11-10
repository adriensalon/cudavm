// Versioned CUDA 12.4 Windows x64 embedded-Runtime boundary. The producer and
// native differential witnesses own the UUIDs, layouts and handshake inputs.
namespace cudavm {
#if defined(_WIN64)
struct runtime_cudart_interface {
    size_t size;
    void* reserved1;
    CUresult (CUDAAPI* primary_context)(CUcontext*, CUdevice);
    std::array<void*, 8> reserved;
};
struct runtime_callback_interface {
    size_t size;
    void* reserved1;
    void (CUDAAPI* enabled_callbacks)(const void**, size_t*);
    std::array<void*, 3> reserved;
    void (CUDAAPI* enabled_domains)(const void**, size_t*);
};
struct runtime_context_interface {
    size_t size;
    CUresult (CUDAAPI* set_current)(CUcontext);
    CUresult (CUDAAPI* get_current)(CUcontext*);
};
struct runtime_handshake_interface {
    size_t size;
    CUresult (CUDAAPI* query)(unsigned, uint64_t, void*);
    void* reserved;
};
struct runtime_storage_interface {
    CUresult (CUDAAPI* insert)(CUcontext, void*, void*, void (CUDAAPI*)(CUcontext, void*, void*));
    CUresult (CUDAAPI* remove)(CUcontext, void*);
    CUresult (CUDAAPI* find)(void**, CUcontext, void*);
};
static_assert(sizeof(runtime_cudart_interface) == 0x58 && sizeof(runtime_callback_interface) == 0x38
    && sizeof(runtime_context_interface) == 0x18 && sizeof(runtime_handshake_interface) == 0x18);

CUresult CUDAAPI runtime_primary_context(CUcontext* output, CUdevice device) {
    return driver_call([&] {
        driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null private primary-context output"); valid_device(device);
        auto& record = global().primary_records[device];
        if (!record.handle) {
            record.handle = reinterpret_cast<CUcontext>(global().next_context);
            global().next_context += 0x100;
        }
        *output = record.handle;
    });
}
void CUDAAPI runtime_enabled_callbacks(const void** output, size_t* count) {
    static const std::array<uint32_t, 1024> disabled{};
    *output = disabled.data(); *count = disabled.size();
}
void CUDAAPI runtime_enabled_domains(const void** output, size_t* count) {
    static const std::array<uint32_t, 14> disabled{};
    *output = disabled.data(); *count = disabled.size();
}
CUresult CUDAAPI runtime_storage_insert(CUcontext handle, void* key, void* value, void (CUDAAPI* destroy_entry)(CUcontext, void*, void*)) {
    return driver_call([&] {
        driver_ready(); auto owner = handle ? lookup(handle) : current();
        require(!owner->local_storage.contains(key), CUDA_ERROR_INVALID_HANDLE, "context-local storage key already exists");
        owner->local_storage.emplace(key, context_local_storage{value, destroy_entry});
    });
}
CUresult CUDAAPI runtime_storage_remove(CUcontext handle, void* key) {
    return driver_call([&] {
        driver_ready(); auto owner = handle ? lookup(handle) : current();
        require(owner->local_storage.erase(key) != 0, CUDA_ERROR_INVALID_HANDLE, "context-local storage key is absent");
    });
}
CUresult CUDAAPI runtime_storage_find(void** output, CUcontext handle, void* key) {
    return driver_call([&] {
        driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null context-local storage output");
        auto owner = handle ? lookup(handle) : current();
        auto found = owner->local_storage.find(key);
        require(found != owner->local_storage.end(), CUDA_ERROR_INVALID_HANDLE, "context-local storage key is absent");
        *output = found->second.value;
    });
}
CUresult CUDAAPI runtime_handshake_query(unsigned, uint64_t, void*);
const runtime_cudart_interface runtime_cudart_exports{sizeof(runtime_cudart_interface), nullptr, runtime_primary_context, {}};
const runtime_callback_interface runtime_callback_exports{sizeof(runtime_callback_interface), nullptr, runtime_enabled_callbacks, {}, runtime_enabled_domains};
const runtime_context_interface runtime_context_exports{sizeof(runtime_context_interface), cuCtxSetCurrent, cuCtxGetCurrent};
const runtime_handshake_interface runtime_handshake_exports{sizeof(runtime_handshake_interface), runtime_handshake_query, nullptr};
const runtime_storage_interface runtime_storage_exports{runtime_storage_insert, runtime_storage_remove, runtime_storage_find};

CUresult CUDAAPI runtime_handshake_query(unsigned request, uint64_t nonce, void* output) {
    return driver_call([&] {
        driver_ready(); require(output && request >= 12040 && request <= 12042, CUDA_ERROR_INVALID_VALUE, "unsupported embedded-Runtime handshake request");
        std::array<uint8_t, 16> result{0x5c,0x67,0xcb,0x03,0x1c,0x18,0x41,0x33,0xe8,0xd1,0x4c,0x1f,0xaa,0x83,0xd3,0x8e};
        if (request == 12041) result[7] = 0x18;
        if (request == 12042) {
            auto hash = runtime_handshake_begin();
            runtime_handshake_append(hash, uint32_t(CUDA_VERSION));
            runtime_handshake_append(hash, uint32_t(request));
            runtime_handshake_append(hash, uint32_t(GetCurrentProcessId()));
            runtime_handshake_append(hash, uint32_t(GetCurrentThreadId()));
            runtime_handshake_append(hash, uint64_t(reinterpret_cast<uintptr_t>(&runtime_cudart_exports)));
            runtime_handshake_append(hash, uint64_t(reinterpret_cast<uintptr_t>(&runtime_handshake_exports)));
            runtime_handshake_append(hash, uint64_t(reinterpret_cast<uintptr_t>(&runtime_handshake_query)));
            runtime_handshake_append(hash, nonce);
            for (size_t device = 0; device < global().devices.size(); ++device) {
                auto uuid = global().provider->device_uuid(int(device), false);
                hash.append({reinterpret_cast<const uint8_t*>(uuid.data()), uuid.size()});
                for (auto attribute : {CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID})
                    runtime_handshake_append(hash, uint32_t(provider_device_attribute(int(device), attribute)));
            }
            result = runtime_handshake_finish(hash);
        }
        std::memcpy(output, result.data(), result.size());
    });
}
#endif
}

extern "C" CUDAVM_API CUresult CUDAAPI cuGetExportTable(const void** output, const CUuuid* id) {
    return driver_call([&] {
        require(output && id, CUDA_ERROR_INVALID_VALUE, "null Driver export-table argument"); driver_ready();
#if defined(_WIN64)
        const std::array<unsigned char, 16> runtime_id{0x6b,0xd5,0xfb,0x6c,0x5b,0xf4,0xe7,0x4a,0x89,0x87,0xd9,0x39,0x12,0xfd,0x9d,0xf9};
        const std::array<unsigned char, 16> callbacks_id{0xa0,0x94,0x79,0x8c,0x2e,0x74,0x2e,0x74,0x93,0xf2,0x08,0x00,0x20,0x0c,0x0a,0x66};
        const std::array<unsigned char, 16> context_id{0x42,0xd8,0x5a,0x81,0x23,0xf6,0xcb,0x47,0x82,0x98,0xf6,0xe7,0x8a,0x3a,0xec,0xdc};
        const std::array<unsigned char, 16> handshake_id{0xd4,0x08,0x20,0x55,0xbd,0xe6,0x70,0x4b,0x8d,0x34,0xba,0x12,0x3c,0x66,0xe1,0xf2};
        const std::array<unsigned char, 16> storage_id{0xc6,0x93,0x33,0x6e,0x11,0x21,0xdf,0x11,0xa8,0xc3,0x68,0xf3,0x55,0xd8,0x95,0x93};
        if (!std::memcmp(id->bytes, runtime_id.data(), 16)) *output = &runtime_cudart_exports;
        else if (!std::memcmp(id->bytes, callbacks_id.data(), 16)) *output = &runtime_callback_exports;
        else if (!std::memcmp(id->bytes, context_id.data(), 16)) *output = &runtime_context_exports;
        else if (!std::memcmp(id->bytes, handshake_id.data(), 16)) *output = &runtime_handshake_exports;
        else if (!std::memcmp(id->bytes, storage_id.data(), 16)) *output = &runtime_storage_exports;
        else { *output = nullptr; throw failure(CUDA_ERROR_INVALID_VALUE, "unknown or unsupported Driver export-table UUID"); }
#else
        *output = nullptr;
        throw failure(CUDA_ERROR_INVALID_VALUE, "Driver export-table UUID is unavailable on this host ABI");
#endif
    });
}
