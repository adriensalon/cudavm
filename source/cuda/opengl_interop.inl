enum_type gl_memory_type(unsigned type) {
    switch (type) {
    case CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD: return 0x9586;
    case CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32: return 0x9587;
    case CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT: return 0x9588;
    case CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE: return 0x958a;
    default: throw failure(CUDA_ERROR_NOT_SUPPORTED, "unsupported GL memory handle type");
    }
}
enum_type gl_semaphore_type(unsigned type) {
    switch (type) {
    case CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD: return 0x9586;
    case CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32: return 0x9587;
    case CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT: return 0x9588;
    case CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE: return 0x9594;
    default: throw failure(CUDA_ERROR_NOT_SUPPORTED, "unsupported GL semaphore handle type");
    }
}
struct opengl_memory : external_memory, std::enable_shared_from_this<opengl_memory> {
    std::shared_ptr<opengl_worker> owner; uint value = 0;
    opengl_memory(std::shared_ptr<opengl_worker> input, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC& d) : owner(std::move(input)) {
        owner->invoke([&] {
            auto& api = *owner->api; api.create_memory(1, &value);
            try {
                int dedicated = (d.flags & CUDA_EXTERNAL_MEMORY_DEDICATED) != 0; api.memory_parameter(value, 0x9581, &dedicated);
#if defined(_WIN32)
                auto bytes = d.type == CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE ? 0ull : d.size;
                if (d.handle.win32.handle) api.memory_import(value, bytes, gl_memory_type(d.type), d.handle.win32.handle);
                else api.memory_name(value, bytes, gl_memory_type(d.type), d.handle.win32.name);
#else
                int fd = dup(d.handle.fd); require(fd >= 0, CUDA_ERROR_INVALID_HANDLE, "duplicate GL external memory fd");
                api.memory_import(value, d.size, gl_memory_type(d.type), fd);
                auto status = api.error(); if (status) { close(fd); throw failure(CUDA_ERROR_INVALID_HANDLE, "GL external memory import failed"); } close(d.handle.fd);
#endif
                api.check("import GL external memory");
            } catch (...) { api.delete_memory(1, &value); value = 0; throw; }
        });
    }
    ~opengl_memory() { if (value) { try { owner->invoke([&] { owner->api->delete_memory(1, &value); }); } catch (...) {} } }
    std::unique_ptr<buffer> map(const CUDA_EXTERNAL_MEMORY_BUFFER_DESC& d) override {
        require(d.size <= uint64_t(std::numeric_limits<size_type>::max()), CUDA_ERROR_INVALID_VALUE, "GL external buffer is too large");
        auto result = std::make_unique<opengl_buffer>(owner, size_t(d.size), 0, shared_from_this());
        owner->invoke([&] {
            auto& api = *owner->api; api.gen(1, &result->value); api.bind(copy_write, result->value);
            api.memory_storage(copy_write, size_type(d.size), value, d.offset); api.check("map GL external buffer"); owner->external_buffers.push_back(result->value);
        }); return result;
    }
};
struct opengl_semaphore : external_semaphore {
    std::shared_ptr<opengl_worker> owner; uint value = 0; bool fence;
    opengl_semaphore(std::shared_ptr<opengl_worker> input, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC& d)
        : owner(std::move(input)), fence(d.type == CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE) {
        owner->invoke([&] {
            auto& api = *owner->api; api.create_semaphore(1, &value);
            try {
#if defined(_WIN32)
                if (d.handle.win32.handle) api.semaphore_import(value, gl_semaphore_type(d.type), d.handle.win32.handle);
                else api.semaphore_name(value, gl_semaphore_type(d.type), d.handle.win32.name);
#else
                int fd = dup(d.handle.fd); require(fd >= 0, CUDA_ERROR_INVALID_HANDLE, "duplicate GL external semaphore fd");
                api.semaphore_import(value, gl_semaphore_type(d.type), fd);
                auto status = api.error(); if (status) { close(fd); throw failure(CUDA_ERROR_INVALID_HANDLE, "GL external semaphore import failed"); } close(d.handle.fd);
#endif
                api.check("import GL external semaphore");
            } catch (...) { api.delete_semaphore(1, &value); value = 0; throw; }
        });
    }
    ~opengl_semaphore() { if (value) { try { owner->invoke([&] { owner->api->delete_semaphore(1, &value); }); } catch (...) {} } }
    void operation(bool signal, uint64_t epoch) {
        owner->invoke([&] {
            auto& api = *owner->api; if (fence) api.semaphore_parameter(value, 0x9595, &epoch);
            auto function = signal ? api.signal_semaphore : api.wait_semaphore;
            function(value, uint(owner->external_buffers.size()), owner->external_buffers.data(), 0, nullptr, nullptr); api.check("GL external synchronization");
        });
    }
    void signal(uint64_t epoch) override { operation(true, epoch); }
    void wait(uint64_t epoch) override { operation(false, epoch); }
};
