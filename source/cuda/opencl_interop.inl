struct opencl_memory : external_memory, std::enable_shared_from_this<opencl_memory> {
    std::shared_ptr<opencl_context> owner; cl_mem value = nullptr;
    opencl_memory(std::shared_ptr<opencl_context> input, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC& d) : owner(std::move(input)) {
        cl_mem_properties properties[7] = {CL_MEM_DEVICE_HANDLE_LIST_KHR, reinterpret_cast<cl_mem_properties>(owner->device), CL_MEM_DEVICE_HANDLE_LIST_END_KHR};
#if defined(_WIN32)
        require(!d.handle.win32.name, CUDA_ERROR_NOT_SUPPORTED, "named OpenCL external memory is not implemented");
        properties[3] = d.type == CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 ? CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR : CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR;
        properties[4] = reinterpret_cast<cl_mem_properties>(d.handle.win32.handle);
#else
        properties[3] = CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR; properties[4] = d.handle.fd;
#endif
        cl_int status = 0; value = owner->create_external_buffer(owner->context, properties, CL_MEM_READ_WRITE, size_t(d.size), nullptr, &status);
        check(status, "import OpenCL external memory");
#if !defined(_WIN32)
        close(d.handle.fd);
#endif
    }
    ~opencl_memory() { if (value) owner->api->clReleaseMemObject(value); }
    std::unique_ptr<buffer> map(const CUDA_EXTERNAL_MEMORY_BUFFER_DESC& d) override {
        cl_buffer_region region{size_t(d.offset), size_t(d.size)}; cl_int status = 0;
        cl_mem mapped = owner->api->clCreateSubBuffer(value, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
        check(status, "map OpenCL external buffer");
        try { return std::make_unique<opencl_buffer>(owner, size_t(d.size), mapped, shared_from_this()); }
        catch (...) { owner->api->clReleaseMemObject(mapped); throw; }
    }
};
struct opencl_external_access {
    opencl_buffer& buffer; bool active;
    opencl_external_access(opencl_buffer& input, bool enabled = true) : buffer(input), active(enabled && bool(input.imported)) {
        if (active) { auto& memory = static_cast<opencl_memory&>(*buffer.imported);
            check(buffer.owner->acquire_external(buffer.owner->queue, 1, &memory.value, 0, nullptr, nullptr), "acquire OpenCL external memory"); }
    }
    ~opencl_external_access() {
        if (active) { auto& memory = static_cast<opencl_memory&>(*buffer.imported);
            buffer.owner->release_external(buffer.owner->queue, 1, &memory.value, 0, nullptr, nullptr); buffer.owner->api->clFlush(buffer.owner->queue); }
    }
};
struct opencl_semaphore : external_semaphore {
    std::shared_ptr<opencl_context> owner; cl_semaphore_khr value = nullptr;
    opencl_semaphore(std::shared_ptr<opencl_context> input, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC& d) : owner(std::move(input)) {
        cl_semaphore_properties_khr properties[8] = {CL_SEMAPHORE_TYPE_KHR, CL_SEMAPHORE_TYPE_BINARY_KHR, CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR,
            reinterpret_cast<cl_semaphore_properties_khr>(owner->device), CL_SEMAPHORE_DEVICE_HANDLE_LIST_END_KHR};
#if defined(_WIN32)
        require(!d.handle.win32.name, CUDA_ERROR_NOT_SUPPORTED, "named OpenCL external semaphores are not implemented");
        properties[5] = d.type == CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 ? CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR : CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR;
        properties[6] = reinterpret_cast<cl_semaphore_properties_khr>(d.handle.win32.handle);
#else
        properties[5] = CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR; properties[6] = d.handle.fd;
#endif
        cl_int status = 0; value = owner->create_semaphore(owner->context, properties, &status); check(status, "import OpenCL external semaphore");
#if !defined(_WIN32)
        close(d.handle.fd);
#endif
    }
    ~opencl_semaphore() { if (value) owner->release_semaphore(value); }
    void signal(uint64_t) override { check(owner->signal_semaphore(owner->queue, 1, &value, nullptr, 0, nullptr, nullptr), "signal OpenCL external semaphore"); check(owner->api->clFlush(owner->queue), "flush OpenCL external signal"); }
    void wait(uint64_t) override { check(owner->wait_semaphore(owner->queue, 1, &value, nullptr, 0, nullptr, nullptr), "wait OpenCL external semaphore"); check(owner->api->clFlush(owner->queue), "flush OpenCL external wait"); }
};
