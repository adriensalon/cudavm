    interop_capabilities caps;
    bool interop_ready = false;
    decltype(&clCreateBufferWithProperties) create_external_buffer = nullptr;
    clEnqueueAcquireExternalMemObjectsKHR_fn acquire_external = nullptr;
    clEnqueueReleaseExternalMemObjectsKHR_fn release_external = nullptr;
    clCreateSemaphoreWithPropertiesKHR_fn create_semaphore = nullptr;
    clReleaseSemaphoreKHR_fn release_semaphore = nullptr;
    clEnqueueWaitSemaphoresKHR_fn wait_semaphore = nullptr;
    clEnqueueSignalSemaphoresKHR_fn signal_semaphore = nullptr;
    void negotiate_interop() {
        if (interop_ready) return;
        size_t size = 0; check(api->clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &size), "query OpenCL extensions");
        std::string extensions(size, '\0'); check(api->clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, size, extensions.data(), nullptr), "query OpenCL extensions");
        if (!extensions.empty() && !extensions.back()) extensions.pop_back();
        auto has = [&](const char* name) { return (" " + extensions + " ").find(" " + std::string(name) + " ") != std::string::npos; };
        cl_platform_id platform = nullptr; check(api->clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr), "query OpenCL platform");
        auto entry = [&](const char* name) { return api->clGetExtensionFunctionAddressForPlatform(platform, name); };
        auto handle_types = [&](cl_device_info query) {
            size_t bytes = 0; check(api->clGetDeviceInfo(device, query, 0, nullptr, &bytes), "query OpenCL external handle types");
            std::vector<cl_uint> result(bytes / sizeof(cl_uint));
            if (bytes) check(api->clGetDeviceInfo(device, query, bytes, result.data(), nullptr), "query OpenCL external handle types");
            return result;
        };
        if (has("cl_khr_external_memory")) {
            create_external_buffer = reinterpret_cast<decltype(create_external_buffer)>(api->driver.find("clCreateBufferWithProperties"));
            acquire_external = reinterpret_cast<decltype(acquire_external)>(entry("clEnqueueAcquireExternalMemObjectsKHR"));
            release_external = reinterpret_cast<decltype(release_external)>(entry("clEnqueueReleaseExternalMemObjectsKHR"));
            if (create_external_buffer && acquire_external && release_external) {
                auto types = handle_types(CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR);
                auto supports = [&](cl_uint type) { return std::find(types.begin(), types.end(), type) != types.end(); };
#if defined(_WIN32)
                if (has("cl_khr_external_memory_win32") && supports(CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR)) caps.memory |= handle_bit(2);
                if (has("cl_khr_external_memory_win32") && supports(CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR)) caps.memory |= handle_bit(3);
#else
                if (has("cl_khr_external_memory_opaque_fd") && supports(CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR)) caps.memory |= handle_bit(1);
#endif
            }
        }
        if (has("cl_khr_semaphore") && has("cl_khr_external_semaphore")) {
            create_semaphore = reinterpret_cast<decltype(create_semaphore)>(entry("clCreateSemaphoreWithPropertiesKHR"));
            release_semaphore = reinterpret_cast<decltype(release_semaphore)>(entry("clReleaseSemaphoreKHR"));
            wait_semaphore = reinterpret_cast<decltype(wait_semaphore)>(entry("clEnqueueWaitSemaphoresKHR"));
            signal_semaphore = reinterpret_cast<decltype(signal_semaphore)>(entry("clEnqueueSignalSemaphoresKHR"));
            if (create_semaphore && release_semaphore && wait_semaphore && signal_semaphore) {
                auto types = handle_types(CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR);
                auto supports = [&](cl_uint type) { return std::find(types.begin(), types.end(), type) != types.end(); };
#if defined(_WIN32)
                if (has("cl_khr_external_semaphore_win32") && supports(CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR)) caps.semaphore |= handle_bit(2);
                if (has("cl_khr_external_semaphore_win32") && supports(CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR)) caps.semaphore |= handle_bit(3);
#else
                if (has("cl_khr_external_semaphore_opaque_fd") && supports(CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR)) caps.semaphore |= handle_bit(1);
#endif
            }
        }
        interop_ready = true;
    }
