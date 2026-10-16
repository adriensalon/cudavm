    interop_capabilities caps;
    std::optional<std::array<char, 16>> identity_uuid;
    std::optional<device_luid_info> identity_luid;
    using memory_parameter_type = void (GL_CALL*)(uint, enum_type, const int*);
    using memory_storage_type = void (GL_CALL*)(enum_type, size_type, uint, uint64_t);
    using semaphore_parameter_type = void (GL_CALL*)(uint, enum_type, const uint64_t*);
    using semaphore_operation_type = void (GL_CALL*)(uint, uint, const uint*, uint, const uint*, const enum_type*);
    gen_type create_memory = nullptr, create_semaphore = nullptr;
    delete_type delete_memory = nullptr, delete_semaphore = nullptr;
    memory_parameter_type memory_parameter = nullptr;
    memory_storage_type memory_storage = nullptr;
    semaphore_parameter_type semaphore_parameter = nullptr;
    semaphore_operation_type wait_semaphore = nullptr, signal_semaphore = nullptr;
#if defined(_WIN32)
    using memory_import_type = void (GL_CALL*)(uint, uint64_t, enum_type, void*);
    using memory_name_type = void (GL_CALL*)(uint, uint64_t, enum_type, const void*);
    using semaphore_import_type = void (GL_CALL*)(uint, enum_type, void*);
    using semaphore_name_type = void (GL_CALL*)(uint, enum_type, const void*);
    memory_name_type memory_name = nullptr;
    semaphore_name_type semaphore_name = nullptr;
#else
    using memory_import_type = void (GL_CALL*)(uint, uint64_t, enum_type, int);
    using semaphore_import_type = void (GL_CALL*)(uint, enum_type, int);
#endif
    memory_import_type memory_import = nullptr;
    semaphore_import_type semaphore_import = nullptr;
    void initialize_interop(opengl_context& context) {
        using integer_type = void (GL_CALL*)(enum_type, int*);
        using string_index_type = const unsigned char* (GL_CALL*)(enum_type, uint);
        auto integer = load<integer_type>(context, "glGetIntegerv"); auto string_index = load<string_index_type>(context, "glGetStringi");
        int count = 0; integer(0x821d, &count); std::vector<std::string> extensions;
        for (int i = 0; i < count; ++i) { auto value = string_index(0x1f03, uint(i)); if (value) extensions.emplace_back(reinterpret_cast<const char*>(value)); }
        auto has = [&](const char* name) { return std::find(extensions.begin(), extensions.end(), name) != extensions.end(); };
        // GL_EXT_external_objects identities describe the devices executing this
        // GL context. A multi-device context cannot supply one CUDA device UUID.
        if (has("GL_EXT_memory_object") || has("GL_EXT_semaphore")) {
            int devices = 0; integer(0x9596, &devices); // GL_NUM_DEVICE_UUIDS_EXT
            if (devices == 1) {
                std::array<char, 16> uuid;
                load<void (GL_CALL*)(enum_type, uint, unsigned char*)>(context, "glGetUnsignedBytei_vEXT")(0x9597, 0, reinterpret_cast<unsigned char*>(uuid.data()));
                identity_uuid = uuid;
#if defined(_WIN32)
                if (has("GL_EXT_memory_object_win32") || has("GL_EXT_semaphore_win32")) {
                    device_luid_info luid; int mask = 0;
                    load<void (GL_CALL*)(enum_type, unsigned char*)>(context, "glGetUnsignedBytevEXT")(0x9599, reinterpret_cast<unsigned char*>(luid.bytes.data()));
                    integer(0x959a, &mask); luid.node_mask = unsigned(mask); identity_luid = luid;
                }
#endif
                check("OpenGL device identity");
            }
        }
        native_spirv = has("GL_ARB_gl_spirv") && shader_binary && specialize;
        native_int64 = has("GL_ARB_gpu_shader_int64");
        if (has("GL_NVX_gpu_memory_info")) {
            int kilobytes=0; integer(0x9047,&kilobytes);
            if(kilobytes>0) memory_bytes=static_cast<size_t>(kilobytes)*1024;
        }
#if defined(_WIN32)
        bool memory_available = has("GL_EXT_memory_object") && has("GL_EXT_memory_object_win32");
        bool semaphore_available = has("GL_EXT_semaphore") && has("GL_EXT_semaphore_win32");
#else
        bool memory_available = has("GL_EXT_memory_object") && has("GL_EXT_memory_object_fd");
        bool semaphore_available = has("GL_EXT_semaphore") && has("GL_EXT_semaphore_fd");
#endif
        if (memory_available) {
            create_memory = load<gen_type>(context, "glCreateMemoryObjectsEXT"); delete_memory = load<delete_type>(context, "glDeleteMemoryObjectsEXT");
            memory_parameter = load<memory_parameter_type>(context, "glMemoryObjectParameterivEXT"); memory_storage = load<memory_storage_type>(context, "glBufferStorageMemEXT");
#if defined(_WIN32)
            memory_import = load<memory_import_type>(context, "glImportMemoryWin32HandleEXT"); memory_name = load<memory_name_type>(context, "glImportMemoryWin32NameEXT");
            for (unsigned type : {2u, 3u, 5u}) caps.memory |= handle_bit(type);
#else
            memory_import = load<memory_import_type>(context, "glImportMemoryFdEXT"); caps.memory = handle_bit(CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD);
#endif
        }
        if (semaphore_available) {
            create_semaphore = load<gen_type>(context, "glGenSemaphoresEXT"); delete_semaphore = load<delete_type>(context, "glDeleteSemaphoresEXT");
            semaphore_parameter = load<semaphore_parameter_type>(context, "glSemaphoreParameterui64vEXT");
            wait_semaphore = load<semaphore_operation_type>(context, "glWaitSemaphoreEXT"); signal_semaphore = load<semaphore_operation_type>(context, "glSignalSemaphoreEXT");
#if defined(_WIN32)
            semaphore_import = load<semaphore_import_type>(context, "glImportSemaphoreWin32HandleEXT"); semaphore_name = load<semaphore_name_type>(context, "glImportSemaphoreWin32NameEXT");
            for (unsigned type : {2u, 3u, 4u}) caps.semaphore |= handle_bit(type);
#else
            semaphore_import = load<semaphore_import_type>(context, "glImportSemaphoreFdEXT"); caps.semaphore = handle_bit(CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD);
#endif
        }
    }
