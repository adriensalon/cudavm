std::shared_ptr<library_entry> library_lookup(CUlibrary handle) {
    auto it = global().libraries.find(handle);
    require(it != global().libraries.end() && !it->second->retiring, CUDA_ERROR_INVALID_HANDLE, "invalid or unloading library"); return it->second;
}
std::shared_ptr<library_kernel_entry> library_kernel_lookup(CUkernel handle) {
    auto it = global().library_kernels.find(handle);
    require(it != global().library_kernels.end(), CUDA_ERROR_INVALID_HANDLE, "invalid or unloaded library kernel");
    library_lookup(it->second->library); return it->second;
}
CUkernel library_find(CUlibrary handle, const char* name) {
    require(name && *name, CUDA_ERROR_INVALID_VALUE, "missing library kernel name"); auto entry = library_lookup(handle);
    if (auto it = entry->kernels.find(name); it != entry->kernels.end()) return it->second;
    if (entry->native) entry->native->find(name);
    else require(std::any_of(entry->descriptors.begin(), entry->descriptors.end(), [&](auto& item) { return item.symbol == name; }), CUDA_ERROR_NOT_FOUND, "kernel is absent from library");
    auto kernel = reinterpret_cast<CUkernel>(global().next_library_kernel); global().next_library_kernel += 0x100;
    auto item = std::make_shared<library_kernel_entry>(); item->library = handle; item->name = name;
    entry->kernels.emplace(name, kernel);
    try { global().library_kernels.emplace(kernel, std::move(item)); } catch (...) { entry->kernels.erase(name); throw; }
    return kernel;
}
std::vector<std::string> library_names(CUlibrary handle) {
    auto entry = library_lookup(handle); if (entry->native) return entry->native->kernel_names();
    std::vector<std::string> names; for (auto& item : entry->descriptors) names.push_back(item.symbol); std::sort(names.begin(), names.end()); return names;
}
CUmodule library_bind(CUlibrary handle, const std::shared_ptr<context>& owner) {
    auto entry = library_lookup(handle);
    std::erase_if(entry->modules, [](auto& item) { return !global().modules.contains(item.second); });
    if (auto it = entry->modules.find(owner->id); it != entry->modules.end()) return it->second;
    auto native = native_call(owner, [&] { return entry->native ? entry->native->bind(*owner->native) :
        owner->native->load_image(entry->image.c_str(), unsigned(entry->options.size()), entry->options.data(), entry->values.data()); }, true);
    library_lookup(handle);
    auto module = add_module(owner, std::move(native)); global().modules.at(module).library = handle;
    entry->modules.emplace(owner->id, module); return module;
}
void bind_context_libraries(const std::shared_ptr<context>& owner) {
    if (global().libraries.empty() || global().provider->loading_mode() != CU_MODULE_EAGER_LOADING) return;
    std::vector<CUlibrary> handles; for (auto& [handle, item] : global().libraries) if (!item->retiring) handles.push_back(handle);
    for (auto handle : handles) library_bind(handle, owner);
}
void configure_library_function(CUfunction handle) {
    auto& function = global().functions.at(handle); auto library = global().modules.at(function.module).library;
    if (!library || library_lookup(library)->native) return;
    auto entry = library_kernel_lookup(library_find(library, function.name.c_str())); auto device = function.owner->device;
    if (auto it = entry->attributes.find(device); it != entry->attributes.end())
        for (auto& [attribute, value] : it->second) function.native->set_attribute(attribute, value);
    if (auto it = entry->caches.find(device); it != entry->caches.end()) function.native->set_cache_config(it->second);
}
CUfunction resolve_library_function(CUfunction handle, const std::shared_ptr<context>& owner) {
    auto kernel = reinterpret_cast<CUkernel>(handle);
    if (!global().library_kernels.contains(kernel)) return handle;
    auto entry = library_kernel_lookup(kernel); return find_function(library_bind(entry->library, owner), entry->name.c_str(), owner);
}
std::shared_ptr<context> launch_context(CUfunction handle, CUstream stream) {
    if (global().library_kernels.contains(reinterpret_cast<CUkernel>(handle)) && stream && stream != CU_STREAM_LEGACY && stream != CU_STREAM_PER_THREAD) {
        auto it = global().streams.find(stream); require(it != global().streams.end(), CUDA_ERROR_INVALID_HANDLE, "invalid library launch stream");
        auto owner = it->second->owner.lock(); require(owner && !owner->retiring, CUDA_ERROR_INVALID_CONTEXT, "library stream context was destroyed"); return owner;
    }
    return current();
}
std::shared_ptr<context> library_query_context(CUlibrary handle, int device) {
    valid_device(device); auto entry = library_lookup(handle); auto& owner = entry->queries[device];
    if (!owner && device == 0 && entry->registration) owner = entry->registration->owner;
    if (!owner) {
        auto value = std::make_shared<context>(); value->device = device; value->native = global().provider->open(device);
        value->id = global().next_context; global().next_context += 0x100; owner = std::move(value);
        if (device == 0 && entry->registration) entry->registration->owner = owner;
    }
    return owner;
}
global_symbol library_managed(CUlibrary handle, const char* name) {
    require(name && *name, CUDA_ERROR_INVALID_VALUE, "missing library managed symbol name");
    auto entry = library_lookup(handle);
    if (auto it = entry->managed.find(name); it != entry->managed.end()) return it->second;
    require(bool(entry->native), CUDA_ERROR_NOT_SUPPORTED, "native library managed symbols are unavailable on this adapter");
    // A private binding registers native managed storage without depending on
    // the lifetime of a public context or changing the caller's current binding.
    auto owner = library_query_context(handle, 0); library_bind(handle, owner);
    auto storage = native_call(owner, [&] { return entry->native->managed(name, *owner->native); }, true);
    library_lookup(handle);
    // Waiting for foreign work can let another host thread publish this name.
    if (auto it = entry->managed.find(name); it != entry->managed.end()) return it->second;
    auto address = add_allocation(owner, std::move(storage)); global().allocations.at(address).library = handle;
    global_symbol result{address, global().allocations.at(address).native->size};
    try {
        entry->managed.emplace(name, result);
        if (entry->registration) entry->registration->storage.emplace(name, global().allocations.at(address).native);
    } catch (...) { entry->managed.erase(name); global().allocations.erase(address); throw; }
    return result;
}
int library_attribute(CUkernel kernel, CUfunction_attribute attribute, int device) {
    valid_device(device); auto item = library_kernel_lookup(kernel); auto entry = library_lookup(item->library);
    if (entry->native) return entry->native->attribute(item->name.c_str(), attribute, device);
    auto owner = library_query_context(item->library, device);
    auto function = find_function(library_bind(item->library, owner), item->name.c_str(), owner);
    return context_configuration(owner, [&] { return global().functions.at(function).native->attribute(attribute); });
}
void library_configure(CUkernel kernel, CUfunction_attribute attribute, int value, int device, bool cache = false) {
    valid_device(device); auto item = library_kernel_lookup(kernel); auto entry = library_lookup(item->library);
    if (entry->native) {
        std::vector<std::shared_ptr<context>> owners;
        for (auto& [id, module] : entry->modules) if (auto it = global().modules.find(module); it != global().modules.end() && it->second.owner->device == device) owners.push_back(it->second.owner);
        for (auto& owner : owners) drain_context_work(owner);
        library_lookup(item->library);
        if (cache) entry->native->set_cache_config(item->name.c_str(), static_cast<CUfunc_cache>(value), device);
        else entry->native->set_attribute(item->name.c_str(), attribute, value, device);
        return;
    }
    auto owner = library_query_context(item->library, device);
    auto function = find_function(library_bind(item->library, owner), item->name.c_str(), owner);
    auto apply = [&](const std::shared_ptr<cudavm::kernel>& native) {
        if (cache) native->set_cache_config(static_cast<CUfunc_cache>(value)); else native->set_attribute(attribute, value);
    };
    context_configuration(owner, [&] { apply(global().functions.at(function).native); });
    if (cache) item->caches[device] = static_cast<CUfunc_cache>(value); else item->attributes[device][attribute] = value;
    std::vector<CUfunction> functions;
    for (auto& [handle, fn] : global().functions) if (fn.name == item->name && fn.owner->device == device && global().modules.at(fn.module).library == item->library && handle != function) functions.push_back(handle);
    for (auto handle : functions) {
        auto it = global().functions.find(handle); if (it == global().functions.end()) continue; auto target = it->second.owner;
        native_call(target, [&] {
            auto found = global().functions.find(handle); if (found == global().functions.end()) return;
            if (cache ? !found->second.cache_override : !found->second.overrides.contains(attribute)) apply(found->second.native);
        }, true);
    }
}
std::vector<kernel_parameter> library_parameters(CUkernel kernel) {
    auto item = library_kernel_lookup(kernel); auto entry = library_lookup(item->library);
    if (entry->native) return entry->native->parameters(item->name.c_str());
    require(!entry->unavailable_parameters.contains(item->name), CUDA_ERROR_NOT_SUPPORTED, "registered PTX parameter layout is unavailable");
    for (auto& descriptor : entry->descriptors) if (descriptor.symbol == item->name) {
        std::vector<kernel_parameter> result; for (auto& p : descriptor.parameters) result.push_back({p.offset, p.size}); return result;
    }
    throw failure(CUDA_ERROR_NOT_FOUND, "library parameter metadata is absent");
}
void library_unload(CUlibrary handle) {
    auto entry = library_lookup(handle);
    std::vector<std::shared_ptr<context>> owners;
    for (auto& [id, module] : entry->modules) if (auto it = global().modules.find(module); it != global().modules.end()) owners.push_back(it->second.owner);
    for (auto& owner : owners) capture_unsafe(owner);
    entry->retiring = true;
    try { for (auto& owner : owners) synchronize_context(owner); }
    catch (...) { entry->retiring = false; throw; }
    std::unordered_set<CUmodule> modules; for (auto& [id, module] : entry->modules) modules.insert(module);
    std::erase_if(global().functions, [&](auto& it) { return modules.contains(it.second.module); });
    std::erase_if(global().allocations, [&](auto& it) { return it.second.library == handle || modules.contains(it.second.module); });
    std::erase_if(global().modules, [&](auto& it) { return modules.contains(it.first); });
    if (entry->native && !entry->registration) entry->native->unload();
    std::erase_if(global().library_kernels, [&](auto& it) { return it.second->library == handle; });
    global().libraries.erase(handle);
}
CUlibrary add_library(const std::shared_ptr<library_entry>& entry, bool eager = true) {
    auto handle = reinterpret_cast<CUlibrary>(global().next_library); global().next_library += 0x100;
    global().libraries.emplace(handle, entry);
    try {
        if (eager && global().provider->loading_mode() == CU_MODULE_EAGER_LOADING) {
            std::vector<std::shared_ptr<context>> owners; for (auto& [id, owner] : global().contexts) owners.push_back(owner);
            for (auto& owner : owners) library_bind(handle, owner);
        }
    } catch (...) { library_unload(handle); throw; }
    return handle;
}
CUlibrary registered_managed_facade(const std::shared_ptr<registered_managed_image>& image) {
    require(!image->retired, CUDA_ERROR_NOT_PERMITTED, "managed registration is retiring");
    if (image->epoch == global().epoch && global().libraries.contains(image->facade)) { library_lookup(image->facade); return image->facade; }
    auto entry = std::make_shared<library_entry>(); entry->registered = true; entry->registration = image; entry->native = image->native;
    if (image->owner) entry->queries.emplace(0, image->owner);
    auto handle = add_library(entry, false);
    try {
        for (auto& [name, native] : image->storage) {
            auto address = add_shared_allocation(image->owner, native); global().allocations.at(address).library = handle;
            entry->managed.emplace(name, global_symbol{address, native->size});
        }
    } catch (...) { std::erase_if(global().allocations, [&](auto& item) { return item.second.library == handle; }); global().libraries.erase(handle); throw; }
    image->facade = handle; image->epoch = global().epoch; return handle;
}
CUlibrary registered_managed_load(const std::shared_ptr<registered_managed_image>& image) {
    require(global().kind == cudavmBackendCuda, CUDA_ERROR_NOT_SUPPORTED, "NVIDIA managed registration requires CUDA");
    if (!image->native) image->native = global().provider->load_library(image->image.data(), nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0);
    require(bool(image->native), CUDA_ERROR_NOT_SUPPORTED, "native managed registration library is unavailable");
    auto handle = registered_managed_facade(image);
    if (!image->ready) {
        try {
            // Retain managed storage before launches, even when a kernel is
            // used before the first compiler-generated host initialization.
            auto owner = library_query_context(handle, 0); library_bind(handle, owner);
            for (auto& name : image->names) (void)library_managed(handle, name.c_str());
            image->ready = true;
        } catch (...) { library_unload(handle); throw; }
    }
    return handle;
}
void restore_registered_images() {
    if (global().kind != cudavmBackendCuda) return;
    std::erase_if(global().registered_images, [](const auto& item) { return item.expired(); });
    // Restore known buffers without native waits or context creation, before
    // another thread can observe the new facade epoch.
    for (auto& weak : global().registered_images) if (auto image = weak.lock(); image && image->ready && !image->retired) (void)registered_managed_facade(image);
}
CUlibrary registered_library_load(const void* image, size_t bytes) {
    require(image && bytes && static_cast<const char*>(image)[bytes - 1] == 0, CUDA_ERROR_INVALID_IMAGE, "invalid registered library image");
    auto entry = std::make_shared<library_entry>(); entry->registered = true;
    entry->image.assign(static_cast<const char*>(image), bytes - 1);
    // Registration owns each context's native module independently. Runtime
    // reset can then retire one binding while preserving other contexts and
    // the contextless kernel identity. CUDA execution still uses its loader.
    if (has_module_descriptor(entry->image)) {
        try { entry->descriptors = decode_module_descriptor(entry->image).entries; }
        catch (const std::exception& e) { throw failure(CUDA_ERROR_INVALID_IMAGE, e.what()); }
    } else {
        require(global().kind == cudavmBackendCuda, CUDA_ERROR_NOT_SUPPORTED, "registered shader libraries require native module metadata");
        auto metadata = nvidia_image_metadata(std::as_bytes(std::span(entry->image.data(), entry->image.size())));
        for (auto& [name, parameters] : metadata.kernels) {
            kernel_descriptor descriptor; descriptor.symbol = name;
            if (!metadata.unavailable.contains(name)) {
                for (auto p : parameters) descriptor.parameters.push_back({uint32_t(p.size), 1, uint32_t(p.offset), 0});
            } else entry->unavailable_parameters.insert(name);
            entry->descriptors.push_back(std::move(descriptor));
        }
    }
    return add_library(entry);
}
void registered_module_release(CUmodule module, const std::shared_ptr<context>& owner) {
    auto& binding = module_lookup(module, owner);
    auto entry = library_lookup(binding.library);
    require(entry->registered, CUDA_ERROR_NOT_PERMITTED, "module is not owned by Runtime registration");
    auto native = binding.native; synchronize_context(owner);
    require(module_lookup(module, owner).native == native, CUDA_ERROR_INVALID_HANDLE, "registered module changed while waiting");
    native->unload();
    std::erase_if(global().functions, [&](auto& item) { return item.second.module == module; });
    std::erase_if(global().allocations, [&](auto& item) { return item.second.module == module; });
    global().modules.erase(module);
    std::erase_if(entry->modules, [&](auto& item) { return item.second == module; });
}
CUlibrary library_load(const void* image, const char* path, CUjit_option* options, void** values, unsigned count,
                       CUlibraryOption* library_options, void** library_values, unsigned library_count) {
    require((image || (path && *path)) && (!count || (options && values)) && (!library_count || (library_options && library_values)), CUDA_ERROR_INVALID_VALUE, "invalid library load arguments");
    for (unsigned i = 0; i < library_count; ++i) {
        require(library_options[i] >= 0 && library_options[i] < CU_LIBRARY_NUM_OPTIONS && !(path && library_options[i] == CU_LIBRARY_BINARY_IS_PRESERVED), CUDA_ERROR_INVALID_VALUE, "invalid library load option");
        if (library_options[i] == CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE) {
            require(library_values[i], CUDA_ERROR_INVALID_VALUE, "null host universal function/data table");
            require(global().kind == cudavmBackendCuda, CUDA_ERROR_NOT_SUPPORTED, "host universal tables require native CUDA");
        }
    }
    auto entry = std::make_shared<library_entry>();
    for (unsigned i = 0; i < count; ++i) require(options[i] >= 0 && options[i] < CU_JIT_NUM_OPTIONS, CUDA_ERROR_INVALID_VALUE, "invalid JIT option identifier");
    entry->native = global().provider->load_library(image, path, options, values, count, library_options, library_values, library_count);
    if (!entry->native) {
        if (path) { std::ifstream file(path, std::ios::binary); require(bool(file), CUDA_ERROR_FILE_NOT_FOUND, "library file not found"); entry->image.assign(std::istreambuf_iterator<char>(file), {}); }
        else entry->image = static_cast<const char*>(image);
        for (unsigned i = 0; i < count; ++i) {
            require(options[i] == CU_JIT_OPTIMIZATION_LEVEL, CUDA_ERROR_NOT_SUPPORTED, "shader library JIT option is unavailable");
            require(reinterpret_cast<uintptr_t>(values[i]) <= 4, CUDA_ERROR_INVALID_VALUE, "invalid optimization level");
            entry->options.push_back(options[i]); entry->values.push_back(values[i]);
        }
        require(has_module_descriptor(entry->image), CUDA_ERROR_NOT_SUPPORTED, "shader libraries require a cudavm native module descriptor");
        try { entry->descriptors = decode_module_descriptor(entry->image).entries; }
        catch (const std::exception& e) { throw failure(CUDA_ERROR_INVALID_IMAGE, e.what()); }
    }
    return add_library(entry);
}
