void** register_nvidia_image(void* input) {
    require(input, CUDA_ERROR_INVALID_VALUE, "null NVIDIA fatbinary wrapper");
    struct wrapper { uint32_t magic, version; const void* image; const void* auxiliary; } value;
    std::memcpy(&value, input, sizeof(value));
    require(value.magic == 0x466243b1 && value.image, CUDA_ERROR_INVALID_IMAGE, "invalid NVIDIA fatbinary wrapper");
    require(value.version == 1 || value.version == 2, CUDA_ERROR_NOT_SUPPORTED, "unsupported NVIDIA fatbinary wrapper version");
    // NVCC's device-link object has already combined the relocatable inputs.
    // Version 2 names that final image; auxiliary points to the prelink inputs
    // used by NVIDIA's optional runtime-link path, not additional host bindings.
    require(value.version != 2 || value.auxiliary, CUDA_ERROR_INVALID_IMAGE, "linked NVIDIA wrapper has no prelink table");
    size_t size = 0; check(cudavmDriverNvidiaImageSize(value.image, &size));
    program_image image{cudavmBackendCuda, {}, {}};
    image.bytes.resize(size + 1); std::memcpy(image.bytes.data(), value.image, size);
    check(cudavmDriverValidateNvidiaImage(image.bytes.data(), size));
    void** handle = nullptr; std::string identity;
    do {
        handle = reinterpret_cast<void**>(global().next_nvidia_registration); global().next_nvidia_registration += 0x100;
        identity = "nvcc-fatbinary:" + std::to_string(reinterpret_cast<uintptr_t>(handle));
    } while (global().programs.contains(identity));
    global().programs.emplace(identity, std::move(image));
    try { global().nvidia_registrations.emplace(handle, nvidia_registration{identity}); }
    catch (...) { global().programs.erase(identity); throw; }
    return handle;
}
const std::string& nvidia_registration_identity(void** handle) {
    auto found = global().nvidia_registrations.find(handle);
    require(found != global().nvidia_registrations.end(), CUDA_ERROR_INVALID_VALUE, "invalid NVIDIA registration handle");
    require(!global().programs.at(found->second.identity).retiring, CUDA_ERROR_NOT_PERMITTED, "NVIDIA registration is unloading");
    return found->second.identity;
}
void register_nvidia_binding(void** handle, const void* token, const char* name, bool kernel) {
    require(token && name && *name, CUDA_ERROR_INVALID_VALUE, "invalid NVIDIA host symbol registration");
    auto identity = nvidia_registration_identity(handle); auto& table = kernel ? global().bindings : global().symbols;
    auto& owners = kernel ? global().nvidia_kernel_owners : global().nvidia_symbol_owners;
    auto found = table.find(token);
    require(found == table.end() || (found->second.symbol == name && (found->second.identity == identity || owners.contains(token))),
        CUDA_ERROR_INVALID_IMAGE, "NVIDIA host symbol has another definition");
    binding value{identity, name};
    auto [owner, added] = owners.try_emplace(token);
    auto& alternatives = owner->second;
    try {
        if (found == table.end()) table.emplace(token, std::move(value));
        else if (found->second.identity != identity) {
            auto previous = std::find_if(alternatives.begin(), alternatives.end(), [&](const auto& item) { return item.identity == identity; });
            if (previous == alternatives.end()) {
                alternatives.push_back(std::move(value)); std::swap(found->second, alternatives.back());
            } else {
                std::swap(found->second, *previous); std::rotate(previous, previous + 1, alternatives.end());
            }
        }
    } catch (...) { if (added) owners.erase(owner); throw; }
}
void register_nvidia_managed(void** handle, void** host, const char* name, size_t bytes) {
    require(host && name && *name && bytes, CUDA_ERROR_INVALID_VALUE, "invalid NVIDIA managed variable registration");
    auto identity = nvidia_registration_identity(handle); auto& image = global().programs.at(identity);
    if (auto found = std::find_if(image.managed.begin(), image.managed.end(), [&](auto& item) { return item.host == host; }); found != image.managed.end()) {
        if (found->symbol != name || found->bytes != bytes) { image.managed_conflict = true; throw failure(CUDA_ERROR_INVALID_IMAGE, "conflicting managed variable registration"); }
        return;
    }
    auto previous = global().symbols.find(host);
    if (previous != global().symbols.end() && previous->second.identity != identity) {
        const auto& other = global().programs.at(previous->second.identity).managed;
        auto shared = std::find_if(other.begin(), other.end(), [&](auto& item) { return item.host == host; });
        if (shared == other.end() || shared->symbol != name || shared->bytes != bytes) {
            // Constructor hooks cannot return errors; poison incomplete images
            // so a rejected binding cannot silently fall back to ordinary load.
            image.managed_conflict = true;
            throw failure(CUDA_ERROR_INVALID_IMAGE, "conflicting coalesced managed variable registration");
        }
    }
    require(!image.managed_image && !global().libraries.contains(identity), CUDA_ERROR_NOT_PERMITTED, "managed variables must be registered before image use");
    image.managed.push_back({host, name, bytes});
    try { register_nvidia_binding(handle, host, name, false); } catch (...) { image.managed.pop_back(); throw; }
}
void initialize_nvidia_managed(void** handle) {
    auto identity = nvidia_registration_identity(handle); auto records = global().programs.at(identity).managed;
    require(!global().programs.at(identity).managed_conflict, CUDA_ERROR_NOT_SUPPORTED, "managed registration contains conflicting host slots");
    auto retained = runtime_managed_image(global().programs.at(identity));
    check(cuInit(0)); auto library = runtime_library(identity);
    std::vector<CUdeviceptr> addresses; addresses.reserve(records.size());
    for (auto& item : records) {
        CUdeviceptr address = 0; size_t bytes = 0;
        auto status = runtime_driver_wait([&] { return cuLibraryGetManaged(&address, &bytes, library, item.symbol.c_str()); });
        nvidia_registration_identity(handle); check(status);
        require(bytes == item.bytes, CUDA_ERROR_INVALID_IMAGE, "managed variable size disagrees with the image"); addresses.push_back(address);
    }
    nvidia_registration_identity(handle);
    // Allocate every symbol alias before publishing any compiler host pointer.
    for (size_t i = 0; i < records.size(); ++i) register_nvidia_binding(handle, reinterpret_cast<void*>(uintptr_t(addresses[i])), records[i].symbol.c_str(), false);
    auto& current = global().programs.at(identity).managed;
    for (size_t i = 0; i < records.size(); ++i) {
        auto pointer = reinterpret_cast<void*>(uintptr_t(addresses[i])); current[i].alias = pointer; *records[i].host = pointer;
    }
}
cudaKernel_t compiler_kernel(const void* token) {
    auto found = global().bindings.find(token);
    symbol_require(token && found != global().bindings.end(), cudaErrorInvalidDeviceFunction, "compiler kernel entry is not registered");
    require(!global().programs.at(found->second.identity).retiring, CUDA_ERROR_NOT_PERMITTED, "compiler kernel program is unloading");
    require(!global().programs.at(found->second.identity).managed_conflict, CUDA_ERROR_NOT_SUPPORTED, "managed registration contains conflicting host slots");
    if (auto entry = global().compiler_entries.find(token); entry != global().compiler_entries.end()) return entry->second;
    auto handle = reinterpret_cast<cudaKernel_t>(global().next_compiler_kernel); global().next_compiler_kernel += 0x100;
    global().compiler_kernels.emplace(handle, token);
    try { global().compiler_entries.emplace(token, handle); } catch (...) { global().compiler_kernels.erase(handle); throw; }
    return handle;
}
CUkernel compiler_kernel_binding(cudaKernel_t handle) {
    auto found = global().compiler_kernels.find(handle);
    symbol_require(found != global().compiler_kernels.end(), cudaErrorInvalidDeviceFunction, "invalid or unregistered compiler kernel");
    // Compiler stubs cache once. Resolve the live host entry on each launch so
    // reset, Driver shutdown and weak-definition retirement cannot strand it.
    auto token = found->second;
    return runtime_registered_kernel(token);
}
void unregister_nvidia_image(void** handle) {
    if (!global().nvidia_registrations.contains(handle)) return;
    auto identity = nvidia_registration_identity(handle); auto& image = global().programs.at(identity); image.retiring = true;
    try {
        if (image.managed_image) {
            // A compiler slot may outlive the image which first supplied its
            // storage. Bind surviving registrations before retiring that keeper,
            // including images whose host initialization has not run yet.
            std::vector<void**> survivors;
            for (auto& [other_handle, registration] : global().nvidia_registrations) {
                auto& other = global().programs.at(registration.identity);
                if (registration.identity == identity || other.retiring || other.managed_conflict) continue;
                if (std::any_of(other.managed.begin(), other.managed.end(), [&](auto& value) {
                    return std::any_of(image.managed.begin(), image.managed.end(), [&](auto& item) { return item.host == value.host; });
                })) survivors.push_back(other_handle);
            }
            for (auto survivor : survivors) initialize_nvidia_managed(survivor);
            // A new Driver epoch can restore this image before Runtime has
            // rebound it. Retire its keeper, rather than just our cached handle.
            auto retained = image.managed_image;
            check(runtime_driver_wait([&] { return cudavmDriverRetireManagedRegistration(retained.get()); }));
        } else if (auto found = global().libraries.find(identity); found != global().libraries.end()) {
            auto library = found->second; auto epoch = global().epoch;
            auto code = runtime_driver_wait([&] { return cuLibraryUnload(library); });
            if (code != CUDA_ERROR_INVALID_HANDLE && !(code == CUDA_ERROR_NOT_INITIALIZED && epoch != cudavmDriverEpoch())) check(code);
        }
    } catch (...) { image.retiring = false; throw; }
    for (auto& item : image.managed) {
        bool shared = false;
        for (auto& [other_identity, other] : global().programs) if (other_identity != identity && !other.managed_conflict)
            shared |= std::any_of(other.managed.begin(), other.managed.end(), [&](auto& value) { return value.host == item.host; });
        if (!shared) *item.host = nullptr;
    }
    std::erase_if(global().function_entries, [&](const auto& item) { return item.second.identity == identity; });
    auto remove_definitions = [&](auto& owners, auto& table) {
        for (auto it = owners.begin(); it != owners.end();) {
            auto& alternatives = it->second; auto& active = table.at(it->first);
            std::erase_if(alternatives, [&](const auto& item) { return item.identity == identity; });
            if (active.identity == identity) {
                if (alternatives.empty()) { table.erase(it->first); it = owners.erase(it); continue; }
                // Moving the next definition avoids allocation after native
                // unload has committed, including under concurrent retirement.
                active = std::move(alternatives.back()); alternatives.pop_back();
            }
            ++it;
        }
    };
    remove_definitions(global().nvidia_kernel_owners, global().bindings);
    remove_definitions(global().nvidia_symbol_owners, global().symbols);
    for (auto it = global().compiler_entries.begin(); it != global().compiler_entries.end();) {
        if (!global().bindings.contains(it->first)) { global().compiler_kernels.erase(it->second); it = global().compiler_entries.erase(it); }
        else ++it;
    }
    for (auto& [owner, cache] : global().modules) cache.modules.erase(identity);
    global().libraries.erase(identity); global().programs.erase(identity); global().nvidia_registrations.erase(handle);
}
