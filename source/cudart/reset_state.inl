void runtime_reset_modules(CUcontext owner, bool unload) {
    if (unload) {
        std::vector<CUlibrary> libraries; for (auto& [identity, library] : global().libraries) libraries.push_back(library);
        check(runtime_driver_wait([&] { return cudavmDriverReleaseRegisteredModules(owner, libraries.data(), libraries.size()); }));
    }
    auto found = global().modules.find(owner);
    if (found != global().modules.end()) {
        global().modules.erase(found);
    }
    std::erase_if(global().function_entries, [&](const auto& entry) { return entry.second.owner == owner; });
}
void runtime_reset() {
    check(cuInit(0));
    int count = 0; check(cuDeviceGetCount(&count));
    require(count != 0, CUDA_ERROR_NO_DEVICE, "selected backend has no GPU devices");
    CUcontext owner = nullptr; check(cuCtxGetCurrent(&owner));
    int device = owner ? cudavmDriverPrimaryDevice(owner) : runtime_device;
    if (owner && device < 0) {
        require(cudavmDriverContextExists(owner), CUDA_ERROR_INVALID_CONTEXT, "current Driver context was destroyed");
        // An explicit Driver context owns its memory and queues. Reset only
        // the modules and function mappings loaded by this Runtime instance.
        runtime_reset_modules(owner, true);
        return;
    }
    check(runtime_driver_wait([&] { return cuDevicePrimaryCtxReset(device); }));
    auto retained = global().primaries.find(device);
    if (retained != global().primaries.end()) {
        runtime_reset_modules(retained->second, false);
        global().primaries.erase(retained);
        auto result = cuDevicePrimaryCtxRelease(device);
        // A legacy Driver reset may already have consumed our only usage.
        if (result != CUDA_ERROR_INVALID_CONTEXT) check(result);
    } else if (owner) runtime_reset_modules(owner, false);
}
