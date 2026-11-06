std::pair<CUmodule, module_image_reference*> image_reference_lookup(std::uintptr_t handle, bool surface) {
    auto owner = current();
    for (auto& [module, entry] : global().modules) {
        auto found = entry.image_references.find(handle);
        if (found == entry.image_references.end()) continue;
        require(found->second.surface == surface, CUDA_ERROR_INVALID_VALUE, "wrong image reference kind");
        module_lookup(module, owner);
        return {module, &found->second};
    }
    throw failure(CUDA_ERROR_INVALID_VALUE, "unknown or retired image reference");
}
std::uintptr_t find_image_reference(CUmodule handle, const char* name, bool surface) {
    require(name, CUDA_ERROR_INVALID_VALUE, "null reference name");
    auto owner = current(); auto& entry = module_lookup(handle, owner);
    for (const auto& [reference, value] : entry.image_references)
        if (value.surface == surface && value.name == name) return reference;
    auto native = entry.native;
    auto reference = native_call(owner, [&] {
        module_lookup(handle, owner); return native->find_image_reference(name, surface);
    }, true);
    auto& module = module_lookup(handle, owner);
    for (const auto& [cached, value] : module.image_references)
        if (value.surface == surface && value.name == name) return cached;
    require(global().next_image_reference < UINTPTR_MAX, CUDA_ERROR_OUT_OF_MEMORY, "image reference handle space exhausted");
    auto result = global().next_image_reference++;
    module.image_references.emplace(result, module_image_reference{name, surface, std::move(reference)});
    return result;
}
template<class F> auto image_reference_call(std::uintptr_t handle, bool surface, F function) {
    auto [module, reference] = image_reference_lookup(handle, surface);
    auto owner = current(); auto native = reference->native;
    return native_call(owner, [&] { module_lookup(module, owner); return function(*native); });
}
void bind_reference_array(std::uintptr_t handle, bool surface, CUarray array, unsigned flags) {
    require(surface ? flags == 0 : flags == CU_TRSA_OVERRIDE_FORMAT, CUDA_ERROR_INVALID_VALUE, "invalid reference binding flags");
    auto resource = array_lookup(array, current());
    if (surface) require(resource->descriptor.Flags & CUDA_ARRAY3D_SURFACE_LDST, CUDA_ERROR_INVALID_VALUE, "surface reference requires surface-capable storage");
    image_reference_call(handle, surface, [&](image_reference& reference) { reference.bind_array(*resource->native, flags); });
    auto [module, reference] = image_reference_lookup(handle, surface);
    reference->array = array; reference->backing = resource->native;
}
