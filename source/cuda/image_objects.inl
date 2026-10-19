// Object descriptors use private registry addresses, never public allocations.
// The shader snapshot binds descriptor and array storage once even when several
// objects alias one array. CUDA objects carry the native driver's opaque value.
auto& image_objects(bool surface) { return surface ? global().surfaces : global().textures; }
std::shared_ptr<image_object_entry> image_object_lookup(uint64_t handle, bool surface) {
    auto& objects = image_objects(surface); auto it = objects.find(handle);
    require(it != objects.end() && !it->second->retiring, CUDA_ERROR_INVALID_VALUE, "invalid image object");
    require(it->second->owner == current(), CUDA_ERROR_INVALID_CONTEXT, "image object belongs to another context");
    return it->second;
}
uint64_t create_image_object(const CUDA_RESOURCE_DESC& resource, const CUDA_TEXTURE_DESC* texture,
                             const CUDA_RESOURCE_VIEW_DESC* view) {
    auto owner = current(); bool surface = !texture;
    require(!resource.flags, CUDA_ERROR_INVALID_VALUE, "image resource flags must be zero");
    require(resource.resType >= CU_RESOURCE_TYPE_ARRAY && resource.resType <= CU_RESOURCE_TYPE_PITCH2D,
            CUDA_ERROR_INVALID_VALUE, "invalid image resource type");
    if (surface) require(resource.resType == CU_RESOURCE_TYPE_ARRAY, CUDA_ERROR_INVALID_VALUE, "surfaces require an array");
    require(resource.resType == CU_RESOURCE_TYPE_ARRAY, CUDA_ERROR_NOT_SUPPORTED, "this image profile requires an array resource");
    auto array = array_lookup(resource.res.array.hArray, owner);
    if (surface) require(array->descriptor.Flags & CUDA_ARRAY3D_SURFACE_LDST, CUDA_ERROR_INVALID_VALUE, "array does not enable surface access");
    if (texture) {
        for (auto mode : texture->addressMode) require(mode >= CU_TR_ADDRESS_MODE_WRAP && mode <= CU_TR_ADDRESS_MODE_BORDER, CUDA_ERROR_INVALID_VALUE, "invalid texture address mode");
        for (auto mode : {texture->filterMode, texture->mipmapFilterMode}) require(mode == CU_TR_FILTER_MODE_POINT || mode == CU_TR_FILTER_MODE_LINEAR, CUDA_ERROR_INVALID_VALUE, "invalid texture filter mode");
        constexpr unsigned flags = CU_TRSF_READ_AS_INTEGER | CU_TRSF_NORMALIZED_COORDINATES | CU_TRSF_SRGB |
            CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION | CU_TRSF_SEAMLESS_CUBEMAP;
        require(!(texture->flags & ~flags), CUDA_ERROR_INVALID_VALUE, "invalid texture flags");
    }
    capture_unsafe(owner);
    auto entry = std::make_shared<image_object_entry>(); entry->owner = owner; entry->array = array; entry->resource = resource;
    if (texture) entry->texture = *texture;
    if (view) entry->view = *view;
    if (global().kind == cudavmBackendCuda) {
        entry->native = context_configuration(owner, [&] { return surface ? owner->native->create_surface(*array->native)
            : owner->native->create_texture(*array->native, *texture, view); });
    } else {
        auto& d = array->descriptor;
        require(d.Height && !d.Depth && !(d.Flags & (CUDA_ARRAY3D_LAYERED | CUDA_ARRAY3D_CUBEMAP)) && d.NumChannels == 1,
                CUDA_ERROR_NOT_SUPPORTED, "shader image objects require a single-channel 2D array");
        require(d.Width <= UINT32_MAX && d.Height <= UINT32_MAX && array->native->size < (1u << 30), CUDA_ERROR_NOT_SUPPORTED, "image exceeds shader address transport");
        if (surface) require(d.Format == CU_AD_FORMAT_SIGNED_INT32 || d.Format == CU_AD_FORMAT_UNSIGNED_INT32 || d.Format == CU_AD_FORMAT_FLOAT,
                CUDA_ERROR_NOT_SUPPORTED, "shader surfaces require 32-bit elements");
        else {
            require(d.Format == CU_AD_FORMAT_FLOAT && texture->filterMode == CU_TR_FILTER_MODE_POINT &&
                !(texture->flags & ~CU_TRSF_READ_AS_INTEGER), CUDA_ERROR_NOT_SUPPORTED, "shader textures require unnormalized point-sampled float elements");
            // Wrap and mirror become clamp when coordinates are unnormalized.
            require(texture->addressMode[0] != CU_TR_ADDRESS_MODE_BORDER && texture->addressMode[1] != CU_TR_ADDRESS_MODE_BORDER,
                    CUDA_ERROR_NOT_SUPPORTED, "shader texture border addressing is unavailable");
            require(!view, CUDA_ERROR_NOT_SUPPORTED, "shader texture resource views are unavailable");
        }
        entry->native = std::make_shared<image_object>();
        entry->native->handle = reserve_device_address(32);
        std::array<uint32_t, 8> descriptor{uint32_t(array->image_address), uint32_t(array->image_address >> 32), uint32_t(d.Width), uint32_t(d.Height), surface ? 2u : 1u};
        entry->descriptor = context_configuration(owner, [&] {
            auto storage = owner->native->allocate(sizeof(descriptor));
            owner->native->write(*storage, 0, std::as_bytes(std::span(descriptor))); return storage;
        });
        global().device_buffers.push_back({owner->device, entry->descriptor});
    }
    auto handle = entry->native->handle;
    require(handle && !image_objects(surface).contains(handle), CUDA_ERROR_UNKNOWN, "native image object handle collision");
    image_objects(surface).emplace(handle, std::move(entry)); return handle;
}
void destroy_image_object(uint64_t handle, bool surface) {
    auto entry = image_object_lookup(handle, surface); capture_unsafe(entry->owner);
    entry->retiring = true; drain_context_work(entry->owner);
    image_objects(surface).erase(handle);
}
