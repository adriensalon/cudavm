// Object descriptors use private registry addresses, never public allocations.
// The shader snapshot binds descriptor and array storage once even when several
// objects alias one array. CUDA objects carry the native driver's opaque value.
auto& image_objects(bool surface) { return surface ? global().surfaces : global().textures; }
uint32_t texture_integer_border(float color, unsigned bits, bool signed_format, bool normalized) {
    unsigned mask = (1u << bits) - 1;
    if (!normalized) {
        auto value = std::bit_cast<uint32_t>(color) & mask;
        return signed_format && (value & (1u << (bits - 1))) ? value | ~mask : value;
    }
    unsigned maximum = (1u << (bits - unsigned(signed_format))) - 1;
    double value = std::isnan(color) ? 0.0 : std::clamp(double(color), signed_format ? -1.0 : 0.0, 1.0);
    double scaled = value * maximum, lower = std::floor(scaled);
    auto integer = int32_t(lower);
    double fraction = scaled - lower;
    if (fraction > 0.5 || (fraction == 0.5 && (integer & 1))) ++integer;
    return uint32_t(integer);
}
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
        require(d.Height && !d.Depth && !(d.Flags & (CUDA_ARRAY3D_LAYERED | CUDA_ARRAY3D_CUBEMAP)),
                CUDA_ERROR_NOT_SUPPORTED, "shader image objects require a 2D array");
        require(d.Width <= UINT32_MAX && d.Height <= UINT32_MAX && array->native->size < (1u << 30), CUDA_ERROR_NOT_SUPPORTED, "image exceeds shader address transport");
        if (surface) require(d.NumChannels == 1 && (d.Format == CU_AD_FORMAT_SIGNED_INT32 || d.Format == CU_AD_FORMAT_UNSIGNED_INT32 || d.Format == CU_AD_FORMAT_FLOAT),
                CUDA_ERROR_NOT_SUPPORTED, "shader surfaces require single-channel 32-bit elements");
        else {
            require((d.Format == CU_AD_FORMAT_FLOAT || d.Format == CU_AD_FORMAT_SIGNED_INT8 || d.Format == CU_AD_FORMAT_UNSIGNED_INT8 ||
                d.Format == CU_AD_FORMAT_SIGNED_INT16 || d.Format == CU_AD_FORMAT_UNSIGNED_INT16) &&
                !(texture->flags & ~(CU_TRSF_READ_AS_INTEGER | CU_TRSF_NORMALIZED_COORDINATES)),
                CUDA_ERROR_NOT_SUPPORTED, "shader textures require float or 8/16-bit integer elements");
            require(!view, CUDA_ERROR_NOT_SUPPORTED, "shader texture resource views are unavailable");
        }
        entry->native = std::make_shared<image_object>();
        std::array<uint32_t, 16> descriptor{uint32_t(array->image_address), uint32_t(array->image_address >> 32), uint32_t(d.Width), uint32_t(d.Height), surface ? 2u : 1u};
        descriptor[12] = d.NumChannels; descriptor[13] = uint32_t(array_element_bytes(d));
        descriptor[14] = descriptor[13] / d.NumChannels;
        descriptor[15] = d.Format == CU_AD_FORMAT_FLOAT ? 0u :
            (d.Format == CU_AD_FORMAT_SIGNED_INT8 || d.Format == CU_AD_FORMAT_SIGNED_INT16 || d.Format == CU_AD_FORMAT_SIGNED_INT32) ? 2u : 1u;
        if (texture) {
            descriptor[5] = unsigned(bool(texture->flags & CU_TRSF_NORMALIZED_COORDINATES)) |
                (texture->filterMode == CU_TR_FILTER_MODE_LINEAR ? 2u : 0u) |
                (descriptor[15] && !(texture->flags & CU_TRSF_READ_AS_INTEGER) ? 4u : 0u);
            for (unsigned i = 0; i < 2; ++i) {
                auto mode = texture->addressMode[i];
                // CUDA treats unnormalized wrap and mirror as clamp. Retain
                // the caller's original descriptor for public queries.
                if (!(descriptor[5] & 1u) && (mode == CU_TR_ADDRESS_MODE_WRAP || mode == CU_TR_ADDRESS_MODE_MIRROR)) mode = CU_TR_ADDRESS_MODE_CLAMP;
                descriptor[6 + i] = uint32_t(mode);
            }
            std::memcpy(descriptor.data() + 8, texture->borderColor, sizeof(texture->borderColor));
            if (descriptor[15]) for (unsigned component = 0; component < 4; ++component)
                descriptor[8 + component] = component < d.NumChannels ? texture_integer_border(texture->borderColor[component],
                    8 * descriptor[14], descriptor[15] == 2, descriptor[5] & 4) : 0;
        }
        entry->native->handle = reserve_device_address(sizeof(descriptor));
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
