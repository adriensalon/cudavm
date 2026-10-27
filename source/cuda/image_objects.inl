// Object descriptors use private registry addresses, never public allocations.
// The shader snapshot binds descriptor and backing storage once even when several
// objects alias one allocation. CUDA objects carry the native driver's opaque value.
auto& image_objects(bool surface) { return surface ? global().surfaces : global().textures; }
uint32_t texture_integer_border(float color, unsigned bits, bool signed_format, bool normalized) {
    unsigned mask = UINT32_MAX >> (32 - bits);
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
    require(resource.resType != CU_RESOURCE_TYPE_MIPMAPPED_ARRAY, CUDA_ERROR_NOT_SUPPORTED, "mipmapped texture resources are unavailable");
    auto entry = std::make_shared<image_object_entry>(); entry->owner = owner; entry->resource = resource;
    CUDA_ARRAY3D_DESCRIPTOR d{}; size_t pitch = 0, offset = 0; CUdeviceptr image_address = 0;
    bool linear = resource.resType == CU_RESOURCE_TYPE_LINEAR;
    if (resource.resType == CU_RESOURCE_TYPE_ARRAY) {
        entry->array = array_lookup(resource.res.array.hArray, owner);
        d = entry->array->descriptor; entry->backing = entry->array->native;
        entry->backing_address = image_address = entry->array->image_address;
        pitch = d.Width * array_element_bytes(d);
        if (surface) require(d.Flags & CUDA_ARRAY3D_SURFACE_LDST, CUDA_ERROR_INVALID_VALUE, "array does not enable surface access");
    } else {
        require(!view, CUDA_ERROR_INVALID_VALUE, "linear and pitched resources cannot have a resource view");
        d.Format = linear ? resource.res.linear.format : resource.res.pitch2D.format;
        d.NumChannels = linear ? resource.res.linear.numChannels : resource.res.pitch2D.numChannels;
        auto element = array_element_bytes(d);
        image_address = linear ? resource.res.linear.devPtr : resource.res.pitch2D.devPtr;
        d.Width = linear ? resource.res.linear.sizeInBytes / element : resource.res.pitch2D.width;
        d.Height = linear ? 1 : resource.res.pitch2D.height;
        require((linear || (d.Width && d.Height)) && d.Width <= SIZE_MAX / element, CUDA_ERROR_INVALID_VALUE, "invalid texture resource extent");
        pitch = linear ? resource.res.linear.sizeInBytes : resource.res.pitch2D.pitchInBytes;
        auto row = d.Width * element;
        require(linear || (pitch >= row && d.Height - 1 <= (SIZE_MAX - row) / pitch),
                CUDA_ERROR_INVALID_VALUE, "invalid texture resource pitch or size");
        auto storage = resolve(image_address, linear ? pitch : (d.Height - 1) * pitch + row, owner);
        offset = storage.offset; entry->backing = storage.value->native; entry->backing_address = image_address - offset;
        auto alignment = global().provider->attribute(owner->device, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT);
        require(alignment > 0 && image_address % alignment == 0, CUDA_ERROR_INVALID_VALUE, "texture resource pointer is misaligned");
        if (!linear) {
            auto pitch_alignment = global().provider->attribute(owner->device, CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT);
            require(pitch_alignment > 0 && pitch % pitch_alignment == 0, CUDA_ERROR_INVALID_VALUE, "texture resource pitch is misaligned");
        }
        auto limit = [&](CUdevice_attribute attribute, size_t value) {
            auto maximum = global().provider->attribute(owner->device, attribute);
            require(maximum > 0 && value <= size_t(maximum), CUDA_ERROR_INVALID_VALUE, "texture resource exceeds device limit");
        };
        if (linear) {
            limit(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH, d.Width);
            require(owner->native->kind() == cudavmBackendCuda || d.Width, CUDA_ERROR_NOT_SUPPORTED, "shader linear textures require at least one complete texel");
        }
        else {
            limit(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH, d.Width);
            limit(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT, d.Height);
            limit(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH, pitch);
        }
    }
    if (texture) {
        for (auto mode : texture->addressMode) require(mode >= CU_TR_ADDRESS_MODE_WRAP && mode <= CU_TR_ADDRESS_MODE_BORDER, CUDA_ERROR_INVALID_VALUE, "invalid texture address mode");
        for (auto mode : {texture->filterMode, texture->mipmapFilterMode}) require(mode == CU_TR_FILTER_MODE_POINT || mode == CU_TR_FILTER_MODE_LINEAR, CUDA_ERROR_INVALID_VALUE, "invalid texture filter mode");
        constexpr unsigned flags = CU_TRSF_READ_AS_INTEGER | CU_TRSF_NORMALIZED_COORDINATES | CU_TRSF_SRGB |
            CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION | CU_TRSF_SEAMLESS_CUBEMAP;
        require(!(texture->flags & ~flags), CUDA_ERROR_INVALID_VALUE, "invalid texture flags");
    }
    capture_unsafe(owner);
    if (texture) entry->texture = *texture;
    if (view) entry->view = *view;
    if (owner->native->kind() == cudavmBackendCuda) {
        entry->native = context_configuration(owner, [&] { return surface ? owner->native->create_surface(*entry->backing)
            : owner->native->create_texture(*entry->backing, resource, offset, *texture, view); });
    } else {
        require(d.Height && !d.Depth && !(d.Flags & (CUDA_ARRAY3D_LAYERED | CUDA_ARRAY3D_CUBEMAP)),
                CUDA_ERROR_NOT_SUPPORTED, "shader image objects require a 2D array");
        require(d.Width <= UINT32_MAX && d.Height <= UINT32_MAX && (linear || pitch <= UINT32_MAX),
            CUDA_ERROR_NOT_SUPPORTED, "image dimensions or row pitch exceed shader descriptor capacity");
        if (!surface) {
            require((d.Format == CU_AD_FORMAT_FLOAT || d.Format == CU_AD_FORMAT_SIGNED_INT8 || d.Format == CU_AD_FORMAT_UNSIGNED_INT8 ||
                d.Format == CU_AD_FORMAT_SIGNED_INT16 || d.Format == CU_AD_FORMAT_UNSIGNED_INT16 ||
                d.Format == CU_AD_FORMAT_SIGNED_INT32 || d.Format == CU_AD_FORMAT_UNSIGNED_INT32) &&
                !(texture->flags & ~(CU_TRSF_READ_AS_INTEGER | CU_TRSF_NORMALIZED_COORDINATES)),
                CUDA_ERROR_NOT_SUPPORTED, "shader textures require float or 8/16/32-bit integer elements");
            require(!view, CUDA_ERROR_NOT_SUPPORTED, "shader texture resource views are unavailable");
        }
        entry->native = std::make_shared<image_object>();
        std::array<uint32_t, 17> descriptor{uint32_t(image_address), uint32_t(image_address >> 32), uint32_t(d.Width), uint32_t(d.Height), surface ? 2u : linear ? 3u : 1u};
        descriptor[16] = uint32_t(pitch);
        descriptor[12] = d.NumChannels; descriptor[13] = uint32_t(array_element_bytes(d));
        descriptor[14] = descriptor[13] / d.NumChannels;
        descriptor[15] = d.Format == CU_AD_FORMAT_FLOAT ? 0u :
            (d.Format == CU_AD_FORMAT_SIGNED_INT8 || d.Format == CU_AD_FORMAT_SIGNED_INT16 || d.Format == CU_AD_FORMAT_SIGNED_INT32) ? 2u : 1u;
        if (texture) {
            bool normalized_integer = descriptor[15] && descriptor[14] < 4 && !(texture->flags & CU_TRSF_READ_AS_INTEGER);
            require(linear || texture->filterMode != CU_TR_FILTER_MODE_LINEAR || !descriptor[15] || normalized_integer,
                    CUDA_ERROR_INVALID_VALUE, "linear filtering requires floating-point reads");
            descriptor[5] = unsigned(!linear && bool(texture->flags & CU_TRSF_NORMALIZED_COORDINATES)) |
                (!linear && texture->filterMode == CU_TR_FILTER_MODE_LINEAR ? 2u : 0u) |
                (normalized_integer ? 4u : 0u);
            for (unsigned i = 0; i < 2; ++i) {
                auto mode = texture->addressMode[i];
                // CUDA treats unnormalized wrap and mirror as clamp. Retain
                // the caller's original descriptor for public queries.
                if (!(descriptor[5] & 1u) && (mode == CU_TR_ADDRESS_MODE_WRAP || mode == CU_TR_ADDRESS_MODE_MIRROR)) mode = CU_TR_ADDRESS_MODE_CLAMP;
                descriptor[6 + i] = uint32_t(mode);
            }
            if (!linear) std::memcpy(descriptor.data() + 8, texture->borderColor, sizeof(texture->borderColor));
            if (descriptor[15]) for (unsigned component = 0; component < 4; ++component)
                descriptor[8 + component] = !linear && component < d.NumChannels ? texture_integer_border(texture->borderColor[component],
                    8 * descriptor[14], descriptor[15] == 2, descriptor[5] & 4) : 0;
        }
        require(global().next_image_address <= uint64_t(UINT32_MAX) - 255, CUDA_ERROR_OUT_OF_MEMORY, "image handle space exhausted");
        entry->native->handle = packet_image_base | global().next_image_address;
        global().next_image_address += 256;
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
    entry->retiring = true; invalidate_native_graphs(); drain_context_work(entry->owner);
    image_objects(surface).erase(handle);
}
