DRIVER_API(cuTexObjectCreate, (CUtexObject* output, const CUDA_RESOURCE_DESC* resource, const CUDA_TEXTURE_DESC* texture, const CUDA_RESOURCE_VIEW_DESC* view),
    driver_ready(); require(output && resource && texture, CUDA_ERROR_INVALID_VALUE, "null texture descriptor or output");
    *output = create_image_object(*resource, texture, view))
DRIVER_API(cuTexObjectDestroy, (CUtexObject handle), driver_ready(); destroy_image_object(handle, false))
DRIVER_API(cuTexObjectGetResourceDesc, (CUDA_RESOURCE_DESC* output, CUtexObject handle),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null texture resource output"); *output = image_object_lookup(handle, false)->resource)
DRIVER_API(cuTexObjectGetTextureDesc, (CUDA_TEXTURE_DESC* output, CUtexObject handle),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null texture descriptor output"); *output = image_object_lookup(handle, false)->texture)
DRIVER_API(cuTexObjectGetResourceViewDesc, (CUDA_RESOURCE_VIEW_DESC* output, CUtexObject handle),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null texture view output"); auto entry = image_object_lookup(handle, false);
    require(entry->view.has_value(), CUDA_ERROR_INVALID_VALUE, "texture has no resource view"); *output = *entry->view)
DRIVER_API(cuSurfObjectCreate, (CUsurfObject* output, const CUDA_RESOURCE_DESC* resource),
    driver_ready(); require(output && resource, CUDA_ERROR_INVALID_VALUE, "null surface descriptor or output"); *output = create_image_object(*resource, nullptr, nullptr))
DRIVER_API(cuSurfObjectDestroy, (CUsurfObject handle), driver_ready(); destroy_image_object(handle, true))
DRIVER_API(cuSurfObjectGetResourceDesc, (CUDA_RESOURCE_DESC* output, CUsurfObject handle),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null surface resource output"); *output = image_object_lookup(handle, true)->resource)
