namespace {
struct runtime_library_options {
    std::vector<CUjit_option> jit;
    std::vector<CUlibraryOption> library;
    runtime_library_options(cudaJitOption* options, void** values, unsigned count, cudaLibraryOption* library_options, void** library_values, unsigned library_count) {
        require((!count || (options && values)) && (!library_count || (library_options && library_values)), CUDA_ERROR_INVALID_VALUE, "missing Runtime library options");
        for (unsigned i = 0; i < count; ++i) jit.push_back(static_cast<CUjit_option>(options[i]));
        for (unsigned i = 0; i < library_count; ++i) library.push_back(static_cast<CUlibraryOption>(library_options[i]));
    }
};
}
RUNTIME_API(cudaLibraryLoadData, (cudaLibrary_t* output, const void* image, cudaJitOption* options, void** values, unsigned count, cudaLibraryOption* library_options, void** library_values, unsigned library_count),
    require(output && image, CUDA_ERROR_INVALID_VALUE, "null Runtime library image or output"); check(cuInit(0)); runtime_library_options converted(options, values, count, library_options, library_values, library_count);
    CUlibrary library = nullptr; check(runtime_driver_wait([&] { return cuLibraryLoadData(&library, image, converted.jit.data(), values, count, converted.library.data(), library_values, library_count); })); *output = reinterpret_cast<cudaLibrary_t>(library))
RUNTIME_API(cudaGetKernel, (cudaKernel_t* output, const void* entry),
    require(output, CUDA_ERROR_INVALID_VALUE, "null registered kernel output"); *output = reinterpret_cast<cudaKernel_t>(runtime_registered_kernel(entry)))
RUNTIME_API(cudaGetFuncBySymbol, (cudaFunction_t* output, const void* entry),
    require(output, CUDA_ERROR_INVALID_VALUE, "null registered function output");
    auto found = global().bindings.find(entry);
    symbol_require(entry && found != global().bindings.end(), cudaErrorInvalidDeviceFunction, "function entry is not registered with this Runtime");
    auto value = found->second; *output = reinterpret_cast<cudaFunction_t>(runtime_function(value, runtime_current())))
RUNTIME_API(cudaLibraryLoadFromFile, (cudaLibrary_t* output, const char* path, cudaJitOption* options, void** values, unsigned count, cudaLibraryOption* library_options, void** library_values, unsigned library_count),
    require(output && path && *path, CUDA_ERROR_INVALID_VALUE, "null Runtime library file or output"); check(cuInit(0)); runtime_library_options converted(options, values, count, library_options, library_values, library_count);
    CUlibrary library = nullptr; check(runtime_driver_wait([&] { return cuLibraryLoadFromFile(&library, path, converted.jit.data(), values, count, converted.library.data(), library_values, library_count); })); *output = reinterpret_cast<cudaLibrary_t>(library))
RUNTIME_API(cudaLibraryUnload, (cudaLibrary_t library), check(cuInit(0)); check(runtime_driver_wait([&] { return cuLibraryUnload(reinterpret_cast<CUlibrary>(library)); })))
RUNTIME_API(cudaLibraryGetKernel, (cudaKernel_t* output, cudaLibrary_t library, const char* name),
    require(output, CUDA_ERROR_INVALID_VALUE, "null Runtime kernel output"); check(cuInit(0)); CUkernel kernel = nullptr;
    check(cuLibraryGetKernel(&kernel, reinterpret_cast<CUlibrary>(library), name)); *output = reinterpret_cast<cudaKernel_t>(kernel))
RUNTIME_API(cudaLibraryGetKernelCount, (unsigned* output, cudaLibrary_t library), require(output, CUDA_ERROR_INVALID_VALUE, "null Runtime library count output"); check(cuInit(0)); check(cuLibraryGetKernelCount(output, reinterpret_cast<CUlibrary>(library))))
RUNTIME_API(cudaLibraryEnumerateKernels, (cudaKernel_t* output, unsigned capacity, cudaLibrary_t library),
    check(cuInit(0)); unsigned count = 0; check(cuLibraryGetKernelCount(&count, reinterpret_cast<CUlibrary>(library)));
    require(output || !capacity, CUDA_ERROR_INVALID_VALUE, "null Runtime kernel array"); count = std::min(capacity, count); std::vector<CUkernel> kernels(count);
    check(cuLibraryEnumerateKernels(kernels.data(), count, reinterpret_cast<CUlibrary>(library))); for (size_t i = 0; i < kernels.size(); ++i) output[i] = reinterpret_cast<cudaKernel_t>(kernels[i]))
RUNTIME_API(cudaLibraryGetGlobal, (void** output, size_t* bytes, cudaLibrary_t library, const char* name),
    require(output || bytes, CUDA_ERROR_INVALID_VALUE, "missing Runtime global outputs"); runtime_current(); CUdeviceptr pointer = 0; size_t size = 0;
    check(runtime_driver_wait([&] { return cuLibraryGetGlobal(&pointer, &size, reinterpret_cast<CUlibrary>(library), name); })); if (output) *output = reinterpret_cast<void*>(uintptr_t(pointer)); if (bytes) *bytes = size)
RUNTIME_API(cudaLibraryGetManaged, (void** output, size_t* bytes, cudaLibrary_t library, const char* name),
    require(output || bytes, CUDA_ERROR_INVALID_VALUE, "missing Runtime managed outputs"); check(cuInit(0)); CUdeviceptr pointer = 0;
    check(runtime_driver_wait([&] { return cuLibraryGetManaged(&pointer, bytes, reinterpret_cast<CUlibrary>(library), name); })); if (output) *output = reinterpret_cast<void*>(uintptr_t(pointer)))
RUNTIME_API(cudaLibraryGetUnifiedFunction, (void** output, cudaLibrary_t library, const char* name),
    check(cuInit(0)); check(cuLibraryGetUnifiedFunction(output, reinterpret_cast<CUlibrary>(library), name)))
RUNTIME_API(cudaKernelSetAttributeForDevice, (cudaKernel_t kernel, cudaFuncAttribute attribute, int value, int device),
    check(cuInit(0)); symbol_require(cudavmDriverIsLibraryKernel(kernel), cudaErrorInvalidDeviceFunction, "invalid Runtime library kernel");
    auto which = attribute == cudaFuncAttributeSharedMemoryMode ? CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE : static_cast<CUfunction_attribute>(attribute);
    check(runtime_driver_wait([&] { return cuKernelSetAttribute(which, value, reinterpret_cast<CUkernel>(kernel), device); })))
