// NVIDIA's compiler-emitted C ABI. Do not throw across constructor/destructor
// stubs. The normal Runtime boundary records failures in the calling thread.
extern "C" CUDAVM_CUDART_API void** CUDARTAPI __cudaRegisterFatBinary(void* wrapper) {
    void** result = nullptr; runtime_call([&] { result = register_nvidia_image(wrapper); }); return result;
}
extern "C" CUDAVM_CUDART_API void CUDARTAPI __cudaRegisterFatBinaryEnd(void** handle) {
    runtime_call([&] { nvidia_registration_identity(handle); });
}
extern "C" CUDAVM_CUDART_API void CUDARTAPI __cudaUnregisterFatBinary(void** handle) {
    runtime_call([&] { unregister_nvidia_image(handle); });
}
extern "C" CUDAVM_CUDART_API void CUDARTAPI __cudaRegisterFunction(void** handle, const char* host, char* device, const char* name,
        int, uint3*, uint3*, dim3*, dim3*, int*) {
    runtime_call([&] { register_nvidia_binding(handle, host, device && *device ? device : name, true); });
}
extern "C" CUDAVM_CUDART_API void CUDARTAPI __cudaRegisterVar(void** handle, char* host, char*, const char* name,
        int, size_t, int, int) {
    runtime_call([&] { register_nvidia_binding(handle, host, name, false); });
}
extern "C" CUDAVM_CUDART_API void CUDARTAPI __cudaRegisterManagedVar(void** handle, void** host, char*, const char* name,
        int, size_t bytes, int, int) {
    runtime_call([&] { register_nvidia_managed(handle, host, name, bytes); });
}
extern "C" CUDAVM_CUDART_API char CUDARTAPI __cudaInitModule(void** handle) {
    return runtime_call([&] { initialize_nvidia_managed(handle); }) == cudaSuccess ? 1 : 0;
}
extern "C" CUDAVM_CUDART_API unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 grid, dim3 block, size_t shared, void* stream) {
    return unsigned(runtime_call([&] { nvidia_launch_frames.push_back({grid, block, shared, reinterpret_cast<cudaStream_t>(stream)}); }));
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI __cudaGetKernel(cudaKernel_t* output, const void* token) {
    return runtime_call([&] { require(output, CUDA_ERROR_INVALID_VALUE, "null compiler kernel output"); *output = compiler_kernel(token); });
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3* grid, dim3* block, size_t* shared, void* stream) {
    return runtime_call([&] {
        require(grid && block && shared && stream, CUDA_ERROR_INVALID_VALUE, "null NVIDIA call configuration output");
        symbol_require(!nvidia_launch_frames.empty(), cudaErrorMissingConfiguration, "NVIDIA call configuration stack is empty");
        auto frame = nvidia_launch_frames.back(); nvidia_launch_frames.pop_back();
        *grid = frame.grid; *block = frame.block; *shared = frame.shared; *static_cast<cudaStream_t*>(stream) = frame.stream;
    });
}
