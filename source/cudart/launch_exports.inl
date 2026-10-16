extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaLaunchKernel_ptsz(const void* token, dim3 grid, dim3 block, void** parameters, size_t shared, cudaStream_t stream) {
    return runtime_launch(token, grid, block, parameters, shared, stream ? stream : cudaStreamPerThread, false, nullptr);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* token, void** parameters) {
    if (!config) return runtime_call([&] { require(false, CUDA_ERROR_INVALID_VALUE, "null Runtime launch descriptor"); });
    return runtime_launch(token, config->gridDim, config->blockDim, parameters, config->dynamicSmemBytes, config->stream, false, config);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaLaunchCooperativeKernel(const void* token, dim3 grid, dim3 block, void** parameters, size_t shared, cudaStream_t stream) {
    return runtime_launch(token, grid, block, parameters, shared, stream, true, nullptr);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaLaunchKernelExC_ptsz(const cudaLaunchConfig_t* config, const void* token, void** parameters) {
    if (!config) return cudaLaunchKernelExC(config, token, parameters);
    auto copy = *config; if (!copy.stream) copy.stream = cudaStreamPerThread;
    return cudaLaunchKernelExC(&copy, token, parameters);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaLaunchCooperativeKernel_ptsz(const void* token, dim3 grid, dim3 block, void** parameters, size_t shared, cudaStream_t stream) {
    return cudaLaunchCooperativeKernel(token, grid, block, parameters, shared, stream ? stream : cudaStreamPerThread);
}
RUNTIME_API(cudaOccupancyMaxPotentialClusterSize, (int* output, const void* token, const cudaLaunchConfig_t* config),
    require(output, CUDA_ERROR_INVALID_VALUE, "null potential cluster size output");
    std::vector<CUlaunchAttribute> attributes; auto converted = driver_launch_config(config, attributes);
    check(cuOccupancyMaxPotentialClusterSize(output, runtime_kernel(token), &converted)))
RUNTIME_API(cudaOccupancyMaxActiveClusters, (int* output, const void* token, const cudaLaunchConfig_t* config),
    require(output, CUDA_ERROR_INVALID_VALUE, "null active cluster count output");
    std::vector<CUlaunchAttribute> attributes; auto converted = driver_launch_config(config, attributes);
    check(cuOccupancyMaxActiveClusters(output, runtime_kernel(token), &converted)))
