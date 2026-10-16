DRIVER_API(cuLaunchKernelEx, (const CUlaunchConfig* config, CUfunction function, void** parameters, void** extra),
    driver_ready(); auto shape = launch_configuration(config);
    launch(function, shape, config->hStream, parameters, extra, launch_context(function, config->hStream)))
DRIVER_API(cuLaunchCooperativeKernel, (CUfunction function, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shared, CUstream stream, void** parameters),
    driver_ready(); launch_shape shape{{gx, gy, gz}, {bx, by, bz}, shared}; shape.options.cooperative = true;
    launch(function, shape, stream, parameters, nullptr, launch_context(function, stream)))
DRIVER_API(cuOccupancyMaxPotentialClusterSize, (int* output, CUfunction function, const CUlaunchConfig* config),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null cluster size output"); *output = cluster_occupancy(function, config, true))
DRIVER_API(cuOccupancyMaxActiveClusters, (int* output, CUfunction function, const CUlaunchConfig* config),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null active cluster output"); *output = cluster_occupancy(function, config, false))
