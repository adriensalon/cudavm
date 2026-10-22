DRIVER_API(cuLaunchKernelEx, (const CUlaunchConfig* config, CUfunction function, void** parameters, void** extra),
    driver_ready(); auto shape = launch_configuration(config);
    auto owner = launch_context(function, config->hStream); auto target = stream_lookup(config->hStream, owner);
    CUlaunchAttributeValue* device_attribute = nullptr;
    for (unsigned i = 0; i < config->numAttrs; ++i) if (config->attrs[i].id == CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE) device_attribute = &config->attrs[i].value;
    require(!device_attribute || target->capture, CUDA_ERROR_INVALID_VALUE, "device-updatable launch attribute requires capture");
    if (device_attribute && device_attribute->deviceUpdatableKernelNode.deviceUpdatable) graph_native_materialize(target->capture->graph, owner);
    launch(function, shape, config->hStream, parameters, extra, owner);
    if (device_attribute) {
        auto node = target->capture_dependencies.back();
        try { graph_launch_attribute(node, CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE, *device_attribute);
            *device_attribute = graph_launch_attribute(node, CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE); }
        catch (...) { target->capture->invalidated = true; throw; }
    })
DRIVER_API(cuLaunchCooperativeKernel, (CUfunction function, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shared, CUstream stream, void** parameters),
    driver_ready(); launch_shape shape{{gx, gy, gz}, {bx, by, bz}, shared}; shape.options.cooperative = true;
    launch(function, shape, stream, parameters, nullptr, launch_context(function, stream)))
DRIVER_API(cuOccupancyMaxPotentialClusterSize, (int* output, CUfunction function, const CUlaunchConfig* config),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null cluster size output"); *output = cluster_occupancy(function, config, true))
DRIVER_API(cuOccupancyMaxActiveClusters, (int* output, CUfunction function, const CUlaunchConfig* config),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null active cluster output"); *output = cluster_occupancy(function, config, false))
