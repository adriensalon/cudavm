graph_parameter_storage& graph_parameter_view(graph_node& node, bool retain = false) {
    if (!node.parameter_view) {
        auto view = std::make_shared<graph_parameter_storage>();
        view->arguments = node.arguments; view->bytes = view->arguments.size();
        for (auto p : node.kernel_layout) view->parameters.push_back(view->arguments.data() + p.offset);
        view->extra = {CU_LAUNCH_PARAM_BUFFER_POINTER, view->arguments.data(), CU_LAUNCH_PARAM_BUFFER_SIZE, &view->bytes, CU_LAUNCH_PARAM_END};
        view->semaphores = node.semaphores; view->signals = node.signals; view->waits = node.waits;
        view->memops = node.memops;
        node.parameter_view = std::move(view);
    }
    if (retain && std::find(node.parameter_history.begin(), node.parameter_history.end(), node.parameter_view) == node.parameter_history.end())
        node.parameter_history.push_back(node.parameter_view);
    return *node.parameter_view;
}
CUDA_KERNEL_NODE_PARAMS graph_kernel_parameters(graph_node& node, bool retain = false) {
    require(node.type == CU_GRAPH_NODE_TYPE_KERNEL, CUDA_ERROR_INVALID_VALUE, "not a kernel graph node");
    auto& view = graph_parameter_view(node, retain);
    CUDA_KERNEL_NODE_PARAMS result{}; result.func = node.function;
    if (node.library_kernel) { result.func = nullptr; result.kern = node.library_kernel; }
    result.gridDimX = node.shape.grid[0]; result.gridDimY = node.shape.grid[1]; result.gridDimZ = node.shape.grid[2];
    result.blockDimX = node.shape.block[0]; result.blockDimY = node.shape.block[1]; result.blockDimZ = node.shape.block[2];
    result.sharedMemBytes = node.shape.shared_bytes; result.ctx = node.parameter_context;
    if (node.kernel_extra) result.extra = view.extra.data();
    else result.kernelParams = view.parameters.empty() ? nullptr : view.parameters.data();
    return result;
}
CUgraphNodeParams graph_node_parameters(graph_node& node) {
    CUgraphNodeParams result{}; result.type = node.type;
    switch (node.type) {
    case CU_GRAPH_NODE_TYPE_EMPTY: break;
    case CU_GRAPH_NODE_TYPE_BATCH_MEM_OP: {
        auto& view = graph_parameter_view(node, true);
        result.memOp = {node.parameter_context, static_cast<unsigned>(view.memops.size()), view.memops.data(), 0}; break;
    }
    case CU_GRAPH_NODE_TYPE_MEM_ALLOC: {
        auto params = graph_memory_parameters(node); std::memcpy(&result.alloc, &params, sizeof(params)); break;
    }
    case CU_GRAPH_NODE_TYPE_MEM_FREE: result.free.dptr = node.memory->address; break;
    case CU_GRAPH_NODE_TYPE_KERNEL: {
        auto params = graph_kernel_parameters(node, true);
        static_assert(sizeof(params) == sizeof(result.kernel)); std::memcpy(&result.kernel, &params, sizeof(params)); break;
    }
    case CU_GRAPH_NODE_TYPE_MEMCPY:
        require(!node.peer_copy, CUDA_ERROR_NOT_SUPPORTED, "peer capture descriptions require distinct endpoint contexts");
        result.memcpy.copyParams = node.copy; result.memcpy.copyCtx = node.parameter_context; break;
    case CU_GRAPH_NODE_TYPE_MEMSET:
        std::memcpy(&result.memset, &node.fill, sizeof(node.fill)); result.memset.ctx = node.parameter_context; break;
    case CU_GRAPH_NODE_TYPE_HOST:
        result.host.fn = node.host.fn; result.host.userData = node.host.userData; result.host.ctx = node.host_context; break;
    case CU_GRAPH_NODE_TYPE_GRAPH:
        result.graph.graph = node.child; result.graph.ownership = CU_GRAPH_CHILD_GRAPH_OWNERSHIP_INVALID; break;
    case CU_GRAPH_NODE_TYPE_EVENT_RECORD: result.eventRecord.event = node.event; break;
    case CU_GRAPH_NODE_TYPE_WAIT_EVENT: result.eventWait.event = node.event; break;
    case CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL: {
        auto& view = graph_parameter_view(node, true);
        result.extSemSignal.extSemArray = view.semaphores.data(); result.extSemSignal.paramsArray = view.signals.data();
        result.extSemSignal.numExtSems = static_cast<unsigned>(view.semaphores.size()); result.extSemSignal.ctx = node.parameter_context; break;
    }
    case CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT: {
        auto& view = graph_parameter_view(node, true);
        result.extSemWait.extSemArray = view.semaphores.data(); result.extSemWait.paramsArray = view.waits.data();
        result.extSemWait.numExtSems = static_cast<unsigned>(view.semaphores.size()); result.extSemWait.ctx = node.parameter_context; break;
    }
    default: throw failure(CUDA_ERROR_NOT_SUPPORTED, "graph parameter query for this node type is not implemented");
    }
    return result;
}
