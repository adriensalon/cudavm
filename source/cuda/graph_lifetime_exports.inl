DRIVER_API(cuUserObjectCreate, (CUuserObject* output, void* data, CUhostFn destroy, unsigned count, unsigned flags),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null user object output"); *output = user_create(data, destroy, count, flags))
DRIVER_API(cuUserObjectRetain, (CUuserObject object, unsigned count), driver_ready(); user_retain(object, count))
DRIVER_API(cuUserObjectRelease, (CUuserObject object, unsigned count), driver_ready(); user_release(object, count))
DRIVER_API(cuGraphRetainUserObject, (CUgraph graph, CUuserObject object, unsigned count, unsigned flags),
    driver_ready(); graph_user_retain(graph, object, count, flags))
DRIVER_API(cuGraphReleaseUserObject, (CUgraph graph, CUuserObject object, unsigned count),
    driver_ready(); graph_user_release(graph, object, count))
DRIVER_API(cuGraphInstantiateWithParams, (CUgraphExec* output, CUgraph graph, CUDA_GRAPH_INSTANTIATE_PARAMS* p),
    driver_ready(); graph_instantiate_params(output, graph, p))
DRIVER_API(cuGraphInstantiateWithParams_ptsz, (CUgraphExec* output, CUgraph graph, CUDA_GRAPH_INSTANTIATE_PARAMS* p),
    driver_ready(); graph_instantiate_params(output, graph, p, true))
