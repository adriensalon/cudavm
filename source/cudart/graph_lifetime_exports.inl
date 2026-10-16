RUNTIME_API(cudaUserObjectCreate, (cudaUserObject_t* output, void* data, cudaHostFn_t destroy, unsigned count, unsigned flags),
    require(output, CUDA_ERROR_INVALID_VALUE, "null user object output"); check(cuInit(0)); CUuserObject value{};
    check(cuUserObjectCreate(&value, data, destroy, count, flags)); *output = graph_cast<cudaUserObject_t>(value))
RUNTIME_API(cudaUserObjectRetain, (cudaUserObject_t object, unsigned count),
    check(cuInit(0)); check(cuUserObjectRetain(graph_cast<CUuserObject>(object), count)))
RUNTIME_API(cudaUserObjectRelease, (cudaUserObject_t object, unsigned count),
    check(cuInit(0)); check(cuUserObjectRelease(graph_cast<CUuserObject>(object), count)))
RUNTIME_API(cudaGraphRetainUserObject, (cudaGraph_t graph, cudaUserObject_t object, unsigned count, unsigned flags),
    check(cuInit(0)); check(cuGraphRetainUserObject(graph_cast<CUgraph>(graph), graph_cast<CUuserObject>(object), count, flags)))
RUNTIME_API(cudaGraphReleaseUserObject, (cudaGraph_t graph, cudaUserObject_t object, unsigned count),
    check(cuInit(0)); check(cuGraphReleaseUserObject(graph_cast<CUgraph>(graph), graph_cast<CUuserObject>(object), count)))
void runtime_graph_instantiate_params(cudaGraphExec_t* output, cudaGraph_t graph, cudaGraphInstantiateParams* p, bool per_thread) {
    if (p) { p->result_out = cudaGraphInstantiateError; p->errNode_out = nullptr; }
    require(output && p, CUDA_ERROR_INVALID_VALUE, "null graph instantiation arguments"); runtime_current();
    CUDA_GRAPH_INSTANTIATE_PARAMS params{}; params.flags = p->flags;
    params.hUploadStream = graph_cast<CUstream>(per_thread && !p->uploadStream ? cudaStreamPerThread : p->uploadStream);
    CUgraphExec handle{}; auto result = cuGraphInstantiateWithParams(&handle, graph_cast<CUgraph>(graph), &params);
    p->result_out = static_cast<cudaGraphInstantiateResult>(params.result_out); p->errNode_out = graph_cast<cudaGraphNode_t>(params.hErrNode_out);
    check(result); *output = graph_cast<cudaGraphExec_t>(handle);
}
RUNTIME_API(cudaGraphInstantiateWithParams, (cudaGraphExec_t* output, cudaGraph_t graph, cudaGraphInstantiateParams* p),
    runtime_graph_instantiate_params(output, graph, p, false))
RUNTIME_API(cudaGraphInstantiateWithParams_ptsz, (cudaGraphExec_t* output, cudaGraph_t graph, cudaGraphInstantiateParams* p),
    runtime_graph_instantiate_params(output, graph, p, true))
