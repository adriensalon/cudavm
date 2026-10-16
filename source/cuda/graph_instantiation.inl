void graph_instantiate_params(CUgraphExec* output, CUgraph graph, CUDA_GRAPH_INSTANTIATE_PARAMS* p, bool per_thread = false) {
    if (p) { p->result_out = CUDA_GRAPH_INSTANTIATE_ERROR; p->hErrNode_out = nullptr; }
    require(output && p, CUDA_ERROR_INVALID_VALUE, "null graph instantiation arguments");
    require(!(p->flags & ~15ull), CUDA_ERROR_INVALID_VALUE, "unknown graph instantiation flags");
    auto handle = graph_instantiate(graph, p->flags & ~static_cast<cuuint64_t>(CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD), p);
    try {
        if (p->flags & CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD)
            graph_launch(handle, per_thread && !p->hUploadStream ? CU_STREAM_PER_THREAD : p->hUploadStream, true);
    } catch (...) { global().graph_execs.erase(handle); throw; }
    *output = handle; p->result_out = CUDA_GRAPH_INSTANTIATE_SUCCESS;
}
