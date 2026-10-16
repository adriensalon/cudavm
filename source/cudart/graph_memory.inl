struct runtime_graph_allocation {
    CUDA_MEM_ALLOC_NODE_PARAMS params{};
    std::vector<CUmemAccessDesc> access;
    template<class Params> explicit runtime_graph_allocation(const Params& p) {
        static_assert(sizeof(p.poolProps) == sizeof(params.poolProps));
        static_assert(sizeof(cudaMemAccessDesc) == sizeof(CUmemAccessDesc));
        require(std::all_of(std::begin(p.poolProps.reserved), std::end(p.poolProps.reserved), [](auto v) { return v == 0; }),
                CUDA_ERROR_INVALID_VALUE, "nonzero graph allocation reserved fields");
        require(!p.accessDescCount || p.accessDescs, CUDA_ERROR_INVALID_VALUE, "null graph memory access descriptors");
        int devices{}; check(cuDeviceGetCount(&devices));
        require(p.accessDescCount <= static_cast<size_t>(devices), CUDA_ERROR_INVALID_VALUE, "too many graph memory access descriptors");
        std::memcpy(&params.poolProps, &p.poolProps, sizeof(params.poolProps));
        access.resize(p.accessDescCount);
        if (!access.empty()) std::memcpy(access.data(), p.accessDescs, access.size() * sizeof(CUmemAccessDesc));
        params.accessDescs = access.empty() ? nullptr : access.data(); params.accessDescCount = access.size(); params.bytesize = p.bytesize;
    }
};
template<class Params> Params runtime_graph_allocation_parameters(const CUDA_MEM_ALLOC_NODE_PARAMS& p) {
    Params result{}; static_assert(sizeof(result.poolProps) == sizeof(p.poolProps));
    std::memcpy(&result.poolProps, &p.poolProps, sizeof(p.poolProps));
    result.accessDescs = reinterpret_cast<const cudaMemAccessDesc*>(p.accessDescs); result.accessDescCount = p.accessDescCount;
    result.bytesize = p.bytesize; result.dptr = reinterpret_cast<void*>(uintptr_t(p.dptr)); return result;
}
