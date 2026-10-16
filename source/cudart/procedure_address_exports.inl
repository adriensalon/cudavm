namespace cudavm {
namespace {
void runtime_entry_point(const char* symbol, void** output, unsigned version,
                         unsigned long long flags, cudaDriverEntryPointQueryResult* status) {
    require(version <= CUDA_VERSION, CUDA_ERROR_INVALID_VALUE, "requested Driver version exceeds the CUDA baseline");
    CUdriverProcAddressQueryResult result;
    check(cuGetProcAddress(symbol, output, static_cast<int>(version), flags, &result));
    if (status) {
        switch (result) {
        case CU_GET_PROC_ADDRESS_SUCCESS: *status = cudaDriverEntryPointSuccess; break;
        case CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND: *status = cudaDriverEntryPointSymbolNotFound; break;
        case CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT: *status = cudaDriverEntryPointVersionNotSufficent; break;
        }
    }
}
unsigned long long entry_point_per_thread_flags(unsigned long long flags) {
    return flags & (cudaEnableLegacyStream | cudaEnablePerThreadDefaultStream) ? flags : flags | cudaEnablePerThreadDefaultStream;
}
}
}
RUNTIME_API(cudaGetDriverEntryPoint, (const char* symbol, void** output, unsigned long long flags, cudaDriverEntryPointQueryResult* status),
    runtime_entry_point(symbol, output, CUDART_VERSION, flags, status))
RUNTIME_API(cudaGetDriverEntryPointByVersion, (const char* symbol, void** output, unsigned version, unsigned long long flags, cudaDriverEntryPointQueryResult* status),
    runtime_entry_point(symbol, output, version, flags, status))
RUNTIME_API(cudaGetDriverEntryPoint_ptsz, (const char* symbol, void** output, unsigned long long flags, cudaDriverEntryPointQueryResult* status),
    runtime_entry_point(symbol, output, CUDART_VERSION, entry_point_per_thread_flags(flags), status))
RUNTIME_API(cudaGetDriverEntryPointByVersion_ptsz, (const char* symbol, void** output, unsigned version, unsigned long long flags, cudaDriverEntryPointQueryResult* status),
    runtime_entry_point(symbol, output, version, entry_point_per_thread_flags(flags), status))
