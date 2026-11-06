RUNTIME_API(cudaIpcGetMemHandle, (cudaIpcMemHandle_t* output, void* pointer),
    require(output, CUDA_ERROR_INVALID_VALUE, "null IPC handle output"); runtime_current(); CUipcMemHandle handle{};
    check(cuIpcGetMemHandle(&handle, reinterpret_cast<uintptr_t>(pointer))); static_assert(sizeof(handle) == sizeof(*output)); std::memcpy(output, &handle, sizeof(handle)))
RUNTIME_API(cudaIpcOpenMemHandle, (void** output, cudaIpcMemHandle_t handle, unsigned flags),
    require(output, CUDA_ERROR_INVALID_VALUE, "null IPC pointer output"); runtime_current(); CUipcMemHandle record{}; std::memcpy(&record, &handle, sizeof(record)); CUdeviceptr pointer{};
    check(cuIpcOpenMemHandle(&pointer, record, flags)); *output = reinterpret_cast<void*>(uintptr_t(pointer)))
RUNTIME_API(cudaIpcCloseMemHandle, (void* pointer),
    runtime_current(); check(runtime_driver_wait([&] { return cuIpcCloseMemHandle(reinterpret_cast<uintptr_t>(pointer)); })))
RUNTIME_API(cudaIpcGetEventHandle, (cudaIpcEventHandle_t* output, cudaEvent_t event),
    require(output, CUDA_ERROR_INVALID_VALUE, "null IPC event handle output"); runtime_current(); CUipcEventHandle record{};
    check(cuIpcGetEventHandle(&record, reinterpret_cast<CUevent>(event))); static_assert(sizeof(record) == sizeof(*output)); std::memcpy(output, &record, sizeof(record)))
RUNTIME_API(cudaIpcOpenEventHandle, (cudaEvent_t* output, cudaIpcEventHandle_t handle),
    require(output, CUDA_ERROR_INVALID_VALUE, "null IPC event import output"); runtime_current(); CUipcEventHandle record{}; std::memcpy(&record, &handle, sizeof(record));
    CUevent event{}; auto status = cuIpcOpenEventHandle(&event, record);
    symbol_require(status != CUDA_ERROR_INVALID_HANDLE, cudaErrorInvalidValue, "invalid IPC event handle");
    check(status); *output = reinterpret_cast<cudaEvent_t>(event))
