DRIVER_API(cuIpcGetMemHandle, (CUipcMemHandle* output, CUdeviceptr pointer),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null IPC handle output"); *output = ipc_export(pointer))
// The pinned public alias names the reference-counted v2 ABI.
DRIVER_API(cuIpcOpenMemHandle, (CUdeviceptr* output, CUipcMemHandle handle, unsigned flags),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null IPC pointer output"); *output = ipc_import(handle, flags, false))
#undef cuIpcOpenMemHandle
DRIVER_API(cuIpcOpenMemHandle, (CUdeviceptr* output, CUipcMemHandle handle, unsigned flags),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null IPC pointer output"); *output = ipc_import(handle, flags, true))
DRIVER_API(cuIpcCloseMemHandle, (CUdeviceptr pointer), driver_ready(); ipc_close(pointer))
