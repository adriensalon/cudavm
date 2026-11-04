#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stddef.h>
#include <string.h>
_Static_assert(sizeof(CUdeviceptr) == 8, "device pointer ABI");
int sdk_c_contract(void) {
    struct { unsigned before; const char* value; unsigned after; } result = {0x12345678, NULL, 0x87654321};
    if (cuGetErrorName(CUDA_ERROR_INVALID_VALUE, &result.value) != CUDA_SUCCESS) return 1;
    if (!result.value || strcmp(result.value, "CUDA_ERROR_INVALID_VALUE")) return 2;
    if (result.before != 0x12345678 || result.after != 0x87654321) return 3;
    return 0;
}
