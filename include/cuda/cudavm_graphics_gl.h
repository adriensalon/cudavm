#pragma once
#include <cuda/cudavm_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

/* An application GL context, without a dependency on GL headers or its profile.
   The caller supplies desktop GL 1.5 buffer entry points. All registration,
   map and unmap calls must run with this context current. Callbacks use the C
   calling convention; returned GL functions use the platform GL convention.
   Callbacks must not reenter cudavm. user_data and the GL context must outlive
   registrations. Buffer ownership transfers at map/unmap; do not access a
   registered buffer through GL while it is mapped by CUDA. The GL 1.5 bridge
   accepts nonempty buffers whose size fits a signed 32-bit GL size query. */
typedef struct cudavmGraphicsGLDesc {
    unsigned int abi_version; /* 1 */
    size_t struct_size;       /* sizeof(cudavmGraphicsGLDesc) */
    void* user_data;
    void* (*get_proc_address)(void* user_data, const char* name);
    int (*is_current)(void* user_data);
} cudavmGraphicsGLDesc;

/* Attach to the current CUDA Driver context. Passing NULL detaches. Replacing
   or detaching a bridge is forbidden while GL resources remain registered.
   Returns a CUresult value. Then use cu/cudaGraphicsGLRegisterBuffer and the
   standard CUDA graphics map/unmap APIs. This bridge stages bytes across the
   ownership boundaries and works with every enabled compute backend. */
CUDAVM_API int cudavmGraphicsGL(const cudavmGraphicsGLDesc* descriptor);

#ifdef __cplusplus
}
#endif
