#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda.h>
#include <cuda_runtime_api.h>
namespace cudavm {
CUresult peer_copy_ptds_driver(CUdeviceptr to, CUcontext to_context, CUdeviceptr from, CUcontext from_context, size_t bytes, bool async) {
    return async ? cuMemcpyPeerAsync(to, to_context, from, from_context, bytes, nullptr) : cuMemcpyPeer(to, to_context, from, from_context, bytes);
}
CUresult peer_copy_ptds_driver_3d(const CUDA_MEMCPY3D_PEER* p, bool async) { return async ? cuMemcpy3DPeerAsync(p, nullptr) : cuMemcpy3DPeer(p); }
// The pinned header does not rewrite the linear Runtime peer-copy spelling.
cudaError_t peer_copy_ptds_runtime(void* to, int to_device, const void* from, int from_device, size_t bytes) { return cudaMemcpyPeerAsync(to, to_device, from, from_device, bytes, cudaStreamPerThread); }
cudaError_t peer_copy_ptds_runtime_3d(const cudaMemcpy3DPeerParms* p, bool async) { return async ? cudaMemcpy3DPeerAsync(p, nullptr) : cudaMemcpy3DPeer(p); }
CUresult peer_copy_ptds_gate(CUhostFn function, void* data) { return cuLaunchHostFunc(nullptr, function, data); }
CUresult peer_copy_ptds_sync() { return cuStreamSynchronize(nullptr); }
}
