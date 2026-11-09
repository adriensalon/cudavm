#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include "test_framework.hpp"
#include <cudaTypedefs.h>
#include <type_traits>
namespace cudavm {
static_assert(std::is_same_v<decltype(&cuMemBatchDecompressAsync), PFN_cuMemBatchDecompressAsync_v12060_ptsz>);
CUresult decompression_ptds(CUmemDecompressParams* records, size_t count, unsigned flags, size_t* index, CUstream stream) {
    return cuMemBatchDecompressAsync(records, count, flags, index, stream);
}
}
