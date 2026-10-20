#pragma once
#include <cstddef>
namespace cudavm {
// Private versioned C boundary. The callback copies bytes into caller-owned
// storage; C++ objects and ownership never cross a compiler DLL boundary.
enum class ptx_jit_status : int { success, invalid, unsupported, out_of_memory };
using ptx_jit_output = bool (*)(const char*, size_t, void*) noexcept;
using ptx_jit_compile = ptx_jit_status (*)(const char*, size_t, unsigned,
    ptx_jit_output, void*, char*, size_t) noexcept;
}
