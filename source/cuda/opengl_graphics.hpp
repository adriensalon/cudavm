#pragma once
#include "backend.hpp"
#include <cuda/cudavm_graphics_gl.h>
#if defined(_WIN32)
#define CUDAVM_GL_CALL __stdcall
#else
#define CUDAVM_GL_CALL
#endif
namespace cudavm {
class opengl_graphics {
    cudavmGraphicsGLDesc descriptor;
    using bind_type = void (CUDAVM_GL_CALL*)(unsigned, unsigned);
    using integer_type = void (CUDAVM_GL_CALL*)(unsigned, int*);
    using query_type = void (CUDAVM_GL_CALL*)(unsigned, unsigned, int*);
    using is_buffer_type = unsigned char (CUDAVM_GL_CALL*)(unsigned);
    using transfer_type = void (CUDAVM_GL_CALL*)(unsigned, ptrdiff_t, ptrdiff_t, void*);
    using write_type = void (CUDAVM_GL_CALL*)(unsigned, ptrdiff_t, ptrdiff_t, const void*);
    using finish_type = void (CUDAVM_GL_CALL*)();
    using error_type = unsigned (CUDAVM_GL_CALL*)();
    bind_type bind;
    integer_type integer;
    query_type query;
    is_buffer_type is_buffer;
    transfer_type read_buffer;
    write_type write_buffer;
    finish_type finish;
    error_type error;
    void ready() const;
    void check(const char*) const;
public:
    explicit opengl_graphics(const cudavmGraphicsGLDesc&);
    size_t size(unsigned buffer) const;
    void synchronize() const;
    void read(unsigned buffer, std::span<std::byte>) const;
    void write(unsigned buffer, std::span<const std::byte>) const;
};
}
