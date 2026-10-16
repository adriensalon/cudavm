#include "opengl_graphics.hpp"
#include <limits>
namespace cudavm {
namespace {
constexpr unsigned array_buffer = 0x8892, array_buffer_binding = 0x8894, buffer_size = 0x8764;
template<class T> T load(const cudavmGraphicsGLDesc& d, const char* name) {
    auto result = reinterpret_cast<T>(d.get_proc_address(d.user_data, name));
    require(result, CUDA_ERROR_NOT_SUPPORTED, name);
    return result;
}
}
opengl_graphics::opengl_graphics(const cudavmGraphicsGLDesc& d) : descriptor(d) {
    require(d.abi_version == 1 && d.struct_size == sizeof(d), CUDA_ERROR_INVALID_VALUE, "invalid GL bridge ABI");
    require(d.get_proc_address && d.is_current, CUDA_ERROR_INVALID_VALUE, "GL bridge callbacks are required");
    ready();
    bind = load<bind_type>(d, "glBindBuffer"); integer = load<integer_type>(d, "glGetIntegerv");
    query = load<query_type>(d, "glGetBufferParameteriv");
    is_buffer = load<is_buffer_type>(d, "glIsBuffer");
    read_buffer = load<transfer_type>(d, "glGetBufferSubData"); write_buffer = load<write_type>(d, "glBufferSubData");
    finish = load<finish_type>(d, "glFinish"); error = load<error_type>(d, "glGetError");
}
void opengl_graphics::ready() const {
    require(descriptor.is_current(descriptor.user_data) != 0, CUDA_ERROR_INVALID_GRAPHICS_CONTEXT, "the attached application GL context must be current");
}
void opengl_graphics::check(const char* operation) const {
    auto result = error();
    if (result) throw failure(result == 0x505 ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN,
                              std::string(operation) + " failed: " + std::to_string(result));
}
size_t opengl_graphics::size(unsigned object) const {
    ready(); check("application GL state");
    require(is_buffer(object) != 0, CUDA_ERROR_INVALID_HANDLE, "GL buffer does not exist in the attached context");
    int previous = 0; integer(array_buffer_binding, &previous);
    bind(array_buffer, object);
    int bytes = 0; query(array_buffer, buffer_size, &bytes);
    bind(array_buffer, static_cast<unsigned>(previous)); check("GL buffer size");
    require(bytes > 0 && uint64_t(bytes) <= uint64_t(std::numeric_limits<ptrdiff_t>::max()), CUDA_ERROR_INVALID_VALUE, "GL buffer has an invalid size");
    return static_cast<size_t>(bytes);
}
void opengl_graphics::synchronize() const { ready(); finish(); check("GL producer completion"); }
void opengl_graphics::read(unsigned object, std::span<std::byte> data) const {
    require(size(object) == data.size(), CUDA_ERROR_INVALID_VALUE, "GL buffer storage changed during mapping");
    synchronize();
    int previous = 0; integer(array_buffer_binding, &previous); bind(array_buffer, object);
    read_buffer(array_buffer, 0, static_cast<ptrdiff_t>(data.size()), data.data());
    bind(array_buffer, static_cast<unsigned>(previous)); check("GL buffer read");
}
void opengl_graphics::write(unsigned object, std::span<const std::byte> data) const {
    require(size(object) == data.size(), CUDA_ERROR_INVALID_VALUE, "GL buffer storage changed while mapped");
    int previous = 0; integer(array_buffer_binding, &previous); bind(array_buffer, object);
    write_buffer(array_buffer, 0, static_cast<ptrdiff_t>(data.size()), data.data());
    bind(array_buffer, static_cast<unsigned>(previous)); finish(); check("GL consumer transfer");
}
}
