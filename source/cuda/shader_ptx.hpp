#pragma once
#include "backend.hpp"
#include "../core/program_image.hpp"
#include "../nvfatbin/container_api.hpp"
#include <string_view>
namespace cudavm {
std::string compile_shader_ptx(std::string_view, cudavmBackend);
std::string compile_shader_cubin(std::string_view, cudavmBackend);
std::string link_shader_images(std::span<const std::string_view>, cudavmBackend);
std::string materialize_program_image(std::string_view, cudavmBackend);
std::string materialize_fatbin(std::string_view, cudavmBackend, unsigned architecture = 0);
std::string_view fatbin_memory_view(const void*);
}
