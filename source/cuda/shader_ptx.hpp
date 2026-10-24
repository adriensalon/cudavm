#pragma once
#include "backend.hpp"
#include "../core/program_image.hpp"
#include <string_view>
namespace cudavm {
std::string compile_shader_ptx(std::string_view, cudavmBackend);
std::string materialize_program_image(std::string_view, cudavmBackend);
}
