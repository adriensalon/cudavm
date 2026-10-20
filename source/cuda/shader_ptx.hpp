#pragma once
#include "backend.hpp"
#include <string_view>
namespace cudavm {
std::string compile_shader_ptx(std::string_view, cudavmBackend);
}
