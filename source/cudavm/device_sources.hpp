#pragma once
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
namespace cudavm {
std::string_view device_source(std::string_view name);
std::string render_device_source(std::string_view name,
    std::initializer_list<std::pair<std::string_view, std::string_view>> values);
}
