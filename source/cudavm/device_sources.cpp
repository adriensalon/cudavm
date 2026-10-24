#include "device_sources.hpp"
#include "embedded_headers.hpp"
namespace cudavm {
std::string_view device_source(std::string_view name) {
    return embedded_source("device/" + std::string(name));
}
std::string render_device_source(std::string_view name,
    std::initializer_list<std::pair<std::string_view, std::string_view>> values) {
    std::string result(device_source(name));
    for (auto [key, value] : values) {
        const auto marker = "@" + std::string(key) + "@";
        size_t at = 0;
        while ((at = result.find(marker, at)) != std::string::npos) {
            result.replace(at, marker.size(), value);
            at += value.size();
        }
    }
    return result;
}
}
