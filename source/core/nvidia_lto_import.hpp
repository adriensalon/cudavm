#pragma once
#include <string>
#include <string_view>

namespace cudavm {
inline bool has_nvidia_lto_ir(std::string_view bytes) {
    return bytes.starts_with(std::string_view("\xed\x43\x4e\x7f", 4));
}
struct nvidia_lto_program {
    std::string bitcode, architecture;
};
nvidia_lto_program decode_nvidia_lto(std::string_view);
}
