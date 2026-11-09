#pragma once
#include <cstdint>
#include <string>

namespace cudavm {
// The implementation owns DXC's isolated LLVM fork. No LLVM or COM types cross
// this boundary. Successful validation writes the DXIL container hash in place.
int32_t validate_directx_dxil(std::string& bytes, std::string& diagnostic);
}
