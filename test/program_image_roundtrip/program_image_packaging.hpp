#pragma once
#include "program_image_fixture.hpp"
#include <nvFatbin.h>

namespace cudavm {
inline void program_image_fatbin_status(nvFatbinResult status) {
    if (status != NVFATBIN_SUCCESS) throw std::runtime_error(std::string("program nvFatbin status ") + nvFatbinGetErrorString(status));
}
inline program_image_bytes program_image_package(program_image_bytes input, bool lto) {
    struct owner { nvFatbinHandle value{}; ~owner() { if (value) nvFatbinDestroy(&value); } } packager;
    const char* options[]{"-compress=false"}; program_image_fatbin_status(nvFatbinCreate(&packager.value, options, 1));
    std::string name = "owned_program_member";
    program_image_fatbin_status(lto ? nvFatbinAddLTOIR(packager.value, input.data(), input.size(), "75", name.c_str(), "")
        : nvFatbinAddCubin(packager.value, input.data(), input.size(), "75", name.c_str()));
    std::fill(input.begin(), input.end(), 0x57); std::fill(name.begin(), name.end(), '!');
    return program_image_result([&](size_t* size) { program_image_fatbin_status(nvFatbinSize(packager.value, size)); },
        [&](unsigned char* bytes) { program_image_fatbin_status(nvFatbinGet(packager.value, bytes)); });
}
}
