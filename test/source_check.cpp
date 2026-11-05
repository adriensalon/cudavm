#include "conformance.hpp"
#include "source_fixture.hpp"
#include <nvrtc.h>
#include <algorithm>
#include <charconv>

namespace cudavm {
void check_source_fixture() {
    struct source_program {
        nvrtcProgram value{};
        ~source_program() { if (value) nvrtcDestroyProgram(&value); }
    } program;
    auto check = [&](nvrtcResult status) {
        if (status == NVRTC_SUCCESS) return;
        size_t size = 0;
        if (program.value) nvrtcGetProgramLogSize(program.value, &size);
        std::string diagnostic(size, '\0');
        if (size) nvrtcGetProgramLog(program.value, diagnostic.data());
        throw std::runtime_error(std::string(nvrtcGetErrorString(status)) + ": " + diagnostic);
    };
    check(nvrtcCreateProgram(&program.value, checked_source, "checked_fixture.cu", 0, nullptr, nullptr));
    const auto architecture = "-arch=compute_" + std::to_string(checked_architecture);
    const char* options[]{architecture.c_str(), "--std=c++17"};
    check(nvrtcCompileProgram(program.value, 2, options));
    size_t size = 0; check(nvrtcGetPTXSize(program.value, &size));
    require(size > 1, "compiled source fixture has a nonempty PTX result");
    std::vector<char> output(size + 32, char(0x5b));
    check(nvrtcGetPTX(program.value, output.data() + 16));
    check(nvrtcDestroyProgram(&program.value));
    for (size_t index = 0; index < output.size(); ++index)
        if (index < 16 || index >= size + 16) require(output[index] == char(0x5b), "source fixture PTX extent guards");
    require(output[size + 15] == 0, "source fixture PTX has its declared terminator");
    const std::string_view ptx(output.data() + 16, size - 1);
    require(ptx.find(".entry ") != ptx.npos && ptx.find(".target sm_" + std::to_string(checked_architecture)) != ptx.npos,
        "owned source fixture output retains its kernel and declared architecture");
    if (checked_minimum_ptx) {
        const auto version = ptx.find(".version ");
        require(version != ptx.npos, "source fixture declares a PTX dialect");
        const auto* first = ptx.data() + version + 9;
        const auto* end = ptx.data() + ptx.size();
        unsigned major = 0, minor = 0;
        const auto whole = std::from_chars(first, end, major);
        require(whole.ec == std::errc{} && whole.ptr != end && *whole.ptr == '.', "source fixture PTX dialect syntax");
        const auto fraction = std::from_chars(whole.ptr + 1, end, minor);
        require(fraction.ec == std::errc{} && minor < 10 && major * 10 + minor >= checked_minimum_ptx,
            "source fixture satisfies its required PTX dialect floor");
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_host_conformance(argc, argv, cudavm::check_source_fixture);
}
