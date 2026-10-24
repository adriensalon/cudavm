// CLI and build artifact paths only. Libraries own compilation and linking.
#include "../core/compiler_client.hpp"
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
using namespace cudavm;
int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error(
            "usage: cudavm-nvcc compile-tu --input source.cu --host host.cpp --bitcode device.bc [--manifest tu.json]\n"
            "       cudavm-nvcc link-device --bitcode device.bc ... --portable program.bc [--target cuda --binary program.ptx --module program.cpp]\n"
            "         [--library-bitcode candidate.bc ... --host-input host.obj --host-input library.lib ...]\n"
            "       cudavm-nvcc --print-resources");
        std::string command = argv[1];
        std::map<std::string, std::vector<const char*>> fields;
        for (int i = 2; i < argc; ++i) {
            std::string key = argv[i];
            if (++i == argc) throw std::runtime_error("missing value for " + key);
            constexpr const char* valid[] = {"--input", "--host", "--bitcode", "--portable", "--ir", "--manifest",
                "--binary", "--module", "--target", "--standard", "--arch", "--depfile", "--include", "--define", "--resources-root",
                "--library-bitcode", "--host-input"};
            bool known = false;
            for (const auto* item : valid) if (key == item) known = true;
            if (!known) throw std::runtime_error("unknown option: " + key);
            auto& values = fields[key];
            if (key != "--bitcode" && key != "--include" && key != "--define"
                && key != "--library-bitcode" && key != "--host-input") values.clear();
            values.push_back(argv[i]);
        }
        auto value = [&](const char* key) -> const char* {
            const auto& values = fields[key];
            return values.empty() ? nullptr : values.back();
        };
        auto& inputs = fields["--bitcode"];
        if (command == "compile-tu" || command == "--print-resources") {
            if (value("--library-bitcode") || value("--host-input"))
                throw std::runtime_error("--library-bitcode and --host-input require link-device");
            compiler_result* result = nullptr;
            int status;
            if (command == "--print-resources") status = cudavm_core_describe_resources_v1(value("--resources-root"), &result);
            else {
                if (inputs.size() > 1) throw std::runtime_error("compile-tu produces one bitcode output");
                if (value("--module")) throw std::runtime_error("--module is a link-device output");
                auto& includes = fields["--include"]; auto& defines = fields["--define"];
                compiler_translation_unit_request options {1, value("--input"), value("--host"), value("--bitcode"),
                    value("--manifest"), value("--depfile"), value("--standard"), value("--arch"), value("--resources-root"),
                    includes.size(), includes.data(), defines.size(), defines.data()};
                status = cudavm_core_compile_translation_unit_v1(&options, &result);
            }
            compiler_result_ptr owned(result);
            std::fputs(cudavm_core_result_output_v1(result), stdout);
            std::fputs(cudavm_core_result_diagnostics_v1(result), stderr);
            return status;
        }
        if (command == "link-device") {
            auto& libraries = fields["--library-bitcode"]; auto& hosts = fields["--host-input"];
            compiler_link_request options {1, inputs.size(), inputs.data(), libraries.size(), libraries.data(),
                hosts.size(), hosts.data(), value("--portable"), value("--ir"), value("--manifest"),
                value("--binary"), value("--target"), value("--module"), value("--arch")};
            compiler_result* result = nullptr;
            auto status = cudavm_core_link_files_v1(&options, &result);
            compiler_result_ptr owned(result);
            std::fputs(cudavm_core_result_diagnostics_v1(result), stderr);
            return status;
        }
        throw std::runtime_error("unknown command: " + command);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "cudavm compiler: %s\n", error.what());
        return 1;
    }
}
