#pragma once
#include <filesystem>
#include <string>
#include <vector>

namespace cudavm {
enum class nvcc_phase { link, compile, device_compile, device_link, archive, preprocess, dependencies, ptx, image };
struct nvcc_options {
    nvcc_phase phase = nvcc_phase::link;
    std::string output, standard = "c++17", architecture = "sm_75", host_compiler;
    std::string runtime = "static", depfile, dependency_target, keep_directory;
    std::vector<std::string> inputs, includes, defines, undefines, host_flags, link_flags, library_paths, libraries;
    unsigned optimization = 0;
    bool verbose = false, version = false, help = false, keep = false, debug = false, shared = false, dep_compile = false;
    bool relocatable = false;
    bool forward_unknown = false;
    bool device_runtime = true;
    bool cuda_input = false;
    std::string host_runtime = "static";
};
nvcc_options parse_nvcc_options(int argc, char** argv);
std::vector<std::string> nvcc_split_arguments(const std::string&);
std::filesystem::path nvcc_executable_path();
std::filesystem::path nvcc_find_program(const std::string&);
int nvcc_execute(const std::vector<std::string>&, bool verbose);
std::string nvcc_quote_argument(const std::string&);
int run_nvcc_driver(int argc, char** argv);
int nvcc_build_device_runtime(const char* output);
}
