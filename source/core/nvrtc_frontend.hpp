#pragma once
#include "compiler.hpp"
#include <map>
#include <utility>
namespace cudavm {
struct nvrtc_options {
    std::string standard = "c++17", architecture = "sm_75";
    bool relocatable = false;
    std::vector<std::string> includes, defines, undefines;
};
struct nvrtc_output {
    std::unique_ptr<llvm::Module> module;
    std::map<std::string, std::string> lowered_names;
};
nvrtc_output import_cuda_source(llvm::LLVMContext&, const std::string& source, const std::string& name,
    const std::vector<std::pair<std::string, std::string>>& headers,
    const std::vector<std::string>& expressions, const nvrtc_options&, std::string& log);
}
