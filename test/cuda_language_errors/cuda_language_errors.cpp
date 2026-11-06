#include "conformance.hpp"
#include <utility>

namespace cudavm {
struct language_rejection_case {
    std::string_view name;
    const char* source;
    const char* entity;
};
constexpr std::array language_rejections{
    language_rejection_case{"host_entities", R"cuda(
#ifdef REJECT_RULE
#line 101 "cuda_language_host_entities.cu"
__host__ unsigned language_host(unsigned value) { return value + 1; }
#else
__device__ unsigned language_host(unsigned value) { return value + 1; }
#endif
extern "C" __global__ void language_control(unsigned* out) { *out = language_host(3); }
)cuda", "language_host"},
    language_rejection_case{"kernel_calls", R"cuda(
#ifdef REJECT_RULE
__global__ void language_child(unsigned* out) { *out = 7; }
#else
__device__ void language_child(unsigned* out) { *out = 7; }
#endif
#line 101 "cuda_language_kernel_calls.cu"
extern "C" __global__ void language_control(unsigned* out) { language_child(out); }
)cuda", "language_child"},
    language_rejection_case{"qualifiers", R"cuda(
#ifdef REJECT_RULE
#line 101 "cuda_language_qualifiers.cu"
extern "C" __global__ unsigned language_control(unsigned* out) { *out = 11; return 1; }
#else
extern "C" __global__ void language_control(unsigned* out) { *out = 11; }
#endif
)cuda", "language_control"},
    language_rejection_case{"storage", R"cuda(
extern "C" __global__ void language_control(unsigned* out) {
#ifdef REJECT_RULE
#line 101 "cuda_language_storage.cu"
    __shared__ unsigned language_shared = 13;
#else
    __shared__ unsigned language_shared;
    if (threadIdx.x == 0) language_shared = 13;
#endif
    __syncthreads();
    *out = language_shared;
}
)cuda", "language_shared"},
    language_rejection_case{"templates", R"cuda(
#ifdef REJECT_RULE
template<class T> __device__ unsigned language_deleted(T) = delete;
#else
template<class T> __device__ unsigned language_deleted(T value) { return unsigned(value) + 17; }
#endif
#line 101 "cuda_language_templates.cu"
extern "C" __global__ void language_control(unsigned* out) { *out = language_deleted(3u); }
)cuda", "language_deleted"}
};
std::string_view language_errors_case = "all";
void execute_cuda_language_errors() {
    bool selected = false;
    for (const auto& test : language_rejections) if (language_errors_case == "all" || language_errors_case == test.name) {
        selected = true;
        const auto file = "cuda_language_" + std::string(test.name) + ".cu";
        compiler_fixture_program program(test.source, file.c_str());
        const std::vector<std::string> options{"-arch=compute_75", "--std=c++17"};
        const auto control = program.success(options);
        require(control.find(".entry language_control") != std::string::npos, "legal CUDA language control retains kernel");
        auto rejected = options; rejected.push_back("-DREJECT_RULE");
        const auto result = program.compile(rejected);
        const auto log = program.text(true);
        require(result == NVRTC_ERROR_COMPILATION, std::string(test.name) + " must reject its illegal rule: " + log);
        require(program.text().empty(), std::string(test.name) + " rejected source publishes no PTX");
        require(log.find(file) != std::string::npos && log.find("101") != std::string::npos
            && log.find(test.entity) != std::string::npos, std::string(test.name) + " diagnostic identifies owned rule/location: " + log);
        require(program.success(options) == control, std::string(test.name) + " repaired legal variant reproduces PTX");
        require(program.text(true).empty(), std::string(test.name) + " successful repair clears rejected-source diagnostic");
    }
    require(selected, "unknown cuda_language_errors case");
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::language_errors_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_cuda_language_errors);
}
