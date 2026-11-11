#include "test_framework.hpp"
#include <nvvm.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <thread>
#include <vector>

namespace cudavm {
template<class T> struct nvvm_guarded {
    alignas(T) std::array<unsigned char, sizeof(T) + 32> bytes;
    nvvm_guarded() { bytes.fill(0xe7); }
    T* data() { return reinterpret_cast<T*>(bytes.data() + 16); }
    T get() const {
        for (size_t i = 0; i < bytes.size(); ++i) if (i < 16 || i >= sizeof(T) + 16) require(bytes[i] == 0xe7, "NVVM scalar output byte guards");
        T result{}; std::memcpy(&result, bytes.data() + 16, sizeof(T)); return result;
    }
};
using nvvm_size_query = nvvmResult (*)(nvvmProgram, size_t*);
using nvvm_data_query = nvvmResult (*)(nvvmProgram, char*);
std::vector<char> nvvm_result_bytes(nvvmProgram program, nvvm_size_query size_query, nvvm_data_query data_query, bool text) {
    nvvm_guarded<size_t> size, repeated;
    require(size_query(program, size.data()) == NVVM_SUCCESS && size_query(program, repeated.data()) == NVVM_SUCCESS, "NVVM result size query");
    const auto count = size.get(); require(count > 0 && repeated.get() == count, "NVVM result size is positive and stable");
    std::vector<char> first(count + 64, '\x27'), second(count + 64, '\x53');
    require(data_query(program, first.data() + 32) == NVVM_SUCCESS && data_query(program, second.data() + 32) == NVVM_SUCCESS, "NVVM result retrieval");
    for (size_t i = 0; i < first.size(); ++i) {
        if (i < 32 || i >= count + 32) require(first[i] == '\x27' && second[i] == '\x53', "NVVM exact result extent and guards");
        else require(first[i] == second[i], "NVVM complete byte writes and repeated retrieval");
    }
    std::vector<char> result(first.begin() + 32, first.end() - 32);
    if (text) require(result.back() == 0 && std::find(result.begin(), result.end() - 1, '\0') == result.end() - 1, "NVVM exact text termination");
    return result;
}
struct nvvm_lifecycle_program {
    nvvmProgram value{}; std::string kernel;
    nvvm_lifecycle_program() {
        nvvm_guarded<nvvmProgram> output; require(nvvmCreateProgram(output.data()) == NVVM_SUCCESS, "NVVM create"); value = output.get(); require(value != nullptr, "NVVM create publishes a program");
    }
    ~nvvm_lifecycle_program() { if (value) nvvmDestroyProgram(&value); }
    nvvm_lifecycle_program(const nvvm_lifecycle_program&) = delete;
    nvvm_lifecycle_program& operator=(const nvvm_lifecycle_program&) = delete;
    void add(unsigned seed) {
        kernel = "nvvm_lifecycle_" + std::to_string(seed);
        std::string ir = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64\"\ntarget triple = \"nvptx64-nvidia-cuda\"\ndefine void @" + kernel + "(i32* %out) {\n store i32 " + std::to_string(seed & 0x7fffffff) + ", i32* %out\n ret void\n}\n!nvvm.annotations = !{!0}\n!0 = !{void (i32*)* @" + kernel + ", !\"kernel\", i32 1}\n!nvvmir.version = !{!1}\n!1 = !{i32 2, i32 0, i32 3, i32 1}\n";
        std::string name = kernel + ".ll"; require(nvvmAddModuleToProgram(value, ir.data(), ir.size(), name.c_str()) == NVVM_SUCCESS, "NVVM eager module add");
        std::fill(ir.begin(), ir.end(), '!'); std::fill(name.begin(), name.end(), '!');
    }
    nvvmResult compile(bool lto = false) const { const char* options[]{"-arch=compute_75", "-gen-lto"}; return nvvmCompileProgram(value, lto ? 2 : 1, options); }
    void verify() const { require(nvvmVerifyProgram(value, 0, nullptr) == NVVM_SUCCESS, "NVVM verifies retained eager input"); }
    std::vector<char> result(bool lto = false) const {
        auto bytes = nvvm_result_bytes(value, nvvmGetCompiledResultSize, nvvmGetCompiledResult, !lto);
        if (!lto) require(std::string(bytes.data()).find(kernel) != std::string::npos, "NVVM PTX retains the accepted module's kernel name"); return bytes;
    }
    void destroy() {
        nvvm_guarded<nvvmProgram> output; std::memcpy(output.data(), &value, sizeof(value));
        require(nvvmDestroyProgram(output.data()) == NVVM_SUCCESS, "NVVM destroy"); value = nullptr; require(output.get() == nullptr, "NVVM destroy clears only handle slot");
    }
};
void nvvm_lifecycle_discovery() {
    nvvm_guarded<int> major, minor, debug_major, debug_minor, llvm;
    require(nvvmVersion(major.data(), minor.data()) == NVVM_SUCCESS && major.get() == 2 && minor.get() == 0, "NVVM API version");
    require(nvvmIRVersion(major.data(), minor.data(), debug_major.data(), debug_minor.data()) == NVVM_SUCCESS && major.get() == 2 && minor.get() == 0 && debug_major.get() == 3 && debug_minor.get() == 1, "NVVM IR/debug versions");
    require(nvvmLLVMVersion("compute_75", llvm.data()) == NVVM_SUCCESS && llvm.get() == 7, "NVVM compute_75 LLVM input dialect");
    for (int status = NVVM_SUCCESS; status <= NVVM_ERROR_CANCELLED; ++status) { const auto text = nvvmGetErrorString(static_cast<nvvmResult>(status)); require(text && *text, "defined NVVM error has description"); }
}
void nvvm_lifecycle_ownership() {
    for (unsigned seed : {0u, 79u, 0xffffffffu}) { nvvm_lifecycle_program program; program.add(seed); program.verify(); require(program.compile() == NVVM_SUCCESS, "NVVM compiles copied input after caller storage dies"); program.result(); program.destroy(); }
}
void nvvm_lifecycle_outputs() {
    for (bool lto : {false, true}) {
        nvvm_lifecycle_program program; program.add(lto ? 29 : 17); program.verify(); require(program.compile(lto) == NVVM_SUCCESS, "NVVM output-mode compilation");
        const auto retained = program.result(lto), before = retained; require(program.result(lto) == retained, "NVVM unchanged program has stable result bytes");
        nvvm_result_bytes(program.value, nvvmGetProgramLogSize, nvvmGetProgramLog, true); program.destroy(); require(retained == before, "NVVM caller-owned result survives program destruction");
    }
}
void nvvm_lifecycle_retry() {
    nvvm_lifecycle_program program;
    require(program.compile() == NVVM_ERROR_NO_MODULE_IN_PROGRAM && nvvmVerifyProgram(program.value, 0, nullptr) == NVVM_ERROR_NO_MODULE_IN_PROGRAM, "empty program compile/verify errors");
    program.add(137); const char* invalid[]{"-lifecycle-invalid-option"};
    require(nvvmCompileProgram(program.value, 1, invalid) == NVVM_ERROR_INVALID_OPTION, "invalid compile option");
    require(nvvm_result_bytes(program.value, nvvmGetProgramLogSize, nvvmGetProgramLog, true).size() > 1, "invalid option diagnostics");
    program.verify(); require(program.compile() == NVVM_SUCCESS, "same NVVM program recovers after empty/option failures"); program.result();
    nvvm_lifecycle_program malformed; const std::string invalid_ir = "not a valid NVVM module";
    require(nvvmAddModuleToProgram(malformed.value, invalid_ir.data(), invalid_ir.size(), "malformed.ll") == NVVM_SUCCESS, "NVVM accepts module bytes before verification");
    require(nvvmVerifyProgram(malformed.value, 0, nullptr) == NVVM_ERROR_INVALID_IR, "malformed verification status");
    require(nvvm_result_bytes(malformed.value, nvvmGetProgramLogSize, nvvmGetProgramLog, true).size() > 1, "malformed verification diagnostic");
    require(malformed.compile() == NVVM_ERROR_COMPILATION, "malformed compilation status"); nvvm_result_bytes(malformed.value, nvvmGetProgramLogSize, nvvmGetProgramLog, true); malformed.destroy();
}
void nvvm_lifecycle_validation() {
    require(nvvmCreateProgram(nullptr) == NVVM_ERROR_INVALID_PROGRAM && nvvmDestroyProgram(nullptr) == NVVM_ERROR_INVALID_PROGRAM, "null NVVM create/destroy arguments");
    require(nvvmCompileProgram(nullptr, 0, nullptr) == NVVM_ERROR_INVALID_PROGRAM && nvvmVerifyProgram(nullptr, 0, nullptr) == NVVM_ERROR_INVALID_PROGRAM, "null NVVM program");
    nvvm_lifecycle_program program; const char input[] = "invalid";
    require(nvvmAddModuleToProgram(program.value, nullptr, 1, "null.ll") == NVVM_ERROR_INVALID_INPUT && nvvmAddModuleToProgram(program.value, input, 0, "empty.ll") == NVVM_ERROR_INVALID_INPUT, "invalid eager module arguments");
    require(nvvmCompileProgram(program.value, -1, nullptr) == NVVM_ERROR_INVALID_INPUT && nvvmVerifyProgram(program.value, 1, nullptr) == NVVM_ERROR_INVALID_INPUT, "invalid NVVM option counts/arrays");
    for (auto query : std::array<nvvm_size_query, 2>{nvvmGetCompiledResultSize, nvvmGetProgramLogSize}) { size_t size{}; require(query(nullptr, &size) == NVVM_ERROR_INVALID_PROGRAM && query(program.value, nullptr) == NVVM_ERROR_INVALID_INPUT, "invalid NVVM size outputs"); }
    for (auto query : std::array<nvvm_data_query, 2>{nvvmGetCompiledResult, nvvmGetProgramLog}) require(query(program.value, nullptr) == NVVM_ERROR_INVALID_INPUT, "null NVVM result storage");
    program.add(193); program.verify(); require(program.compile() == NVVM_SUCCESS, "valid NVVM program after rejected arguments"); program.result(); program.destroy();
}
void nvvm_lifecycle_threading() {
    std::array<std::thread, 4> workers; std::array<std::exception_ptr, 4> failures{};
    const auto join = [&] { for (auto& worker : workers) if (worker.joinable()) worker.join(); };
    try { for (unsigned worker = 0; worker < workers.size(); ++worker) workers[worker] = std::thread([&, worker] {
        try { nvvm_lifecycle_program program; program.add(211 + worker * 19); program.verify(); require(program.compile() == NVVM_SUCCESS, "concurrent independent NVVM compilation"); program.result(); program.destroy(); }
        catch (...) { failures[worker] = std::current_exception(); }
    }); } catch (...) { join(); throw; }
    join(); for (const auto& failure : failures) if (failure) std::rethrow_exception(failure);
}
std::string nvvm_lifecycle_case = "all";
void execute_nvvm_lifecycle() {
    bool selected = false;
    const auto run = [&](const char* name, auto execute) { if (nvvm_lifecycle_case == "all" || nvvm_lifecycle_case == name) { execute(); selected = true; } };
    run("discovery", nvvm_lifecycle_discovery); run("ownership", nvvm_lifecycle_ownership); run("outputs", nvvm_lifecycle_outputs); run("retry", nvvm_lifecycle_retry); run("validation", nvvm_lifecycle_validation); run("threading", nvvm_lifecycle_threading);
    require(selected, "unknown nvvm_lifecycle case: " + nvvm_lifecycle_case);
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::nvvm_lifecycle_case = argv[1];
    return cudavm::run_host_test(argc, argv, cudavm::execute_nvvm_lifecycle);
}
