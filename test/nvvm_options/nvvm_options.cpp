#include "test_framework.hpp"
#include <nvvm.h>
#include <algorithm>
#include <utility>

namespace cudavm {
void options_nvvm(nvvmResult result) {
    require(result == NVVM_SUCCESS, std::string("libNVVM option API: ") + nvvmGetErrorString(result));
}
template<class T> struct nvvm_options_output {
    static_assert(alignof(T) <= 16);
    std::array<unsigned char, 16> before;
    T value{};
    std::array<unsigned char, 16> after;
    nvvm_options_output() { before.fill(0xa9); after.fill(0x3d); }
    T get() const {
        require(std::all_of(before.begin(), before.end(), [](auto byte) { return byte == 0xa9; })
            && std::all_of(after.begin(), after.end(), [](auto byte) { return byte == 0x3d; }), "libNVVM options scalar output guards");
        return value;
    }
};
std::string nvvm_options_ir(unsigned major = 2) {
    return R"ir(target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare float @llvm.fmuladd.f32(float, float, float)
define void @nvvm_options_kernel(float* %out, float %a, float %b, float %c) {
  %value = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
  store float %value, float* %out
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (float*, float, float, float)* @nvvm_options_kernel, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 )ir" + std::to_string(major) + ", i32 0, i32 3, i32 1}\n";
}
struct nvvm_options_program {
    nvvmProgram value{};
    nvvm_options_program() {
        nvvm_options_output<nvvmProgram> output; options_nvvm(nvvmCreateProgram(&output.value)); value = output.get();
        require(value != nullptr, "libNVVM options create publishes handle");
    }
    ~nvvm_options_program() { if (value) nvvmDestroyProgram(&value); }
    nvvm_options_program(const nvvm_options_program&) = delete;
    nvvm_options_program& operator=(const nvvm_options_program&) = delete;
    void add(const std::string& ir = nvvm_options_ir()) const { options_nvvm(nvvmAddModuleToProgram(value, ir.data(), ir.size(), "nvvm_options.ll")); }
    nvvmResult run(bool verify, const std::vector<std::string>& options = {}) const {
        std::vector<const char*> pointers; for (const auto& option : options) pointers.push_back(option.c_str());
        return verify ? nvvmVerifyProgram(value, static_cast<int>(pointers.size()), pointers.data())
            : nvvmCompileProgram(value, static_cast<int>(pointers.size()), pointers.data());
    }
    std::string text(bool log = false) const {
        const auto size_query = log ? nvvmGetProgramLogSize : nvvmGetCompiledResultSize;
        const auto text_query = log ? nvvmGetProgramLog : nvvmGetCompiledResult;
        nvvm_options_output<size_t> size, repeated;
        options_nvvm(size_query(value, &size.value)); options_nvvm(size_query(value, &repeated.value));
        const auto count = size.get(); require(count > 0 && count == repeated.get(), "libNVVM text size includes NUL and is stable");
        std::vector<char> first(count + 32, char(0x23)), second(count + 32, char(0x67));
        options_nvvm(text_query(value, first.data() + 16)); options_nvvm(text_query(value, second.data() + 16));
        for (size_t i = 0; i < first.size(); ++i) {
            if (i < 16 || i >= count + 16) require(first[i] == char(0x23) && second[i] == char(0x67), "libNVVM text exact output guards");
            else require(first[i] == second[i], "libNVVM text retrieval writes every reported byte");
        }
        require(first[count + 15] == 0 && std::find(first.begin() + 16, first.begin() + count + 15, '\0') == first.begin() + count + 15, "libNVVM text exact terminal NUL");
        return {first.begin() + 16, first.begin() + count + 15};
    }
    std::string success(const std::vector<std::string>& options = {}) const {
        const auto result = run(false, options);
        require(result == NVVM_SUCCESS, std::string(nvvmGetErrorString(result)) + ": " + text(true));
        const auto output = text(); require(output.find("nvvm_options_kernel") != std::string::npos, "compiled PTX retains kernel"); return output;
    }
};
void nvvm_options_architectures() {
    nvvm_options_program program; program.add();
    for (const auto* architecture : {"75", "80", "87", "89", "90"}) {
        const std::vector<std::string> options{std::string("-arch=compute_") + architecture};
        options_nvvm(program.run(true, options));
        require(program.text(true).empty(), "valid architecture verification has no diagnostic");
        require(program.success(options).find(std::string(".target sm_") + architecture) != std::string::npos, "public architecture option selects PTX target");
    }
    require(program.success().find(".target sm_75") != std::string::npos, "default NVVM target is compute_75");
}
void nvvm_options_contraction() {
    nvvm_options_program program; program.add();
    for (const auto* optimization : {"-opt=0", "-opt=3"}) {
        const auto separate = program.success({optimization, "-fma=0", "-ftz=0", "-prec-div=1", "-prec-sqrt=1", "-jump-table-density=101"});
        require(separate.find("mul.rn.f32") != std::string::npos && separate.find("add.rn.f32") != std::string::npos
            && separate.find("fma.") == std::string::npos && separate.find("mad.") == std::string::npos, "disabled contraction emits independent rounded multiply and add");
        program.success({optimization, "-fma=1"}); // Permission to fuse does not require fusion.
    }
}
void nvvm_options_rejection() {
    nvvm_options_program program; program.add();
    const auto original = program.success();
    for (bool verify : {false, true}) {
        for (const auto* option : {"-conformance-invalid-option", "-opt=1", "-fma=2", "-arch=compute_0", "-ftz=2", "-prec-div=2", "-jump-table-density=102"}) {
            require(program.run(verify, {option}) == NVVM_ERROR_INVALID_OPTION, "malformed public option has invalid-option status");
            require(!program.text(true).empty(), "invalid option has retrievable guarded diagnostic");
            options_nvvm(program.run(true));
            require(program.text(true).empty(), "valid verification clears preceding option diagnostic");
        }
    }
    require(program.success() == original, "failed option attempts preserve module inputs for identical successful retry");
}
void nvvm_options_verification() {
    nvvm_options_program empty;
    require(empty.run(true) == NVVM_ERROR_NO_MODULE_IN_PROGRAM && empty.run(false) == NVVM_ERROR_NO_MODULE_IN_PROGRAM, "empty verification/compilation has no-module status");
    empty.add(); options_nvvm(empty.run(true));
    require(empty.text().empty() && empty.text(true).empty(), "verification alone emits no code or errors");
    empty.success();
    nvvm_options_program invalid;
    invalid.add(R"ir(target triple = "nvptx64-nvidia-cuda"
define void @broken_dominance(i1 %condition) {
entry:
  br i1 %condition, label %left, label %right
left:
  %value = add i32 1, 2
  br label %right
right:
  %use = add i32 %value, 3
  ret void
}
!nvvmir.version = !{!0}
!0 = !{i32 2, i32 0, i32 3, i32 1}
)ir");
    require(invalid.run(true) == NVVM_ERROR_INVALID_IR && !invalid.text(true).empty(), "non-dominating SSA definition fails verification with IR-specific status");
    require(invalid.run(false) == NVVM_ERROR_COMPILATION && !invalid.text(true).empty(), "same invalid IR fails compilation with compilation-specific status");
    nvvm_options_program version;
    version.add(nvvm_options_ir(99));
    for (bool verify : {false, true})
        require(version.run(verify) == NVVM_ERROR_IR_VERSION_MISMATCH && !version.text(true).empty(), "unsupported NVVM IR major version has version-mismatch status");
}
void nvvm_options_validation() {
    nvvm_options_program program; program.add();
    const char* null_option = nullptr;
    for (bool verify : {false, true}) {
        const auto run = verify ? nvvmVerifyProgram : nvvmCompileProgram;
        require(run(nullptr, 0, nullptr) == NVVM_ERROR_INVALID_PROGRAM, "null option program has invalid-program status");
        require(run(program.value, -1, nullptr) == NVVM_ERROR_INVALID_INPUT && run(program.value, 1, nullptr) == NVVM_ERROR_INVALID_INPUT
            && run(program.value, 1, &null_option) == NVVM_ERROR_INVALID_INPUT, "malformed option arrays have invalid-input status");
    }
    options_nvvm(program.run(true)); program.success();
}
std::string_view nvvm_options_case = "all";
void execute_nvvm_options() {
    const std::array<std::pair<std::string_view, void (*)()>, 5> cases{{
        {"architectures", nvvm_options_architectures}, {"contraction", nvvm_options_contraction},
        {"rejection", nvvm_options_rejection}, {"verification", nvvm_options_verification}, {"validation", nvvm_options_validation}}};
    bool selected = false;
    for (const auto& [name, execute] : cases) if (nvvm_options_case == "all" || nvvm_options_case == name) { selected = true; execute(); }
    require(selected, "unknown nvvm_options case");
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::nvvm_options_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_nvvm_options);
}
