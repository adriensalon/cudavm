#include "test_framework.hpp"
#include "nvvm_fixture.hpp"
#include <nvvm.h>

namespace cudavm {
std::string_view nvvm_import_case = "dialects";
void nvvm_import_check(nvvmResult result) {
    require(result == NVVM_SUCCESS, std::string("NVVM import API: ") + nvvmGetErrorString(result));
}
struct nvvm_import_program {
    nvvmProgram value{};
    nvvm_import_program() { nvvm_import_check(nvvmCreateProgram(&value)); }
    ~nvvm_import_program() { if (value) nvvmDestroyProgram(&value); }
    nvvm_import_program(const nvvm_import_program&) = delete;
    void add(std::string bytes) const {
        std::string name = "owned_nvvm_import.ll";
        nvvm_import_check(nvvmAddModuleToProgram(value, bytes.data(), bytes.size(), name.c_str()));
        std::fill(bytes.begin(), bytes.end(), '!'); std::fill(name.begin(), name.end(), '!');
    }
    nvvmResult run(bool verify, bool lto = false) const {
        const char* options[]{"-arch=compute_75", "-opt=0", "-gen-lto"};
        return verify ? nvvmVerifyProgram(value, 2, options) : nvvmCompileProgram(value, lto ? 3 : 2, options);
    }
    std::vector<unsigned char> bytes(bool log = false, bool text = true) const {
        const auto query = log ? nvvmGetProgramLogSize : nvvmGetCompiledResultSize;
        const auto get = log ? nvvmGetProgramLog : nvvmGetCompiledResult;
        struct output { size_t before = 0x193759bd, value{}, after = 0xabcdef01; } first_size, second_size;
        nvvm_import_check(query(value, &first_size.value)); nvvm_import_check(query(value, &second_size.value));
        require(first_size.before == 0x193759bd && first_size.after == 0xabcdef01
            && second_size.before == 0x193759bd && second_size.after == 0xabcdef01, "NVVM import size guards");
        const auto count = first_size.value;
        require(count > 0 && count < 64 * 1024 * 1024 && count == second_size.value, "NVVM import stable bounded extent");
        std::vector<unsigned char> first(count + 32, 0x35), second(count + 32, 0x79);
        nvvm_import_check(get(value, reinterpret_cast<char*>(first.data() + 16)));
        nvvm_import_check(get(value, reinterpret_cast<char*>(second.data() + 16)));
        for (size_t i = 0; i < first.size(); ++i) {
            if (i < 16 || i >= count + 16) require(first[i] == 0x35 && second[i] == 0x79, "NVVM import exact output guards");
            else require(first[i] == second[i], "NVVM import complete byte writes and repeated output");
        }
        first.erase(first.end() - 16, first.end()); first.erase(first.begin(), first.begin() + 16);
        if (text) require(first.back() == 0 && std::find(first.begin(), first.end() - 1, 0) == first.end() - 1,
            "NVVM import text has exact terminal NUL");
        return first;
    }
};
std::string nvvm_import_replace(std::string source, std::string_view from, std::string_view to) {
    const auto position = source.find(from); require(position != std::string::npos, "owned NVVM mutation exists");
    source.replace(position, from.size(), to); return source;
}
std::string nvvm_import_control() { return std::string(nvvm_import_source); }
void nvvm_import_accept(const std::string& input, bool downstream = false) {
    std::vector<unsigned char> image;
    {
        nvvm_import_program program; program.add(input);
        const auto verified = program.run(true);
        const auto verification_log = program.bytes(true);
        require(verified == NVVM_SUCCESS, "NVVM positive verification: " + std::string(verification_log.begin(), verification_log.end()));
        const auto compiled = program.run(false);
        const auto log = program.bytes(true);
        require(compiled == NVVM_SUCCESS, "NVVM positive compilation: " + std::string(log.begin(), log.end()));
        const auto text = program.bytes();
        require(std::string_view(reinterpret_cast<const char*>(text.data()), text.size() - 1).find("nvvm_import_control")
            != std::string_view::npos, "NVVM public PTX retains the input kernel");
        if (downstream) {
            nvvm_import_check(program.run(false, true)); image = program.bytes(false, false);
        }
    }
    if (downstream) {
        linker_fixture_program linker({"-arch=sm_75", "-lto", "-ptx", "-O0"});
        linker.add(image, "copied_nvvm_result", NVJITLINK_INPUT_LTOIR); std::fill(image.begin(), image.end(), 0x57);
        linker.complete();
        const auto text = compiler_fixture_bytes([&](size_t* size) {
            linker.check(nvJitLinkGetLinkedPtxSize(linker.value, size), "NVVM downstream PTX size");
        }, [&](unsigned char* data) {
            linker.check(nvJitLinkGetLinkedPtx(linker.value, reinterpret_cast<char*>(data)), "NVVM downstream PTX");
        });
        require(text.back() == 0 && std::string_view(reinterpret_cast<const char*>(text.data()), text.size() - 1)
            .find("nvvm_import_control") != std::string_view::npos, "NVVM copied LTO image survives producer/caller retirement");
    }
}
void nvvm_import_reject(const std::string& input, std::string_view reason, bool version = false) {
    nvvm_import_program program; program.add(input);
    for (bool verify : {true, false}) {
        const auto expected = version ? NVVM_ERROR_IR_VERSION_MISMATCH : verify ? NVVM_ERROR_INVALID_IR : NVVM_ERROR_COMPILATION;
        const auto status = program.run(verify); const auto log = program.bytes(true);
        const std::string diagnostic(log.begin(), log.end() - 1);
        require(status == expected, "NVVM reject status for " + std::string(reason) + ": " + nvvmGetErrorString(status) + "; " + diagnostic);
        require(!diagnostic.empty() && diagnostic.find(reason) != std::string::npos,
            "NVVM reject identifies input rule: " + diagnostic);
        require(program.bytes() == std::vector<unsigned char>{0}, "NVVM failed import publishes no output");
    }
    nvvm_import_accept(nvvm_import_control());
}
void nvvm_import_dialects() {
    const auto opaque = nvvm_import_control(); nvvm_import_accept(opaque);
    auto typed = nvvm_import_replace(opaque, "ptr @nvvm_import_control", "void (i32 addrspace(1)*)* @nvvm_import_control");
    while (typed.find("ptr addrspace(1)") != std::string::npos)
        typed = nvvm_import_replace(typed, "ptr addrspace(1)", "i32 addrspace(1)*");
    nvvm_import_accept(typed);
    const auto start = opaque.find("declare"); require(start != std::string::npos, "NVVM byval prefix");
    const auto legacy = opaque.substr(0, start) + R"ir(
%"record byval" = type { i32, i32 }
define void @nvvm_import_control(i32* %out, %"record byval"* byval align 8 %record, i32* byval %scalar) {
 ; i32* byval inside a comment is not an attribute.
 %field = getelementptr %"record byval", %"record byval"* %record, i32 0, i32 1
 %a = load i32, i32* %field
 %b = load i32, i32* %scalar
 %sum = add i32 %a, %b
 store i32 %sum, i32* %out
 ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, %"record byval"*, i32*)* @nvvm_import_control, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
!owned.note = !{!2}
!2 = !{!"i32* byval"}
)ir";
    nvvm_import_accept(legacy);
    auto explicit_types = nvvm_import_replace(legacy, "* byval align", "* byval(%\"record byval\") align");
    explicit_types = nvvm_import_replace(explicit_types, "i32* byval %scalar", "i32* byval(i32) %scalar");
    nvvm_import_accept(explicit_types);
    nvvm_import_reject(nvvm_import_replace(legacy, "i32* byval %scalar", "ptr byval %scalar"), "byval");
}
void nvvm_import_bitcode_cases() {
    nvvm_import_accept(nvvm_import_control(), true);
    const std::string bitcode(reinterpret_cast<const char*>(nvvm_import_bitcode), sizeof(nvvm_import_bitcode));
    nvvm_import_accept(bitcode, true);
    for (size_t count : {size_t{4}, bitcode.size() / 2, bitcode.size() - 1})
        nvvm_import_reject(bitcode.substr(0, count), "NVVM:");
}
void nvvm_import_versions() {
    const auto control = nvvm_import_control();
    nvvm_import_accept(nvvm_import_replace(control, "i32 2, i32 0, i32 3, i32 1", "i32 2, i32 0"));
    for (auto tuple : {"i32 1, i32 0", "i32 2, i32 1", "i32 2, i32 0, i32 3, i32 0"})
        nvvm_import_reject(nvvm_import_replace(control, "i32 2, i32 0, i32 3, i32 1", tuple), "expected IR 2.0", true);
    nvvm_import_reject(nvvm_import_replace(control, "!nvvmir.version = !{!1}", "!owned.version = !{!1}"), "expected IR 2.0", true);
    nvvm_import_reject(nvvm_import_replace(control, "i32 2, i32 0, i32 3, i32 1", "i32 2"), "version tuple");
    nvvm_import_reject(nvvm_import_replace(control, "i32 2, i32 0, i32 3, i32 1", "i64 2, i32 0"), "version field");
}
void nvvm_import_layout() {
    const auto control = nvvm_import_control();
    for (auto widths : {"e-p3:32:32-p5:32:32-", "e-p3:64:64-p5:64:64-"})
        nvvm_import_accept(nvvm_import_replace(control, "e-p:64:64-", std::string(widths) + "p:64:64-"));
    nvvm_import_reject(nvvm_import_replace(control, "nvptx64-nvidia-cuda", "x86_64-unknown-linux-gnu"), "nvptx64 target");
    nvvm_import_reject(nvvm_import_replace(control, "e-p:64:64-", "E-p:64:64-"), "little-endian");
    nvvm_import_reject(nvvm_import_replace(control, "e-p:64:64-", "e-p:32:32-"), "64-bit CUDA addresses");
    nvvm_import_reject(nvvm_import_replace(control, "e-p:64:64-", "e-ni:1-p:64:64-"), "integral 64-bit");
    nvvm_import_reject(nvvm_import_replace(control, "i64:64", "i64:32"), "integer data layout");
}
void nvvm_import_metadata() {
    const auto control = nvvm_import_control();
    nvvm_import_reject(nvvm_import_replace(control, "ptr @nvvm_import_control, !\"kernel\", i32 1", "ptr @nvvm_import_control, !\"kernel\""), "annotations tuple");
    nvvm_import_reject(nvvm_import_replace(control, "ptr @nvvm_import_control", "i32 7"), "requires a function");
    nvvm_import_reject(nvvm_import_replace(control, "!\"kernel\", i32 1", "!\"kernel\", i64 1"), "kernel annotation");
    nvvm_import_reject(nvvm_import_replace(control, "!\"kernel\", i32 1", "!\"maxntidx\", i32 32"), "kernel annotation");
    nvvm_import_reject(control + "\n!cudavm.claim = !{!2}\n!2 = !{i32 1}\n", "reserved cudavm");
    nvvm_import_reject(control + "\n!llvm.module.flags = !{!2}\n!2 = !{i32 2, !\"Debug Info Version\", i32 3}\n", "debug module flags");
}
void nvvm_import_operations() {
    const auto control = nvvm_import_control();
    nvvm_import_reject(control + "\n@owned_shared = addrspace(3) global i32 0\n", "shared storage requires undef");
    nvvm_import_reject(control + "\n@owned_space = addrspace(7) global i32 0\n", "global address space");
    nvvm_import_reject(control + "\ndeclare i32 @llvm.nvvm.owned.unknown()\n", "llvm.nvvm.owned.unknown");
    nvvm_import_reject(control + "\ndeclare ptr @llvm.returnaddress(i32 immarg)\n", "unsupported LLVM/NVVM intrinsic");
    nvvm_import_reject(control + "\nmodule asm \"owned module assembly\"\n", "module assembly");
    nvvm_import_reject(nvvm_import_replace(control, "  ret void", "  call void asm sideeffect \"owned unsupported assembly\", \"\"()\n  ret void"), "inline assembly");
}
void execute_nvvm_import() {
    if (nvvm_import_case == "dialects") nvvm_import_dialects();
    else if (nvvm_import_case == "bitcode") nvvm_import_bitcode_cases();
    else if (nvvm_import_case == "versions") nvvm_import_versions();
    else if (nvvm_import_case == "layout") nvvm_import_layout();
    else if (nvvm_import_case == "metadata") nvvm_import_metadata();
    else { require(nvvm_import_case == "operations", "known NVVM import case"); nvvm_import_operations(); }
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::nvvm_import_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_nvvm_import);
}
