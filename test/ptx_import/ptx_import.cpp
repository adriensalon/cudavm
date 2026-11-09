#include "conformance.hpp"
#include <nvPTXCompiler.h>

namespace cudavm {
std::string_view ptx_import_case = "versions";
std::string ptx_import_header(std::string_view version = "8.4", std::string_view target = "sm_75") {
    return ".version " + std::string(version) + "\n.target " + std::string(target) + "\n.address_size 64\n";
}
std::string ptx_import_kernel(std::string_view body = "ret;", std::string_view name = "ptx_import_rejected") {
    return ptx_import_header() + ".visible .entry " + std::string(name) + "() { " + std::string(body) + " }\n";
}
std::string ptx_import_log(const linker_fixture_program& program) {
    auto bytes = compiler_fixture_bytes([&](size_t* size) {
        program.check(nvJitLinkGetErrorLogSize(program.value, size), "PTX rejection log size");
    }, [&](unsigned char* data) {
        program.check(nvJitLinkGetErrorLog(program.value, reinterpret_cast<char*>(data)), "PTX rejection log");
    });
    require(bytes.back() == 0 && std::find(bytes.begin(), bytes.end() - 1, 0) == bytes.end() - 1,
        "PTX diagnostic has exactly one terminal NUL");
    return {bytes.begin(), bytes.end() - 1};
}
void ptx_import_no_result(const linker_fixture_program& program) {
    struct { size_t before = 0x193759bd, value{}, after = 0xabcdef01; } output;
    require(nvJitLinkGetLinkedCubinSize(program.value, &output.value) == NVJITLINK_ERROR_INTERNAL,
        "PTX link publishes no binary before completion");
    require(output.before == 0x193759bd && output.after == 0xabcdef01, "failed PTX size query guards");
}
std::string ptx_import_text(const std::vector<unsigned char>& bytes) {
    linker_fixture_program downstream({"-arch=sm_90", "-lto", "-ptx", "-O0"});
    downstream.add(bytes, "retained_ptx_input", NVJITLINK_INPUT_CUBIN); downstream.complete();
    const auto text = compiler_fixture_bytes([&](size_t* size) {
        downstream.check(nvJitLinkGetLinkedPtxSize(downstream.value, size), "reimported PTX size");
    }, [&](unsigned char* data) {
        downstream.check(nvJitLinkGetLinkedPtx(downstream.value, reinterpret_cast<char*>(data)), "reimported PTX text");
    });
    require(text.back() == 0 && std::find(text.begin(), text.end() - 1, 0) == text.end() - 1,
        "reimported PTX has exact terminal NUL");
    return {text.begin(), text.end() - 1};
}
void ptx_import_accept(std::string input, size_t extent = 0) {
    std::vector<unsigned char> image;
    {
        linker_fixture_program program({"-arch=sm_90", "-r"});
        program.check(nvJitLinkAddData(program.value, NVJITLINK_INPUT_PTX, input.data(), extent ? extent : input.size(),
            "accepted_profile.ptx"), "accept declared PTX profile");
        std::fill(input.begin(), input.end(), '!'); program.complete(); image = program.image();
    }
    require(ptx_import_text(image).find("ptx_import_control") != std::string::npos,
        "accepted PTX kernel survives caller mutation, producer destruction and public reimport");
}
void ptx_import_reject(const std::string& source, std::string_view reason,
        nvJitLinkResult expected = NVJITLINK_ERROR_PTX_COMPILE) {
    std::vector<unsigned char> image;
    {
        linker_fixture_program program({"-arch=sm_90", "-r"});
        const auto sentinel = ptx_import_kernel("ret;", "ptx_import_sentinel");
        program.add({sentinel.begin(), sentinel.end()}, "sentinel.ptx", NVJITLINK_INPUT_PTX);
        require(nvJitLinkAddData(program.value, NVJITLINK_INPUT_PTX, source.data(), source.size(),
            "owned_rejected_profile.ptx") == expected, "PTX rejection status for " + std::string(reason));
        const auto diagnostic = ptx_import_log(program);
        require(diagnostic.find("owned_rejected_profile.ptx") != std::string::npos
            && diagnostic.find("PTX ") != std::string::npos && diagnostic.find(reason) != std::string::npos,
            "PTX rejection identifies owned source and rule: " + diagnostic);
        ptx_import_no_result(program);
        const auto control = ptx_import_kernel("ret;", "ptx_import_control");
        program.add({control.begin(), control.end()}, "control.ptx", NVJITLINK_INPUT_PTX);
        program.complete(); image = program.image();
    }
    const auto text = ptx_import_text(image);
    require(text.find("ptx_import_sentinel") != std::string::npos && text.find("ptx_import_control") != std::string::npos
        && text.find("ptx_import_rejected") == std::string::npos,
        "failed PTX addition preserves accepted input, excludes failed input and permits repair");
}
void ptx_import_parse(const std::string& source, nvPTXCompileResult expected, const char* architecture = "-arch=compute_90") {
    struct owner { nvPTXCompilerHandle value{}; ~owner() { if (value) nvPTXCompilerDestroy(&value); } } program;
    require(nvPTXCompilerCreate(&program.value, source.size(), source.data()) == NVPTXCOMPILE_SUCCESS, "PTX parse-only create");
    const char* options[]{architecture};
    const auto status = nvPTXCompilerCompile(program.value, 1, options);
    require(status == expected, "PTX parse-only expected status " + std::to_string(expected) + ", got " + std::to_string(status));
    struct { size_t before = 0x193759bd, value = 71, after = 0xabcdef01; } size;
    const auto query = nvPTXCompilerGetCompiledProgramSize(program.value, &size.value);
    require(size.before == 0x193759bd && size.after == 0xabcdef01, "PTX parse-only size guards");
    if (status == NVPTXCOMPILE_PARSE_ONLY_SUCCESS)
        require(query == NVPTXCOMPILE_SUCCESS && size.value == 0, "parse-only success has no binary bytes");
    else {
        require(query == NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE, "failed PTX import has no result");
        const auto log = compiler_fixture_bytes([&](size_t* output) {
            require(nvPTXCompilerGetErrorLogSize(program.value, output) == NVPTXCOMPILE_SUCCESS, "PTX Compiler rejection log size");
        }, [&](unsigned char* output) {
            require(nvPTXCompilerGetErrorLog(program.value, reinterpret_cast<char*>(output)) == NVPTXCOMPILE_SUCCESS,
                "PTX Compiler rejection log");
        });
        require(log.size() > 1 && log.back() == 0, "failed PTX Compiler parse has diagnostic text");
    }
}
void ptx_import_versions() {
    for (unsigned major : {6u, 7u, 8u}) for (unsigned minor = 0; minor <= (major == 6 ? 5u : major == 7 ? 8u : 4u); ++minor) {
        const auto source = ptx_import_header(std::to_string(major) + "." + std::to_string(minor), "sm_50")
            + ".visible .entry ptx_import_control() { ret; }\n";
        ptx_import_parse(source, NVPTXCOMPILE_PARSE_ONLY_SUCCESS);
    }
    for (auto version : {"5.0", "6.6", "7.9", "8.5", "9.0", "8.40"}) {
        const auto source = ptx_import_header(version) + ".visible .entry ptx_import_rejected() { ret; }\n";
        ptx_import_parse(source, NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION);
        ptx_import_reject(source, "unsupported PTX version");
    }
}
void ptx_import_targets() {
    const std::array<std::pair<const char*, const char*>, 15> targets{{
        {"sm_50", "6.0"}, {"sm_52", "6.0"}, {"sm_53", "6.0"}, {"sm_60", "6.0"}, {"sm_61", "6.0"},
        {"sm_62", "6.0"}, {"sm_70", "6.0"}, {"sm_72", "6.1"}, {"sm_75", "6.3"}, {"sm_80", "7.0"},
        {"sm_86", "7.1"}, {"sm_87", "7.4"}, {"sm_89", "7.8"}, {"sm_90", "7.8"}, {"sm_75", "8.4"}}};
    for (const auto& [target, version] : targets)
        ptx_import_parse(ptx_import_header(version, target) + ".entry ptx_import_control() { ret; }", NVPTXCOMPILE_PARSE_ONLY_SUCCESS);
    ptx_import_reject(ptx_import_header("6.0", "sm_75"), "target is unavailable");
    for (auto target : {"sm_90a", "sm_91"})
        ptx_import_reject(ptx_import_header("8.4", target), "unsupported PTX target", NVJITLINK_ERROR_UNRECOGNIZED_INPUT);
    ptx_import_reject(ptx_import_header("8.4", "sm_75, texmode_independent"), "target option", NVJITLINK_ERROR_UNRECOGNIZED_INPUT);
    ptx_import_reject(".version 8.4\n.target sm_75\n.address_size 32\n", "64-bit addresses", NVJITLINK_ERROR_UNRECOGNIZED_INPUT);
    const auto newer = ptx_import_header("8.4", "sm_80") + ".entry ptx_import_control() { ret; }";
    ptx_import_parse(newer, NVPTXCOMPILE_ERROR_COMPILATION_FAILURE, "-arch=compute_75");
    ptx_import_parse(newer, NVPTXCOMPILE_PARSE_ONLY_SUCCESS);
    linker_fixture_program older({"-arch=sm_75", "-r"});
    require(nvJitLinkAddData(older.value, NVJITLINK_INPUT_PTX, newer.data(), newer.size(), "newer.ptx")
        == NVJITLINK_ERROR_ARCH_MISMATCH, "PTX link compilation target cannot be older than input target");
}
void ptx_import_declarations() {
    ptx_import_accept(ptx_import_header() + R"ptx(
.file 1 "owned_source.cu"
.visible .entry ptx_import_control(.param .b8 byte, .param .u16 halfword, .param .s32 word,
 .param .f32 scalar, .param .f64 wide, .param .align 16 .b8 record[32],
 .param .u64 .ptr .global .align 16 pointer) {
 .loc 1 7 0
 ret;
}
.section .debug_info { .b8 0, 1, 2, 3 }
)ptx");
    for (const auto& [parameters, reason] : std::array<std::pair<const char*, const char*>, 5>{{
        {".param .u64 p, .param .u64 p", "duplicate parameter"}, {".param .u32 .ptr p", "pointer annotations"},
        {".param .u64 .ptr .align 3 p", "alignment"}, {".param .b8 p[0]", "storage extent"},
        {".param .align 3 .b8 p[16]", "alignment"}}})
        ptx_import_reject(ptx_import_header() + ".entry ptx_import_rejected(" + parameters + ") { ret; }", reason);
}
void ptx_import_instructions() {
    ptx_import_accept(ptx_import_kernel(".reg .b64 %p; mov.u64 %p, 0; "
        "prefetch.global.L1 [%p]; prefetch.global.L2 [%p]; prefetch.L1 [%p]; prefetch.L2 [%p]; prefetchu.L1 [%p]; ret;",
        "ptx_import_control"));
    ptx_import_reject(ptx_import_kernel(".reg .b64 %p; prefetch.global.L3 [%p]; ret;"),
        "unsupported cache prefetch instruction", NVJITLINK_ERROR_UNRECOGNIZED_INPUT);
    ptx_import_reject(ptx_import_kernel(".reg .b64 %p; prefetch.global.L1 [%p], 1; ret;"), "operand count");
    ptx_import_reject(ptx_import_kernel(".reg .b32 %p; prefetch.global.L1 [%p]; ret;"), "width");
    for (const auto* instruction : {"ld.global.wb.u32 %r,[%p];", "st.global.ca.u32 [%p],%r;"})
        ptx_import_reject(ptx_import_kernel(std::string(".reg .b64 %p; .reg .u32 %r; ") + instruction + " ret;"),
            "unsupported memory instruction", NVJITLINK_ERROR_UNRECOGNIZED_INPUT);
    for (const auto& [body, reason] : std::array<std::pair<const char*, const char*>, 8>{{
        {".reg .b32 %r; .reg .b32 %r; ret;", "duplicate register"},
        {".reg .b32 %r; mov.u32 %unknown, 1; ret;", "undeclared register"},
        {".reg .b32 %r; add.u32 %r, 1; ret;", "operand count"},
        {".reg .b64 %r; add.u32 %r, %r, 1; ret;", "width"},
        {".reg .f32 %f; add.u32 %f, 1, 2; ret;", "register type"},
        {"bra missing; ret;", "undefined branch"}, {"label: ret; label: ret;", "duplicate label"},
        {"invented.u32; ret;", "unsupported instruction"}}})
        ptx_import_reject(ptx_import_kernel(body), reason,
            std::string_view(body).starts_with("invented") ? NVJITLINK_ERROR_UNRECOGNIZED_INPUT : NVJITLINK_ERROR_PTX_COMPILE);
    ptx_import_reject(ptx_import_header() + ".invented_directive 17;", "module declaration", NVJITLINK_ERROR_UNRECOGNIZED_INPUT);
}
void ptx_import_lexical() {
    const auto control = ptx_import_kernel("ret;", "ptx_import_control");
    ptx_import_accept(control); ptx_import_accept(control + '\0');
    ptx_import_accept(" /* owned leading comment */\n" + control + "// trailing comment\n");
    ptx_import_accept(control + "\ninvalid bytes beyond explicit extent", control.size());
    ptx_import_reject(control + '\0' + ptx_import_kernel(), "unrecognized character");
    ptx_import_reject(control + "trailing_garbage", "module declaration", NVJITLINK_ERROR_UNRECOGNIZED_INPUT);
    ptx_import_reject(control + "/* unterminated", "unterminated comment");
    ptx_import_reject(ptx_import_header() + ".file 1 \"unterminated\n", "string literal", NVJITLINK_ERROR_UNRECOGNIZED_INPUT);
    ptx_import_reject(ptx_import_header() + ".entry ptx_import_rejected() {", "unterminated");
}
void execute_ptx_import() {
    if (ptx_import_case == "versions") ptx_import_versions();
    else if (ptx_import_case == "targets") ptx_import_targets();
    else if (ptx_import_case == "declarations") ptx_import_declarations();
    else if (ptx_import_case == "instructions") ptx_import_instructions();
    else { require(ptx_import_case == "lexical", "known PTX import case"); ptx_import_lexical(); }
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::ptx_import_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_ptx_import);
}
