#include "conformance.hpp"
#include <nvFatbin.h>
#include <nvPTXCompiler.h>
#include <chrono>
#include <filesystem>
#include <fstream>

namespace cudavm {
using jit_composition_bytes = std::vector<unsigned char>;
std::string_view jit_composition_case = "ptx";
constexpr uint32_t jit_composition_bias = 0x13579bdf;

template<class Query, class Get> jit_composition_bytes jit_composition_result(Query query, Get get) {
    struct output { size_t before = 0x193759bd, value{}, after = 0xabcdef01; } first_size, second_size;
    query(&first_size.value); query(&second_size.value);
    require(first_size.before == 0x193759bd && first_size.after == 0xabcdef01
        && second_size.before == 0x193759bd && second_size.after == 0xabcdef01,
        "composition result size guards");
    const size_t count = first_size.value;
    require(count > 0 && count < 64 * 1024 * 1024 && count == second_size.value,
        "composition result size is stable, bounded and nonempty");
    jit_composition_bytes first(count + 32, 0x35), second(count + 32, 0x79);
    get(first.data() + 16); get(second.data() + 16);
    for (size_t i = 0; i < first.size(); ++i) {
        if (i < 16 || i >= count + 16) require(first[i] == 0x35 && second[i] == 0x79,
            "composition Get exact extent guards");
        else require(first[i] == second[i], "composition Get writes every byte and repeat retrieval agrees");
    }
    return {first.begin() + 16, first.end() - 16};
}

std::array<std::string, 3> jit_composition_ptx() {
    const std::string prefix = ".version 6.5\n.target sm_75\n.address_size 64\n";
    return {prefix + R"ptx(
.extern .func (.param .b32 result) composition_transform(.param .b32 input);
.extern .global .align 4 .u32 composition_bias;
.visible .entry composition_results(.param .u64 out, .param .u64 in, .param .u32 count) {
 .reg .b32 %i,%block,%width,%count,%value,%bias,%a,%b,%c,%d;
 .reg .b64 %out,%in; .reg .pred %p;
 .param .b32 argument, result;
 mov.u32 %i,%tid.x; mov.u32 %block,%ctaid.x; mov.u32 %width,%ntid.x;
 mad.lo.u32 %i,%block,%width,%i; ld.param.u32 %count,[count];
 setp.ge.u32 %p,%i,%count; @%p ret;
 ld.param.u64 %out,[out]; ld.param.u64 %in,[in];
 mad.wide.u32 %in,%i,4,%in; ld.global.u32 %value,[%in];
 ld.global.u32 %bias,[composition_bias];
 st.param.b32 [argument],%value; call.uni (result),composition_transform,(argument);
 ld.param.b32 %a,[result]; xor.b32 %b,%value,%bias;
 add.u32 %c,%value,%i; st.param.b32 [argument],%c;
 call.uni (result),composition_transform,(argument); ld.param.b32 %c,[result];
 add.u32 %c,%c,%bias; mad.lo.u32 %d,%i,7,%value;
 mad.wide.u32 %out,%i,16,%out;
 st.global.u32 [%out],%a; st.global.u32 [%out+4],%b;
 st.global.u32 [%out+8],%c; st.global.u32 [%out+12],%d; ret;
}
)ptx", prefix + R"ptx(
.visible .func (.param .b32 result) composition_transform(.param .b32 input) {
 .reg .b32 %value; ld.param.b32 %value,[input];
 mad.lo.u32 %value,%value,9,23; st.param.b32 [result],%value; ret;
}
)ptx", prefix + ".visible .global .align 4 .u32 composition_bias = 0x13579bdf;\n"};
}

std::array<jit_composition_bytes, 3> jit_composition_source() {
    const std::array<std::string, 3> sources{R"cuda(
extern "C" __device__ unsigned composition_transform(unsigned);
extern "C" __device__ unsigned composition_bias;
extern "C" __global__ void composition_results(unsigned* out, const unsigned* in, unsigned count) {
 const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i >= count) return;
 const unsigned value = in[i], bias = composition_bias; out += i * 4;
 out[0] = composition_transform(value); out[1] = value ^ bias;
 out[2] = composition_transform(value + i) + bias; out[3] = i * 7u + value;
}
)cuda", R"cuda(
extern "C" __device__ unsigned composition_transform(unsigned value) { return value * 9u + 23u; }
)cuda", R"cuda(
extern "C" { __device__ unsigned composition_bias = 0x13579bdf; }
)cuda"};
    std::array<jit_composition_bytes, 3> units;
    for (size_t i = 0; i < units.size(); ++i) {
        compiler_fixture_program program(sources[i], "composition.cu");
        const auto status = program.compile({"-arch=compute_75", "--std=c++17", "-dlto"});
        require(status == NVRTC_SUCCESS, "composition source unit: " + program.text(true));
        units[i] = program.lto();
    }
    return units;
}

jit_composition_bytes jit_composition_compiled(const std::string& source) {
    const auto check = [](nvPTXCompileResult result) {
        require(result == NVPTXCOMPILE_SUCCESS, "composition PTX Compiler status " + std::to_string(result));
    };
    struct program { nvPTXCompilerHandle value{}; ~program() { if (value) nvPTXCompilerDestroy(&value); } } owner;
    check(nvPTXCompilerCreate(&owner.value, source.size(), source.data()));
    const char* options[]{"-arch=sm_75", "-c", "-O0"};
    check(nvPTXCompilerCompile(owner.value, 3, options));
    return jit_composition_result([&](size_t* size) { check(nvPTXCompilerGetCompiledProgramSize(owner.value, size)); },
        [&](unsigned char* data) { check(nvPTXCompilerGetCompiledProgram(owner.value, data)); });
}
jit_composition_bytes jit_composition_pack(const std::string& source) {
    const auto check = [](nvFatbinResult result) {
        require(result == NVFATBIN_SUCCESS, "composition packager status " + std::to_string(result));
    };
    struct container { nvFatbinHandle value{}; ~container() { if (value) nvFatbinDestroy(&value); } } owner;
    const char* options[]{"-compress=false"}; check(nvFatbinCreate(&owner.value, options, 1));
    check(nvFatbinAddPTX(owner.value, source.data(), source.size(), "75", "composition_storage", ""));
    return jit_composition_result([&](size_t* size) { check(nvFatbinSize(owner.value, size)); },
        [&](unsigned char* data) { check(nvFatbinGet(owner.value, data)); });
}
struct jit_composition_file {
    std::filesystem::path root, path;
    explicit jit_composition_file(const jit_composition_bytes& bytes) {
        root = std::filesystem::current_path() / ("nvjitlink_composition_"
            + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        require(std::filesystem::create_directory(root), "create unique composition file directory");
        path = root / "unit.ptx";
        try {
            std::ofstream stream(path, std::ios::binary);
            stream.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size())); stream.close();
            require(bool(stream), "write exact composition input bytes");
        } catch (...) { cleanup(); throw; }
    }
    ~jit_composition_file() { cleanup(); }
    void cleanup() const {
        std::error_code error; std::filesystem::remove(path, error); std::filesystem::remove(root, error);
    }
    jit_composition_file(const jit_composition_file&) = delete;
};
jit_composition_bytes jit_composition_output(const linker_fixture_program& linker, unsigned kind) {
    return jit_composition_result([&](size_t* size) {
        linker.check(kind == 0 ? nvJitLinkGetLinkedCubinSize(linker.value, size)
            : kind == 1 ? nvJitLinkGetLinkedLTOIRSize(linker.value, size)
            : nvJitLinkGetLinkedPtxSize(linker.value, size), "composition output size");
    }, [&](unsigned char* data) {
        linker.check(kind == 0 ? nvJitLinkGetLinkedCubin(linker.value, data)
            : kind == 1 ? nvJitLinkGetLinkedLTOIR(linker.value, data)
            : nvJitLinkGetLinkedPtx(linker.value, reinterpret_cast<char*>(data)), "composition output retrieval");
    });
}
jit_composition_bytes jit_composition_link(std::string_view profile, bool reverse, bool automatic, bool text = false) {
    const auto sources = jit_composition_ptx();
    std::array<jit_composition_bytes, 3> units;
    std::array<nvJitLinkInputType, 3> kinds{NVJITLINK_INPUT_PTX, NVJITLINK_INPUT_PTX, NVJITLINK_INPUT_PTX};
    for (size_t i = 0; i < units.size(); ++i) units[i] = {sources[i].begin(), sources[i].end()};
    if (profile == "mixed") {
        units[1] = jit_composition_compiled(sources[1]); kinds[1] = NVJITLINK_INPUT_CUBIN;
        units[2] = jit_composition_pack(sources[2]); kinds[2] = NVJITLINK_INPUT_FATBIN;
    } else if (profile == "retained") {
        linker_fixture_program producer({"-arch=sm_75", "-r", "-lto"});
        producer.add(units[0], "unresolved_root", NVJITLINK_INPUT_PTX); producer.complete();
        // Both public binary output routes must remain suitable for later linking.
        units[0] = jit_composition_output(producer, reverse ? 1 : 0);
        kinds[0] = reverse ? NVJITLINK_INPUT_LTOIR : NVJITLINK_INPUT_CUBIN;
    } else if (profile == "source") {
        units = jit_composition_source(); kinds.fill(NVJITLINK_INPUT_LTOIR);
    } else require(profile == "ptx" || profile == "files", "known composition profile");
    std::vector<std::string> options{"-arch=sm_75", "-lto", "-O0"};
    if (text) options.push_back("-ptx");
    linker_fixture_program linker(options);
    for (size_t position = 0; position < units.size(); ++position) {
        const size_t i = reverse ? units.size() - 1 - position : position;
        const auto kind = automatic ? NVJITLINK_INPUT_ANY : kinds[i];
        if (profile == "files") {
            jit_composition_file file(units[i]); auto path = file.path.string();
            linker.check(nvJitLinkAddFile(linker.value, kind, path.c_str()), "composition AddFile");
            std::fill(path.begin(), path.end(), '!');
            require(std::filesystem::remove(file.path), "remove accepted input before completion");
        } else {
            std::string label = "composition_input";
            linker.add(units[i], label.c_str(), kind); std::fill(label.begin(), label.end(), '!');
        }
        std::fill(units[i].begin(), units[i].end(), 0x57);
    }
    linker.complete(); auto result = jit_composition_output(linker, text ? 2 : 0);
    if (text) {
        require(result.back() == 0 && std::find(result.begin(), result.end() - 1, 0) == result.end() - 1,
            "composition PTX has exact final NUL");
        require(std::string_view(reinterpret_cast<const char*>(result.data()), result.size() - 1)
            .find("composition_results") != std::string_view::npos, "linked text retains root kernel");
    }
    return result;
}
void jit_composition_execute_image(const loaded_program& program) {
    constexpr unsigned capacity = 73, guards = 9, fields = 4;
    constexpr uint32_t guard = 0xa95b371d;
    const auto bias_address = program.global("composition_bias", sizeof(uint32_t));
    std::array<uint32_t, 3> bias_words{guard, guard, guard};
    CONFORMANCE_CUDA(cuMemcpyDtoH(bias_words.data() + 1, bias_address, sizeof(uint32_t)));
    require(bias_words == std::array<uint32_t, 3>{guard, jit_composition_bias, guard}, "linked storage initializer and host guards");
    std::vector<uint32_t> input(capacity + 2 * guards, guard), preserved(input.size());
    std::vector<uint32_t> expected(capacity * fields + 2 * guards, guard), actual(expected.size());
    device_buffer source(input.size() * 4), output(expected.size() * 4);
    auto in = source.value + guards * 4, out = output.value + guards * 4;
    for (unsigned phase = 0; phase < 2; ++phase) {
        const uint32_t bias = phase ? 0xfffffff1u : jit_composition_bias;
        CONFORMANCE_CUDA(cuMemcpyHtoD(bias_address, &bias, sizeof(bias)));
        constexpr std::array<uint32_t, 8> edges{0, 1, 0x7fffffff, 0x80000000, 0xffffffff, 0xfffffffe, 0x193759bd, 0xabcdef01};
        for (unsigned i = 0; i < capacity; ++i) input[guards + i] = i < edges.size()
            ? edges[(i + phase) % edges.size()] : i * 0x7654321u + phase * 0x193759bdu;
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        for (unsigned threads : {32u, 47u}) for (unsigned count : {0u, 1u, 33u, capacity}) {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
            void* arguments[]{&out, &in, &count};
            CONFORMANCE_CUDA(cuLaunchKernel(program.function("composition_results"), (capacity + threads - 1) / threads,
                1, 1, threads, 1, 1, 0, nullptr, arguments, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
            for (unsigned i = 0; i < count; ++i) {
                const uint32_t value = input[guards + i];
                const std::array<uint32_t, 4> row{value * 9u + 23u, value ^ bias,
                    (value + i) * 9u + 23u + bias, i * 7u + value};
                std::copy(row.begin(), row.end(), expected.begin() + guards + i * fields);
            }
            expect_words(actual, expected, "composition values, inactive capacity and output guards");
            CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
            expect_words(preserved, input, "composition input and guards preserved");
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(bias_words.data() + 1, bias_address, sizeof(uint32_t)));
        require(bias_words == std::array<uint32_t, 3>{guard, bias, guard}, "linked input storage preserved after execution");
    }
}
void execute_jit_composition() {
    for (bool reverse : {false, true}) for (bool automatic : {false, true}) {
        const auto image = jit_composition_link(jit_composition_case, reverse, automatic);
        for (bool library : {false, true}) {
            loaded_program program(image.data(), library); jit_composition_execute_image(program);
            CONFORMANCE_CUDA(cuCtxSynchronize());
        }
    }
}
void jit_composition_inputs() {
    for (auto profile : {"ptx", "files", "mixed", "retained", "source"})
        for (bool reverse : {false, true}) jit_composition_link(profile, reverse, reverse, true);
}
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "inputs")
        return cudavm::run_host_conformance(argc, argv, cudavm::jit_composition_inputs);
    if (argc == 3) { cudavm::jit_composition_case = argv[2]; argc = 2; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_jit_composition, 0, cudavm::probe_execution);
}
