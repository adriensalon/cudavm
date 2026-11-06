#include "conformance.hpp"
#include <utility>

namespace cudavm {
std::string_view separate_compilation_case = "functions";
std::array<std::string, 3> separate_compilation_sources(std::string_view profile) {
    if (profile == "functions") {
        const std::string header = R"cuda(
extern "C" __device__ unsigned separate_leaf(unsigned);
extern "C" __device__ unsigned separate_middle(unsigned);
)cuda";
        return {header + R"cuda(
extern "C" __device__ unsigned separate_leaf(unsigned value) { return value * 7u + 11u; }
)cuda", header + R"cuda(
extern "C" __device__ unsigned separate_middle(unsigned value) { return separate_leaf(value ^ 0x13579bdfu) + 17u; }
)cuda", header + R"cuda(
extern "C" __global__ void separate_results(unsigned* out, const unsigned* in, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned value = in[i]; out += i * 4;
    out[0] = separate_leaf(value); out[1] = separate_middle(value);
    out[2] = separate_leaf(separate_middle(value + i));
    out[3] = separate_middle(separate_leaf(value ^ i));
}
)cuda"};
    }
    if (profile == "globals") {
        const std::string header = R"cuda(
extern "C" __device__ unsigned separate_words[4];
extern "C" __constant__ unsigned separate_bias;
extern "C" __device__ unsigned separate_lookup(unsigned);
)cuda";
        return {R"cuda(
extern "C" {
__device__ unsigned separate_words[4] = {3, 17, 29, 43};
__constant__ unsigned separate_bias = 59;
}
)cuda", header + R"cuda(
extern "C" __device__ unsigned separate_lookup(unsigned value) {
    return value + separate_words[value & 3] + separate_bias;
}
)cuda", header + R"cuda(
extern "C" __global__ void separate_results(unsigned* out, const unsigned* in, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned value = in[i]; out += i * 4;
    out[0] = separate_words[i & 3]; out[1] = separate_lookup(value);
    out[2] = separate_bias; out[3] = separate_lookup(i) ^ separate_words[(i + 1) & 3];
}
)cuda"};
    }
    require(profile == "aggregate", "unknown separate-compilation source profile");
    const std::string header = R"cuda(
struct alignas(16) separate_record { unsigned long long wide; unsigned tag, salt; };
static_assert(sizeof(separate_record) == 16 && alignof(separate_record) == 16, "record device ABI");
extern "C" __device__ unsigned separate_twist(unsigned);
extern "C" __device__ separate_record separate_transform(separate_record, unsigned);
)cuda";
    return {header + R"cuda(
extern "C" __device__ unsigned separate_twist(unsigned value) { return value * 3u + 5u; }
)cuda", header + R"cuda(
extern "C" __device__ separate_record separate_transform(separate_record value, unsigned delta) {
    value.wide += (static_cast<unsigned long long>(delta) << 32) | separate_twist(value.tag);
    value.tag ^= delta; value.salt += delta * 11u; return value;
}
)cuda", header + R"cuda(
extern "C" __global__ void separate_results(unsigned* out, const unsigned* in, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned value = in[i]; out += i * 4;
    separate_record record{(static_cast<unsigned long long>(value) << 32) | (value ^ 0xabcdef01u), value + 7u, value ^ 0x193759bdu};
    record = separate_transform(record, i * 13u + 17u);
    out[0] = unsigned(record.wide); out[1] = unsigned(record.wide >> 32);
    out[2] = record.tag; out[3] = record.salt;
}
)cuda"};
}
std::array<std::vector<unsigned char>, 3> separate_compile_units(std::string_view profile, const std::array<std::string, 3>& sources) {
    std::array<std::vector<unsigned char>, 3> result;
    for (size_t i = 0; i < sources.size(); ++i) {
        const auto name = "separate_" + std::string(profile) + "_" + std::to_string(i) + ".cu";
        compiler_fixture_program program(sources[i], name.c_str());
        const auto status = program.compile({"-arch=compute_75", "--std=c++17", "-dlto"});
        require(status == NVRTC_SUCCESS, name + ": " + nvrtcGetErrorString(status) + "; " + program.text(true));
        result[i] = program.lto();
    }
    return result;
}
std::array<std::vector<unsigned char>, 3> separate_compile_units(std::string_view profile) {
    return separate_compile_units(profile, separate_compilation_sources(profile));
}
std::vector<unsigned char> separate_link_units(const std::array<std::vector<unsigned char>, 3>& units, bool reverse) {
    linker_fixture_program linker;
    for (unsigned position = 0; position < units.size(); ++position) {
        const unsigned i = reverse ? unsigned(units.size()) - position - 1 : position;
        auto temporary = units[i];
        const auto name = "separate_unit_" + std::to_string(i);
        linker.add(temporary, name.c_str());
        std::fill(temporary.begin(), temporary.end(), 0x35);
    }
    linker.complete();
    return linker.image();
}
std::array<uint32_t, 4> separate_compilation_oracle(std::string_view profile, uint32_t value, unsigned i,
    const std::array<uint32_t, 4>& table, uint32_t bias) {
    if (profile == "functions") return {value * 7u + 11u, (value ^ 0x13579bdfu) * 7u + 28u,
        (((value + i) ^ 0x13579bdfu) * 7u + 28u) * 7u + 11u,
        (((value ^ i) * 7u + 11u) ^ 0x13579bdfu) * 7u + 28u};
    if (profile == "globals") return {table[i & 3], value + table[value & 3] + bias, bias,
        (i + table[i & 3] + bias) ^ table[(i + 1) & 3]};
    const uint32_t delta = i * 13u + 17u;
    const uint64_t wide = (uint64_t(value) << 32 | (value ^ 0xabcdef01u))
        + ((uint64_t(delta) << 32) | ((value + 7u) * 3u + 5u));
    return {uint32_t(wide), uint32_t(wide >> 32), (value + 7u) ^ delta, (value ^ 0x193759bdu) + delta * 11u};
}
void separate_execute_program(const loaded_program& program) {
    constexpr unsigned capacity = 137, fields = 4, guards = 11;
    constexpr uint32_t guard = 0xa95b371d;
    constexpr std::array<uint32_t, 8> edges{0, 1, 0x7fffffff, 0x80000000, 0xffffffff, 0xfffffffe, 0x193759bd, 0xabcdef01};
    std::array<uint32_t, 4> table{3, 17, 29, 43}; uint32_t bias = 59;
    CUdeviceptr global_words{}, global_bias{};
    if (separate_compilation_case == "globals") {
        global_words = program.global("separate_words", sizeof(table)); global_bias = program.global("separate_bias", sizeof(bias));
        std::array<uint32_t, 8> actual; actual.fill(guard);
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data() + 2, global_words, sizeof(table)));
        require(actual[0] == guard && actual[1] == guard && actual[6] == guard && actual[7] == guard
            && std::equal(table.begin(), table.end(), actual.begin() + 2), "separately defined global initializer and host guards");
        uint32_t initial_bias{}; CONFORMANCE_CUDA(cuMemcpyDtoH(&initial_bias, global_bias, sizeof(initial_bias)));
        require(initial_bias == bias, "separately defined constant initializer");
    }
    std::vector<uint32_t> input(capacity + guards * 2, guard), preserved(input.size());
    std::vector<uint32_t> expected((capacity + 3) * fields + guards * 2, guard), actual(expected.size());
    device_buffer source(input.size() * 4), output(expected.size() * 4);
    auto from = source.value + guards * 4, to = output.value + guards * 4;
    for (unsigned phase = 0; phase < 2; ++phase) {
        if (global_words) {
            for (unsigned i = 0; i < table.size(); ++i) table[i] = (0x193759bdu * (i + 1)) ^ (phase * 0xabcdef01u);
            bias = phase ? 0xffffffefu : 71u;
            CONFORMANCE_CUDA(cuMemcpyHtoD(global_words, table.data(), sizeof(table)));
            CONFORMANCE_CUDA(cuMemcpyHtoD(global_bias, &bias, sizeof(bias)));
        }
        for (unsigned i = 0; i < capacity; ++i)
            input[guards + i] = i < edges.size() ? edges[(i + phase) % edges.size()] : i * 0x7654321u + phase * 0x193759bdu;
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        for (unsigned threads : {32u, 61u}) for (unsigned count : {0u, 1u, 33u, capacity}) {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
            void* arguments[]{&to, &from, &count};
            CONFORMANCE_CUDA(cuLaunchKernel(program.function("separate_results"), (capacity + threads - 1) / threads, 1, 1,
                threads, 1, 1, 0, nullptr, arguments, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
            for (unsigned i = 0; i < count; ++i) {
                const auto row = separate_compilation_oracle(separate_compilation_case, input[guards + i], i, table, bias);
                std::copy(row.begin(), row.end(), expected.begin() + guards + i * fields);
            }
            expect_words(actual, expected, "separate compilation linked values and complete output guards");
            CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
            expect_words(preserved, input, "separate compilation preserves input and guards");
        }
    }
}
void execute_separate_compilation() {
    const auto units = separate_compile_units(separate_compilation_case);
    for (bool reverse : {false, true}) {
        const auto image = separate_link_units(units, reverse);
        for (bool library : {false, true}) {
            loaded_program program(image.data(), library);
            separate_execute_program(program);
            CONFORMANCE_CUDA(cuCtxSynchronize());
        }
    }
}
void separate_compilation_source_check() {
    for (auto profile : {"functions", "globals", "aggregate"}) separate_compile_units(profile);
}
void separate_compilation_abi_validation() {
    auto sources = separate_compilation_sources("aggregate");
    const std::string original = "unsigned tag, salt;";
    const auto position = sources[1].find(original);
    require(position != std::string::npos, "aggregate rejection fixture owns its record definition");
    // The size/alignment stay identical; a field's ABI type changes only in the
    // defining TU. Matching allocation size must not make the call compatible.
    sources[1].replace(position, original.size(), "unsigned short tag, padding; unsigned salt;");
    const auto units = separate_compile_units("aggregate_mismatch", sources);
    for (bool reverse : {false, true}) {
        linker_fixture_program linker;
        for (unsigned position = 0; position < units.size(); ++position) {
            const unsigned i = reverse ? unsigned(units.size()) - position - 1 : position;
            linker.add(units[i], "aggregate_abi_mismatch");
        }
        require(nvJitLinkComplete(linker.value) == NVJITLINK_ERROR_FINALIZE, "incompatible aggregate fields fail final device linking in either input order");
        size_t size{}; linker.check(nvJitLinkGetErrorLogSize(linker.value, &size), "ABI rejection log size");
        require(size > 1 && size < 64 * 1024 * 1024, "aggregate ABI rejection has bounded diagnostic text");
        std::vector<char> log(size); linker.check(nvJitLinkGetErrorLog(linker.value, log.data()), "ABI rejection log");
        require(std::string_view(log.data(), size).find("separate_transform") != std::string_view::npos,
            "aggregate ABI rejection identifies the incompatible function");
    }
}
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "sources")
        return cudavm::run_host_conformance(argc, argv, cudavm::separate_compilation_source_check);
    if (argc == 2 && std::string_view(argv[1]) == "abi_validation")
        return cudavm::run_host_conformance(argc, argv, cudavm::separate_compilation_abi_validation);
    if (argc == 3) { cudavm::separate_compilation_case = argv[2]; argc = 2; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_separate_compilation, 0, cudavm::probe_execution);
}
