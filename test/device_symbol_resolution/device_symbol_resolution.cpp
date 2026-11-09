#include "test_framework.hpp"
#include "symbol_sources.hpp"

namespace cudavm {
std::string_view device_symbol_case = "weak";
using symbol_units = std::vector<std::vector<unsigned char>>;
symbol_units compile_symbol_sources(std::string_view profile, const std::vector<std::string>& sources) {
    symbol_units result;
    for (size_t i = 0; i < sources.size(); ++i) {
        const auto name = "symbols_" + std::string(profile) + "_" + std::to_string(i) + ".cu";
        compiler_fixture_program program(sources[i], name.c_str());
        const auto status = program.compile({"-arch=compute_75", "--std=c++17", "-dlto"});
        require(status == NVRTC_SUCCESS, name + ": " + nvrtcGetErrorString(status) + "; " + program.text(true));
        result.push_back(program.lto());
    }
    return result;
}
void add_symbol_units(const linker_fixture_program& linker, const symbol_units& units, bool reverse, bool strong = true) {
    for (size_t position = 0; position < units.size(); ++position) {
        const auto i = reverse ? units.size() - position - 1 : position;
        if (!strong && i == 1) continue;
        auto temporary = units[i];
        const auto name = "symbol_unit_" + std::to_string(i);
        linker.add(temporary, name.c_str());
        std::fill(temporary.begin(), temporary.end(), 0x53);
    }
}
template<class T> T symbol_read(CUdeviceptr address) {
    struct { std::array<unsigned char, 16> before; T value{}; std::array<unsigned char, 16> after; } output;
    output.before.fill(0xa9); output.after.fill(0x37);
    CONFORMANCE_CUDA(cuMemcpyDtoH(&output.value, address, sizeof(T)));
    require(std::all_of(output.before.begin(), output.before.end(), [](auto byte) { return byte == 0xa9; })
        && std::all_of(output.after.begin(), output.after.end(), [](auto byte) { return byte == 0x37; }), "device symbol read exact output guards");
    return output.value;
}
struct symbol_oracle_state {
    bool strong;
    uint32_t value{}, bias{}, counter{};
    std::array<uint32_t, 4> table{3, 17, 29, 43};
    CUdeviceptr value_address{}, bias_address{}, counter_address{}, values_address{}, pointer_address{}, word_address{};
    uint64_t function_pointer{};
};
symbol_oracle_state initialize_symbol_state(const loaded_program& program, bool strong) {
    symbol_oracle_state state{strong};
    if (device_symbol_case == "weak") {
        state.value_address = program.global("symbol_value", 4); state.bias_address = program.global("symbol_bias", 4);
        state.value = strong ? 71u : 17u; state.bias = strong ? 89u : 29u;
        require(symbol_read<uint32_t>(state.value_address) == state.value && symbol_read<uint32_t>(state.bias_address) == state.bias,
            "strong/weak storage initializers select the expected provider");
    } else if (device_symbol_case == "comdat") {
        state.counter_address = program.global("symbol_counter", 4);
        require(symbol_read<uint32_t>(state.counter_address) == 17, "coalesced inline storage initializer");
    } else if (device_symbol_case == "aliases") {
        state.pointer_address = program.global("symbol_alias_pointer", 8);
        state.function_pointer = symbol_read<uint64_t>(state.pointer_address);
        require(state.function_pointer != 0, "stored strong function alias is nonnull");
    } else {
        require(device_symbol_case == "relocations", "unknown device-symbol execution profile");
        state.values_address = program.global("symbol_values", 16);
        state.pointer_address = program.global("symbol_pointer", 8); state.word_address = program.global("symbol_address", 8);
        require(symbol_read<std::array<uint32_t, 4>>(state.values_address) == state.table, "separately defined relocation backing initializer");
        require(symbol_read<uint64_t>(state.pointer_address) == state.values_address + 8
            && symbol_read<uint64_t>(state.word_address) == state.values_address + 4, "cross-unit pointer and integer address relocations");
    }
    return state;
}
std::array<uint32_t, 8> device_symbol_oracle(uint32_t value, unsigned i, symbol_oracle_state& state) {
    if (device_symbol_case == "weak") {
        const uint32_t multiply = state.strong ? 7u : 3u, add = state.strong ? 11u : 5u;
        return {value * multiply + add, (value ^ i) * multiply + add + state.value + state.bias,
            state.value, state.bias, (value + i) * multiply + add,
            i * multiply + add + state.value + state.bias, value ^ i, state.value + state.bias};
    }
    if (device_symbol_case == "aliases") return {value * 13u + 17u, (value ^ i) * 13u + 17u,
        (value + i) * 13u + 17u, (value + i * 3u) * 13u + 17u,
        (value ^ 0x193759bdu) * 13u + 17u, i * 39u + 17u, value ^ i, 1u};
    if (device_symbol_case == "relocations") return {state.table[2] + value, state.table[1] ^ value,
        state.table[0] + i, state.table[3] ^ i, uint32_t(state.values_address + 8), uint32_t((state.values_address + 8) >> 32),
        uint32_t(state.values_address + 4), uint32_t((state.values_address + 4) >> 32)};
    const uint32_t initial = state.counter;
    const uint32_t after_left = initial + value + 7u;
    const uint32_t after_right = after_left + (value ^ i) + 10u;
    const uint32_t after_direct = after_right + i + 7u;
    state.counter = after_direct + i + 12u;
    return {initial, after_left, after_right, after_right, after_direct, 1u, after_direct, state.counter};
}
void verify_symbol_state(const symbol_oracle_state& state) {
    if (state.value_address) require(symbol_read<uint32_t>(state.value_address) == state.value
        && symbol_read<uint32_t>(state.bias_address) == state.bias, "strong/weak storage preserves host updates");
    if (state.counter_address) require(symbol_read<uint32_t>(state.counter_address) == state.counter, "COMDAT storage final serial state");
    if (state.values_address) {
        require(symbol_read<std::array<uint32_t, 4>>(state.values_address) == state.table, "relocation backing bytes remain unchanged");
        require(symbol_read<uint64_t>(state.pointer_address) == state.values_address + 8
            && symbol_read<uint64_t>(state.word_address) == state.values_address + 4, "relocation slots retain expected pointer bits");
    } else if (state.pointer_address) require(symbol_read<uint64_t>(state.pointer_address) == state.function_pointer, "function alias pointer slot remains unchanged");
}
void execute_symbol_program(const loaded_program& program, bool strong) {
    constexpr unsigned capacity = 137, fields = 8, guards = 13;
    constexpr uint32_t guard = 0x97531bad;
    constexpr std::array<uint32_t, 8> edges{0, 1, 0x7fffffff, 0x80000000, 0xffffffff, 0xfffffffe, 0x193759bd, 0xabcdef01};
    auto state = initialize_symbol_state(program, strong);
    std::vector<uint32_t> input(capacity + guards * 2, guard), preserved(input.size());
    std::vector<uint32_t> expected((capacity + 3) * fields + guards * 2, guard), actual(expected.size());
    device_buffer source(input.size() * 4), output(expected.size() * 4);
    auto from = source.value + guards * 4, to = output.value + guards * 4;
    const std::vector<unsigned> sizes = device_symbol_case == "comdat" ? std::vector<unsigned>{1} : std::vector<unsigned>{32, 61};
    for (unsigned phase = 0; phase < 2; ++phase) {
        if (state.value_address && phase) {
            state.value = 0xffffffe7u; state.bias = 0x193759bdu;
            CONFORMANCE_CUDA(cuMemcpyHtoD(state.value_address, &state.value, 4));
            CONFORMANCE_CUDA(cuMemcpyHtoD(state.bias_address, &state.bias, 4));
        }
        if (state.values_address && phase) {
            for (unsigned i = 0; i < state.table.size(); ++i) state.table[i] = 0xfffffff0u + i * 0x193759bdu;
            CONFORMANCE_CUDA(cuMemcpyHtoD(state.values_address, state.table.data(), sizeof(state.table)));
        }
        for (unsigned i = 0; i < capacity; ++i)
            input[guards + i] = i < edges.size() ? edges[(i + phase) % edges.size()] : (i * 0x7654321u) ^ (phase * 0x193759bdu);
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        for (const auto threads : sizes) for (unsigned count : {0u, 1u, 33u, capacity}) {
            if (state.counter_address) {
                state.counter = phase ? 0xffffffe7u : 17u;
                CONFORMANCE_CUDA(cuMemcpyHtoD(state.counter_address, &state.counter, 4));
            }
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
            void* arguments[]{&to, &from, &count};
            const auto blocks = state.counter_address ? 1u : (capacity + threads - 1) / threads;
            CONFORMANCE_CUDA(cuLaunchKernel(program.function("symbol_results"), blocks, 1, 1, threads, 1, 1, 0, nullptr, arguments, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
            for (unsigned i = 0; i < count; ++i) {
                const auto row = device_symbol_oracle(input[guards + i], i, state);
                std::copy(row.begin(), row.end(), expected.begin() + guards + i * fields);
            }
            expect_words(actual, expected, "device symbol resolution values, identities, addresses and complete output guards");
            CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
            expect_words(preserved, input, "device symbols preserve input and guards");
            verify_symbol_state(state);
        }
    }
}
void execute_device_symbol_resolution() {
    const auto units = compile_symbol_sources(device_symbol_case, symbol_sources(device_symbol_case));
    const std::vector<bool> providers = device_symbol_case == "weak" ? std::vector<bool>{false, true} : std::vector<bool>{true};
    for (bool strong : providers) for (bool reverse : {false, true}) {
        std::vector<unsigned char> image;
        {
            linker_fixture_program linker;
            add_symbol_units(linker, units, reverse, strong); linker.complete(); image = linker.image();
        }
        for (bool library : {false, true}) {
            loaded_program program(image.data(), library);
            execute_symbol_program(program, strong); CONFORMANCE_CUDA(cuCtxSynchronize());
        }
    }
}
void symbol_source_check() {
    for (auto profile : {"weak", "comdat", "aliases", "relocations"}) compile_symbol_sources(profile, symbol_sources(profile));
}
void symbol_link_failures() {
    const std::vector<std::string> sources{
        "extern \"C\" __device__ unsigned symbol_required(unsigned);\nextern \"C\" __global__ void symbol_results(unsigned* out) { *out = symbol_required(17); }\n",
        "extern \"C\" __device__ unsigned symbol_required(unsigned value) { return value + 3u; }\n",
        "extern \"C\" __device__ unsigned symbol_required(unsigned value) { return value + 9u; }\n"};
    const auto units = compile_symbol_sources("failures", sources), preserved = units;
    linker_fixture_program valid;
    valid.add(units[0], "valid_root"); valid.add(units[1], "valid_definition"); valid.complete();
    const auto snapshot = valid.image();
    for (bool duplicate : {false, true}) for (bool reverse : {false, true}) {
        linker_fixture_program invalid;
        const symbol_units selected = duplicate ? units : symbol_units{units[0]};
        add_symbol_units(invalid, selected, reverse);
        require(nvJitLinkComplete(invalid.value) == NVJITLINK_ERROR_FINALIZE, "invalid device symbol closure fails final linking");
        size_t size{}; invalid.check(nvJitLinkGetErrorLogSize(invalid.value, &size), "failure log size");
        require(size > 1 && size < 64 * 1024 * 1024, "symbol failure has bounded diagnostic text");
        std::vector<char> log(size); invalid.check(nvJitLinkGetErrorLog(invalid.value, log.data()), "failure log");
        require(std::string_view(log.data(), size).find("symbol_required") != std::string_view::npos, "symbol failure identifies unresolved/duplicate name");
        require(valid.image() == snapshot && units == preserved, "failed link preserves another completed handle and every caller input");
    }
}
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "sources")
        return cudavm::run_host_conformance(argc, argv, cudavm::symbol_source_check);
    if (argc == 2 && std::string_view(argv[1]) == "failures")
        return cudavm::run_host_conformance(argc, argv, cudavm::symbol_link_failures);
    if (argc == 3) { cudavm::device_symbol_case = argv[2]; argc = 2; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_device_symbol_resolution, 0, cudavm::probe_execution);
}
