#include "test_framework.hpp"
#include "lto_fixture.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>

namespace cudavm {
using lto_bytes = std::vector<unsigned char>;
std::string_view lto_case = "data";
std::array<lto_bytes, 2> lto_inputs() {
    return {lto_bytes(std::begin(lto_kernel_fixture), std::end(lto_kernel_fixture)),
        lto_bytes(std::begin(lto_helper_fixture), std::end(lto_helper_fixture))};
}
struct lto_input_file {
    std::filesystem::path root, path;
    explicit lto_input_file(const lto_bytes& bytes) {
        root = std::filesystem::current_path() / ("cudavm_lto_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        require(std::filesystem::create_directory(root), "create owned LTO fixture directory");
        path = root / "input.ltoir";
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
        require(bool(file), "write genuine LTO fixture file");
    }
    ~lto_input_file() {
        std::error_code error; std::filesystem::remove(path, error); std::filesystem::remove(root, error);
    }
};
lto_bytes link_lto_inputs(std::string_view route, bool reverse) {
    linker_fixture_program program({"-arch=sm_75", "-lto", "-r"});
    auto inputs = lto_inputs();
    for (unsigned ordinal = 0; ordinal < 2; ++ordinal) {
        const auto index = reverse ? 1 - ordinal : ordinal;
        auto& bytes = inputs[index];
        if (route == "files") {
            lto_input_file file(bytes);
            program.check(nvJitLinkAddFile(program.value, NVJITLINK_INPUT_LTOIR, file.path.string().c_str()), "add genuine LTO file");
            // The public call must own the imported IR before the file is changed
            // and removed; destruction also removes its empty owned directory.
            { std::ofstream changed(file.path, std::ios::binary | std::ios::trunc); changed << "changed caller file"; }
        } else program.add(bytes, index ? "genuine_helper.ltoir" : "genuine_kernel.ltoir",
            route == "automatic" ? NVJITLINK_INPUT_ANY : NVJITLINK_INPUT_LTOIR);
        std::fill(bytes.begin(), bytes.end(), 0x97);
    }
    program.complete(); return program.image();
}
void lto_results_case(const loaded_program& program) {
    constexpr unsigned blocks = 3, guards = 16;
    constexpr uint32_t guard = 0x8c5917a3;
    constexpr uint32_t values[]{0, 1, 0xffffffffu, 0x80000000u, 0x7fffffffu, 0x12345678u};
    for (unsigned threads : {1u, 33u, 64u}) {
        const unsigned capacity = blocks * threads;
        std::vector<uint32_t> input(guards * 2 + capacity, guard), preserved(input.size());
        for (unsigned i = 0; i < capacity; ++i) input[guards + i] = values[i % std::size(values)] ^ (i * 0x10001u);
        std::vector<uint32_t> initial(guards * 2 + capacity + 1, guard), actual(initial.size());
        device_buffer source(input.size() * 4), destination(initial.size() * 4);
        TEST_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        auto in = source.value + guards * 4, out = destination.value + guards * 4;
        for (unsigned count : {0u, 1u, capacity - 1, capacity, capacity + 7}) {
            auto expected = initial;
            for (unsigned i = 0; i < std::min(count, capacity); ++i)
                expected[guards + i] = uint32_t(uint64_t(input[guards + i]) * 9 + 37);
            TEST_CUDA(cuMemcpyHtoD(destination.value, initial.data(), initial.size() * 4));
            void* arguments[]{&out, &in, &count};
            TEST_CUDA(cuLaunchKernel(program.function("lto_results"), blocks, 1, 1, threads, 1, 1, 0, nullptr, arguments, nullptr));
            TEST_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
            expect_words(actual, expected, "genuine NVIDIA LTO cross-unit unsigned results, guards and inactive records");
            TEST_CUDA(cuMemcpyDtoH(preserved.data(), source.value, input.size() * 4));
            expect_words(preserved, input, "genuine NVIDIA LTO preserves input allocation and guards");
        }
    }
}
void execute_lto_ir_import() {
    require(lto_case == "data" || lto_case == "files" || lto_case == "automatic", "unknown LTO import route");
    for (bool reverse : {false, true}) {
        // The compiler handle and caller inputs die before either public loader
        // sees its owned copy of the completed result.
        const auto image = link_lto_inputs(lto_case, reverse);
        for (bool library : {false, true}) {
            loaded_program program(image.data(), library); lto_results_case(program);
        }
    }
}
void lto_put(lto_bytes& bytes, unsigned offset, uint32_t value, unsigned width = 4) {
    for (unsigned i = 0; i < width; ++i) bytes[offset + i] = static_cast<unsigned char>(value >> (i * 8));
}
void lto_reject(linker_fixture_program& program, const lto_bytes& bytes, size_t extent,
    nvJitLinkResult expected = NVJITLINK_ERROR_INVALID_INPUT, nvJitLinkInputType kind = NVJITLINK_INPUT_LTOIR) {
    const auto status = nvJitLinkAddData(program.value, kind, bytes.data(), extent, "owned_bad_lto_input");
    require(status == expected, "genuine LTO malformed input status expected=" + std::to_string(expected) + " actual=" + std::to_string(status));
    if (extent) {
        const auto log = compiler_fixture_bytes([&](size_t* size) { program.check(nvJitLinkGetErrorLogSize(program.value, size), "LTO error log size"); },
            [&](unsigned char* out) { program.check(nvJitLinkGetErrorLog(program.value, reinterpret_cast<char*>(out)), "LTO error log"); });
        require(log.back() == 0 && log.size() > 1, "malformed LTO input retains a terminated diagnostic");
    }
    struct { size_t before = 0x193759bd, value = 17, after = 0xabcdef01; } result;
    require(nvJitLinkGetLinkedPtxSize(program.value, &result.value) == NVJITLINK_ERROR_INTERNAL,
        "failed LTO addition publishes no completed result");
    require(result.before == 0x193759bd && result.after == 0xabcdef01, "failed LTO result query guards");
}
void lto_format_case() {
    auto inputs = lto_inputs();
    linker_fixture_program program({"-arch=sm_75", "-lto", "-ptx"});
    program.add(inputs[1], "accepted_helper.ltoir");
    for (size_t extent = 0; extent < 44; ++extent) lto_reject(program, inputs[0], extent);
    for (unsigned offset : {0u, 4u, 14u, 16u, 20u, 24u, 26u, 28u, 30u, 34u, 38u, 40u, 44u}) {
        auto bytes = inputs[0]; nvJitLinkResult expected = NVJITLINK_ERROR_INVALID_INPUT;
        switch (offset) {
        case 0: bytes[0] ^= 1; break;
        case 4: bytes[4] = 2; expected = NVJITLINK_ERROR_UNRECOGNIZED_INPUT; break;
        case 14: lto_put(bytes, 14, 2, 2); expected = NVJITLINK_ERROR_UNRECOGNIZED_INPUT; break;
        case 16: lto_put(bytes, 16, 23); break;
        case 20: lto_put(bytes, 20, uint32_t(bytes.size() + 4)); break;
        case 24: lto_put(bytes, 24, 3, 2); expected = NVJITLINK_ERROR_UNRECOGNIZED_INPUT; break;
        case 26: lto_put(bytes, 26, 751, 2); break;
        case 28: lto_put(bytes, 28, 1, 2); break;
        case 30: lto_put(bytes, 30, 3, 2); expected = NVJITLINK_ERROR_UNRECOGNIZED_INPUT; break;
        case 34: lto_put(bytes, 34, 8, 2); break;
        case 38: lto_put(bytes, 38, 1, 2); break;
        case 40: lto_put(bytes, 40, 0); break;
        default: bytes[44] ^= 1; break;
        }
        lto_reject(program, bytes, bytes.size(), expected);
    }
    lto_reject(program, inputs[0], inputs[0].size() - 1);
    for (auto kind : {NVJITLINK_INPUT_PTX, NVJITLINK_INPUT_CUBIN, NVJITLINK_INPUT_FATBIN})
        lto_reject(program, inputs[0], inputs[0].size(), NVJITLINK_ERROR_INCORRECT_INPUT_TYPE, kind);
    // Rejecting arbitrary LLVM bitcode is distinct from decoding a versioned
    // NVIDIA wrapper. A matching four-byte bitcode signature is insufficient.
    lto_bytes bare(24); bare[0] = 'B'; bare[1] = 'C'; bare[2] = 0xc0; bare[3] = 0xde;
    lto_reject(program, bare, bare.size());
    program.add(inputs[0], "repaired_kernel.ltoir");
    std::fill(inputs[0].begin(), inputs[0].end(), 0x97); std::fill(inputs[1].begin(), inputs[1].end(), 0x35);
    program.complete();
    const auto ptx = compiler_fixture_bytes([&](size_t* size) { program.check(nvJitLinkGetLinkedPtxSize(program.value, size), "LTO PTX size"); },
        [&](unsigned char* out) { program.check(nvJitLinkGetLinkedPtx(program.value, reinterpret_cast<char*>(out)), "LTO PTX bytes"); });
    require(ptx.back() == 0 && std::string_view(reinterpret_cast<const char*>(ptx.data()), ptx.size()).find(".entry lto_results") != std::string_view::npos,
        "LTO failure repair preserves the accepted helper and materializes the genuine kernel");
    const auto fresh = lto_inputs();
    linker_fixture_program disabled({"-arch=sm_75"});
    for (auto kind : {NVJITLINK_INPUT_LTOIR, NVJITLINK_INPUT_ANY})
        require(nvJitLinkAddData(disabled.value, kind, fresh[0].data(), fresh[0].size(), "disabled_lto") == NVJITLINK_ERROR_LTO_NOT_ENABLED,
            "explicit and automatic genuine LTO require LTO mode");
    linker_fixture_program mismatch({"-arch=sm_80", "-lto"});
    require(nvJitLinkAddData(mismatch.value, NVJITLINK_INPUT_LTOIR, fresh[0].data(), fresh[0].size(), "mismatched_lto") == NVJITLINK_ERROR_ARCH_MISMATCH,
        "genuine LTO retains its compiled source architecture");
}
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "format") return cudavm::run_host_test(argc, argv, cudavm::lto_format_case);
    if (argc == 3) { cudavm::lto_case = argv[2]; --argc; }
    return cudavm::run_test(argc, argv, cudavm::execute_lto_ir_import, 0, cudavm::probe_execution);
}
