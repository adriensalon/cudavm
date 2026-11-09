#include "test_framework.hpp"
#include "fatbinary_fixture.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>

namespace cudavm {
using fatbinary_bytes = std::vector<unsigned char>;
std::string_view fatbinary_import_case = "plain";
fatbinary_bytes fatbinary_input(bool compressed) {
    return compressed ? fatbinary_bytes(std::begin(fatbinary_compressed), std::end(fatbinary_compressed))
        : fatbinary_bytes(std::begin(fatbinary_plain), std::end(fatbinary_plain));
}
uint64_t fatbinary_field(const fatbinary_bytes& bytes, size_t position, unsigned width = 4) {
    require(width <= 8 && position <= bytes.size() && width <= bytes.size() - position, "bounded NVIDIA fixture header read");
    uint64_t value = 0; for (unsigned i = 0; i < width; ++i) value |= uint64_t(bytes[position + i]) << (i * 8); return value;
}
void fatbinary_edit(fatbinary_bytes& bytes, size_t position, uint64_t value, unsigned width = 4) {
    require(width <= 8 && position <= bytes.size() && width <= bytes.size() - position, "bounded NVIDIA fixture header edit");
    for (unsigned i = 0; i < width; ++i) bytes[position + i] = static_cast<unsigned char>(value >> (i * 8));
}
size_t fatbinary_member(const fatbinary_bytes& bytes, bool compressed = false) {
    size_t position = size_t(fatbinary_field(bytes, 6, 2));
    while (position < bytes.size()) {
        const auto header = fatbinary_field(bytes, position + 4), packed = fatbinary_field(bytes, position + 8, 8);
        require(header >= 64 && header <= bytes.size() - position && packed <= bytes.size() - position - header,
            "owned NVIDIA fixture member extent");
        if (compressed ? (fatbinary_field(bytes, position + 40, 8) & 0x2000) != 0
            : fatbinary_field(bytes, position, 2) == 1) return position;
        position += size_t(header + packed);
    }
    throw std::runtime_error("missing owned NVIDIA fixture member");
}
struct fatbinary_file {
    std::filesystem::path root, path;
    explicit fatbinary_file(const fatbinary_bytes& bytes) {
        root = std::filesystem::current_path() / ("fatbinary_import_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        require(std::filesystem::create_directory(root), "create unique NVIDIA file directory"); path = root / "input.fatbin";
        try {
            std::ofstream output(path, std::ios::binary);
            output.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size())); output.close();
            require(bool(output), "write exact NVIDIA fixture extent");
        } catch (...) { cleanup(); throw; }
    }
    ~fatbinary_file() { cleanup(); }
    void cleanup() const { std::error_code error; std::filesystem::remove(path, error); std::filesystem::remove(root, error); }
    fatbinary_file(const fatbinary_file&) = delete;
};
template<class Program> void fatbinary_execute_image(const Program& program) {
    constexpr uint32_t guard = 0xa95b371d;
    const auto seed_address = program.global("seed", 4), bias_address = program.global("bias", 4);
    std::array<uint32_t, 3> seed_words{guard, guard, guard}, bias_words{guard, guard, guard};
    CONFORMANCE_CUDA(cuMemcpyDtoH(seed_words.data() + 1, seed_address, 4));
    CONFORMANCE_CUDA(cuMemcpyDtoH(bias_words.data() + 1, bias_address, 4));
    require(seed_words == std::array<uint32_t, 3>{guard, 13, guard}
        && bias_words == std::array<uint32_t, 3>{guard, 5, guard}, "NVIDIA container global/constant initialization and host guards");
    device_buffer output(37 * 4); auto out = output.value + 13 * 4;
    std::array<uint32_t, 37> expected, actual;
    for (unsigned phase = 0; phase < 2; ++phase) {
        const uint32_t seed = phase ? 0xfffffff1u : 13u, bias = phase ? 29u : 5u;
        CONFORMANCE_CUDA(cuMemcpyHtoD(seed_address, &seed, 4)); CONFORMANCE_CUDA(cuMemcpyHtoD(bias_address, &bias, 4));
        for (unsigned add : {0u, 1u, 17u, 0x7fffffffu, 0x80000000u, 0xffffffffu}) {
            expected.fill(guard); CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), sizeof(expected)));
            void* parameters[]{&out, &add};
            CONFORMANCE_CUDA(cuLaunchKernel(program.function("entry"), 1, 1, 1, 1, 1, 1, 0, nullptr, parameters, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, sizeof(actual)));
            expected[13] = seed + bias + add + 7u;
            expect_words(actual, expected, "NVIDIA container selected program result and entire output guards");
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(seed_words.data() + 1, seed_address, 4));
        CONFORMANCE_CUDA(cuMemcpyDtoH(bias_words.data() + 1, bias_address, 4));
        require(seed_words == std::array<uint32_t, 3>{guard, seed, guard}
            && bias_words == std::array<uint32_t, 3>{guard, bias, guard}, "NVIDIA container preserves symbol inputs and host guards");
    }
}
struct fatbinary_module {
    CUmodule value{};
    explicit fatbinary_module(const void* image, decltype(&cuModuleLoadFatBinary) loader = cuModuleLoadFatBinary) {
        struct { uint64_t before = 0x193759bd; CUmodule value{}; uint64_t after = 0xabcdef01; } output;
        CONFORMANCE_CUDA(loader(&output.value, image)); value = output.value;
        require(output.before == 0x193759bd && output.after == 0xabcdef01 && value, "fatbinary module load output guards");
    }
    ~fatbinary_module() { if (value) cuModuleUnload(value); }
    fatbinary_module(const fatbinary_module&) = delete;
    CUfunction function(const char* name) const { CUfunction function{}; CONFORMANCE_CUDA(cuModuleGetFunction(&function, value, name)); return function; }
    CUdeviceptr global(const char* name, size_t expected) const {
        CUdeviceptr pointer{}; size_t size{}; CONFORMANCE_CUDA(cuModuleGetGlobal(&pointer, &size, value, name));
        require(pointer && size == expected, "fatbinary module symbol address and extent"); return pointer;
    }
};
void fatbinary_execute_copy(fatbinary_bytes bytes) {
    for (bool library : {false, true}) {
        auto temporary = bytes; loaded_program program(temporary.data(), library);
        std::fill(temporary.begin(), temporary.end(), 0x57);
        fatbinary_execute_image(program); CONFORMANCE_CUDA(cuCtxSynchronize());
    }
}
void execute_fatbinary_import() {
    if (fatbinary_import_case == "plain" || fatbinary_import_case == "compressed" || fatbinary_import_case == "load_fatbinary") {
        // Only native CUDA needs this architecture prerequisite. Shader profiles
        // execute the PTX member through their common IR import path.
        CUdevice device{}; char name[256]{}; int major{}, minor{};
        CONFORMANCE_CUDA(cuCtxGetDevice(&device)); CONFORMANCE_CUDA(cuDeviceGetName(name, sizeof(name), device));
        if (std::string_view(name).ends_with(" (CUDA)")) {
            CONFORMANCE_CUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
            CONFORMANCE_CUDA(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
            require_capability(major == 7 && minor == 5, "NVIDIA fatbinary native cubin execution profile requires SM75");
        }
        if (fatbinary_import_case == "load_fatbinary") {
            void* entry{}; CUdriverProcAddressQueryResult status{};
            CONFORMANCE_CUDA(cuGetProcAddress("cuModuleLoadFatBinary", &entry, 1999, CU_GET_PROC_ADDRESS_DEFAULT, &status));
            require(!entry && status == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "fatbinary loader rejects procedure revisions before its introduction");
            CONFORMANCE_CUDA(cuGetProcAddress("cuModuleLoadFatBinary", &entry, 2000, CU_GET_PROC_ADDRESS_DEFAULT, &status));
            require(entry && status == CU_GET_PROC_ADDRESS_SUCCESS, "fatbinary module loader is available through its pinned ABI revision");
            const auto resolved = reinterpret_cast<decltype(&cuModuleLoadFatBinary)>(entry);
            for (bool compressed : {false, true}) {
                auto input = fatbinary_input(compressed); fatbinary_module program(input.data(), compressed ? resolved : cuModuleLoadFatBinary);
                std::fill(input.begin(), input.end(), 0x57);
                fatbinary_execute_image(program); CONFORMANCE_CUDA(cuCtxSynchronize());
            }
            CUmodule rejected{};
            require(cuModuleLoadFatBinary(nullptr, fatbinary_plain) == CUDA_ERROR_INVALID_VALUE
                && cuModuleLoadFatBinary(&rejected, nullptr) == CUDA_ERROR_INVALID_VALUE, "fatbinary module loader rejects null arguments");
        } else fatbinary_execute_copy(fatbinary_input(fatbinary_import_case == "compressed"));
        return;
    }
    require(fatbinary_import_case == "linker" || fatbinary_import_case == "files", "known NVIDIA fatbinary execution case");
    for (bool compressed : {false, true}) for (auto kind : {NVJITLINK_INPUT_FATBIN, NVJITLINK_INPUT_ANY}) {
        fatbinary_bytes image;
        {
            linker_fixture_program linker({"-arch=sm_75", "-O0"}); auto input = fatbinary_input(compressed);
            if (fatbinary_import_case == "files") {
                fatbinary_file file(input); auto path = file.path.string();
                linker.check(nvJitLinkAddFile(linker.value, kind, path.c_str()), "NVIDIA AddFile");
                std::fill(path.begin(), path.end(), '!'); require(std::filesystem::remove(file.path), "retire NVIDIA file before Complete");
            } else {
                std::string label = "owned_nvidia_container"; linker.add(input, label.c_str(), kind);
                std::fill(label.begin(), label.end(), '!');
            }
            std::fill(input.begin(), input.end(), 0x57); linker.complete(); image = linker.image();
        }
        fatbinary_execute_copy(std::move(image));
    }
}
void fatbinary_reject(const fatbinary_bytes& input, std::string_view reason,
        nvJitLinkResult expected = NVJITLINK_ERROR_INVALID_INPUT) {
    fatbinary_bytes image;
    {
        linker_fixture_program linker({"-arch=sm_75", "-r"});
        const std::string sentinel = ".version 6.5\n.target sm_75\n.address_size 64\n.visible .entry fatbinary_sentinel(){ret;}\n";
        linker.add({sentinel.begin(), sentinel.end()}, "sentinel.ptx", NVJITLINK_INPUT_PTX);
        require(nvJitLinkAddData(linker.value, NVJITLINK_INPUT_FATBIN, input.data(), input.size(), "owned_invalid_nvidia.fatbin")
            == expected, "NVIDIA container rejection status for " + std::string(reason));
        const auto log = compiler_fixture_bytes([&](size_t* size) {
            linker.check(nvJitLinkGetErrorLogSize(linker.value, size), "NVIDIA rejection log size");
        }, [&](unsigned char* data) {
            linker.check(nvJitLinkGetErrorLog(linker.value, reinterpret_cast<char*>(data)), "NVIDIA rejection log");
        });
        require(log.back() == 0, "NVIDIA rejection log termination");
        const std::string diagnostic(log.begin(), log.end() - 1);
        require(diagnostic.find("owned_invalid_nvidia.fatbin") != std::string::npos && diagnostic.find(reason) != std::string::npos,
            "NVIDIA input diagnostic identifies owned label and rule: " + diagnostic);
        struct { size_t before = 0x193759bd, size{}, after = 0xabcdef01; } output;
        require(nvJitLinkGetLinkedCubinSize(linker.value, &output.size) == NVJITLINK_ERROR_INTERNAL
            && output.before == 0x193759bd && output.after == 0xabcdef01, "NVIDIA rejected input publishes no result and preserves guards");
        linker.add(fatbinary_input(false), "repaired_nvidia.fatbin", NVJITLINK_INPUT_FATBIN); linker.complete(); image = linker.image();
    }
    linker_fixture_program downstream({"-arch=sm_75", "-lto", "-ptx", "-O0"});
    downstream.add(image, "retained_after_repair", NVJITLINK_INPUT_CUBIN); downstream.complete();
    const auto text = compiler_fixture_bytes([&](size_t* size) {
        downstream.check(nvJitLinkGetLinkedPtxSize(downstream.value, size), "NVIDIA repaired PTX size");
    }, [&](unsigned char* data) {
        downstream.check(nvJitLinkGetLinkedPtx(downstream.value, reinterpret_cast<char*>(data)), "NVIDIA repaired PTX");
    });
    require(text.back() == 0, "NVIDIA repaired PTX termination");
    const std::string_view source(reinterpret_cast<const char*>(text.data()), text.size() - 1);
    require(source.find("fatbinary_sentinel") != std::string_view::npos && source.find(".entry entry") != std::string_view::npos,
        "failed NVIDIA addition preserves sentinel and permits valid input through downstream link");
}
void fatbinary_import_host() {
    const auto original = fatbinary_input(false);
    const auto edit = [&](size_t position, uint64_t value, unsigned width, std::string_view reason,
            nvJitLinkResult expected = NVJITLINK_ERROR_INVALID_INPUT) {
        auto input = original; fatbinary_edit(input, position, value, width); fatbinary_reject(input, reason, expected);
    };
    const auto first = size_t(fatbinary_field(original, 6, 2));
    if (fatbinary_import_case == "extents") {
        auto input = original; input.pop_back(); fatbinary_reject(input, "size mismatch");
        input = original; input.push_back(0); fatbinary_reject(input, "size mismatch");
        edit(6, 0, 2, "extent"); edit(8, UINT64_MAX, 8, "extent");
        edit(first + 4, 63, 4, "entry extent"); edit(first + 8, UINT64_MAX, 8, "entry extent");
    } else if (fatbinary_import_case == "versions") {
        edit(4, 2, 2, "version", NVJITLINK_ERROR_UNRECOGNIZED_INPUT);
        edit(first + 2, 0x102, 2, "entry version", NVJITLINK_ERROR_UNRECOGNIZED_INPUT);
    } else if (fatbinary_import_case == "metadata") {
        edit(first + 28, 0, 4, "architecture");
        edit(first + 40, fatbinary_field(original, first + 40, 8) & ~uint64_t{1}, 8, "64-bit", NVJITLINK_ERROR_UNRECOGNIZED_INPUT);
        const auto ptx = fatbinary_member(original);
        edit(ptx + 24, 0, 4, "PTX version metadata");
        edit(ptx + 28, fatbinary_field(original, ptx + 28) + 1, 4, "PTX metadata mismatch");
    } else {
        require(fatbinary_import_case == "compression", "known NVIDIA host case");
        const auto compressed = fatbinary_input(true); const auto member = fatbinary_member(compressed, true);
        auto input = compressed;
        fatbinary_edit(input, member + 16, fatbinary_field(compressed, member + 8, 8) + 1);
        fatbinary_reject(input, "compressed NVIDIA fatbinary extent");
        input = compressed; fatbinary_edit(input, member + 56, UINT64_MAX, 8); fatbinary_reject(input, "oversized");
        input = compressed; fatbinary_edit(input, member + 56, 1, 8); fatbinary_reject(input, "compression");
    }
}
}
int main(int argc, char** argv) {
    if (argc == 2) { cudavm::fatbinary_import_case = argv[1]; return cudavm::run_host_conformance(argc, argv, cudavm::fatbinary_import_host); }
    if (argc == 3) { cudavm::fatbinary_import_case = argv[2]; argc = 2; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_fatbinary_import, 0, cudavm::probe_execution);
}
