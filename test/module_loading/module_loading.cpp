#include "test_framework.hpp"
#include "module_fixtures.hpp"
#include "module_oracle.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>

namespace cudavm {
using module_loading_bytes = std::vector<unsigned char>;
std::string_view module_loading_case = "memory";
template<class T> struct module_loading_output {
    static_assert(alignof(T) <= 16);
    std::array<unsigned char, 16> before; T value{}; std::array<unsigned char, 16> after;
    module_loading_output() { before.fill(0x35); after.fill(0x79); }
    T get() const {
        require(std::all_of(before.begin(), before.end(), [](auto byte) { return byte == 0x35; })
            && std::all_of(after.begin(), after.end(), [](auto byte) { return byte == 0x79; }), "module query exact output guards"); return value;
    }
};
module_loading_bytes module_loading_image(bool source) {
    if (!source) { module_loading_bytes bytes(module_loading_ptx.begin(), module_loading_ptx.end()); bytes.push_back(0); return bytes; }
    compiler_fixture_program program(std::string(module_loading_source), "module_loading.cu");
    const auto status = program.compile({"-arch=sm_75", "--std=c++17"});
    require(status == NVRTC_SUCCESS, "module source compilation: " + program.text(true));
    return compiler_fixture_bytes([&](size_t* size) { compiler_fixture_program::check(nvrtcGetCUBINSize(program.value, size)); },
        [&](unsigned char* bytes) { compiler_fixture_program::check(nvrtcGetCUBIN(program.value, reinterpret_cast<char*>(bytes))); });
}
struct module_loading_file {
    std::filesystem::path root, path;
    explicit module_loading_file(const module_loading_bytes& bytes) {
        root = std::filesystem::current_path() / ("module_loading_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        require(std::filesystem::create_directory(root), "create owned module file directory"); path = root / "input.image";
        try {
            std::ofstream output(path, std::ios::binary); output.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
            output.close(); require(bool(output), "write exact caller module bytes");
        } catch (...) { cleanup(); throw; }
    }
    ~module_loading_file() { cleanup(); }
    void cleanup() const { std::error_code error; std::filesystem::remove(path, error); std::filesystem::remove(root, error); }
    module_loading_file(const module_loading_file&) = delete;
};
struct module_loading_program {
    CUmodule value{};
    module_loading_program(const module_loading_bytes& image, unsigned route = 0, unsigned optimization = 0) {
        auto temporary = image; module_loading_output<CUmodule> output;
        if (route == 1) {
            module_loading_file file(temporary); auto path = file.path.string();
            TEST_CUDA(cuModuleLoad(&output.value, path.c_str())); value = output.value;
            std::fill(path.begin(), path.end(), '!'); require(std::filesystem::remove(file.path), "retire module file before lookup/execution");
        } else if (route == 2) TEST_CUDA(cuModuleLoadDataEx(&output.value, temporary.data(), 0, nullptr, nullptr));
        else if (route == 3) {
            CUjit_option option = CU_JIT_OPTIMIZATION_LEVEL; void* setting = reinterpret_cast<void*>(uintptr_t(optimization));
            TEST_CUDA(cuModuleLoadDataEx(&output.value, temporary.data(), 1, &option, &setting));
        } else TEST_CUDA(cuModuleLoadData(&output.value, temporary.data()));
        value = output.value; require(output.get() != nullptr, "module load publishes guarded handle");
        std::fill(temporary.begin(), temporary.end(), 0x57);
    }
    ~module_loading_program() { if (value) cuModuleUnload(value); }
    module_loading_program(const module_loading_program&) = delete;
    CUfunction function(const char* name) const {
        module_loading_output<CUfunction> output; TEST_CUDA(cuModuleGetFunction(&output.value, value, name));
        require(output.get() != nullptr, "module function lookup publishes nonnull handle"); return output.value;
    }
    CUdeviceptr global(const char* name) const {
        module_loading_output<CUdeviceptr> pointer; module_loading_output<size_t> size;
        TEST_CUDA(cuModuleGetGlobal(&pointer.value, &size.value, value, name));
        require(pointer.get() != 0 && size.get() == 4, "module symbol address/extent agrees with declared scalar"); return pointer.value;
    }
    void close() { TEST_CUDA(cuModuleUnload(value)); value = nullptr; }
};
void module_loading_execute(const module_loading_program& program, uint32_t expected_seed = 13) {
    constexpr uint32_t guard = 0xa95b371d; constexpr unsigned capacity = 67, guards = 9;
    const auto seed_address = program.global("module_seed"), bias_address = program.global("module_bias");
    std::array<uint32_t, 3> seed{guard, guard, guard}, bias{guard, guard, guard};
    TEST_CUDA(cuMemcpyDtoH(seed.data() + 1, seed_address, 4)); TEST_CUDA(cuMemcpyDtoH(bias.data() + 1, bias_address, 4));
    require(seed == std::array<uint32_t, 3>{guard, expected_seed, guard} && bias == std::array<uint32_t, 3>{guard, 29, guard},
        "module independent storage initialization and host guards");
    std::vector<uint32_t> input(capacity + guards * 2, guard), expected(input.size(), guard), actual(input.size()), preserved(input.size());
    for (unsigned i = 0; i < capacity; ++i) input[guards + i] = i < 4 ? std::array<uint32_t, 4>{0, 0xffffffff, 0x80000000, 0x7fffffff}[i]
        : i * 0x193759bdu;
    device_buffer source(input.size() * 4), output(expected.size() * 4);
    TEST_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
    auto in = source.value + guards * 4, out = output.value + guards * 4;
    for (unsigned phase = 0; phase < 2; ++phase) {
        const uint32_t seed_value = phase ? 0xfffffff1u : expected_seed;
        TEST_CUDA(cuMemcpyHtoD(seed_address, &seed_value, 4));
        for (bool beta : {false, true}) for (unsigned count : {0u, 1u, 33u, capacity}) {
            std::fill(expected.begin(), expected.end(), guard);
            TEST_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
            void* parameters[]{&out, &in, &count};
            TEST_CUDA(cuLaunchKernel(program.function(beta ? "module_beta" : "module_alpha"), 2, 1, 1, 47, 1, 1,
                0, nullptr, parameters, nullptr));
            TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
            for (unsigned i = 0; i < count; ++i) expected[guards + i] = module_boundary_expected(beta, input[guards + i], seed_value, i);
            expect_words(actual, expected, "module entry execution, untouched capacity and output guards");
            TEST_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
            expect_words(preserved, input, "module execution preserves input and all guards");
        }
    }
}
void module_loading_queries(const module_loading_program& program) {
    module_loading_output<unsigned> count; TEST_CUDA(cuModuleGetFunctionCount(&count.value, program.value));
    require(count.get() == 2, "module function inventory has the two declared kernel entries");
    const auto alpha = program.function("module_alpha"), beta = program.function("module_beta");
    require(alpha != beta && alpha == program.function("module_alpha"), "module function identities are distinct and lookup is stable");
    for (unsigned capacity : {0u, 1u, 2u, 3u}) {
        struct { uint64_t before = 0x193759bd; std::array<CUfunction, 3> values{}; uint64_t after = 0xabcdef01; } output;
        TEST_CUDA(cuModuleEnumerateFunctions(output.values.data(), capacity, program.value));
        require(output.before == 0x193759bd && output.after == 0xabcdef01, "module enumeration output guards");
        for (unsigned i = 0; i < output.values.size(); ++i)
            require(i < std::min(capacity, 2u) ? output.values[i] == alpha || output.values[i] == beta : output.values[i] == nullptr,
                "module enumeration writes only its admitted kernel handles");
        if (capacity >= 2) require(output.values[0] != output.values[1], "complete module enumeration has no duplicates");
    }
    for (const auto& [function, name] : std::array<std::pair<CUfunction, const char*>, 2>{{{alpha, "module_alpha"}, {beta, "module_beta"}}}) {
        module_loading_output<CUmodule> owner; module_loading_output<const char*> text;
        TEST_CUDA(cuFuncGetModule(&owner.value, function)); TEST_CUDA(cuFuncGetName(&text.value, function));
        require(owner.get() == program.value && text.get() && std::string_view(text.value) == name, "function name and module owner agree with lookup");
        module_loading_output<size_t> parameters; TEST_CUDA(cuFuncGetParamCount(function, &parameters.value));
        require(parameters.get() == 3, "declared three-parameter module kernel ABI");
        for (size_t index = 0; index < 3; ++index) {
            module_loading_output<size_t> offset, bytes;
            TEST_CUDA(cuFuncGetParamInfo(function, index, &offset.value, &bytes.value));
            require(offset.get() == index * 8 && bytes.get() == (index < 2 ? 8u : 4u), "independent module parameter offsets and extents");
        }
    }
    for (const char* name : {"module_seed", "module_bias"}) {
        module_loading_output<CUdeviceptr> pointer; module_loading_output<size_t> bytes;
        TEST_CUDA(cuModuleGetGlobal(&pointer.value, nullptr, program.value, name));
        TEST_CUDA(cuModuleGetGlobal(nullptr, &bytes.value, program.value, name));
        require(pointer.get() == program.global(name) && bytes.get() == 4, "module global single-output query variants");
    }
}
void module_loading_validation(const module_loading_program& program, const module_loading_bytes& image) {
    module_loading_output<CUmodule> module;
    require(cuModuleLoadData(nullptr, image.data()) == CUDA_ERROR_INVALID_VALUE
        && cuModuleLoadData(&module.value, nullptr) == CUDA_ERROR_INVALID_VALUE
        && cuModuleLoadDataEx(&module.value, image.data(), 1, nullptr, nullptr) == CUDA_ERROR_INVALID_VALUE,
        "module invalid load outputs/image/JIT arrays"); module.get();
    module_loading_output<CUfunction> function; module_loading_output<CUdeviceptr> pointer; module_loading_output<size_t> size;
    require(cuModuleGetFunction(nullptr, program.value, "module_alpha") == CUDA_ERROR_INVALID_VALUE
        && cuModuleGetFunction(&function.value, program.value, "owned_missing_kernel") == CUDA_ERROR_NOT_FOUND,
        "module missing function and null lookup output statuses"); function.get();
    require(cuModuleGetGlobal(&pointer.value, &size.value, program.value, "owned_missing_global") == CUDA_ERROR_NOT_FOUND
        && cuModuleGetGlobal(nullptr, nullptr, program.value, "module_seed") == CUDA_ERROR_INVALID_VALUE,
        "module missing global and missing output statuses"); pointer.get(); size.get();
    require(cuModuleGetFunctionCount(nullptr, program.value) == CUDA_ERROR_INVALID_VALUE
        && cuModuleEnumerateFunctions(nullptr, 1, program.value) == CUDA_ERROR_INVALID_VALUE,
        "module count and enumeration null outputs");
}
void execute_module_loading() {
    module_loading_output<CUmoduleLoadingMode> before, after;
    TEST_CUDA(cuModuleGetLoadingMode(&before.value));
    require(before.get() == CU_MODULE_EAGER_LOADING || before.value == CU_MODULE_LAZY_LOADING, "module loading mode is a defined value");
    for (bool source : {false, true}) {
        const auto image = module_loading_image(source);
        if (module_loading_case == "isolation") {
            module_loading_program first(image), second(image);
            const auto first_seed = first.global("module_seed"), second_seed = second.global("module_seed");
            require(first_seed != second_seed, "simultaneous module loads own separate live storage");
            uint32_t changed = 71; TEST_CUDA(cuMemcpyHtoD(first_seed, &changed, 4));
            module_loading_execute(second); first.close();
            uint32_t actual{}; TEST_CUDA(cuMemcpyDtoH(&actual, second_seed, 4));
            require(actual == 0xfffffff1u, "unloading another module preserves the live module's storage");
            module_loading_program fresh(image); module_loading_execute(fresh); fresh.close(); second.close(); continue;
        }
        const unsigned routes = module_loading_case == "memory" || module_loading_case == "options" ? 2 : 1;
        for (unsigned index = 0; index < routes; ++index) {
            const auto route = module_loading_case == "files" ? 1u : module_loading_case == "options" ? 3u : index ? 2u : 0u;
            module_loading_program program(image, route, index ? 3u : 0u);
            if (module_loading_case == "queries") module_loading_queries(program);
            else if (module_loading_case == "validation") module_loading_validation(program, image);
            else require(module_loading_case == "memory" || module_loading_case == "files" || module_loading_case == "options", "known module loading case");
            module_loading_execute(program); TEST_CUDA(cuCtxSynchronize()); program.close();
        }
    }
    TEST_CUDA(cuModuleGetLoadingMode(&after.value)); require(after.get() == before.value, "module loading mode query is stable across loads/unloads");
}
void module_loading_sources() { module_loading_image(true); }
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "sources") return cudavm::run_host_test(argc, argv, cudavm::module_loading_sources);
    if (argc == 3) { cudavm::module_loading_case = argv[2]; argc = 2; }
    return cudavm::run_test(argc, argv, cudavm::execute_module_loading, 0, cudavm::probe_execution);
}
