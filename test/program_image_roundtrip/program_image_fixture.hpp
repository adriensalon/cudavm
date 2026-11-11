#pragma once
#include "test_framework.hpp"
#include "module_fixtures.hpp"
#include "../module_loading/module_oracle.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace cudavm {
using program_image_bytes = std::vector<unsigned char>;
template<class T> struct program_image_output {
    static_assert(alignof(T) <= 16);
    std::array<unsigned char, 16> before; T value{}; std::array<unsigned char, 16> after;
    program_image_output() { before.fill(0x35); after.fill(0x79); }
    T get() const {
        require(std::all_of(before.begin(), before.end(), [](auto byte) { return byte == 0x35; })
            && std::all_of(after.begin(), after.end(), [](auto byte) { return byte == 0x79; }), "program result scalar output guards"); return value;
    }
};
template<class Query, class Get> program_image_bytes program_image_result(Query query, Get get) {
    program_image_output<size_t> first_size, second_size; query(&first_size.value); query(&second_size.value);
    const auto count = first_size.get();
    require(count > 0 && count < 64 * 1024 * 1024 && count == second_size.get(), "public program result stable bounded extent");
    program_image_bytes first(count + 32, 0x35), second(count + 32, 0x79); get(first.data() + 16); get(second.data() + 16);
    for (size_t i = 0; i < first.size(); ++i) {
        if (i < 16 || i >= count + 16) require(first[i] == 0x35 && second[i] == 0x79, "public binary result exact byte extent guards");
        else require(first[i] == second[i], "public binary result complete writes and repeatability");
    }
    return {first.begin() + 16, first.end() - 16};
}
#ifndef CUDAVM_TEST_HOST_ONLY
inline void program_image_runtime_status(cudaError_t status) {
    require(status == cudaSuccess, std::string("program Runtime status ") + cudaGetErrorString(status));
}
#endif
#ifdef CUDAVM_TEST_NVJITLINK
inline program_image_bytes program_image_linked_result(const linker_fixture_program& linker, bool lto = false) {
    return program_image_result([&](size_t* size) {
        linker.check(lto ? nvJitLinkGetLinkedLTOIRSize(linker.value, size) : nvJitLinkGetLinkedCubinSize(linker.value, size), "public linked result size");
    }, [&](unsigned char* bytes) {
        linker.check(lto ? nvJitLinkGetLinkedLTOIR(linker.value, bytes) : nvJitLinkGetLinkedCubin(linker.value, bytes), "public linked result bytes");
    });
}
#endif
#ifdef CUDAVM_TEST_NVRTC
inline program_image_bytes program_image_nvrtc_produce(std::string_view producer) {
    require(producer == "nvrtc_cubin" || producer == "nvrtc_rdc" || producer == "nvrtc_lto", "known public NVRTC image producer");
    compiler_fixture_program program(std::string(module_loading_source), "program_image.cu");
    const bool lto = producer == "nvrtc_lto";
    std::vector<std::string> options{lto ? "-arch=compute_75" : "-arch=sm_75", "--std=c++17"};
    if (lto) options.push_back("-dlto"); if (producer == "nvrtc_rdc") options.push_back("-rdc=true");
    const auto status = program.compile(options); require(status == NVRTC_SUCCESS, "public program NVRTC compile: " + program.text(true));
    return program_image_result([&](size_t* size) {
        compiler_fixture_program::check(lto ? nvrtcGetLTOIRSize(program.value, size) : nvrtcGetCUBINSize(program.value, size));
    }, [&](unsigned char* bytes) {
        compiler_fixture_program::check(lto ? nvrtcGetLTOIR(program.value, reinterpret_cast<char*>(bytes)) : nvrtcGetCUBIN(program.value, reinterpret_cast<char*>(bytes)));
    });
}
#endif
struct program_image_file {
    std::filesystem::path root, path;
    explicit program_image_file(const program_image_bytes& bytes) {
        root = std::filesystem::current_path() / ("program_image_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        require(std::filesystem::create_directory(root), "create owned program image directory"); path = root / "input.image";
        try {
            std::ofstream output(path, std::ios::binary); output.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
            output.close(); require(bool(output), "write exact public program bytes");
        } catch (...) { cleanup(); throw; }
    }
    ~program_image_file() { cleanup(); }
    void cleanup() const { std::error_code error; std::filesystem::remove(path, error); std::filesystem::remove(root, error); }
    program_image_file(const program_image_file&) = delete;
};
#ifdef CUDAVM_TEST_NVJITLINK
inline program_image_bytes program_image_link(program_image_bytes input, nvJitLinkInputType kind, bool file, bool relocatable = false) {
    std::vector<std::string> options{"-arch=sm_75", "-lto", "-O0"}; if (relocatable) options.push_back("-r");
    linker_fixture_program linker(options);
    if (file) {
        program_image_file input_file(input); auto path = input_file.path.string();
        linker.check(nvJitLinkAddFile(linker.value, kind, path.c_str()), "program image AddFile");
        std::fill(path.begin(), path.end(), '!'); require(std::filesystem::remove(input_file.path), "remove program input file before link completion");
    } else {
        std::string name = "owned_program_input"; linker.add(input, name.c_str(), kind); std::fill(name.begin(), name.end(), '!');
    }
    std::fill(input.begin(), input.end(), 0x57); linker.complete(); return program_image_linked_result(linker);
}
#endif

#ifndef CUDAVM_TEST_HOST_ONLY
struct program_image_loaded {
    CUmodule module{}; CUlibrary library{}; bool runtime = false;
    program_image_loaded(const program_image_bytes& image, unsigned route) : runtime(route == 5 || route == 6) {
        require(route < 8, "known program image loader route");
        auto input = image; program_image_output<CUmodule> module_output; program_image_output<CUlibrary> library_output;
        program_image_output<cudaLibrary_t> runtime_output;
        const bool file = route == 2 || route == 4 || route == 6;
        std::unique_ptr<program_image_file> input_file;
        if (file) input_file = std::make_unique<program_image_file>(input);
        auto path = input_file ? input_file->path.string() : std::string();
        if (route == 0) TEST_CUDA(cuModuleLoadData(&module_output.value, input.data()));
        else if (route == 1) TEST_CUDA(cuModuleLoadDataEx(&module_output.value, input.data(), 0, nullptr, nullptr));
        else if (route == 2) TEST_CUDA(cuModuleLoad(&module_output.value, path.c_str()));
        else if (route == 3) TEST_CUDA(cuLibraryLoadData(&library_output.value, input.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
        else if (route == 4) TEST_CUDA(cuLibraryLoadFromFile(&library_output.value, path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
        else if (route == 5) program_image_runtime_status(cudaLibraryLoadData(&runtime_output.value, input.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
        else if (route == 6) program_image_runtime_status(cudaLibraryLoadFromFile(&runtime_output.value, path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
        else TEST_CUDA(cuModuleLoadFatBinary(&module_output.value, input.data()));
        module = module_output.get(); library = runtime ? reinterpret_cast<CUlibrary>(runtime_output.get()) : library_output.get();
        require((route < 3 || route == 7) ? module != nullptr : library != nullptr, "public loader publishes guarded program handle");
        std::fill(input.begin(), input.end(), 0x57); std::fill(path.begin(), path.end(), '!');
        if (input_file) require(std::filesystem::remove(input_file->path), "remove file before program lookup and execution");
    }
    ~program_image_loaded() { if (module) cuModuleUnload(module); if (library) cuLibraryUnload(library); }
    program_image_loaded(const program_image_loaded&) = delete;
    CUfunction function(const char* name) const {
        program_image_output<CUfunction> output;
        if (module) TEST_CUDA(cuModuleGetFunction(&output.value, module, name));
        else { CUkernel kernel{}; TEST_CUDA(cuLibraryGetKernel(&kernel, library, name)); TEST_CUDA(cuKernelGetFunction(&output.value, kernel)); }
        require(output.get() != nullptr, "public program function lookup"); return output.value;
    }
    CUdeviceptr global(const char* name) const {
        program_image_output<CUdeviceptr> pointer; program_image_output<size_t> size;
        if (module) TEST_CUDA(cuModuleGetGlobal(&pointer.value, &size.value, module, name));
        else TEST_CUDA(cuLibraryGetGlobal(&pointer.value, &size.value, library, name));
        require(pointer.get() && size.get() == 4, "public program scalar storage address/extent"); return pointer.value;
    }
    void launch(const char* name, void** parameters) const {
        if (runtime) {
            cudaKernel_t kernel{}; program_image_runtime_status(cudaLibraryGetKernel(&kernel, reinterpret_cast<cudaLibrary_t>(library), name));
            program_image_runtime_status(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(1), dim3(31), parameters, 0, nullptr));
        } else TEST_CUDA(cuLaunchKernel(function(name), 1, 1, 1, 31, 1, 1, 0, nullptr, parameters, nullptr));
    }
};
inline void program_image_execute(const program_image_loaded& program) {
    constexpr unsigned capacity = 17, guards = 7; constexpr uint32_t guard = 0xa95b371d;
    const auto seed_address = program.global("module_seed"), bias_address = program.global("module_bias");
    std::array<uint32_t, 3> seed{guard, guard, guard}, bias{guard, guard, guard};
    TEST_CUDA(cuMemcpyDtoH(seed.data() + 1, seed_address, 4)); TEST_CUDA(cuMemcpyDtoH(bias.data() + 1, bias_address, 4));
    require(seed == std::array<uint32_t, 3>{guard, 13, guard} && bias == std::array<uint32_t, 3>{guard, 29, guard}, "public program copied storage initializers and host guards");
    std::vector<uint32_t> input(capacity + 2 * guards, guard), expected(input.size()), actual(input.size()), preserved(input.size());
    constexpr std::array<uint32_t, 4> edges{0, 0xffffffff, 0x80000000, 0x7fffffff};
    for (unsigned i = 0; i < capacity; ++i) input[guards + i] = i < edges.size() ? edges[i] : i * 0x193759bdu;
    device_buffer source(input.size() * 4), output(input.size() * 4); auto in = source.value + guards * 4, out = output.value + guards * 4;
    TEST_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
    for (uint32_t seed_value : {13u, 0xfffffff1u}) {
        TEST_CUDA(cuMemcpyHtoD(seed_address, &seed_value, 4));
        for (bool beta : {false, true}) for (unsigned count : {0u, 1u, capacity}) {
            std::fill(expected.begin(), expected.end(), guard); TEST_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
            void* parameters[]{&out, &in, &count}; program.launch(beta ? "module_beta" : "module_alpha", parameters);
            TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
            for (unsigned i = 0; i < count; ++i) expected[guards + i] = module_boundary_expected(beta, input[guards + i], seed_value, i);
            expect_words(actual, expected, "public program image route results and complete output guards");
        }
        TEST_CUDA(cuMemcpyDtoH(seed.data() + 1, seed_address, 4)); TEST_CUDA(cuMemcpyDtoH(bias.data() + 1, bias_address, 4));
        require(seed == std::array<uint32_t, 3>{guard, seed_value, guard} && bias == std::array<uint32_t, 3>{guard, 29, guard}, "public program route preserves module inputs");
    }
    TEST_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4)); expect_words(preserved, input, "public program image route preserves source and guards");
}
inline void program_image_load_all(const program_image_bytes& image, bool fatbinary = false) {
    for (unsigned route = 0; route < (fatbinary ? 8u : 7u); ++route) {
        std::cout << "program_image loader=" << route << std::endl;
        program_image_loaded program(image, route); program_image_execute(program); TEST_CUDA(cuCtxSynchronize());
    }
}
#endif
}
