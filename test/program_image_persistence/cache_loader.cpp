#include "program_image_fixture.hpp"
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace cudavm {
std::filesystem::path program_image_cache_directory;
program_image_bytes program_image_cache_loader_read(std::string_view name) {
    std::ifstream input(program_image_cache_directory / name, std::ios::binary | std::ios::ate);
    require(bool(input), "open independent compiler-process result");
    const auto size = input.tellg(); require(size > 0 && size < 64 * 1024 * 1024, "bounded persisted cache input");
    program_image_bytes image(static_cast<size_t>(size)); input.seekg(0);
    input.read(reinterpret_cast<char*>(image.data()), std::streamsize(image.size()));
    require(bool(input) && input.peek() == std::char_traits<char>::eof(), "read complete isolated cache result");
    return image;
}
void program_image_cache_compiler_unavailable() {
#if defined(_WIN32)
    require(GetModuleHandleW(L"cudavm_core.dll") == nullptr, "isolated loader has no resident core compiler");
    auto compiler = LoadLibraryW(L"cudavm_core.dll"); const bool absent = compiler == nullptr;
    if (compiler) FreeLibrary(compiler);
#else
#if defined(__APPLE__)
    constexpr auto name = "libcudavm_core.dylib";
#else
    constexpr auto name = "libcudavm_core.so";
#endif
    auto compiler = dlopen(name, RTLD_NOW | RTLD_LOCAL); const bool absent = compiler == nullptr;
    if (compiler) dlclose(compiler);
#endif
    require(absent, "core compiler is genuinely unavailable through the isolated process search paths");
}
void program_image_cache_probe() {
    auto image = program_image_cache_loader_read("probe.image"); loaded_program program(image.data(), false);
    std::fill(image.begin(), image.end(), 0x57);
    device_buffer output(3 * sizeof(uint32_t));
    constexpr uint32_t guard = 0xa95b371d; std::array<uint32_t, 3> words{guard, guard, guard};
    CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, words.data(), sizeof(words)));
    auto address = output.value + sizeof(uint32_t); void* parameters[]{&address};
    CONFORMANCE_CUDA(cuLaunchKernel(program.function("cached_availability_probe"), 1, 1, 1, 1, 1, 1, 0, nullptr, parameters, nullptr));
    CONFORMANCE_CUDA(cuMemcpyDtoH(words.data(), output.value, sizeof(words)));
    require(words == std::array<uint32_t, 3>{guard, 0x43564d31, guard}, "compiler-independent cached availability probe and complete guards");
}
void execute_program_image_cache() {
    program_image_cache_compiler_unavailable();
    const auto image = program_image_cache_loader_read("cached.image"); program_image_load_all(image);
    require(program_image_cache_loader_read("cached.image") == image, "compiler-free execution preserves persisted cache bytes");
    // Retained IR is a distinct path. This control proves that the harness has
    // not accidentally found the original build's compiler through a side path.
    const auto portable = program_image_cache_loader_read("portable.image");
    program_image_output<CUmodule> module; program_image_output<CUlibrary> library;
    program_image_output<cudaLibrary_t> runtime_library;
    require(cuModuleLoadData(&module.value, portable.data()) == CUDA_ERROR_JIT_COMPILER_NOT_FOUND,
        "an uncached module requires the absent optional compiler");
    require(cuLibraryLoadData(&library.value, portable.data(), nullptr, nullptr, 0, nullptr, nullptr, 0) == CUDA_ERROR_JIT_COMPILER_NOT_FOUND,
        "an uncached library requires the absent optional compiler");
    require(cudaLibraryLoadData(&runtime_library.value, portable.data(), nullptr, nullptr, 0, nullptr, nullptr, 0)
        == cudaErrorJitCompilerNotFound, "Runtime reports the absent optional compiler for retained IR");
    require(!module.get() && !library.get() && !runtime_library.get(), "missing compiler publishes no handle and preserves guarded outputs");
    program_image_cache_compiler_unavailable();
    // The failed materializations leave an independent cached image usable.
    program_image_loaded repaired(image, 0); program_image_execute(repaired);
}
}
int main(int argc, char** argv) {
    if (argc != 3) return 1;
    cudavm::program_image_cache_directory = argv[2];
    try { cudavm::program_image_cache_compiler_unavailable(); }
    catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
    return cudavm::run_conformance(2, argv, cudavm::execute_program_image_cache, 0, cudavm::program_image_cache_probe);
}
