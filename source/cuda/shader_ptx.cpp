#include "shader_ptx.hpp"
#include "dynamic_library.hpp"
#include "../cudavm/compiler_service.hpp"
namespace cudavm {
std::string compile_shader_ptx(std::string_view input, cudavmBackend backend) {
    // Retry discovery after an unavailable service; a failed first load must
    // not prevent a later application from deploying the optional compiler.
#if defined(_WIN32)
    static dynamic_library compiler({"cudavm_compiler.dll"}, true, CUDA_ERROR_JIT_COMPILER_NOT_FOUND);
#elif defined(__APPLE__)
    static dynamic_library compiler({"libcudavm_compiler.dylib"}, true, CUDA_ERROR_JIT_COMPILER_NOT_FOUND);
#else
    static dynamic_library compiler({"libcudavm_compiler.so"}, true, CUDA_ERROR_JIT_COMPILER_NOT_FOUND);
#endif
    auto compile = compiler.get<compiler_compile_image>("cudavm_compiler_compile_image_v1");
    std::array<char, 8192> diagnostic{};
    std::string image;
    auto output = [](const char* bytes, size_t size, void* user) noexcept {
        try { static_cast<std::string*>(user)->assign(bytes, size); return true; } catch (...) { return false; }
    };
    compiler_image_request request{1, compiler_input_kind::ptx, input.data(), input.size(), unsigned(backend)};
    auto status = compile(&request, output, &image, diagnostic.data(), diagnostic.size());
    if (status != compiler_status::success) throw failure(status == compiler_status::unsupported ? CUDA_ERROR_NOT_SUPPORTED :
        status == compiler_status::out_of_memory ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_INVALID_PTX, diagnostic.data());
    return image;
}
}
