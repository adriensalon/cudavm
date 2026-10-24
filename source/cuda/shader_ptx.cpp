#include "shader_ptx.hpp"
#include "dynamic_library.hpp"
#include "../core/image_api.hpp"
#include "../core/module_descriptor.hpp"
namespace cudavm {
namespace {
std::string compile_image(std::string_view input, cudavmBackend backend, compiler_input_kind kind) {
    // Retry discovery after an unavailable service; a failed first load must
    // not prevent a later application from deploying the optional compiler.
#if defined(_WIN32)
    static dynamic_library compiler({"cudavm_core.dll"}, true, CUDA_ERROR_JIT_COMPILER_NOT_FOUND);
#elif defined(__APPLE__)
    static dynamic_library compiler({"libcudavm_core.dylib"}, true, CUDA_ERROR_JIT_COMPILER_NOT_FOUND);
#else
    static dynamic_library compiler({"libcudavm_core.so"}, true, CUDA_ERROR_JIT_COMPILER_NOT_FOUND);
#endif
    auto compile = compiler.get<compiler_compile_image>("cudavm_core_compile_image_v1");
    std::array<char, 8192> diagnostic{};
    std::string image;
    auto output = [](const char* bytes, size_t size, void* user) noexcept {
        try { static_cast<std::string*>(user)->assign(bytes, size); return true; } catch (...) { return false; }
    };
    compiler_image_request request{1, kind, input.data(), input.size(), unsigned(backend)};
    auto status = compile(&request, output, &image, diagnostic.data(), diagnostic.size());
    if (status != compiler_status::success) throw failure(status == compiler_status::unsupported ? CUDA_ERROR_NOT_SUPPORTED :
        status == compiler_status::out_of_memory ? CUDA_ERROR_OUT_OF_MEMORY :
        kind == compiler_input_kind::ptx ? CUDA_ERROR_INVALID_PTX : CUDA_ERROR_INVALID_IMAGE, diagnostic.data());
    return image;
}
}
std::string compile_shader_ptx(std::string_view input, cudavmBackend backend) {
    return compile_image(input, backend, compiler_input_kind::ptx);
}
std::string materialize_program_image(std::string_view input, cudavmBackend backend) {
    try {
        auto image = decode_program_image(input);
        if (auto* artifact = find_program_image_artifact(image, unsigned(backend))) return artifact->code;
    } catch (const std::bad_alloc&) { throw failure(CUDA_ERROR_OUT_OF_MEMORY, "program image allocation failed"); }
    catch (const std::exception& error) { throw failure(CUDA_ERROR_INVALID_IMAGE, error.what()); }
    return compile_image(input, backend, compiler_input_kind::program_image);
}
std::string materialize_fatbin(std::string_view input, cudavmBackend backend, unsigned architecture) {
    std::vector<fatbin_record> records;
    try { records = decode_fatbin(input); }
    catch (const std::bad_alloc&) { throw failure(CUDA_ERROR_OUT_OF_MEMORY, "fatbinary allocation failed"); }
    catch (const std::exception& error) { throw failure(CUDA_ERROR_INVALID_IMAGE, error.what()); }
    auto* selected = select_fatbin(records, architecture, false, backend == cudavmBackendCuda);
    require(selected, CUDA_ERROR_NO_BINARY_FOR_GPU, "fatbinary has no compatible executable image");
    if (has_program_image(std::string_view(selected->code))) return materialize_program_image(selected->code, backend);
    if (selected->kind == fatbin_kind::ptx && backend != cudavmBackendCuda && !has_module_descriptor(selected->code))
        return compile_shader_ptx(selected->code, backend);
    return selected->code;
}
std::string_view fatbin_memory_view(const void* image) {
    try { return fatbin_memory(image); }
    catch (const std::exception& error) { throw failure(CUDA_ERROR_INVALID_IMAGE, error.what()); }
}
}
