#include "shader_ptx.hpp"
#include "dynamic_library.hpp"
#include "../core/image_api.hpp"
#include "../core/module_descriptor.hpp"
namespace cudavm {
namespace {
dynamic_library& image_compiler() {
    // Retry discovery after an unavailable service; a failed first load must
    // not prevent a later application from deploying the optional compiler.
#if defined(_WIN32)
    static dynamic_library compiler({"cudavm_core.dll"}, true, CUDA_ERROR_JIT_COMPILER_NOT_FOUND);
#elif defined(__APPLE__)
    static dynamic_library compiler({"libcudavm_core.dylib"}, true, CUDA_ERROR_JIT_COMPILER_NOT_FOUND);
#else
    static dynamic_library compiler({"libcudavm_core.so"}, true, CUDA_ERROR_JIT_COMPILER_NOT_FOUND);
#endif
    return compiler;
}
std::string compile_image(std::string_view input, cudavmBackend backend, compiler_input_kind kind) {
    auto& compiler = image_compiler();
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
std::string compile_shader_cubin(std::string_view input, cudavmBackend backend) {
    return compile_image(input, backend, compiler_input_kind::cubin);
}
std::string link_shader_images(std::span<const std::string_view> inputs, cudavmBackend backend) {
    std::vector<std::string> ptx; unsigned target = 0; size_t bytes = 0;
    for (auto input : inputs) {
        require(!has_nvidia_cubin(input), CUDA_ERROR_NOT_SUPPORTED, "unfinished shader device links require relocatable PTX in every input");
        std::vector<fatbin_record> records;
        try { records = decode_fatbin(input); }
        catch (const std::bad_alloc&) { throw failure(CUDA_ERROR_OUT_OF_MEMORY, "NVIDIA device link input allocation failed"); }
        catch (const nvidia_container_error& error) { throw failure(error.unsupported ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_INVALID_IMAGE, error.what()); }
        catch (const std::exception& error) { throw failure(CUDA_ERROR_INVALID_IMAGE, error.what()); }
        auto* selected = select_fatbin(records, 0, true, false);
        require(selected && selected->kind == fatbin_kind::ptx, CUDA_ERROR_NOT_SUPPORTED, "unfinished shader device links require PTX in every input");
        require(selected->code.size() <= fatbin_limit - bytes, CUDA_ERROR_INVALID_IMAGE, "oversized NVIDIA device link PTX inputs");
        bytes += selected->code.size();
        target = std::max(target, selected->architecture); ptx.push_back(selected->code);
    }
    auto architecture = "sm_" + std::to_string(target);
    auto& compiler = image_compiler();
    auto import = compiler.get<decltype(&cudavm_core_program_import_v1)>("cudavm_core_program_import_v1");
    auto link = compiler.get<decltype(&cudavm_core_program_link_v1)>("cudavm_core_program_link_v1");
    auto prepare = compiler.get<decltype(&cudavm_core_program_prepare_v1)>("cudavm_core_program_prepare_v1");
    auto emit = compiler.get<decltype(&cudavm_core_program_emit_v1)>("cudavm_core_program_emit_v1");
    auto release = compiler.get<decltype(&cudavm_core_program_release_v1)>("cudavm_core_program_release_v1");
    auto release_result = compiler.get<decltype(&cudavm_core_result_release_v1)>("cudavm_core_result_release_v1");
    auto diagnostics = compiler.get<decltype(&cudavm_core_result_diagnostics_v1)>("cudavm_core_result_diagnostics_v1");
    auto output = compiler.get<decltype(&cudavm_core_result_output_v1)>("cudavm_core_result_output_v1");
    auto size = compiler.get<decltype(&cudavm_core_result_size_v1)>("cudavm_core_result_size_v1");
    using owned_program = std::unique_ptr<compiler_program, decltype(release)>;
    auto check_result = [&](compiler_status status, compiler_result* result) {
        std::unique_ptr<compiler_result, decltype(release_result)> owned(result, release_result);
        if (status != compiler_status::success) throw failure(status == compiler_status::out_of_memory ? CUDA_ERROR_OUT_OF_MEMORY
            : status == compiler_status::unsupported || status == compiler_status::unsupported_ptx_version ? CUDA_ERROR_NOT_SUPPORTED
            : CUDA_ERROR_INVALID_IMAGE, diagnostics(result));
    };
    std::vector<owned_program> programs; std::vector<const compiler_program*> borrowed;
    for (size_t i = 0; i < ptx.size(); ++i) {
        auto name = "nvcc-rdc:" + std::to_string(i);
        compiler_input_request_v3 request;
        request.base.base = {3, compiler_input_kind::ptx, ptx[i].data(), ptx[i].size(), name.c_str(), nullptr};
        request.base.architecture = architecture.c_str(); request.relocatable = 1;
        compiler_program* program = nullptr; compiler_result* result = nullptr;
        auto status = import(&request.base.base, &program, &result);
        owned_program owned(program, release); check_result(status, result);
        borrowed.push_back(program); programs.push_back(std::move(owned));
    }
    compiler_program* program = nullptr; compiler_result* result = nullptr;
    auto status = link(borrowed.size(), borrowed.data(), &program, &result);
    owned_program linked(program, release); check_result(status, result);
    compiler_pipeline_request pipeline{1, 1, 1, 0};
    status = prepare(linked.get(), &pipeline, &program, &result);
    owned_program prepared(program, release); check_result(status, result);
    compiler_emit_request emission;
    emission.kind = compiler_output_kind::native_image; emission.environment = compiler_environment(backend);
    status = emit(prepared.get(), &emission, &result);
    std::unique_ptr<compiler_result, decltype(release_result)> emitted(result, release_result);
    if (status != compiler_status::success) { emitted.release(); check_result(status, result); }
    return {output(result), size(result)};
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
    catch (const nvidia_container_error& error) { throw failure(error.unsupported ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_INVALID_IMAGE, error.what()); }
    catch (const std::exception& error) { throw failure(CUDA_ERROR_INVALID_IMAGE, error.what()); }
    auto* selected = select_fatbin(records, architecture, false, backend == cudavmBackendCuda);
    if (!selected && has_nvidia_fatbin(input)
        && std::none_of(records.begin(), records.end(), [](const auto& entry) { return entry.kind == fatbin_kind::ptx; }))
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "NVIDIA fatbinary has no supported PTX, SM75 cubin or program image member");
    require(selected, CUDA_ERROR_NO_BINARY_FOR_GPU, "fatbinary has no compatible executable image");
    if (has_program_image(std::string_view(selected->code))) return materialize_program_image(selected->code, backend);
    if (selected->kind == fatbin_kind::cubin && backend != cudavmBackendCuda)
        return compile_shader_cubin(selected->code, backend);
    if (selected->kind == fatbin_kind::ptx && backend != cudavmBackendCuda && !has_module_descriptor(selected->code)) {
        // Member framing must not hide appended descriptors when the result
        // crosses a NUL-terminated module/file loader boundary.
        auto ptx = std::string_view(selected->code);
        if (!ptx.empty() && ptx.back() == 0) ptx.remove_suffix(1);
        return compile_shader_ptx(ptx, backend);
    }
    return selected->code;
}
std::string_view fatbin_memory_view(const void* image) {
    try { return fatbin_memory(image); }
    catch (const nvidia_container_error& error) { throw failure(error.unsupported ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_INVALID_IMAGE, error.what()); }
    catch (const std::exception& error) { throw failure(CUDA_ERROR_INVALID_IMAGE, error.what()); }
}
}
