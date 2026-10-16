#pragma once
#include "dynamic_library.hpp"
#if defined(CUDAVM_SPIRV_CROSS_LIBRARY)
#include <spirv_cross/spirv_cross_c.h>
#endif

namespace cudavm {
inline constexpr bool native_glsl_configured =
#if defined(CUDAVM_SPIRV_CROSS_LIBRARY)
    true;
#else
    false;
#endif

// Translate the emitted native instruction graph, with its resource ABI and
// specialized workgroup dimensions, into the driver's GLSL ingestion format.
// No legacy CUDA instruction interpreter or source payload participates.
inline std::string native_glsl(std::span<const uint32_t> words) {
#if defined(CUDAVM_SPIRV_CROSS_LIBRARY)
#if defined(CUDAVM_NATIVE_SPIRV_OPTIMIZER)
    static dynamic_library optimizer({"cudavm_spirv_optimizer.dll","cudavm_spirv_optimizer.so",CUDAVM_NATIVE_SPIRV_OPTIMIZER},true,CUDA_ERROR_JIT_COMPILER_NOT_FOUND);
    using optimize_type=int (*)(const uint32_t*,size_t,void (*)(const uint32_t*,size_t,void*),void*,char*,size_t);
    struct optimized_words {std::vector<uint32_t> code;std::exception_ptr error;} optimized;
    auto receive=+[](const uint32_t* data,size_t count,void* opaque) noexcept {
        auto& result=*static_cast<optimized_words*>(opaque);
        try {result.code.assign(data,data+count);} catch(...) {result.error=std::current_exception();}
    };
    char diagnostic[2048]{};auto status=optimizer.get<optimize_type>("cudavmOptimizeNativeSpirv")(words.data(),words.size(),receive,&optimized,diagnostic,sizeof(diagnostic));
    if(optimized.error)std::rethrow_exception(optimized.error);
    if(status)throw failure(status==2 ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_INVALID_IMAGE,std::string("native OpenGL SPIR-V optimization failed: ")+diagnostic);
    words=optimized.code;
#endif
    static dynamic_library api({"spirv-cross-c-shared.dll","libspirv-cross-c-shared.so",CUDAVM_SPIRV_CROSS_LIBRARY},true,CUDA_ERROR_JIT_COMPILER_NOT_FOUND);
    auto create=api.get<decltype(&spvc_context_create)>("spvc_context_create");
    auto destroy=api.get<decltype(&spvc_context_destroy)>("spvc_context_destroy");
    auto error=api.get<decltype(&spvc_context_get_last_error_string)>("spvc_context_get_last_error_string");
    spvc_context context{};
    require(create(&context)==SPVC_SUCCESS && context,CUDA_ERROR_OUT_OF_MEMORY,"native GLSL compiler context unavailable");
    struct guard { spvc_context value; decltype(destroy) release; ~guard(){release(value);} } owned{context,destroy};
    auto check=[&](spvc_result result) {
        if(result!=SPVC_SUCCESS) throw failure(result==SPVC_ERROR_UNSUPPORTED_SPIRV ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_INVALID_IMAGE,
            std::string("native SPIR-V to GLSL emission failed: ")+error(context));
    };
    spvc_parsed_ir ir{};spvc_compiler compiler{};spvc_compiler_options options{};
    check(api.get<decltype(&spvc_context_parse_spirv)>("spvc_context_parse_spirv")(context,words.data(),words.size(),&ir));
    check(api.get<decltype(&spvc_context_create_compiler)>("spvc_context_create_compiler")(context,SPVC_BACKEND_GLSL,ir,SPVC_CAPTURE_MODE_TAKE_OWNERSHIP,&compiler));
    check(api.get<decltype(&spvc_compiler_create_compiler_options)>("spvc_compiler_create_compiler_options")(compiler,&options));
    check(api.get<decltype(&spvc_compiler_options_set_uint)>("spvc_compiler_options_set_uint")(options,SPVC_COMPILER_OPTION_GLSL_VERSION,430));
    auto set_bool=api.get<decltype(&spvc_compiler_options_set_bool)>("spvc_compiler_options_set_bool");
    check(set_bool(options,SPVC_COMPILER_OPTION_GLSL_ES,SPVC_FALSE));
    check(set_bool(options,SPVC_COMPILER_OPTION_GLSL_VULKAN_SEMANTICS,SPVC_FALSE));
    check(api.get<decltype(&spvc_compiler_install_compiler_options)>("spvc_compiler_install_compiler_options")(compiler,options));
    const char* source{};check(api.get<decltype(&spvc_compiler_compile)>("spvc_compiler_compile")(compiler,&source));
    require(source && *source,CUDA_ERROR_INVALID_IMAGE,"native GLSL emission returned empty source");
    return source;
#else
    (void)words;throw failure(CUDA_ERROR_NOT_SUPPORTED,"native GLSL emission is not configured");
#endif
}
}
