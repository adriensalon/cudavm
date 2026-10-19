cudavm_add_component(nvrtc compiler_api.cpp compiler_resources.cpp cuda_frontend.cpp nvrtc_api.cpp nvrtc_frontend.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/embedded_headers.cpp"
    "${engine_dir}/nvptx_emitter.cpp"
    "${engine_dir}/native_emission.cpp"
    "${engine_dir}/spirv_emitter.cpp"
    "${engine_dir}/shader_lowering.cpp"
    "${engine_dir}/dxil_emitter.cpp"
    "${engine_dir}/module_descriptor.cpp"
    "${engine_dir}/device_passes.cpp" "${engine_dir}/host_embedding.cpp" "${engine_dir}/compiler_io.cpp")
set(cudavm_install_targets ${cudavm_install_targets} PARENT_SCOPE)
foreach(linkage shared static)
    if(NOT TARGET cudavm_nvrtc_${linkage})
        continue()
    endif()
    cudavm_compiler_dependencies(cudavm_nvrtc_${linkage} clangCodeGen ${source_libraries} ${link_libraries})
    target_include_directories(cudavm_nvrtc_${linkage} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_definitions(cudavm_nvrtc_${linkage} PRIVATE
        CUDAVM_INCLUDES_FROM_BIN="${includes_from_bin}"
        CUDAVM_INCLUDES_FROM_LIB="${includes_from_lib}"
        CUDAVM_PACKAGE_FROM_BIN="${package_from_bin}"
        CUDAVM_PACKAGE_FROM_LIB="${package_from_lib}"
        CUDAVM_INSTALL_INCLUDE_DIR="${CMAKE_INSTALL_INCLUDEDIR}"
        CUDAVM_BUILD_MODULE_DIR="${CMAKE_CURRENT_BINARY_DIR}"
        CUDAVM_BUILD_NVCC_DIR="${PROJECT_BINARY_DIR}/source/nvcc"
        CUDAVM_CUDA_INCLUDE_DIR="${CUDAVM_CUDA_INCLUDE_DIR}"
        CUDAVM_CUDA_CRT_INCLUDE_DIR="${CUDAVM_CUDA_CRT_INCLUDE_DIR}"
        CUDAVM_PUBLIC_INCLUDE_DIR="${PROJECT_SOURCE_DIR}/include")
endforeach()
# Device-only NVRTC defaults live in the library. Preserve the exact licensed
# header bytes; no installed compiler/resource path is used by this frontend.
file(READ "${PROJECT_SOURCE_DIR}/include/cudart/vector_types.h" nvrtc_vector_types)
file(READ "${PROJECT_SOURCE_DIR}/include/nvcc/crt/host_defines.h" nvrtc_host_defines)
file(READ "${CUDAVM_CLANG_RESOURCE_DIR}/include/__clang_cuda_builtin_vars.h" nvrtc_builtin_variables)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/include/cudart/vector_types.h"
    "${PROJECT_SOURCE_DIR}/include/nvcc/crt/host_defines.h"
    "${CUDAVM_CLANG_RESOURCE_DIR}/include/__clang_cuda_builtin_vars.h")
set(nvrtc_headers_source "// Embedded, unmodified NVIDIA and LLVM headers; license notices retained.\n#include \"embedded_headers.hpp\"\nnamespace cudavm {\ninline constexpr char nvrtc_vector_types[] = R\"cudavm(${nvrtc_vector_types})cudavm\";\ninline constexpr char nvrtc_host_defines[] = R\"cudavm(${nvrtc_host_defines})cudavm\";\ninline constexpr char nvrtc_builtin_variables[] = R\"cudavm(${nvrtc_builtin_variables})cudavm\";\n")
set(nvrtc_image_entries)
foreach(header device_types.h driver_types.h builtin_types.h texture_types.h surface_types.h
        texture_indirect_functions.h surface_indirect_functions.h cuda_runtime_api.h __clang_cuda_texture_intrinsics.h)
    if(header STREQUAL "__clang_cuda_texture_intrinsics.h")
        set(header_path "${CUDAVM_CLANG_RESOURCE_DIR}/include/${header}")
    else()
        set(header_path "${PROJECT_SOURCE_DIR}/include/cudart/${header}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${header_path}")
    file(READ "${header_path}" header_contents)
    string(MAKE_C_IDENTIFIER "${header}" header_id)
    string(APPEND nvrtc_headers_source "inline constexpr char nvrtc_image_${header_id}[] = R\"cudavm(${header_contents})cudavm\";\n")
    string(APPEND nvrtc_image_entries "{\"${header}\", nvrtc_image_${header_id}, sizeof(nvrtc_image_${header_id})-1},\n")
endforeach()
string(APPEND nvrtc_headers_source "inline constexpr embedded_file nvrtc_image_headers[] = {\n${nvrtc_image_entries}};\n}\n")
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/nvrtc_headers.hpp" CONTENT "${nvrtc_headers_source}" @ONLY)
# The complete Clang builtin-header set is an in-memory resource, not an SDK
# directory. User and pinned NVIDIA headers keep their ordinary include paths.
file(GLOB_RECURSE builtin_headers CONFIGURE_DEPENDS RELATIVE "${CUDAVM_CLANG_RESOURCE_DIR}/include"
    "${CUDAVM_CLANG_RESOURCE_DIR}/include/*")
set(embedded_source "// Embedded LLVM headers, with original license notices.\n#include \"embedded_headers.hpp\"\nnamespace cudavm {\n")
set(embedded_entries)
set(header_index 0)
foreach(header IN LISTS builtin_headers)
    set(header_path "${CUDAVM_CLANG_RESOURCE_DIR}/include/${header}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${header_path}")
    file(READ "${header_path}" header_contents)
    string(FIND "${header_contents}" ")cudavm_header\"" delimiter_collision)
    if(NOT delimiter_collision EQUAL -1)
        message(FATAL_ERROR "Embedded header contains the raw-string delimiter: ${header}")
    endif()
    string(APPEND embedded_source "static constexpr char header_${header_index}[] = R\"cudavm_header(${header_contents})cudavm_header\";\n")
    string(APPEND embedded_entries "{\"include/${header}\", header_${header_index}, sizeof(header_${header_index}) - 1},\n")
    math(EXPR header_index "${header_index} + 1")
endforeach()
file(READ "${wrapper_path}" header_contents)
string(APPEND embedded_source "static constexpr char wrapper[] = R\"cudavm_header(${header_contents})cudavm_header\";\nstatic constexpr embedded_file files[] = {\n${embedded_entries}{\"cuda_runtime_wrapper.h\", wrapper, sizeof(wrapper)-1}\n};\nstd::span<const embedded_file> embedded_headers() { return files; }\n}\n")
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/embedded_headers.cpp" CONTENT "${embedded_source}" @ONLY)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/embedded_headers.cpp"
        PROPERTIES COMPILE_OPTIONS -Wno-overlength-strings)
    set_source_files_properties(nvrtc_frontend.cpp PROPERTIES COMPILE_OPTIONS -Wno-overlength-strings)
endif()
install(FILES "${CUDAVM_LLVM_LICENSE}"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licences" RENAME LLVM-LICENSE.txt)
