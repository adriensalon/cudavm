# Shared compiler implementation and sole module owning static Clang/LLVM.
# Facades link the matching variant; the Driver discovers the shared service lazily.
cudavm_add_component(core compiler_api.cpp program_api.cpp translation_unit_api.cpp linker_api.cpp
    compiler_resources.cpp device_sources.cpp cuda_frontend.cpp nvrtc_frontend.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/embedded_headers.cpp"
    ptx_import.cpp image_api.cpp device_numeric_ir.cpp device_services.cpp nvptx_emitter.cpp native_emission.cpp
    spirv_emitter.cpp shader_lowering.cpp dxil_emitter.cpp module_descriptor.cpp
    device_passes.cpp host_embedding.cpp compiler_io.cpp)
set(cudavm_install_targets ${cudavm_install_targets} PARENT_SCOPE)
foreach(linkage shared static)
    set(target cudavm_core_${linkage})
    if(NOT TARGET ${target})
        continue()
    endif()
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_definitions(${target} PRIVATE
        CUDAVM_INCLUDES_FROM_BIN="${includes_from_bin}"
        CUDAVM_INCLUDES_FROM_LIB="${includes_from_lib}"
        CUDAVM_PACKAGE_FROM_BIN="${package_from_bin}"
        CUDAVM_PACKAGE_FROM_LIB="${package_from_lib}"
        CUDAVM_INSTALL_INCLUDE_DIR="${CMAKE_INSTALL_INCLUDEDIR}"
        CUDAVM_BUILD_MODULE_DIR="${PROJECT_BINARY_DIR}"
        CUDAVM_BUILD_NVCC_DIR="${PROJECT_BINARY_DIR}/source/nvcc"
        CUDAVM_CUDA_INCLUDE_DIR="${CUDAVM_CUDA_INCLUDE_DIR}"
        CUDAVM_CUDA_CRT_INCLUDE_DIR="${CUDAVM_CUDA_CRT_INCLUDE_DIR}"
        CUDAVM_PUBLIC_INCLUDE_DIR="${PROJECT_SOURCE_DIR}/include")
    cudavm_core_dependencies(${target} clangCodeGen ${source_libraries} ${link_libraries})
endforeach()
if(TARGET cudavm_core_shared)
    list(APPEND CUDAVM_SHADER_COMPILER_FILES "$<TARGET_FILE:cudavm_core_shared>")
    set(CUDAVM_SHADER_COMPILER_FILES ${CUDAVM_SHADER_COMPILER_FILES} PARENT_SCOPE)
endif()
# Runtime compilation uses embedded compiler headers, the common public CUDA
# surface and private device sources. AOT also retains installed SDK search paths.
set(embedded_includes)
set(embedded_entries)
set(embedded_index 0)
macro(cudavm_embed_file path resource)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${path}")
    file(READ "${path}" embedded_contents)
    string(FIND "${embedded_contents}" ")cudavm_resource\"" delimiter_collision)
    if(NOT delimiter_collision EQUAL -1)
        message(FATAL_ERROR "Embedded source contains the raw-string delimiter: ${path}")
    endif()
    set(embedded_symbol "embedded_${embedded_index}")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/embedded_file.inc.in"
        "${CMAKE_CURRENT_BINARY_DIR}/embedded/${embedded_index}.inc" @ONLY)
    string(APPEND embedded_includes "#include \"embedded/${embedded_index}.inc\"\n")
    string(APPEND embedded_entries "    {\"${resource}\", ${embedded_symbol}, sizeof(${embedded_symbol}) - 1},\n")
    math(EXPR embedded_index "${embedded_index} + 1")
endmacro()
file(GLOB_RECURSE builtin_headers CONFIGURE_DEPENDS RELATIVE "${CUDAVM_CLANG_RESOURCE_DIR}/include"
    "${CUDAVM_CLANG_RESOURCE_DIR}/include/*")
foreach(header IN LISTS builtin_headers)
    cudavm_embed_file("${CUDAVM_CLANG_RESOURCE_DIR}/include/${header}" "include/${header}")
endforeach()
cudavm_embed_file("${wrapper_path}" "cuda_runtime_wrapper.h")
foreach(module cuda cudart cudadevrt nvcc)
    file(GLOB_RECURSE public_headers CONFIGURE_DEPENDS RELATIVE "${PROJECT_SOURCE_DIR}/include/${module}"
        "${PROJECT_SOURCE_DIR}/include/${module}/*")
    foreach(header IN LISTS public_headers)
        cudavm_embed_file("${PROJECT_SOURCE_DIR}/include/${module}/${header}" "public/${module}/${header}")
    endforeach()
endforeach()
# CUDA numeric and cooperative headers use CCCL's nv/target dispatch macros.
# These are a build input only; never search an installed toolkit at runtime.
find_path(CUDAVM_CCCL_INCLUDE_DIR nv/target
    HINTS "$ENV{CUDA_PATH}/include" "$ENV{CUDA_PATH}/include/cccl" REQUIRED)
file(GLOB_RECURSE cccl_headers CONFIGURE_DEPENDS RELATIVE "${CUDAVM_CCCL_INCLUDE_DIR}"
    "${CUDAVM_CCCL_INCLUDE_DIR}/nv/*")
foreach(header IN LISTS cccl_headers)
    cudavm_embed_file("${CUDAVM_CCCL_INCLUDE_DIR}/${header}" "public/${header}")
endforeach()
file(GLOB device_sources CONFIGURE_DEPENDS RELATIVE "${PROJECT_SOURCE_DIR}/source/cudadevrt"
    "${PROJECT_SOURCE_DIR}/source/cudadevrt/*.cu" "${PROJECT_SOURCE_DIR}/source/cudadevrt/*.cu.in")
foreach(source IN LISTS device_sources)
    cudavm_embed_file("${PROJECT_SOURCE_DIR}/source/cudadevrt/${source}" "device/${source}")
endforeach()
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/embedded_headers.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/embedded_headers.cpp" @ONLY)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/embedded_headers.cpp"
        PROPERTIES COMPILE_OPTIONS -Wno-overlength-strings)
    set_source_files_properties(nvrtc_frontend.cpp PROPERTIES COMPILE_OPTIONS -Wno-overlength-strings)
endif()
install(FILES "${CUDAVM_LLVM_LICENSE}"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licences" RENAME LLVM-LICENSE.txt)
