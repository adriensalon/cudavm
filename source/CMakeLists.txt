# SDK modules share one configure scope; their C++ ownership stays module-local.
set(cudavm_install_targets)
function(cudavm_add_component component)
    string(TOUPPER "${component}" export_component)
    foreach(linkage shared static)
        string(TOUPPER "${linkage}" library_kind)
        set(target "cudavm_${component}_${linkage}")
        add_library(${target} ${library_kind} ${ARGN})
        add_library(cudavm::${component}_${linkage} ALIAS ${target})
        set_target_properties(${target} PROPERTIES
            OUTPUT_NAME "cudavm_${component}$<$<STREQUAL:${linkage},static>:_static>")
        set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
            LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}" ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
        target_compile_features(${target} PRIVATE cxx_std_20)
        target_include_directories(${target} PUBLIC
            "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
        # Preserve NVIDIA include spellings while retaining module ownership.
        set(header_modules ${component})
        if(component STREQUAL "core")
            set(header_modules) # The compiler service has no user-facing headers.
        endif()
        if(component STREQUAL "cuda" OR component STREQUAL "cudart")
            list(APPEND header_modules cuda cudart cudadevrt nvcc)
        endif()
        list(REMOVE_DUPLICATES header_modules)
        foreach(module IN LISTS header_modules)
            target_include_directories(${target} SYSTEM PUBLIC
                "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/${module}>"
                "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${module}>")
        endforeach()
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
        if(linkage STREQUAL "shared")
            target_compile_definitions(${target} PRIVATE CUDAVM_BUILDING_${export_component})
            set_target_properties(${target} PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
        else()
            target_compile_definitions(${target} PUBLIC CUDAVM_STATIC)
        endif()
        if(WIN32)
            set(crt MultiThreaded)
            target_compile_definitions(${target} PRIVATE NDEBUG _ITERATOR_DEBUG_LEVEL=0)
            set_property(TARGET ${target} PROPERTY MSVC_RUNTIME_LIBRARY "${crt}")
            if(linkage STREQUAL "static")
                set_property(TARGET ${target} PROPERTY INTERFACE_CUDAVM_MSVC_RUNTIME_LIBRARY "${crt}")
                target_compile_definitions(${target} INTERFACE NDEBUG _ITERATOR_DEBUG_LEVEL=0)
            endif()
        endif()
        if(MSVC)
            target_compile_options(${target} PRIVATE /W4 /permissive-)
        else()
            target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        endif()
        set_property(TARGET ${target} PROPERTY INTERFACE_CUDAVM_LINKAGE ${linkage})
        set_property(TARGET ${target} PROPERTY CUDAVM_COMPILED_BACKENDS "${CUDAVM_ENABLED_BACKENDS}")
        set_property(TARGET ${target} APPEND PROPERTY COMPATIBLE_INTERFACE_STRING CUDAVM_LINKAGE CUDAVM_MSVC_RUNTIME_LIBRARY)
        list(APPEND cudavm_install_targets ${target})
    endforeach()
    set(cudavm_install_targets ${cudavm_install_targets} PARENT_SCOPE)
endfunction()

# Driver and backend services.
if(NOT WIN32)
    find_package(Threads REQUIRED)
endif()
set(shader_compiler_files)
set(backend_sources)
foreach(backend IN LISTS CUDAVM_ENABLED_BACKENDS)
    list(APPEND backend_sources "cuda/${backend}_backend.cpp")
endforeach()
cudavm_add_component(cuda cuda/driver_api.cpp cuda/procedure_address.cpp cuda/opengl_graphics.cpp cuda/registry.cpp cuda/host_memory.cpp
    cuda/shader_module.cpp cuda/shader_services.cpp cuda/shader_ptx.cpp ${backend_sources})
foreach(linkage shared static)
    foreach(backend IN LISTS CUDAVM_ENABLED_BACKENDS)
        string(TOUPPER "${backend}" name)
        target_compile_definitions(cudavm_cuda_${linkage} PRIVATE CUDAVM_ENABLE_${name}=1)
    endforeach()
    foreach(dependency opencl vulkan directx)
        if(TARGET cudavm_${dependency}_headers)
            target_link_libraries(cudavm_cuda_${linkage} PRIVATE "$<BUILD_INTERFACE:cudavm_${dependency}_headers>")
        endif()
    endforeach()
    if(WIN32 AND CUDAVM_ENABLE_OPENGL)
        target_link_libraries(cudavm_cuda_${linkage} PRIVATE user32 gdi32)
    elseif(NOT WIN32)
        target_link_libraries(cudavm_cuda_${linkage} PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
    endif()
endforeach()
if(WIN32 AND CUDAVM_ENABLE_DIRECTX)
    foreach(linkage shared static)
        cudavm_directx_dependencies(cudavm_cuda_${linkage})
    endforeach()
endif()

# Runtime and LLVM-independent fatbinary packaging.
cudavm_add_component(cudart cudart/runtime_api.cpp)
foreach(linkage shared static)
    target_link_libraries(cudavm_cudart_${linkage} PUBLIC cudavm_cuda_${linkage})
    if(UNIX)
        set_property(TARGET cudavm_cudart_${linkage} PROPERTY INSTALL_RPATH "$ORIGIN")
    endif()
endforeach()
cudavm_add_component(nvfatbin nvfatbin/nvfatbin_api.cpp)
set(core_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/core")
# Shared compiler implementation and sole module owning static Clang/LLVM.
# Facades link the matching variant; the Driver discovers the shared service lazily.
cudavm_add_component(core core/result_api.cpp core/program_api.cpp core/frontend_api.cpp core/linker_api.cpp core/image_api.cpp
    core/compiler.cpp core/program.cpp
    core/device_sources.cpp core/cuda_frontend.cpp core/nvrtc_frontend.cpp
    "${core_binary_dir}/embedded_headers.cpp"
    core/ptx_import.cpp core/cubin_import.cpp core/nvvm_import.cpp core/nvidia_lto_import.cpp core/device_numeric_ir.cpp core/device_services.cpp core/nvptx_emitter.cpp core/native_emission.cpp
    core/spirv_emitter.cpp core/shader_lowering.cpp core/dxil_emitter.cpp core/module_descriptor.cpp
    core/device_passes.cpp core/host_embedding.cpp)
foreach(linkage shared static)
    set(target cudavm_core_${linkage})
    target_include_directories(${target} PRIVATE "${core_binary_dir}")
    target_compile_definitions(${target} PRIVATE
        CUDAVM_INCLUDES_FROM_BIN="${includes_from_bin}"
        CUDAVM_INCLUDES_FROM_LIB="${includes_from_lib}"
        CUDAVM_SDK_NOTICE_FROM_BIN="${sdk_notice_from_bin}"
        CUDAVM_SDK_NOTICE_FROM_LIB="${sdk_notice_from_lib}"
        CUDAVM_INSTALL_INCLUDE_DIR="${CMAKE_INSTALL_INCLUDEDIR}"
        CUDAVM_BUILD_MODULE_DIR="${PROJECT_BINARY_DIR}"
        CUDAVM_BUILD_NVCC_DIR="${PROJECT_BINARY_DIR}/source/nvcc"
        CUDAVM_CUDA_INCLUDE_DIR="${CUDAVM_CUDA_INCLUDE_DIR}"
        CUDAVM_CUDA_CRT_INCLUDE_DIR="${CUDAVM_CUDA_CRT_INCLUDE_DIR}"
        CUDAVM_PUBLIC_INCLUDE_DIR="${PROJECT_SOURCE_DIR}/include")
    cudavm_core_dependencies(${target} clangCodeGen ${source_libraries} ${link_libraries})
endforeach()
list(APPEND shader_compiler_files "$<TARGET_FILE:cudavm_core_shared>")
# Runtime compilation uses embedded compiler headers, the common public CUDA
# surface and private device sources. AOT also retains installed SDK search paths.
set(embedded_resources)
file(GLOB_RECURSE builtin_headers CONFIGURE_DEPENDS RELATIVE "${clang_resource_dir}/include"
    "${clang_resource_dir}/include/*")
foreach(header IN LISTS builtin_headers)
    list(APPEND embedded_resources "${clang_resource_dir}/include/${header}" "include/${header}")
endforeach()
list(APPEND embedded_resources "${wrapper_path}" "cuda_runtime_wrapper.h")
foreach(module cuda cudart cudadevrt nvcc)
    file(GLOB_RECURSE public_headers CONFIGURE_DEPENDS RELATIVE "${PROJECT_SOURCE_DIR}/include/${module}"
        "${PROJECT_SOURCE_DIR}/include/${module}/*")
    foreach(header IN LISTS public_headers)
        list(APPEND embedded_resources "${PROJECT_SOURCE_DIR}/include/${module}/${header}" "public/${module}/${header}")
    endforeach()
endforeach()
# CUDA numeric and cooperative headers use CCCL's nv/target dispatch macros.
# The root resolves this build-only dependency through external/cccl.
# Embed it here so runtime compilation never searches an installed toolkit.
file(GLOB_RECURSE cccl_headers CONFIGURE_DEPENDS RELATIVE "${CUDAVM_CCCL_INCLUDE_DIR}"
    "${CUDAVM_CCCL_INCLUDE_DIR}/nv/*")
foreach(header IN LISTS cccl_headers)
    list(APPEND embedded_resources "${CUDAVM_CCCL_INCLUDE_DIR}/${header}" "public/${header}")
endforeach()
file(GLOB device_sources CONFIGURE_DEPENDS RELATIVE "${PROJECT_SOURCE_DIR}/source/cudadevrt"
    "${PROJECT_SOURCE_DIR}/source/cudadevrt/*.cu" "${PROJECT_SOURCE_DIR}/source/cudadevrt/*.cu.in")
foreach(source IN LISTS device_sources)
    list(APPEND embedded_resources "${PROJECT_SOURCE_DIR}/source/cudadevrt/${source}" "device/${source}")
endforeach()
set(embedded_includes)
set(embedded_entries)
set(embedded_index 0)
while(embedded_resources)
    list(POP_FRONT embedded_resources path resource)
    set(symbol "embedded_${embedded_index}")
    cudavm_embed_file("${path}" "${core_binary_dir}/embedded/${embedded_index}.inc" "${symbol}")
    string(APPEND embedded_includes "#include \"embedded/${embedded_index}.inc\"\n")
    string(APPEND embedded_entries "    {\"${resource}\", ${symbol}, sizeof(${symbol}) - 1},\n")
    math(EXPR embedded_index "${embedded_index} + 1")
endwhile()
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/core/embedded_headers.cpp.in"
    "${core_binary_dir}/embedded_headers.cpp" @ONLY)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set_source_files_properties("${core_binary_dir}/embedded_headers.cpp"
        PROPERTIES COMPILE_OPTIONS -Wno-overlength-strings)
    set_source_files_properties(core/nvrtc_frontend.cpp PROPERTIES COMPILE_OPTIONS -Wno-overlength-strings)
endif()
install(FILES "${CUDAVM_LLVM_LICENSE}"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licences" RENAME LLVM-LICENSE.txt)

# Compiler facades share linkage and loader-relative compiler dependencies.
foreach(component nvrtc nvptxcompiler nvjitlink nvvm)
    cudavm_add_component(${component} "${component}/${component}_api.cpp")
    foreach(linkage shared static)
        target_link_libraries(cudavm_${component}_${linkage} PUBLIC cudavm::core_${linkage})
        if(APPLE)
            set_property(TARGET cudavm_${component}_${linkage} PROPERTY INSTALL_RPATH "@loader_path")
        elseif(UNIX)
            set_property(TARGET cudavm_${component}_${linkage} PROPERTY INSTALL_RPATH "$ORIGIN")
        endif()
    endforeach()
endforeach()

add_executable(cudavm-nvcc nvcc/main.cpp nvcc/device_inventory.cpp)
add_executable(cudavm::nvcc ALIAS cudavm-nvcc)
if(NOT DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY OR CMAKE_RUNTIME_OUTPUT_DIRECTORY STREQUAL "")
    set_property(TARGET cudavm-nvcc PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/nvcc")
endif()
if(NOT DEFINED CMAKE_ARCHIVE_OUTPUT_DIRECTORY OR CMAKE_ARCHIVE_OUTPUT_DIRECTORY STREQUAL "")
    set_property(TARGET cudavm-nvcc PROPERTY ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/nvcc")
endif()
set_target_properties(cudavm-nvcc PROPERTIES MSVC_RUNTIME_LIBRARY MultiThreaded)
target_compile_features(cudavm-nvcc PRIVATE cxx_std_20)
target_link_libraries(cudavm-nvcc PRIVATE cudavm::core_shared)
target_include_directories(cudavm-nvcc SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/include/cuda")
# Device discovery is optional; ordinary compiler requests do not load Driver.
set(driver_install_dir "${CMAKE_INSTALL_LIBDIR}")
if(WIN32)
    set(driver_install_dir "${CMAKE_INSTALL_BINDIR}")
endif()
file(RELATIVE_PATH driver_from_bin "/${CMAKE_INSTALL_BINDIR}" "/${driver_install_dir}")
target_compile_definitions(cudavm-nvcc PRIVATE
    CUDAVM_NVCC_BUILD_DIRECTORY="$<TARGET_FILE_DIR:cudavm-nvcc>"
    CUDAVM_NVCC_BUILD_DRIVER="$<TARGET_FILE:cudavm::cuda_shared>"
    CUDAVM_NVCC_DRIVER_DIRECTORY="${driver_from_bin}"
    CUDAVM_NVCC_DRIVER_FILENAME="$<TARGET_FILE_NAME:cudavm::cuda_shared>")
target_link_libraries(cudavm-nvcc PRIVATE ${CMAKE_DL_LIBS})
if(WIN32)
    target_compile_definitions(cudavm-nvcc PRIVATE NDEBUG _ITERATOR_DEBUG_LEVEL=0)
endif()
if(UNIX)
    file(RELATIVE_PATH libraries_from_bin "/${CMAKE_INSTALL_BINDIR}" "/${CMAKE_INSTALL_LIBDIR}")
    if(APPLE)
        set_property(TARGET cudavm-nvcc PROPERTY INSTALL_RPATH "@loader_path/${libraries_from_bin}")
    else()
        set_property(TARGET cudavm-nvcc PROPERTY INSTALL_RPATH "$ORIGIN/${libraries_from_bin}")
    endif()
endif()
foreach(linkage shared static)
    set_property(TARGET cudavm_cuda_${linkage} PROPERTY CUDAVM_RUNTIME_FILES "${shader_compiler_files}")
endforeach()
cudavm_prepare_target(cudavm-nvcc)
install(TARGETS cudavm-nvcc RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

set(cudavm_install_targets ${cudavm_install_targets} PARENT_SCOPE)
