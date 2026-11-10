if(NOT CUDAVM_DIRECTX_ACTION STREQUAL "configure")
    cmake_minimum_required(VERSION 3.24)
endif()

function(_cudavm_directx_isolate_static)
    execute_process(COMMAND "${NM}" --defined-only --extern-only "${OUTPUT}"
        OUTPUT_VARIABLE symbols COMMAND_ERROR_IS_FATAL ANY)
    string(REGEX MATCHALL " [A-Za-z] (llvm_[A-Za-z0-9_]+|LLVM[A-Za-z0-9_]+)" exports "${symbols}")
    list(REMOVE_DUPLICATES exports)
    if(NOT exports)
        return()
    endif()
    set(mapping "")
    foreach(symbol IN LISTS exports)
        string(REGEX REPLACE "^ [A-Za-z] " "" symbol "${symbol}")
        string(APPEND mapping "${symbol} cudavm_directx_${symbol}\n")
    endforeach()
    file(WRITE "${OUTPUT}.symbols" "${mapping}")
    execute_process(COMMAND "${OBJCOPY}" --strip-debug --remove-section=.llvm_addrsig
        "--redefine-syms=${OUTPUT}.symbols"
        "${OUTPUT}" "${OUTPUT}.isolated" COMMAND_ERROR_IS_FATAL ANY)
    file(RENAME "${OUTPUT}.isolated" "${OUTPUT}")
    file(REMOVE "${OUTPUT}.symbols")
endfunction()

function(_cudavm_directx_configure_upstream)
    include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../CMakeLists.txt")
    find_library(atl_library NAMES atls PATHS ENV LIB NO_DEFAULT_PATH REQUIRED)
    link_libraries("${atl_library}")
    set(symbols "#pragma once\n#define _ATL_NO_DEFAULT_LIBS\n#define llvm cudavm_directx_llvm\n#define clang cudavm_directx_clang\n")
    file(GLOB_RECURSE c_headers "${CMAKE_SOURCE_DIR}/include/llvm-c/*.h")
    set(c_functions)
    foreach(header IN LISTS c_headers)
        file(READ "${header}" contents)
        string(REGEX MATCHALL "LLVM[A-Z][A-Za-z0-9]*[ \r\n]*\\(" functions "${contents}")
        foreach(function IN LISTS functions)
            string(REGEX MATCH "LLVM[A-Za-z0-9_]+" function "${function}")
            list(APPEND c_functions "${function}")
        endforeach()
    endforeach()
    file(READ "${CMAKE_SOURCE_DIR}/include/llvm-c/Core.h" core_header)
    string(REGEX MATCHALL "macro\\([A-Za-z0-9_]+\\)" value_classes "${core_header}")
    foreach(value_class IN LISTS value_classes)
        string(REGEX REPLACE "macro\\(([A-Za-z0-9_]+)\\)" "LLVMIsA\\1" function "${value_class}")
        list(APPEND c_functions "${function}")
    endforeach()
    list(REMOVE_DUPLICATES c_functions)
    foreach(function IN LISTS c_functions)
        string(APPEND symbols "#define ${function} cudavm_directx_${function}\n")
    endforeach()
    file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/cudavm_directx_symbols.hpp" CONTENT "${symbols}" @ONLY)
    add_compile_options("/FI${CMAKE_BINARY_DIR}/cudavm_directx_symbols.hpp")
endfunction()

function(_cudavm_directx_add_validator)
    add_library(cudavm_directx_validator STATIC "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/validator.cpp")
    target_compile_features(cudavm_directx_validator PRIVATE cxx_std_17)
    target_compile_options(cudavm_directx_validator PRIVATE /EHsc /GR)
    set_target_properties(cudavm_directx_validator PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>" MSVC_RUNTIME_LIBRARY MultiThreaded)
    target_include_directories(cudavm_directx_validator PRIVATE
        "${LLVM_MAIN_SRC_DIR}/include" "${LLVM_BINARY_DIR}/include")
    llvm_map_components_to_libnames(validator_libraries
        DxilValidation DxilHash DxilContainer DxilRootSignature DXIL HLSL
        DxcSupport MSSupport BitReader BitWriter Analysis TransformUtils ipa Support)
    find_library(atl_library NAMES atls PATHS ENV LIB NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(cudavm_directx_validator PRIVATE ${validator_libraries} "${atl_library}"
        ole32 advapi32 uuid version user32 shell32 oleaut32 shlwapi)
    set(queue ${validator_libraries})
    set(visited)
    set(archives)
    while(queue)
        list(POP_FRONT queue library)
        string(REGEX REPLACE "^\\$<LINK_ONLY:([^>]+)>$" "\\1" library "${library}")
        if(library IN_LIST visited OR NOT TARGET "${library}")
            continue()
        endif()
        list(APPEND visited "${library}")
        get_target_property(kind "${library}" TYPE)
        if(kind STREQUAL "STATIC_LIBRARY")
            list(APPEND archives "$<TARGET_FILE:${library}>")
        endif()
        get_target_property(dependencies "${library}" INTERFACE_LINK_LIBRARIES)
        if(dependencies)
            list(APPEND queue ${dependencies})
        endif()
    endwhile()
    list(REMOVE_DUPLICATES archives)
    list(APPEND archives "${atl_library}")
    cudavm_merge_static_libraries(cudavm_directx_validator ${archives})
    get_filename_component(tool_directory "${CMAKE_AR}" DIRECTORY)
    find_program(symbol_tool NAMES llvm-nm HINTS "${tool_directory}" REQUIRED)
    find_program(object_tool NAMES llvm-objcopy HINTS "${tool_directory}" REQUIRED)
    add_custom_command(TARGET cudavm_directx_validator POST_BUILD
        COMMAND "${CMAKE_COMMAND}" "-DNM=${symbol_tool}" "-DOBJCOPY=${object_tool}"
            "-DOUTPUT=$<TARGET_FILE:cudavm_directx_validator>"
            -DCUDAVM_DIRECTX_ACTION=isolate_static -P "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
        VERBATIM)
endfunction()

function(_cudavm_directx_build_contract output args_output)
    set(c_compiler "${CMAKE_C_COMPILER}")
    set(cxx_compiler "${CMAKE_CXX_COMPILER}")
    if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        get_filename_component(compiler_directory "${CMAKE_CXX_COMPILER}" DIRECTORY)
        find_program(directx_clang_cl NAMES clang-cl HINTS "${compiler_directory}" NO_CACHE REQUIRED)
        set(c_compiler "${directx_clang_cl}")
        set(cxx_compiler "${directx_clang_cl}")
    endif()
    set(settings
        CMAKE_BUILD_TYPE=Release CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
        LLVM_USE_CRT_RELEASE=MT BUILD_SHARED_LIBS=OFF
        LLVM_TARGETS_TO_BUILD=None LLVM_ENABLE_EH=ON LLVM_ENABLE_RTTI=ON
        LLVM_DEFAULT_TARGET_TRIPLE=dxil-ms-dx
        LLVM_ENABLE_ASSERTIONS=OFF LLVM_ENABLE_WARNINGS=OFF LLVM_ENABLE_ZLIB=OFF LLVM_ENABLE_TERMINFO=OFF
        LLVM_INCLUDE_TESTS=OFF CLANG_INCLUDE_TESTS=OFF HLSL_INCLUDE_TESTS=OFF
        LLVM_INCLUDE_EXAMPLES=OFF LLVM_INCLUDE_DOCS=OFF
        HLSL_BUILD_DXILCONV=OFF HLSL_SUPPORT_QUERY_GIT_COMMIT_INFO=OFF
        HLSL_EMBED_VERSION=ON ENABLE_SPIRV_CODEGEN=OFF SPIRV_BUILD_TESTS=OFF
        CLANG_FORMAT_EXE=OFF
        CLANG_ENABLE_STATIC_ANALYZER=OFF CLANG_ENABLE_ARCMT=OFF
        "CMAKE_C_COMPILER=${c_compiler}"
        "CMAKE_CXX_COMPILER=${cxx_compiler}"
        CUDAVM_DIRECTX_ACTION=configure
        "CMAKE_PROJECT_LLVM_INCLUDE=${CMAKE_CURRENT_FUNCTION_LIST_FILE}")
    set(contract "${CUDAVM_DIRECTX_REVISION};${settings};${CMAKE_GENERATOR};${CMAKE_GENERATOR_PLATFORM};${CMAKE_GENERATOR_TOOLSET};${CMAKE_CXX_COMPILER_VERSION}")
    foreach(file CMakeLists.txt ../../CMakeLists.txt validator.cpp ../../source/cuda/directx_validation.hpp)
        file(SHA256 "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/${file}" digest)
        string(APPEND contract ";${file}:${digest}")
    endforeach()
    string(SHA256 contract "${contract}")
    set(args)
    foreach(setting IN LISTS settings)
        list(APPEND args "-D${setting}")
    endforeach()
    set(${output} "${contract}" PARENT_SCOPE)
    set(${args_output} "${args}" PARENT_SCOPE)
endfunction()

function(cudavm_directx_dependencies target)
    get_target_property(kind ${target} TYPE)
    if(NOT kind STREQUAL "STATIC_LIBRARY")
        target_link_libraries(${target} PRIVATE "$<BUILD_INTERFACE:cudavm_directx_validator>")
        return()
    endif()
    target_link_libraries(${target} PRIVATE ole32 advapi32 uuid version user32 shell32 oleaut32 shlwapi)
    cudavm_merge_static_libraries(${target} "$<TARGET_FILE:cudavm_directx_validator>")
endfunction()

set(CUDAVM_DIRECTX_VERSION "1.9.2607")
set(CUDAVM_DIRECTX_REVISION "0d3ee6b551b8fa768fbf825300ebab81047ef6a8")

if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
    if(CUDAVM_DIRECTX_ACTION STREQUAL "isolate_static")
        _cudavm_directx_isolate_static()
    elseif(CUDAVM_DIRECTX_ACTION STREQUAL "prepare")
        set(source "${CUDAVM_DIRECTX_ROOT}/src")

        if(NOT EXISTS "${source}/.git")
            if(CUDAVM_DIRECTX_VERIFY_ONLY)
                message(FATAL_ERROR "Pinned DXC source is missing: ${source}")
            endif()

            if(EXISTS "${source}")
                message(FATAL_ERROR "Preserving existing non-Git DXC source directory: ${source}")
            endif()

            execute_process(COMMAND "${GIT_EXECUTABLE}" -c core.autocrlf=input
                clone --depth 1 --single-branch --branch "v${CUDAVM_DIRECTX_VERSION}"
                "${CUDAVM_DIRECTX_GIT_REPOSITORY}" "${source}"
                COMMAND_ERROR_IS_FATAL ANY)
        endif()

        set(git "${GIT_EXECUTABLE}" -c "safe.directory=${source}")
        execute_process(COMMAND ${git} rev-parse HEAD WORKING_DIRECTORY "${source}"
            OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)

        if(NOT revision STREQUAL CUDAVM_DIRECTX_REVISION)
            message(FATAL_ERROR "DXC source must be v${CUDAVM_DIRECTX_VERSION} (${CUDAVM_DIRECTX_REVISION}), found ${revision}")
        endif()

        execute_process(COMMAND ${git} status --porcelain --untracked-files=all
            WORKING_DIRECTORY "${source}" OUTPUT_VARIABLE changes COMMAND_ERROR_IS_FATAL ANY)

        if(NOT changes STREQUAL "")
            message(FATAL_ERROR "Preserving local DXC source changes; use a clean checkout: ${changes}")
        endif()
    else()
        message(FATAL_ERROR "Unknown CUDAVM_DIRECTX_ACTION")
    endif()

    return()
endif()

if(CUDAVM_DIRECTX_ACTION STREQUAL "configure")
    _cudavm_directx_configure_upstream()
    cmake_language(DEFER CALL _cudavm_directx_add_validator)
    return()
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND NOT CUDAVM_DIRECTX_ACTION STREQUAL "build")
    message(FATAL_ERROR "Configure cudavm from the repository root; external dependencies are not standalone projects")
endif()

if(NOT WIN32 AND NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    return()
endif()

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    if(NOT CUDAVM_ENABLE_DIRECTX AND NOT CUDAVM_ENABLE_OPENCL)
        return()
    endif()

    include(FetchContent)
    FetchContent_Declare(windows_sdk
        URL https://api.nuget.org/v3-flatcontainer/microsoft.windows.sdk.cpp/10.0.26100.1/microsoft.windows.sdk.cpp.10.0.26100.1.nupkg
        URL_HASH SHA256=b4730a467a8f29145fc0136b2b3f626767e985f6aa6e32f1352953c38d6ff5d2
        DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.tmp/downloads"
        DOWNLOAD_NAME windows_sdk_10.0.26100.1.zip
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE
        SOURCE_SUBDIR c/Include)
    FetchContent_MakeAvailable(windows_sdk)
    set(windows_headers "${windows_sdk_SOURCE_DIR}/c/Include/10.0.26100.0")
    add_library(cudavm_directx_headers INTERFACE)
    target_include_directories(cudavm_directx_headers SYSTEM INTERFACE
        "$<BUILD_INTERFACE:${windows_headers}/um>"
        "$<BUILD_INTERFACE:${windows_headers}/shared>"
        "$<BUILD_INTERFACE:${windows_headers}/winrt>")

    if(NOT CUDAVM_ENABLE_DIRECTX)
        return()
    endif()

    set(directx_root "${PROJECT_SOURCE_DIR}/.tmp/directx/${CUDAVM_DIRECTX_VERSION}")
    set(directx_build_dir "${directx_root}/build")
    set(CUDAVM_DIRECTX_GIT_REPOSITORY "https://github.com/microsoft/DirectXShaderCompiler.git" CACHE STRING "DXC repository or local Git mirror")
    set(CUDAVM_DIRECTX_BUILD_JOBS 4 CACHE STRING "Maximum concurrent DXC compilation jobs")
    _cudavm_directx_build_contract(contract unused_args)
    set(receipt "${directx_build_dir}/cudavm_directx.txt")
    set(archive "${directx_build_dir}/lib/cudavm_directx_validator.lib")
    set(actual_contract "")

    if(EXISTS "${receipt}")
        file(READ "${receipt}" actual_contract)
    endif()

    if(NOT actual_contract STREQUAL contract OR NOT EXISTS "${archive}")
        message(STATUS "Bootstrapping static DirectX validator ${CUDAVM_DIRECTX_VERSION}")
        set(bootstrap_args -S "${CMAKE_CURRENT_SOURCE_DIR}" -B "${directx_root}" -G "${CMAKE_GENERATOR}"
            -DCUDAVM_DIRECTX_ACTION=build
            "-DGIT_EXECUTABLE:FILEPATH=${GIT_EXECUTABLE}"
            "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}" "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DCUDAVM_DIRECTX_GIT_REPOSITORY=${CUDAVM_DIRECTX_GIT_REPOSITORY}"
            "-DCUDAVM_DIRECTX_BUILD_JOBS=${CUDAVM_DIRECTX_BUILD_JOBS}")

        if(CMAKE_MAKE_PROGRAM)
            list(APPEND bootstrap_args "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}")
        endif()

        if(CMAKE_GENERATOR_PLATFORM)
            list(APPEND bootstrap_args -A "${CMAKE_GENERATOR_PLATFORM}")
        endif()

        if(CMAKE_GENERATOR_TOOLSET)
            list(APPEND bootstrap_args -T "${CMAKE_GENERATOR_TOOLSET}")
        endif()

        execute_process(COMMAND "${CMAKE_COMMAND}" ${bootstrap_args} COMMAND_ERROR_IS_FATAL ANY)
        execute_process(COMMAND "${CMAKE_COMMAND}" --build "${directx_root}" --config Release
            COMMAND_ERROR_IS_FATAL ANY)
    endif()

    file(READ "${receipt}" actual_contract)

    if(NOT actual_contract STREQUAL contract)
        message(FATAL_ERROR "Static DXIL validator does not match the selected source/build profile")
    endif()

    execute_process(COMMAND "${CMAKE_COMMAND}" "-DCUDAVM_DIRECTX_ROOT=${directx_root}"
        "-DGIT_EXECUTABLE:FILEPATH=${GIT_EXECUTABLE}"
        -DCUDAVM_DIRECTX_ACTION=prepare -DCUDAVM_DIRECTX_VERIFY_ONLY=ON
        -P "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" COMMAND_ERROR_IS_FATAL ANY)
    add_library(cudavm_directx_validator STATIC IMPORTED GLOBAL)
    set_target_properties(cudavm_directx_validator PROPERTIES
        IMPORTED_CONFIGURATIONS Release IMPORTED_LOCATION_RELEASE "${archive}"
        IMPORTED_LOCATION "${archive}" INTERFACE_LINK_LIBRARIES "ole32;advapi32;uuid;version;user32;shell32;oleaut32;shlwapi")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${receipt}"
        "${CMAKE_CURRENT_SOURCE_DIR}/validator.cpp"
        "${PROJECT_SOURCE_DIR}/source/cuda/directx_validation.hpp")
    install(FILES "${directx_root}/src/LICENSE.TXT"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/licences" RENAME DXC_LICENSE.txt)
    install(FILES "${directx_root}/src/ThirdPartyNotices.txt"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/licences" RENAME DXC_THIRD_PARTY_NOTICES.txt)
    return()
endif()

project(cudavm_directx LANGUAGES C CXX)

if(NOT WIN32)
    message(FATAL_ERROR "cudavm's DirectX backend requires Windows")
endif()

find_path(CUDAVM_DIRECTX_ATL_INCLUDE_DIR atlbase.h PATHS ENV INCLUDE NO_DEFAULT_PATH)

if(NOT CUDAVM_DIRECTX_ATL_INCLUDE_DIR)
    message(FATAL_ERROR "DXC requires Visual Studio C++ ATL headers. Install the C++ ATL component and configure from its C++ toolchain environment")
endif()

include(ExternalProject)
set(CUDAVM_DIRECTX_GIT_REPOSITORY "https://github.com/microsoft/DirectXShaderCompiler.git" CACHE STRING "DXC source repository or local Git mirror")
set(CUDAVM_DIRECTX_BUILD_JOBS 4 CACHE STRING "Maximum concurrent DXC compilation jobs")
_cudavm_directx_build_contract(contract directx_args)
set(prepare "${CMAKE_COMMAND}" "-DCUDAVM_DIRECTX_ROOT=${CMAKE_BINARY_DIR}"
    "-DGIT_EXECUTABLE:FILEPATH=${GIT_EXECUTABLE}"
    "-DCUDAVM_DIRECTX_GIT_REPOSITORY=${CUDAVM_DIRECTX_GIT_REPOSITORY}"
    -DCUDAVM_DIRECTX_ACTION=prepare -P "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
ExternalProject_Add(cudavm_directx
    SOURCE_DIR "${CMAKE_BINARY_DIR}/src" BINARY_DIR "${CMAKE_BINARY_DIR}/build"
    DOWNLOAD_COMMAND ${prepare} UPDATE_COMMAND "" PATCH_COMMAND ""
    CMAKE_ARGS ${directx_args}
    BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --config Release
        --parallel "${CUDAVM_DIRECTX_BUILD_JOBS}" --target cudavm_directx_validator
    BUILD_ALWAYS TRUE INSTALL_COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${CMAKE_BINARY_DIR}/contract.txt" "${CMAKE_BINARY_DIR}/build/cudavm_directx.txt"
    USES_TERMINAL_CONFIGURE TRUE USES_TERMINAL_BUILD TRUE)
file(WRITE "${CMAKE_BINARY_DIR}/contract.txt" "${contract}")
ExternalProject_Add_Step(cudavm_directx verify_source
    COMMAND ${prepare} DEPENDEES download DEPENDERS configure ALWAYS TRUE)
