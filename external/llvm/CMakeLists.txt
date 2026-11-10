cmake_minimum_required(VERSION 3.24)
include_guard(GLOBAL)

function(cudavm_llvm_json_string output value)
    string(REPLACE "\\" "\\\\" escaped "${value}")
    string(REPLACE "\"" "\\\"" escaped "${escaped}")
    string(REPLACE "\n" "\\n" escaped "${escaped}")
    string(REPLACE "\r" "\\r" escaped "${escaped}")
    set(${output} "\"${escaped}\"" PARENT_SCOPE)
endfunction()

function(cudavm_llvm_source_contract output)
    set(contract "{}")
    string(JSON contract SET "${contract}" schema_version 2)
    foreach(field version revision)
        string(TOUPPER "${field}" upper)
        cudavm_llvm_json_string(value "${CUDAVM_LLVM_${upper}}")
        string(JSON contract SET "${contract}" "${field}" "${value}")
    endforeach()
    set(patches "[]")
    set(index 0)
    foreach(name IN LISTS CUDAVM_LLVM_PATCH_NAMES)
        set(path "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cudavm_${name}.patch")
        file(SHA256 "${path}" digest)
        set(patch "{}")
        string(JSON patch SET "${patch}" name "\"cudavm_${name}.patch\"")
        string(JSON patch SET "${patch}" sha256 "\"${digest}\"")
        string(JSON patches SET "${patches}" ${index} "${patch}")
        math(EXPR index "${index} + 1")
    endforeach()
    string(JSON contract SET "${contract}" patches "${patches}")
    foreach(field targets experimental_targets)
        if(field STREQUAL "targets")
            set(values ${CUDAVM_LLVM_TARGETS})
        else()
            set(values ${CUDAVM_LLVM_EXPERIMENTAL_TARGETS})
        endif()
        set(array "[]")
        set(index 0)
        foreach(value IN LISTS values)
            string(JSON array SET "${array}" ${index} "\"${value}\"")
            math(EXPR index "${index} + 1")
        endforeach()
        string(JSON contract SET "${contract}" "${field}" "${array}")
    endforeach()
    set(${output} "${contract}" PARENT_SCOPE)
endfunction()

function(_cudavm_llvm_build_contract output args_output)
    cudavm_llvm_source_contract(contract)
    set(config "{}")
    set(llvm_args)
    list(JOIN CUDAVM_LLVM_TARGETS "|" targets_arg)
    list(JOIN CUDAVM_LLVM_EXPERIMENTAL_TARGETS "|" experimental_targets_arg)
    set(settings
        CMAKE_BUILD_TYPE=Release
        CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
        LLVM_ENABLE_PROJECTS=clang
        "LLVM_TARGETS_TO_BUILD=${targets_arg}"
        "LLVM_EXPERIMENTAL_TARGETS_TO_BUILD=${experimental_targets_arg}"
        LLVM_ENABLE_RTTI=OFF LLVM_ENABLE_EH=OFF LLVM_ENABLE_PIC=ON
        LLVM_ENABLE_ASSERTIONS=OFF LLVM_BUILD_TOOLS=OFF CLANG_BUILD_TOOLS=OFF
        LLVM_INCLUDE_TESTS=OFF CLANG_INCLUDE_TESTS=OFF LLVM_INCLUDE_EXAMPLES=OFF
        LLVM_INCLUDE_BENCHMARKS=OFF LLVM_INCLUDE_DOCS=OFF
        LLVM_ENABLE_ZLIB=OFF LLVM_ENABLE_ZSTD=OFF LLVM_ENABLE_LIBXML2=OFF
        CLANG_ENABLE_LIBXML2=OFF LLVM_ENABLE_Z3_SOLVER=OFF LLVM_ENABLE_CURL=OFF
        LLVM_ENABLE_LIBPFM=OFF LLVM_ENABLE_DIA_SDK=OFF
        LLVM_BUILD_LLVM_DYLIB=OFF LLVM_LINK_LLVM_DYLIB=OFF
        CLANG_LINK_CLANG_DYLIB=OFF BUILD_SHARED_LIBS=OFF)
    foreach(language C CXX)
        list(APPEND settings "CMAKE_${language}_COMPILER=${CMAKE_${language}_COMPILER}")
        list(APPEND settings "CMAKE_${language}_FLAGS_RELEASE=${CMAKE_${language}_FLAGS_RELEASE}")
        if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            list(APPEND settings "CMAKE_${language}_FLAGS=-fms-runtime-lib=static")
        else()
            list(APPEND settings "CMAKE_${language}_FLAGS=")
        endif()
    endforeach()
    foreach(setting IN LISTS settings)
        string(REGEX MATCH "^([^=]+)=(.*)$" unused "${setting}")
        set(key "${CMAKE_MATCH_1}")
        set(value "${CMAKE_MATCH_2}")
        list(APPEND llvm_args "-D${key}:STRING=${value}")
        string(REPLACE "|" ";" value "${value}")
        cudavm_llvm_json_string(quoted "${value}")
        string(JSON config SET "${config}" "${key}" "${quoted}")
    endforeach()
    foreach(key CMAKE_GENERATOR CMAKE_GENERATOR_PLATFORM CMAKE_GENERATOR_TOOLSET
            CMAKE_C_COMPILER_ID CMAKE_C_COMPILER_VERSION
            CMAKE_CXX_COMPILER_ID CMAKE_CXX_COMPILER_VERSION)
        cudavm_llvm_json_string(quoted "${${key}}")
        string(JSON config SET "${config}" "${key}" "${quoted}")
    endforeach()
    if(CMAKE_MAKE_PROGRAM)
        list(APPEND llvm_args "-DCMAKE_MAKE_PROGRAM:FILEPATH=${CMAKE_MAKE_PROGRAM}")
    endif()
    string(JSON contract SET "${contract}" configuration "${config}")
    string(SHA256 fingerprint "${contract}")
    string(JSON contract SET "${contract}" fingerprint "\"${fingerprint}\"")
    set(${output} "${contract}" PARENT_SCOPE)
    set(${args_output} "${llvm_args}" PARENT_SCOPE)
endfunction()

function(_cudavm_llvm_prepare_source root verify_only)
    get_filename_component(root "${root}" ABSOLUTE)
    set(source "${root}/src")
    if(EXISTS "${source}" AND NOT EXISTS "${source}/.git")
        file(GLOB entries "${source}/*" "${source}/.*")
        if(entries)
            message(FATAL_ERROR "Refusing to replace an existing non-Git directory: ${source}")
        endif()
    endif()
    if(NOT EXISTS "${source}/.git")
        if(verify_only OR NOT CUDAVM_LLVM_GIT_REPOSITORY)
            message(FATAL_ERROR "Pinned LLVM source is missing: ${source}")
        endif()
        file(MAKE_DIRECTORY "${root}")
        execute_process(COMMAND "${GIT_EXECUTABLE}" -c core.autocrlf=input
            clone --no-checkout --no-hardlinks --depth 1 --filter=blob:none
            --branch "${CUDAVM_LLVM_TAG}" "${CUDAVM_LLVM_GIT_REPOSITORY}" "${source}"
            COMMAND_ERROR_IS_FATAL ANY)
        set(git "${GIT_EXECUTABLE}" -c "safe.directory=${source}" -c core.autocrlf=input)
        execute_process(COMMAND ${git} sparse-checkout init --cone
            WORKING_DIRECTORY "${source}" COMMAND_ERROR_IS_FATAL ANY)
        execute_process(COMMAND ${git} sparse-checkout set llvm clang cmake libc third-party
            WORKING_DIRECTORY "${source}" COMMAND_ERROR_IS_FATAL ANY)
        execute_process(COMMAND ${git} -c advice.detachedHead=false
            checkout --detach "${CUDAVM_LLVM_REVISION}"
            WORKING_DIRECTORY "${source}" COMMAND_ERROR_IS_FATAL ANY)
    endif()
    set(git "${GIT_EXECUTABLE}" -c "safe.directory=${source}" -c core.autocrlf=input -c core.safecrlf=false)
    execute_process(COMMAND ${git} rev-parse HEAD WORKING_DIRECTORY "${source}"
        OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    if(NOT revision STREQUAL CUDAVM_LLVM_REVISION)
        message(FATAL_ERROR "LLVM checkout must be ${CUDAVM_LLVM_TAG} (${CUDAVM_LLVM_REVISION}), found ${revision}; use a new build root")
    endif()
    execute_process(COMMAND ${git} ls-files -v WORKING_DIRECTORY "${source}"
        OUTPUT_VARIABLE index_flags COMMAND_ERROR_IS_FATAL ANY)
    if(index_flags MATCHES "(^|\n)[a-z] ")
        message(FATAL_ERROR "LLVM index has assume-unchanged entries that could hide source edits; preserving the index. Use a clean build root")
    endif()
    execute_process(COMMAND ${git} ls-files --others
        WORKING_DIRECTORY "${source}" OUTPUT_VARIABLE untracked
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    if(NOT untracked STREQUAL "")
        message(FATAL_ERROR "LLVM checkout contains untracked source files; preserving local changes: ${untracked}")
    endif()
    set(diff_args --no-ext-diff --no-textconv --no-color --src-prefix=a/ --dst-prefix=b/
        --unified=3 --diff-algorithm=myers --full-index --binary)
    execute_process(COMMAND ${git} diff ${diff_args} HEAD WORKING_DIRECTORY "${source}"
        OUTPUT_VARIABLE actual COMMAND_ERROR_IS_FATAL ANY)
    string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef nonce)
    set(index "${root}/contract_index_${nonce}")
    set(index_git "${CMAKE_COMMAND}" -E env "GIT_INDEX_FILE=${index}" ${git})
    execute_process(COMMAND ${index_git} read-tree HEAD WORKING_DIRECTORY "${source}"
        COMMAND_ERROR_IS_FATAL ANY)
    set(expected "")
    set(prefix -1)
    if(actual STREQUAL expected)
        set(prefix 0)
    endif()
    set(count 0)
    foreach(name IN LISTS CUDAVM_LLVM_PATCH_NAMES)
        execute_process(COMMAND ${index_git} apply --cached
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cudavm_${name}.patch"
            WORKING_DIRECTORY "${source}" COMMAND_ERROR_IS_FATAL ANY)
        execute_process(COMMAND ${index_git} diff --cached ${diff_args} HEAD
            WORKING_DIRECTORY "${source}" OUTPUT_VARIABLE expected COMMAND_ERROR_IS_FATAL ANY)
        math(EXPR count "${count} + 1")
        if(actual STREQUAL expected)
            set(prefix ${count})
        endif()
    endforeach()
    file(REMOVE "${index}")
    if(prefix LESS 0)
        message(FATAL_ERROR "LLVM checkout differs from the ordered cudavm patch set; preserving local changes. Use a clean build root for incompatible source changes")
    endif()
    if(verify_only AND NOT prefix EQUAL count)
        message(FATAL_ERROR "LLVM SDK source is not fully patched; rebuild the dependency")
    endif()
    set(position 0)
    foreach(name IN LISTS CUDAVM_LLVM_PATCH_NAMES)
        if(position GREATER_EQUAL prefix)
            execute_process(COMMAND ${git} apply --check "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cudavm_${name}.patch"
                WORKING_DIRECTORY "${source}" COMMAND_ERROR_IS_FATAL ANY)
            execute_process(COMMAND ${git} apply "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cudavm_${name}.patch"
                WORKING_DIRECTORY "${source}" COMMAND_ERROR_IS_FATAL ANY)
        endif()
        math(EXPR position "${position} + 1")
    endforeach()
    execute_process(COMMAND ${git} diff ${diff_args} HEAD WORKING_DIRECTORY "${source}"
        OUTPUT_VARIABLE actual COMMAND_ERROR_IS_FATAL ANY)
    if(NOT actual STREQUAL expected)
        message(FATAL_ERROR "LLVM source failed exact patch-set verification; preserving checkout")
    endif()
    if(NOT verify_only)
        cudavm_llvm_source_contract(contract)
        file(CONFIGURE OUTPUT "${root}/source_contract.json" CONTENT "${contract}\n" @ONLY)
        file(REMOVE "${root}/build/cudavm_llvm.json")
    endif()
    message(STATUS "Verified cudavm LLVM ${CUDAVM_LLVM_VERSION}, ordered nvptx/spirv/dxil patches: ${source}")
endfunction()

function(_cudavm_llvm_verify_build root publish)
    get_filename_component(root "${root}" ABSOLUTE)
    set(source "${root}/src")
    _cudavm_llvm_prepare_source("${root}" TRUE)
    file(READ "${root}/build_contract.json" contract)
    string(JSON recorded_fingerprint GET "${contract}" fingerprint)
    string(JSON unsigned_contract REMOVE "${contract}" fingerprint)
    string(SHA256 actual_fingerprint "${unsigned_contract}")
    if(NOT recorded_fingerprint STREQUAL actual_fingerprint)
        message(FATAL_ERROR "LLVM build contract fingerprint is invalid")
    endif()
    cudavm_llvm_source_contract(expected_source)
    foreach(field schema_version version revision patches targets experimental_targets)
        string(JSON actual GET "${contract}" "${field}")
        string(JSON expected GET "${expected_source}" "${field}")
        if(NOT actual STREQUAL expected)
            message(FATAL_ERROR "LLVM SDK source contract is stale: ${field}")
        endif()
    endforeach()
    if(NOT publish)
        if(NOT EXISTS "${root}/build/cudavm_llvm.json")
            message(FATAL_ERROR "LLVM SDK receipt is missing")
        endif()
        file(READ "${root}/build/cudavm_llvm.json" receipt)
        if(NOT receipt STREQUAL contract)
            message(FATAL_ERROR "LLVM SDK receipt is stale; rebuild the dependency")
        endif()
    endif()
    string(JSON configuration GET "${contract}" configuration)
    string(JSON length LENGTH "${configuration}")
    math(EXPR last "${length} - 1")
    file(STRINGS "${root}/build/CMakeCache.txt" cache)
    set(configured_source "")
    foreach(line IN LISTS cache)
        if(line MATCHES "^CMAKE_HOME_DIRECTORY:[^=]+=(.*)$")
            set(configured_source "${CMAKE_MATCH_1}")
        endif()
    endforeach()
    if(NOT configured_source STREQUAL "${source}/llvm")
        message(FATAL_ERROR "LLVM build is configured against a different source checkout: ${configured_source}")
    endif()
    foreach(index RANGE ${last})
        string(JSON key MEMBER "${configuration}" ${index})
        if(key MATCHES "^CMAKE_(C|CXX)_COMPILER_(ID|VERSION)$")
            continue()
        endif()
        string(JSON expected GET "${configuration}" "${key}")
        set(found FALSE)
        foreach(line IN LISTS cache)
            if(line MATCHES "^${key}:[^=]+=(.*)$")
                set(actual "${CMAKE_MATCH_1}")
                set(found TRUE)
                break()
            endif()
        endforeach()
        if(NOT found OR NOT actual STREQUAL expected)
            message(FATAL_ERROR "LLVM build configuration differs for ${key}: expected '${expected}', found '${actual}'")
        endif()
    endforeach()
    foreach(name LLVMNVPTXCodeGen LLVMSPIRVCodeGen LLVMDirectXCodeGen clangCodeGen clangTooling LLVMPasses)
        if(NOT EXISTS "${root}/build/lib/${name}.lib" AND NOT EXISTS "${root}/build/lib/lib${name}.a")
            message(FATAL_ERROR "LLVM receipt cannot be published: missing ${name} archive")
        endif()
    endforeach()
    foreach(path lib/cmake/llvm/LLVMConfig.cmake lib/cmake/clang/ClangConfig.cmake
            lib/clang/22/include/__clang_cuda_runtime_wrapper.h)
        if(NOT EXISTS "${root}/build/${path}")
            message(FATAL_ERROR "LLVM receipt cannot be published: missing ${path}")
        endif()
    endforeach()
    if(NOT EXISTS "${root}/build/bin/llc" AND NOT EXISTS "${root}/build/bin/llc.exe")
        message(FATAL_ERROR "LLVM receipt cannot be published: missing llc validation tool")
    endif()
    if(publish)
        file(CONFIGURE OUTPUT "${root}/build/cudavm_llvm.json" CONTENT "${contract}" @ONLY)
        message(STATUS "Published coherent cudavm LLVM SDK receipt")
    endif()
endfunction()

function(cudavm_core_dependencies target)
    if(NOT target MATCHES "^cudavm_core_(shared|static)$")
        message(FATAL_ERROR "Only compiler module variants may link Clang/LLVM targets")
    endif()
    target_include_directories(${target} PRIVATE "${engine_dir}")
    target_include_directories(${target} SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
    target_compile_definitions(${target} PRIVATE ${LLVM_DEFINITIONS} LLVM_BUILD_STATIC CLANG_BUILD_STATIC)
    set_property(TARGET ${target} PROPERTY MSVC_RUNTIME_LIBRARY MultiThreaded)
    if(MSVC)
        target_compile_options(${target} PRIVATE /GR-)
    else()
        target_compile_options(${target} PRIVATE -fno-rtti)
        if(WIN32)
            target_compile_options(${target} PRIVATE -fms-runtime-lib=static)
        else()
            target_link_options(${target} PRIVATE "LINKER:--exclude-libs,ALL")
        endif()
    endif()
    get_target_property(kind ${target} TYPE)
    if(NOT kind STREQUAL "STATIC_LIBRARY")
        target_link_libraries(${target} PRIVATE "$<BUILD_INTERFACE:${ARGN}>")
        return()
    endif()
    set(queue ${ARGN})
    set(visited)
    set(archives)
    set(system_libraries)
    while(queue)
        list(POP_FRONT queue item)
        string(REGEX REPLACE "^\\$<LINK_ONLY:([^>]+)>$" "\\1" item "${item}")
        if(item IN_LIST visited OR item STREQUAL "")
            continue()
        endif()
        list(APPEND visited "${item}")
        if(TARGET "${item}")
            get_target_property(kind "${item}" TYPE)
            if(kind STREQUAL "STATIC_LIBRARY")
                list(APPEND archives "$<TARGET_FILE:${item}>")
            elseif(NOT kind STREQUAL "INTERFACE_LIBRARY")
                message(FATAL_ERROR "Compiler dependency ${item} must be static, found ${kind}")
            endif()
            get_target_property(dependencies "${item}" INTERFACE_LINK_LIBRARIES)
            if(dependencies)
                list(APPEND queue ${dependencies})
            endif()
        else()
            list(APPEND system_libraries "${item}")
        endif()
    endwhile()
    list(REMOVE_DUPLICATES archives)
    list(REMOVE_DUPLICATES system_libraries)
    target_link_libraries(${target} PRIVATE ${system_libraries})
    cudavm_merge_static_libraries(${target} ${archives})
endfunction()

set(CUDAVM_LLVM_VERSION "22.1.4")
set(CUDAVM_LLVM_TAG "llvmorg-${CUDAVM_LLVM_VERSION}")
set(CUDAVM_LLVM_REVISION "35990504507d79e0b9deb809c8ee5e1b34ceef20")

set(CUDAVM_LLVM_PATCH_NAMES nvptx spirv dxil)
set(CUDAVM_LLVM_TARGETS NVPTX SPIRV)
set(CUDAVM_LLVM_EXPERIMENTAL_TARGETS DirectX)
set(CUDAVM_LLVM_REQUIRED_TARGETS
    clangCodeGen clangTooling LLVMPasses LLVMNVPTXCodeGen LLVMSPIRVCodeGen
    LLVMDirectXCodeGen clang-resource-headers llc)

if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
    if(NOT CUDAVM_LLVM_ACTION MATCHES "^(prepare|verify|publish)$")
        message(FATAL_ERROR "Set CUDAVM_LLVM_ACTION to prepare, verify or publish")
    endif()

    if(NOT CUDAVM_LLVM_ROOT)
        message(FATAL_ERROR "LLVM maintenance requires a private build root")
    endif()

    if(CUDAVM_LLVM_ACTION STREQUAL "prepare")
        _cudavm_llvm_prepare_source("${CUDAVM_LLVM_ROOT}" "${CUDAVM_LLVM_VERIFY_ONLY}")
    elseif(CUDAVM_LLVM_ACTION STREQUAL "verify")
        _cudavm_llvm_verify_build("${CUDAVM_LLVM_ROOT}" FALSE)
    else()
        _cudavm_llvm_verify_build("${CUDAVM_LLVM_ROOT}" TRUE)
    endif()

    return()
endif()

if(CMAKE_SCRIPT_MODE_FILE OR NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_LIST_DIR)
    return()
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND NOT CUDAVM_LLVM_ACTION STREQUAL "build")
    message(FATAL_ERROR "Configure cudavm from the repository root; external dependencies are not standalone projects")
endif()

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(llvm_root "${PROJECT_SOURCE_DIR}/.tmp/llvm/${CUDAVM_LLVM_VERSION}-native")
    set(llvm_build_dir "${llvm_root}/build")
    set(CUDAVM_LLVM_GIT_REPOSITORY "https://github.com/llvm/llvm-project.git" CACHE STRING "LLVM repository or local Git mirror")
    set(CUDAVM_LLVM_BUILD_JOBS 4 CACHE STRING "Maximum concurrent LLVM compilation jobs")
    _cudavm_llvm_build_contract(expected_contract unused_llvm_args)
    string(JSON expected_fingerprint GET "${expected_contract}" fingerprint)
    set(sdk_matches FALSE)

    if(EXISTS "${llvm_build_dir}/cudavm_llvm.json")
        file(READ "${llvm_build_dir}/cudavm_llvm.json" llvm_contract)
        string(JSON sdk_fingerprint ERROR_VARIABLE receipt_error GET "${llvm_contract}" fingerprint)

        if(NOT receipt_error AND sdk_fingerprint STREQUAL expected_fingerprint)
            set(sdk_matches TRUE)
        endif()
    endif()

    if(NOT sdk_matches OR NOT EXISTS "${llvm_build_dir}/lib/cmake/llvm/LLVMConfig.cmake")
        set(bootstrap_args -S "${PROJECT_SOURCE_DIR}/external/llvm" -B "${llvm_root}" -G "${CMAKE_GENERATOR}"
            -DCUDAVM_LLVM_ACTION=build
            "-DGIT_EXECUTABLE:FILEPATH=${GIT_EXECUTABLE}"
            "-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}"
            "-DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}"
            "-DCUDAVM_LLVM_GIT_REPOSITORY:STRING=${CUDAVM_LLVM_GIT_REPOSITORY}"
            "-DCUDAVM_LLVM_BUILD_JOBS:STRING=${CUDAVM_LLVM_BUILD_JOBS}")

        if(CMAKE_MAKE_PROGRAM)
            list(APPEND bootstrap_args "-DCMAKE_MAKE_PROGRAM:FILEPATH=${CMAKE_MAKE_PROGRAM}")
        endif()

        if(CMAKE_GENERATOR_PLATFORM)
            list(APPEND bootstrap_args -A "${CMAKE_GENERATOR_PLATFORM}")
        endif()

        if(CMAKE_GENERATOR_TOOLSET)
            list(APPEND bootstrap_args -T "${CMAKE_GENERATOR_TOOLSET}")
        endif()

        message(STATUS "Bootstrapping cudavm LLVM ${CUDAVM_LLVM_VERSION}, NVPTX/SPIRV/DirectX")
        execute_process(COMMAND "${CMAKE_COMMAND}" ${bootstrap_args} COMMAND_ERROR_IS_FATAL ANY)
        execute_process(COMMAND "${CMAKE_COMMAND}" --build "${llvm_root}" --config Release
            COMMAND_ERROR_IS_FATAL ANY)
    endif()

    file(READ "${llvm_build_dir}/cudavm_llvm.json" llvm_contract)
    string(JSON sdk_fingerprint GET "${llvm_contract}" fingerprint)

    if(NOT sdk_fingerprint STREQUAL expected_fingerprint)
        message(FATAL_ERROR "LLVM SDK does not match cudavm's complete source/build contract")
    endif()

    execute_process(COMMAND "${CMAKE_COMMAND}" "-DCUDAVM_LLVM_ROOT=${llvm_root}"
        "-DGIT_EXECUTABLE:FILEPATH=${GIT_EXECUTABLE}"
        -DCUDAVM_LLVM_ACTION=verify
        -P "${PROJECT_SOURCE_DIR}/external/llvm/CMakeLists.txt" COMMAND_ERROR_IS_FATAL ANY)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/external/llvm/CMakeLists.txt"
        "${PROJECT_SOURCE_DIR}/external/llvm/cudavm_nvptx.patch"
        "${PROJECT_SOURCE_DIR}/external/llvm/cudavm_spirv.patch"
        "${PROJECT_SOURCE_DIR}/external/llvm/cudavm_dxil.patch"
        "${llvm_build_dir}/cudavm_llvm.json"
        "${llvm_build_dir}/CMakeCache.txt"
        "${llvm_root}/build_contract.json"
        "${llvm_root}/source_contract.json"
        "${llvm_root}/src/.git/HEAD")
    set(CMAKE_MAP_IMPORTED_CONFIG_DEBUG Release)
    set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
    set(LLVM_DIR "${llvm_build_dir}/lib/cmake/llvm" CACHE PATH "Selected cudavm LLVM package" FORCE)
    set(Clang_DIR "${llvm_build_dir}/lib/cmake/clang" CACHE PATH "Selected cudavm Clang package" FORCE)
    find_package(LLVM ${CUDAVM_LLVM_VERSION} EXACT REQUIRED CONFIG NO_DEFAULT_PATH PATHS "${LLVM_DIR}")
    find_package(Clang REQUIRED CONFIG NO_DEFAULT_PATH PATHS "${Clang_DIR}")
    set(enabled_targets ${LLVM_TARGETS_TO_BUILD})
    list(SORT enabled_targets)

    if(LLVM_ENABLE_RTTI OR NOT enabled_targets STREQUAL "DirectX;NVPTX;SPIRV")
        message(FATAL_ERROR "cudavm requires its RTTI-disabled static NVPTX/SPIRV/DirectX LLVM profile")
    endif()

    set(llvm_queue clangCodeGen clangTooling LLVMPasses LLVMNVPTXCodeGen LLVMSPIRVCodeGen LLVMDirectXCodeGen)

    foreach(item IN LISTS llvm_queue)
        if(NOT TARGET "${item}")
            message(FATAL_ERROR "LLVM SDK is missing required imported target ${item}")
        endif()
    endforeach()

    set(llvm_seen)

    while(llvm_queue)
        list(POP_FRONT llvm_queue item)
        string(REGEX REPLACE "^\\$<LINK_ONLY:([^>]+)>$" "\\1" item "${item}")

        if(item IN_LIST llvm_seen OR NOT TARGET "${item}")
            continue()
        endif()

        list(APPEND llvm_seen "${item}")
        get_target_property(kind "${item}" TYPE)

        if(kind STREQUAL "STATIC_LIBRARY")
            get_target_property(archive "${item}" IMPORTED_LOCATION_RELEASE)

            if(NOT archive)
                get_target_property(archive "${item}" IMPORTED_LOCATION)
            endif()

            if(NOT archive OR NOT EXISTS "${archive}")
                message(FATAL_ERROR "LLVM SDK is incomplete: missing static archive for ${item}")
            endif()
        elseif(NOT kind STREQUAL "INTERFACE_LIBRARY")
            message(FATAL_ERROR "LLVM SDK component ${item} must be static, found ${kind}")
        endif()

        get_target_property(dependencies "${item}" INTERFACE_LINK_LIBRARIES)

        if(dependencies)
            list(APPEND llvm_queue ${dependencies})
        endif()
    endwhile()

    set(clang_resource_dir "${llvm_build_dir}/lib/clang/${LLVM_VERSION_MAJOR}")
    set(wrapper_source "${clang_resource_dir}/include/__clang_cuda_runtime_wrapper.h")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${wrapper_source}")
    file(READ "${wrapper_source}" wrapper)

    if(NOT wrapper MATCHES "#if __has_include\\(\"curand_mtgp32_kernel.h\"\\)" OR NOT wrapper MATCHES "return __builtin_fmal\\(x, y, z\\)")
        message(FATAL_ERROR "Clang resource headers must come from the LLVM build with cudavm_nvptx.patch")
    endif()

    set(wrapper_path "${PROJECT_BINARY_DIR}/cuda_runtime_wrapper.h")
    configure_file("${PROJECT_SOURCE_DIR}/source/core/cuda_runtime_wrapper.h.in" "${wrapper_path}" @ONLY)
    set(CUDAVM_LLVM_LICENSE "${LLVM_BUILD_MAIN_SRC_DIR}/../LICENSE.TXT")

    file(RELATIVE_PATH includes_from_bin "/${CMAKE_INSTALL_BINDIR}" "/${CMAKE_INSTALL_INCLUDEDIR}")
    file(RELATIVE_PATH includes_from_lib "/${CMAKE_INSTALL_LIBDIR}" "/${CMAKE_INSTALL_INCLUDEDIR}")
    file(RELATIVE_PATH sdk_notice_from_bin "/${CMAKE_INSTALL_BINDIR}" "/${CMAKE_INSTALL_DATADIR}/licences/LLVM_LICENSE.txt")
    file(RELATIVE_PATH sdk_notice_from_lib "/${CMAKE_INSTALL_LIBDIR}" "/${CMAKE_INSTALL_DATADIR}/licences/LLVM_LICENSE.txt")
    set(device_libraries LLVMCore LLVMSupport LLVMAnalysis LLVMipo LLVMPasses LLVMScalarOpts
        LLVMTransformUtils LLVMBitReader LLVMBitWriter LLVMLinker LLVMObject)
    set(source_libraries clangTooling clangFrontend clangSerialization clangDriver
        clangParse clangSema clangAnalysis clangAST clangLex clangBasic ${device_libraries})
    set(link_libraries ${device_libraries} LLVMAsmParser LLVMTarget LLVMCodeGen LLVMNVPTXCodeGen LLVMSPIRVCodeGen LLVMDirectXCodeGen)
    set(engine_dir "${PROJECT_SOURCE_DIR}/source/core")
    get_property(llvm_imported_targets DIRECTORY PROPERTY IMPORTED_TARGETS)

    foreach(target IN LISTS llvm_imported_targets)
        set_property(TARGET ${target} PROPERTY IMPORTED_GLOBAL TRUE)
    endforeach()

    foreach(variable LLVM_INCLUDE_DIRS CLANG_INCLUDE_DIRS LLVM_DEFINITIONS LLVM_BUILD_MAIN_SRC_DIR
            CUDAVM_LLVM_LICENSE includes_from_bin includes_from_lib sdk_notice_from_bin sdk_notice_from_lib
            CMAKE_MAP_IMPORTED_CONFIG_DEBUG CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO
            device_libraries source_libraries link_libraries engine_dir wrapper_path llvm_build_dir clang_resource_dir)
        set(${variable} "${${variable}}" PARENT_SCOPE)
    endforeach()

    return()
endif()

project(cudavm_llvm LANGUAGES C CXX)
include(ExternalProject)
set(CUDAVM_LLVM_GIT_REPOSITORY "https://github.com/llvm/llvm-project.git" CACHE STRING
    "LLVM source repository or local Git mirror; its release commit is verified")
set(CUDAVM_LLVM_BUILD_JOBS 4 CACHE STRING "Maximum concurrent LLVM compilation jobs")
_cudavm_llvm_build_contract(build_contract llvm_args)
file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/build_contract.json"
    CONTENT "${build_contract}\n" @ONLY)
set(prepare "${CMAKE_COMMAND}" "-DCUDAVM_LLVM_ROOT=${CMAKE_BINARY_DIR}"
    "-DGIT_EXECUTABLE:FILEPATH=${GIT_EXECUTABLE}"
    "-DCUDAVM_LLVM_GIT_REPOSITORY=${CUDAVM_LLVM_GIT_REPOSITORY}"
    -DCUDAVM_LLVM_ACTION=prepare -P "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
ExternalProject_Add(cudavm_llvm
    SOURCE_DIR "${CMAKE_BINARY_DIR}/src"
    SOURCE_SUBDIR llvm
    BINARY_DIR "${CMAKE_BINARY_DIR}/build"
    DOWNLOAD_COMMAND ${prepare}
    UPDATE_COMMAND "" PATCH_COMMAND ""
    USES_TERMINAL_DOWNLOAD TRUE
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD TRUE
    LIST_SEPARATOR "|"
    CMAKE_ARGS ${llvm_args}
    BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --config Release
        --parallel "${CUDAVM_LLVM_BUILD_JOBS}" --target ${CUDAVM_LLVM_REQUIRED_TARGETS}
    BUILD_ALWAYS TRUE
    INSTALL_COMMAND "${CMAKE_COMMAND}" "-DCUDAVM_LLVM_ROOT=${CMAKE_BINARY_DIR}"
        "-DGIT_EXECUTABLE:FILEPATH=${GIT_EXECUTABLE}"
        -DCUDAVM_LLVM_ACTION=publish -P "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
ExternalProject_Add_Step(cudavm_llvm source_contract
    COMMAND ${prepare}
    DEPENDEES download DEPENDERS configure
    ALWAYS TRUE
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cudavm_nvptx.patch"
        "${CMAKE_CURRENT_SOURCE_DIR}/cudavm_spirv.patch"
        "${CMAKE_CURRENT_SOURCE_DIR}/cudavm_dxil.patch"
        "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
        "${CMAKE_BINARY_DIR}/build_contract.json")
