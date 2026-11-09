cmake_minimum_required(VERSION 3.24)
include_guard(GLOBAL)

# Pinned upstream release; frontend patches are owned by cudavm.
set(CUDAVM_LLVM_VERSION "22.1.4")
set(CUDAVM_LLVM_TAG "llvmorg-${CUDAVM_LLVM_VERSION}")
set(CUDAVM_LLVM_REVISION "35990504507d79e0b9deb809c8ee5e1b34ceef20")

set(CUDAVM_LLVM_PATCH_NAMES nvptx spirv dxil)
set(CUDAVM_LLVM_TARGETS NVPTX SPIRV)
set(CUDAVM_LLVM_EXPERIMENTAL_TARGETS DirectX)
set(CUDAVM_LLVM_REQUIRED_TARGETS
    clangCodeGen clangTooling LLVMPasses LLVMNVPTXCodeGen LLVMSPIRVCodeGen
    LLVMDirectXCodeGen clang-resource-headers llc)

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

# The same profile produces ExternalProject arguments and the SDK fingerprint.
# Parallelism and mirror location do not change the compiler artifacts.
function(cudavm_llvm_build_contract output args_output)
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

function(cudavm_llvm_prepare_source root verify_only)
    find_package(Git REQUIRED)
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
    # Scope trust to this owned checkout; never change the user's global Git config.
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
    # Construct the exact expected diff with an isolated index. This accounts for
    # shared-file hunks without staging, reverting or rewriting the live checkout.
    string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef nonce)
    set(index "${root}/contract-index-${nonce}")
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
        file(CONFIGURE OUTPUT "${root}/source-contract.json" CONTENT "${contract}\n" @ONLY)
        # Publish only after a successful build, including a repeated build of the
        # same profile. Interrupted rebuilds must never retain a success receipt.
        file(REMOVE "${root}/build/cudavm-llvm.json")
    endif()
    message(STATUS "Verified cudavm LLVM ${CUDAVM_LLVM_VERSION}, ordered nvptx/spirv/dxil patches: ${source}")

endfunction()

function(cudavm_llvm_verify_build root publish)
    get_filename_component(root "${root}" ABSOLUTE)
    set(source "${root}/src")
    cudavm_llvm_prepare_source("${root}" TRUE)
    file(READ "${root}/build-contract.json" contract)
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
        if(NOT EXISTS "${root}/build/cudavm-llvm.json")
            message(FATAL_ERROR "LLVM SDK receipt is missing")
        endif()
        file(READ "${root}/build/cudavm-llvm.json" receipt)
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
            continue() # These are compiler-detection fields, not cache entries.
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
        file(CONFIGURE OUTPUT "${root}/build/cudavm-llvm.json" CONTENT "${contract}" @ONLY)
        message(STATUS "Published coherent cudavm LLVM SDK receipt")
    endif()
endfunction()

# Consumer SDK setup and static archive composition share the pinned dependency contract.
function(cudavm_core_dependencies target output_dir)
    if(NOT target MATCHES "^cudavm_core_(shared|static)$")
        message(FATAL_ERROR "Only compiler module variants may link Clang/LLVM targets")
    endif()
    target_include_directories(${target} PRIVATE "${engine_dir}")
    target_include_directories(${target} SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
    # Clang's GNU-driver Windows build does not export this definition through
    # its imported targets. These dependencies are always static.
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
        # Build-time LLVM targets never leak into the installed SDK.
        target_link_libraries(${target} PRIVATE "$<BUILD_INTERFACE:${ARGN}>")
        return()
    endif()
    # A static compiler is self-contained. Rebundle LLVM only here, leaving the
    # facades small and avoiding LLVM targets in either consumer link graph.
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
    if(WIN32)
        get_filename_component(tool_directory "${CMAKE_AR}" DIRECTORY)
        find_program(CUDAVM_ARCHIVE_TOOL NAMES llvm-lib lib HINTS "${tool_directory}" REQUIRED)
    else()
        set(CUDAVM_ARCHIVE_TOOL "${CMAKE_AR}")
    endif()
    # Static archivers do not relink when imported dependency archives change.
    # A generated member gives those files a normal dependency edge.
    set(archive_list "${output_dir}/${target}_archives_$<CONFIG>.txt")
    file(GENERATE OUTPUT "${archive_list}" CONTENT "$<JOIN:${archives},\n>\n")
    set(dependency_member "${output_dir}/${target}_llvm_dependencies.cpp")
    add_custom_command(OUTPUT "${dependency_member}"
        COMMAND ${CMAKE_COMMAND} -E touch "${dependency_member}"
        DEPENDS ${archives} "${archive_list}" "${CMAKE_CURRENT_FUNCTION_LIST_FILE}" VERBATIM)
    target_sources(${target} PRIVATE "${dependency_member}")
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} "-DTOOL=${CUDAVM_ARCHIVE_TOOL}" "-DOUTPUT=$<TARGET_FILE:${target}>"
            "-DARCHIVE_LIST=${archive_list}" "-DWINDOWS=${WIN32}"
            -DCUDAVM_LLVM_ACTION=merge_static -P "${CMAKE_CURRENT_FUNCTION_LIST_FILE}" VERBATIM)
endfunction()



function(cudavm_merge_static)
    # Merge indexed archive members, never nested archives. Only compiler_static
    # invokes this; installed consumers need no LLVM CMake package or archive paths.
    if(DEFINED ARCHIVE_LIST)
        file(STRINGS "${ARCHIVE_LIST}" archives)
    else()
        string(REPLACE "|" ";" archives "${ARCHIVES}")
    endif()
    set(merged "${OUTPUT}.merged")
    if(WINDOWS)
        set(response "/NOLOGO\n/OUT:\"${merged}\"\n\"${OUTPUT}\"\n")
        foreach(archive IN LISTS archives)
            string(APPEND response "\"${archive}\"\n")
        endforeach()
        file(WRITE "${OUTPUT}.rsp" "${response}")
        execute_process(COMMAND "${TOOL}" "@${OUTPUT}.rsp"
            COMMAND_ERROR_IS_FATAL ANY)
        file(REMOVE "${OUTPUT}.rsp")
    else()
        set(script "CREATE \"${merged}\"\nADDLIB \"${OUTPUT}\"\n")
        foreach(archive IN LISTS archives)
            string(APPEND script "ADDLIB \"${archive}\"\n")
        endforeach()
        string(APPEND script "SAVE\nEND\n")
        file(WRITE "${OUTPUT}.mri" "${script}")
        execute_process(COMMAND "${TOOL}" -M INPUT_FILE "${OUTPUT}.mri" COMMAND_ERROR_IS_FATAL ANY)
        file(REMOVE "${OUTPUT}.mri")
    endif()
    file(RENAME "${merged}" "${OUTPUT}")
endfunction()

# Include this file for the contract helpers; execute it with -P for maintenance.
if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
    if(NOT CUDAVM_LLVM_ACTION MATCHES "^(prepare|verify|publish|merge_static)$")
        message(FATAL_ERROR "Set CUDAVM_LLVM_ACTION to prepare, verify, publish or merge_static")
    endif()
    if(CUDAVM_LLVM_ACTION STREQUAL "merge_static")
        cudavm_merge_static()
        return()
    endif()
    if(NOT CUDAVM_LLVM_ROOT)
        message(FATAL_ERROR "LLVM maintenance requires a private build root")
    endif()
    if(CUDAVM_LLVM_ACTION STREQUAL "prepare")
        cudavm_llvm_prepare_source("${CUDAVM_LLVM_ROOT}" "${CUDAVM_LLVM_VERIFY_ONLY}")
    elseif(CUDAVM_LLVM_ACTION STREQUAL "verify")
        cudavm_llvm_verify_build("${CUDAVM_LLVM_ROOT}" FALSE)
    else()
        cudavm_llvm_verify_build("${CUDAVM_LLVM_ROOT}" TRUE)
    endif()
    return()
endif()
if(CMAKE_SCRIPT_MODE_FILE OR NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_LIST_DIR)
    return()
endif()

# A parent build consumes its coherent SDK; a standalone configure bootstraps it.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # LLVM source, patches and generated libraries are owned by cudavm.
    cudavm_llvm_source_contract(source_contract)
    string(SHA256 source_key "${source_contract}")
    string(SUBSTRING "${source_key}" 0 12 source_key)
    set(llvm_root "${PROJECT_SOURCE_DIR}/.tmp/llvm/${CUDAVM_LLVM_VERSION}-${source_key}")
    set(CUDAVM_LLVM_BUILD_DIR "${llvm_root}/build" CACHE PATH "Coherent static three-target LLVM/Clang SDK")
    set(CUDAVM_CLANG_RESOURCE_DIR "${CUDAVM_LLVM_BUILD_DIR}/lib/clang/22" CACHE PATH "Matching patched Clang resource headers embedded in nvrtc")
    option(CUDAVM_LLVM_BOOTSTRAP "Build cudavm's pinned LLVM dependency when it is missing or stale" ON)
    set(CUDAVM_LLVM_GIT_REPOSITORY "https://github.com/llvm/llvm-project.git" CACHE STRING "LLVM repository or local Git mirror")
    set(CUDAVM_LLVM_BUILD_JOBS 4 CACHE STRING "Maximum concurrent LLVM compilation jobs")
    cudavm_llvm_build_contract(expected_contract unused_llvm_args)
    string(JSON expected_fingerprint GET "${expected_contract}" fingerprint)
    set(sdk_matches FALSE)
    if(EXISTS "${CUDAVM_LLVM_BUILD_DIR}/cudavm-llvm.json")
        file(READ "${CUDAVM_LLVM_BUILD_DIR}/cudavm-llvm.json" llvm_contract)
        string(JSON sdk_fingerprint ERROR_VARIABLE receipt_error GET "${llvm_contract}" fingerprint)
        if(NOT receipt_error AND sdk_fingerprint STREQUAL expected_fingerprint)
            set(sdk_matches TRUE)
        endif()
    endif()
    if(NOT sdk_matches OR NOT EXISTS "${CUDAVM_LLVM_BUILD_DIR}/lib/cmake/llvm/LLVMConfig.cmake")
        if(CUDAVM_LLVM_BOOTSTRAP AND CUDAVM_LLVM_BUILD_DIR STREQUAL "${llvm_root}/build")
            set(bootstrap_args -S "${PROJECT_SOURCE_DIR}/external/llvm" -B "${llvm_root}" -G "${CMAKE_GENERATOR}"
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
        else()
            message(FATAL_ERROR "LLVM SDK is missing or stale (revision, three patches, targets or build profile). Build cmake -S external/llvm -B <clean-root> and cmake --build <clean-root>, then set CUDAVM_LLVM_BUILD_DIR=<clean-root>/build and CUDAVM_CLANG_RESOURCE_DIR=<clean-root>/build/lib/clang/22")
        endif()
    endif()
    file(READ "${CUDAVM_LLVM_BUILD_DIR}/cudavm-llvm.json" llvm_contract)
    string(JSON sdk_fingerprint GET "${llvm_contract}" fingerprint)
    if(NOT sdk_fingerprint STREQUAL expected_fingerprint)
        message(FATAL_ERROR "LLVM SDK does not match cudavm's complete source/build contract")
    endif()
    get_filename_component(sdk_root "${CUDAVM_LLVM_BUILD_DIR}" DIRECTORY)
    execute_process(COMMAND "${CMAKE_COMMAND}" "-DCUDAVM_LLVM_ROOT=${sdk_root}"
        -DCUDAVM_LLVM_ACTION=verify
        -P "${PROJECT_SOURCE_DIR}/external/llvm/CMakeLists.txt" COMMAND_ERROR_IS_FATAL ANY)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/external/llvm/CMakeLists.txt"
        "${PROJECT_SOURCE_DIR}/external/llvm/cudavm_nvptx.patch"
        "${PROJECT_SOURCE_DIR}/external/llvm/cudavm_spirv.patch"
        "${PROJECT_SOURCE_DIR}/external/llvm/cudavm_dxil.patch"
        "${CUDAVM_LLVM_BUILD_DIR}/cudavm-llvm.json"
        "${CUDAVM_LLVM_BUILD_DIR}/CMakeCache.txt"
        "${sdk_root}/build-contract.json"
        "${sdk_root}/source-contract.json"
        "${sdk_root}/src/.git/HEAD")
    set(CMAKE_MAP_IMPORTED_CONFIG_DEBUG Release)
    set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
    set(LLVM_DIR "${CUDAVM_LLVM_BUILD_DIR}/lib/cmake/llvm" CACHE PATH "Selected cudavm LLVM package" FORCE)
    set(Clang_DIR "${CUDAVM_LLVM_BUILD_DIR}/lib/cmake/clang" CACHE PATH "Selected cudavm Clang package" FORCE)
    find_package(LLVM ${CUDAVM_LLVM_VERSION} EXACT REQUIRED CONFIG NO_DEFAULT_PATH PATHS "${LLVM_DIR}")
    find_package(Clang REQUIRED CONFIG NO_DEFAULT_PATH PATHS "${Clang_DIR}")
    set(enabled_targets ${LLVM_TARGETS_TO_BUILD})
    list(SORT enabled_targets)
    if(LLVM_ENABLE_RTTI OR NOT enabled_targets STREQUAL "DirectX;NVPTX;SPIRV")
        message(FATAL_ERROR "cudavm requires its RTTI-disabled static NVPTX/SPIRV/DirectX LLVM profile")
    endif()
    # Check the entire static dependency closure, including archives that CMake's
    # generated package exports even when they have not actually been built.
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
    # Wrapper changes belong to cudavm_nvptx.patch and clang-resource-headers.
    file(REAL_PATH "${CUDAVM_CLANG_RESOURCE_DIR}" resource_root)
    file(REAL_PATH "${CUDAVM_LLVM_BUILD_DIR}/lib/clang/22" sdk_resource_root)
    if(WIN32)
        string(TOLOWER "${resource_root}" resource_root)
        string(TOLOWER "${sdk_resource_root}" sdk_resource_root)
    endif()
    if(NOT resource_root STREQUAL sdk_resource_root)
        message(FATAL_ERROR "CUDAVM_CLANG_RESOURCE_DIR must select this SDK's matching lib/clang/22 directory")
    endif()
    set(wrapper_source "${CUDAVM_CLANG_RESOURCE_DIR}/include/__clang_cuda_runtime_wrapper.h")
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
    file(RELATIVE_PATH sdk_notice_from_bin "/${CMAKE_INSTALL_BINDIR}" "/${CMAKE_INSTALL_DATADIR}/licences/LLVM-LICENSE.txt")
    file(RELATIVE_PATH sdk_notice_from_lib "/${CMAKE_INSTALL_LIBDIR}" "/${CMAKE_INSTALL_DATADIR}/licences/LLVM-LICENSE.txt")
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
            device_libraries source_libraries link_libraries engine_dir wrapper_path)
        set(${variable} "${${variable}}" PARENT_SCOPE)
    endforeach()
    return()
endif()

project(cudavm_llvm LANGUAGES C CXX)
include(ExternalProject)
set(CUDAVM_LLVM_GIT_REPOSITORY "https://github.com/llvm/llvm-project.git" CACHE STRING
    "LLVM source repository or local Git mirror; its release commit is verified")
set(CUDAVM_LLVM_BUILD_JOBS 4 CACHE STRING "Maximum concurrent LLVM compilation jobs")
cudavm_llvm_build_contract(build_contract llvm_args)
file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/build-contract.json"
    CONTENT "${build_contract}\n" @ONLY)
set(prepare "${CMAKE_COMMAND}" "-DCUDAVM_LLVM_ROOT=${CMAKE_BINARY_DIR}"
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
        -DCUDAVM_LLVM_ACTION=publish -P "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
# Verify on every build, including interrupted patch application. Never reset
# an occupied checkout. A changed contract also forces LLVM configuration.
ExternalProject_Add_Step(cudavm_llvm source_contract
    COMMAND ${prepare}
    DEPENDEES download DEPENDERS configure
    ALWAYS TRUE
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cudavm_nvptx.patch"
        "${CMAKE_CURRENT_SOURCE_DIR}/cudavm_spirv.patch"
        "${CMAKE_CURRENT_SOURCE_DIR}/cudavm_dxil.patch"
        "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
        "${CMAKE_BINARY_DIR}/build-contract.json")
