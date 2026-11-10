cmake_minimum_required(VERSION 3.24)

foreach(argument build config fixture work backend mode tracer)
    if(NOT DEFINED ${argument} OR "${${argument}}" STREQUAL "")
        message(FATAL_ERROR "Missing static Runtime test argument: ${argument}")
    endif()
endforeach()

file(READ "${fixture}/provenance.json" provenance)
string(JSON hashes GET "${provenance}" sha256)
string(JSON count LENGTH "${hashes}")
math(EXPR last "${count} - 1")

foreach(index RANGE 0 ${last})
    string(JSON name MEMBER "${hashes}" ${index})
    string(JSON expected GET "${hashes}" "${name}")
    file(SHA256 "${fixture}/${name}" actual)

    if(NOT actual STREQUAL expected)
        message(FATAL_ERROR "NVIDIA static Runtime producer input changed without regenerated provenance: ${name}")
    endif()
endforeach()

file(MAKE_DIRECTORY "${work}")
file(REAL_PATH "${work}" work_root)
string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef nonce)
set(root "${work_root}/client_${backend}_${mode}_${nonce}")
cmake_path(IS_PREFIX work_root "${root}" NORMALIZE owned)

if(NOT owned OR root STREQUAL work_root)
    message(FATAL_ERROR "Static Runtime deployment must stay under its owned work directory")
endif()

set(original "${root}/original SDK")
set(relocated "${root}/relocated SDK")
file(MAKE_DIRECTORY "${original}/bin")

if(NOT backend STREQUAL "native")
    foreach(component cuda_driver cuda_compiler)
        execute_process(COMMAND "${CMAKE_COMMAND}" --install "${build}" --config "${config}"
            --prefix "${original}" --component "${component}"
            RESULT_VARIABLE installed OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 45)

        if(NOT installed STREQUAL "0")
            message(FATAL_ERROR "Install ${component} failed: ${installed}\n${output}\n${error}")
        endif()
    endforeach()

    file(WRITE "${original}/bin/runtime_client.exe.local" "")
endif()

file(COPY_FILE "${fixture}/windows_x64/runtime_client.exe" "${original}/bin/runtime_client.exe")
file(RENAME "${original}" "${relocated}")

if(EXISTS "${original}")
    message(FATAL_ERROR "Original installation remains discoverable after relocation")
endif()

file(GLOB runtime_dlls "${relocated}/bin/*cudart*.dll")

if(runtime_dlls)
    message(FATAL_ERROR "The embedded Runtime must not be replaced by a deployed Runtime DLL")
endif()

set(ENV{PATH} "$ENV{SystemRoot}/System32;$ENV{SystemRoot}")

foreach(variable CUDA_PATH CUDA_HOME CUDAVM_CUDA_DRIVER CUDAVM_DEVICES)
    unset(ENV{${variable}})
endforeach()

set(ENV{CUDAVM_BACKENDS} "${backend}")

if(backend STREQUAL "native")
    set(driver "$ENV{SystemRoot}/System32/nvcuda.dll")

    if(NOT EXISTS "${driver}")
        message("SKIP: compatibility profile unavailable")
        file(REMOVE_RECURSE "${root}")
        return()
    endif()
else()
    set(driver "${relocated}/bin/nvcuda.dll")

    if(NOT EXISTS "${driver}" OR NOT EXISTS "${relocated}/bin/cudavm_core.dll")
        message(FATAL_ERROR "Installed shader Driver and compiler are required")
    endif()
endif()

set(command "${relocated}/bin/runtime_client.exe" "${driver}" "${backend}")

if(mode STREQUAL "audit")
    set(command "${tracer}" "${relocated}/bin/runtime_client.exe" "${root}/runtime.trace" "${driver}" "${backend}")
endif()

execute_process(COMMAND ${command} WORKING_DIRECTORY "${relocated}/bin"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 120)
message("${output}${error}")
file(SHA256 "${relocated}/bin/runtime_client.exe" after)
string(JSON before GET "${hashes}" "windows_x64/runtime_client.exe")

if(NOT after STREQUAL before)
    message(FATAL_ERROR "Candidate execution changed the pinned NVIDIA executable")
endif()

if(result STREQUAL "77")
    message("SKIP: compatibility profile unavailable")
elseif(NOT result STREQUAL "0")
    message(FATAL_ERROR "Unchanged NVIDIA static Runtime client failed: ${result}; deployment retained at ${relocated}")
elseif(mode STREQUAL "audit")
    file(READ "${root}/runtime.trace" trace)

    foreach(witness "call LoadLibraryExA path=nvcuda.dll flags=800"
            "call cuGetProcAddress" "call cuGetExportTable uuid=6bd5fb6c5bf4e74a8987d93912fd9df9"
            "call cuGetExportTable uuid=d4082055bde6704b8d34ba123c66e1f2"
            "call table:c693336e1121df11a8c368f355d89593:0"
            "call driver:cuLibraryLoadData" "call driver:cuLaunchKernel" "exit 0")
        string(FIND "${trace}" "${witness}" found)

        if(found EQUAL -1)
            message(FATAL_ERROR "Runtime audit is missing witness '${witness}'; log retained at ${root}/runtime.trace")
        endif()
    endforeach()

    file(COPY_FILE "${root}/runtime.trace" "${work_root}/${backend}_runtime.trace")
endif()

file(REMOVE_RECURSE "${root}")
