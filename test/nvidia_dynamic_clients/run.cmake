cmake_minimum_required(VERSION 3.24)

foreach(argument build config fixture work backend mode)
    if(NOT DEFINED ${argument} OR "${${argument}}" STREQUAL "")
        message(FATAL_ERROR "Missing NVIDIA-client test argument: ${argument}")
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
        message(FATAL_ERROR "Independent NVIDIA client/producer input changed without regenerated provenance: ${name}")
    endif()
endforeach()

file(MAKE_DIRECTORY "${work}")
file(REAL_PATH "${work}" work_root)
string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef nonce)
set(root "${work_root}/client_${backend}_${mode}_${nonce}")
cmake_path(IS_PREFIX work_root "${root}" NORMALIZE owned)

if(NOT owned OR root STREQUAL work_root)
    message(FATAL_ERROR "Compatibility test deployment must stay under its owned work directory")
endif()

set(original "${root}/original SDK")
set(relocated "${root}/relocated SDK")

foreach(component cuda_driver cuda_compiler)
    execute_process(COMMAND "${CMAKE_COMMAND}" --install "${build}" --config "${config}"
        --prefix "${original}" --component "${component}"
        RESULT_VARIABLE installed OUTPUT_VARIABLE install_output ERROR_VARIABLE install_error TIMEOUT 45)
    if(NOT installed STREQUAL "0")
        message(FATAL_ERROR "Install ${component} failed: ${installed}\n${install_output}\n${install_error}")
    endif()
endforeach()

if(NOT EXISTS "${original}/bin/nvcuda.dll" OR NOT EXISTS "${original}/lib/cuda.lib"
        OR NOT EXISTS "${original}/bin/cudavm_core.dll")
    message(FATAL_ERROR "Installed Driver profile lacks its NVIDIA loader/import names or compiler dependency")
endif()

file(COPY_FILE "${fixture}/windows_x64/driver_client.exe" "${original}/bin/driver_client.exe")
file(RENAME "${original}" "${relocated}")

if(EXISTS "${original}")
    message(FATAL_ERROR "Original installation is still discoverable after relocation")
endif()

set(ENV{PATH} "$ENV{SystemRoot}/System32;$ENV{SystemRoot}")
unset(ENV{CUDA_PATH})
unset(ENV{CUDA_HOME})
unset(ENV{CUDAVM_CUDA_DRIVER})
set(ENV{CUDAVM_BACKENDS} "${backend}")
set(client_mode "${mode}")

if(mode STREQUAL "discovery")
    set(ENV{CUDAVM_BACKENDS} "")
elseif(mode STREQUAL "blocked")
    set(ENV{CUDAVM_BACKENDS} cuda)
    set(ENV{CUDAVM_CUDA_DRIVER} "${relocated}/bin/nvcuda.dll")
elseif(mode STREQUAL "missing_compiler")
    file(REMOVE "${relocated}/bin/cudavm_core.dll")
    set(client_mode ptx)
endif()

execute_process(COMMAND "${relocated}/bin/driver_client.exe" "${client_mode}" "${relocated}/bin/nvcuda.dll" "${backend}"
    WORKING_DIRECTORY "${relocated}/bin" RESULT_VARIABLE result
    OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 120)
message("${output}${error}")
file(SHA256 "${relocated}/bin/driver_client.exe" after)
string(JSON before GET "${hashes}" "windows_x64/driver_client.exe")

if(NOT after STREQUAL before)
    message(FATAL_ERROR "Candidate execution changed the independent client binary")
endif()

if(result STREQUAL "77")
    message("SKIP: compatibility profile unavailable")
elseif(mode STREQUAL "missing_compiler")
    if(NOT result STREQUAL "1" OR NOT error MATCHES "cuModuleLoadData: CUDA_ERROR_JIT_COMPILER_NOT_FOUND")
        message(FATAL_ERROR "Missing deployed compiler did not produce the exact Driver error: ${result}")
    endif()
    message("PASS: absent compiler is not borrowed from the build tree")
elseif(NOT result STREQUAL "0")
    message(FATAL_ERROR "Unchanged NVIDIA-linked client failed: ${result}; deployment retained at ${relocated}")
endif()

file(REMOVE_RECURSE "${root}")
