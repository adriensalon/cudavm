cmake_minimum_required(VERSION 3.24)

function(_cudavm_record_provenance path value)
    string(REPLACE "\\" "\\\\" quoted "${value}")
    string(REPLACE "\"" "\\\"" quoted "${quoted}")
    string(REPLACE "\r" "\\r" quoted "${quoted}")
    string(REPLACE "\n" "\\n" quoted "${quoted}")
    string(JSON provenance SET "${provenance}" ${path} "\"${quoted}\"")
    set(provenance "${provenance}" PARENT_SCOPE)
endfunction()

foreach(argument toolkit work)
    if(NOT DEFINED ${argument} OR "${${argument}}" STREQUAL "")
        message(FATAL_ERROR "Specify -D${argument}=... when regenerating the genuine static Runtime client")
    endif()
endforeach()

get_filename_component(toolkit "${toolkit}" ABSOLUTE)
set(fixture "${CMAKE_CURRENT_LIST_DIR}")
set(nvcc "${toolkit}/bin/nvcc.exe")
file(STRINGS "${toolkit}/include/cuda.h" version REGEX "^#define CUDA_VERSION ")

if(NOT version MATCHES "12040$" OR NOT EXISTS "${nvcc}")
    message(FATAL_ERROR "This producer requires genuine NVIDIA CUDA 12.4")
endif()

if(NOT DEFINED ENV{VCToolsVersion} OR NOT DEFINED ENV{WindowsSDKVersion})
    message(FATAL_ERROR "Run in a Windows x64 Visual Studio C++ environment")
endif()

file(MAKE_DIRECTORY "${work}")
file(REAL_PATH "${work}" work)
string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef nonce)
set(root "${work}/static_runtime_${nonce}")
cmake_path(IS_PREFIX work "${root}" NORMALIZE owned)

if(NOT owned OR root STREQUAL work OR EXISTS "${root}")
    message(FATAL_ERROR "Producer work must stay inside its new owned directory")
endif()

file(MAKE_DIRECTORY "${root}" "${fixture}/windows_x64")
set(output "${fixture}/windows_x64/runtime_client.exe")
set(command "${nvcc}" --cudart=static --cudadevrt=none --use-local-env --allow-unsupported-compiler
    -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -std=c++17 -O3 -gencode=arch=compute_75,code=sm_75
    -Xcompiler /MT -Xlinker /MANIFEST:NO --keep --keep-dir "${root}"
    "${fixture}/runtime_client.cu" -o "${output}")
execute_process(COMMAND ${command} WORKING_DIRECTORY "${root}" COMMAND_ERROR_IS_FATAL ANY)
file(GLOB images "${root}/*.cubin")

if(NOT images)
    message(FATAL_ERROR "Producer did not retain its original NVIDIA cubin")
endif()

file(REMOVE "${fixture}/windows_x64/runtime_client.lib" "${fixture}/windows_x64/runtime_client.exp")
execute_process(COMMAND "${nvcc}" --version OUTPUT_VARIABLE compiler_version
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
set(provenance [[{"platform":"windows-x86_64","runtime_linkage":"genuine NVIDIA CUDA 12.4 cudart_static.lib","runtime_version":12040,"device_runtime":"none","default_stream":"legacy","application_manifest":"none; application-local DLL redirection marker supplied at deployment","generator":"produce.cmake","command":[],"original_cubin_sha256":{},"sha256":{},"producer_sha256":{}}]])

_cudavm_record_provenance(host_crt "Microsoft static release CRT, Visual C++ tools $ENV{VCToolsVersion}")
_cudavm_record_provenance(windows_sdk "$ENV{WindowsSDKVersion}")
_cudavm_record_provenance(nvcc "${compiler_version}")
file(SHA256 "${CMAKE_CURRENT_LIST_FILE}" generator_hash)
_cudavm_record_provenance(generator_sha256 "${generator_hash}")
set(index 0)

foreach(argument IN LISTS command)
    string(REPLACE "${toolkit}" "<NVIDIA-toolkit>" argument "${argument}")
    string(REPLACE "${fixture}" "<fixture>" argument "${argument}")
    string(REPLACE "${root}" "<producer-work>" argument "${argument}")
    _cudavm_record_provenance("command;${index}" "${argument}")
    math(EXPR index "${index} + 1")
endforeach()

foreach(image IN LISTS images)
    get_filename_component(name "${image}" NAME)
    file(SHA256 "${image}" digest)
    _cudavm_record_provenance("original_cubin_sha256;${name}" "${digest}")
endforeach()

foreach(name runtime_client.cu windows_x64/runtime_client.exe)
    file(SHA256 "${fixture}/${name}" digest)
    _cudavm_record_provenance("sha256;${name}" "${digest}")
endforeach()

foreach(name bin/nvcc.exe include/cuda.h include/cuda_runtime.h lib/x64/cudart_static.lib)
    file(SHA256 "${toolkit}/${name}" digest)
    _cudavm_record_provenance("producer_sha256;${name}" "${digest}")
endforeach()

file(WRITE "${fixture}/provenance.json" "${provenance}\n")
file(REAL_PATH "${root}" resolved_root)
cmake_path(IS_PREFIX work "${resolved_root}" NORMALIZE owned)

if(NOT owned OR resolved_root STREQUAL work OR IS_SYMLINK "${root}")
    message(FATAL_ERROR "Refusing cleanup outside the producer's work directory")
endif()

file(REMOVE_RECURSE "${resolved_root}")
