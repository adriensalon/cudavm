cmake_minimum_required(VERSION 3.24)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "Configure cudavm from the repository root; external dependencies are not standalone projects")
endif()

if(NOT CUDAVM_ENABLE_OPENCL)
    return()
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
FetchContent_Declare(opencl_headers
    URL https://github.com/KhronosGroup/OpenCL-Headers/archive/refs/tags/v2026.05.29.tar.gz
    URL_HASH SHA256=d9e6c48357de5002da11ce45de600e0c3ffe6ab4f628a3b9fe2b38603161658a
    DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/_deps"
    DOWNLOAD_NAME opencl.tar.gz
    DOWNLOAD_NO_PROGRESS FALSE
    USES_TERMINAL_DOWNLOAD TRUE
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    SOURCE_SUBDIR CL)
FetchContent_MakeAvailable(opencl_headers)

add_library(cudavm_opencl_headers INTERFACE)
target_include_directories(cudavm_opencl_headers SYSTEM INTERFACE
    "$<BUILD_INTERFACE:${opencl_headers_SOURCE_DIR}>")
install(FILES "${opencl_headers_SOURCE_DIR}/LICENSE"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licences" RENAME OpenCL_Headers_LICENSE.txt)
