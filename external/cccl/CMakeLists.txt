cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "Configure cudavm from the repository root; external dependencies are not standalone projects")
endif()

include(FetchContent)
FetchContent_Declare(cccl
    URL https://github.com/NVIDIA/cccl/archive/refs/tags/v3.4.2.tar.gz
    URL_HASH SHA256=544f5f14c59cae6cfdad3156aa720f2dd44fbd6feb3556d10bdb4fe6aa3cdf4b
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    SOURCE_SUBDIR libcudacxx/include)
FetchContent_MakeAvailable(cccl)

set(CUDAVM_CCCL_INCLUDE_DIR "${cccl_SOURCE_DIR}/libcudacxx/include")
if(NOT EXISTS "${CUDAVM_CCCL_INCLUDE_DIR}/nv/target")
    message(FATAL_ERROR "CCCL 3.4.2 source is missing libcudacxx/include/nv/target")
endif()
set(CUDAVM_CCCL_INCLUDE_DIR "${CUDAVM_CCCL_INCLUDE_DIR}" PARENT_SCOPE)
install(FILES "${cccl_SOURCE_DIR}/LICENSE"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licences" RENAME CCCL-LICENSE.txt)
