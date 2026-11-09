cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "Configure cudavm from the repository root; external dependencies are not standalone projects")
endif()
if(NOT CUDAVM_ENABLE_VULKAN)
    return()
endif()

include(FetchContent)
FetchContent_Declare(vulkan_headers
    URL https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/vulkan-sdk-1.4.363.0.tar.gz
    URL_HASH SHA256=4a078be12bef21cfebc09d878b77a63cff9d68f899254a0b00d0e37ef73e7f7e
    DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.tmp/downloads"
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    SOURCE_SUBDIR include)
FetchContent_MakeAvailable(vulkan_headers)

add_library(cudavm_vulkan_headers INTERFACE)
target_include_directories(cudavm_vulkan_headers SYSTEM INTERFACE
    "$<BUILD_INTERFACE:${vulkan_headers_SOURCE_DIR}/include>")
install(FILES "${vulkan_headers_SOURCE_DIR}/LICENSE.md"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licences/Vulkan-Headers")
install(DIRECTORY "${vulkan_headers_SOURCE_DIR}/LICENSES/"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licences/Vulkan-Headers")
