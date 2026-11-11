if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    get_filename_component(host_tool_directory "${CMAKE_CXX_COMPILER}" DIRECTORY)
    find_program(public_nvcc_host NAMES clang-cl HINTS "${host_tool_directory}" REQUIRED)
    add_custom_target(cudavm_test_public_sdk DEPENDS cudavm_nvcc cudavm_core_shared cudavm_cuda_shared
        cudavm_cudart_shared cudavm_cudart_static cudavm_cudadevrt_static)
    add_dependencies(cudavm_test cudavm_test_public_sdk)

    foreach(mode discovery protocol execution archives cmake cmake_rdc)
        set(slice nvcc_driver)
        set(backend host)

        if(mode STREQUAL "discovery")
            set(slice nvidia_sdk_discovery)
        elseif(mode MATCHES "^cmake")
            set(slice cuda_build_integration)
            set(backend vulkan)
        elseif(mode STREQUAL "execution" OR mode STREQUAL "archives")
            set(backend vulkan)
        endif()

        if(backend STREQUAL "vulkan" AND NOT "vulkan" IN_LIST test_backends)
            continue()
        endif()

        set(case "test.${slice}.${backend}.${mode}")
        add_test(NAME "${case}" COMMAND "${CMAKE_COMMAND}"
            "-Dbuild=${CMAKE_BINARY_DIR}" "-Dconfig=$<CONFIG>" "-Dwork=${CMAKE_CURRENT_BINARY_DIR}/public_sdk"
            "-Dmode=${mode}" "-Dcmake=${CMAKE_COMMAND}" "-Dninja=${CMAKE_MAKE_PROGRAM}" "-Dhost=${public_nvcc_host}"
            -P "${CMAKE_CURRENT_LIST_DIR}/run.cmake")
        set_tests_properties("${case}" PROPERTIES RUN_SERIAL TRUE TIMEOUT 420
            SKIP_REGULAR_EXPRESSION "SKIP: selected Vulkan backend has no available device"
            LABELS "test;${slice};${backend};A01.discovery;C01.options;C01.link;C02.inputs")
    endforeach()
endif()
