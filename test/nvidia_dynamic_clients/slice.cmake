function(_cudavm_add_nvidia_driver_client_case slice backend mode)
    set(case "conformance.${slice}.${backend}.${mode}")
    add_test(NAME "${case}" COMMAND "${CMAKE_COMMAND}"
        "-Dbuild=${CMAKE_BINARY_DIR}" "-Dconfig=$<CONFIG>"
        "-Dfixture=${CMAKE_CURRENT_LIST_DIR}"
        "-Dwork=${CMAKE_CURRENT_BINARY_DIR}/nvidia_dynamic_clients"
        "-Dbackend=${backend}" "-Dmode=${mode}"
        -P "${CMAKE_CURRENT_LIST_DIR}/run.cmake")
    set_tests_properties("${case}" PROPERTIES
        LABELS "conformance;${slice};${backend};A01.exports;A01.discovery;C02.inputs"
        RUN_SERIAL TRUE TIMEOUT 180 SKIP_REGULAR_EXPRESSION "SKIP: compatibility profile unavailable")
endfunction()

if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64)$")
    add_custom_target(cudavm_test_nvidia_dynamic_clients
        DEPENDS cudavm_cuda_shared cudavm_core_shared)
    add_dependencies(cudavm_test cudavm_test_nvidia_dynamic_clients)

    _cudavm_add_nvidia_driver_client_case(nvidia_library_discovery host discovery)
    if("cuda" IN_LIST conformance_backends)
        _cudavm_add_nvidia_driver_client_case(native_cuda_coexistence host blocked)
    endif()
    foreach(backend cuda vulkan directx)
        if(backend IN_LIST conformance_backends)
            foreach(mode ptx cubin)
                _cudavm_add_nvidia_driver_client_case(nvidia_dynamic_clients "${backend}" "${mode}")
            endforeach()
            if(NOT backend STREQUAL "cuda")
                _cudavm_add_nvidia_driver_client_case(nvidia_library_discovery "${backend}" missing_compiler)
            endif()
        endif()
    endforeach()
endif()
