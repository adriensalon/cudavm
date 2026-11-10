if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64)$")
    add_executable(cudavm_test_runtime_trace "${CMAKE_CURRENT_LIST_DIR}/runtime_trace.cpp")
    add_executable(cudavm_test_runtime_exports "${CMAKE_CURRENT_LIST_DIR}/runtime_exports.cpp")

    foreach(target cudavm_test_runtime_trace cudavm_test_runtime_exports)
        target_compile_features(${target} PRIVATE cxx_std_20)
        set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" MSVC_RUNTIME_LIBRARY MultiThreaded)
        cudavm_prepare_target(${target})
    endforeach()

    target_include_directories(cudavm_test_runtime_exports PRIVATE "${PROJECT_SOURCE_DIR}/include/cuda")
    add_custom_target(cudavm_test_nvidia_static_runtime_clients
        DEPENDS cudavm_cuda_shared cudavm_core_shared cudavm_test_runtime_trace cudavm_test_runtime_exports)
    add_dependencies(cudavm_test cudavm_test_nvidia_static_runtime_clients)

    foreach(backend native vulkan)
        if(NOT backend STREQUAL "native" AND NOT backend IN_LIST conformance_backends)
            continue()
        endif()

        foreach(mode execute audit)
            set(case "conformance.nvidia_static_runtime_clients.${backend}.${mode}")
            add_test(NAME "${case}" COMMAND "${CMAKE_COMMAND}"
                "-Dbuild=${CMAKE_BINARY_DIR}" "-Dconfig=$<CONFIG>"
                "-Dfixture=${CMAKE_CURRENT_LIST_DIR}"
                "-Dwork=${CMAKE_CURRENT_BINARY_DIR}/nvidia_static_runtime_clients"
                "-Dbackend=${backend}" "-Dmode=${mode}"
                "-Dtracer=$<TARGET_FILE:cudavm_test_runtime_trace>"
                -P "${CMAKE_CURRENT_LIST_DIR}/run.cmake")
            set_tests_properties("${case}" PROPERTIES RUN_SERIAL TRUE TIMEOUT 180
                LABELS "conformance;nvidia_static_runtime_clients;${backend};A01.discovery;A01.registration;C02.inputs"
                SKIP_REGULAR_EXPRESSION "SKIP: compatibility profile unavailable")
        endforeach()

        add_test(NAME "conformance.runtime_exports.${backend}.contracts"
            COMMAND cudavm_test_runtime_exports "$<IF:$<STREQUAL:${backend},native>,native,$<TARGET_FILE:cudavm_cuda_shared>>")
        set_tests_properties("conformance.runtime_exports.${backend}.contracts" PROPERTIES
            ENVIRONMENT "CUDAVM_BACKENDS=${backend}" RUN_SERIAL TRUE TIMEOUT 120 SKIP_RETURN_CODE 77
            LABELS "conformance;runtime_exports;${backend};A01.discovery;H01.lifetime")
    endforeach()
endif()
