# Host API clients use the configured targets, with no SDK discovery adapter.
add_custom_target(cudavm-conformance-host_linkage)
add_dependencies(cudavm-conformance cudavm-conformance-host_linkage)
foreach(mode shared static)
    set(target cudavm-conformance-host_linkage-${mode})
    add_executable(${target} "${CMAKE_CURRENT_LIST_DIR}/host_linkage.cpp" "${CMAKE_CURRENT_LIST_DIR}/host_linkage.c")
    target_compile_features(${target} PRIVATE cxx_std_20 c_std_11)
    target_link_libraries(${target} PRIVATE cudavm::cudart_${mode} cudavm::cuda_${mode})
    foreach(component nvrtc nvvm nvjitlink nvfatbin nvptxcompiler)
        target_link_libraries(${target} PRIVATE cudavm::${component}_${mode})
        string(TOUPPER "${component}" define)
        target_compile_definitions(${target} PRIVATE HOST_WITH_${define})
    endforeach()
    cudavm_prepare_target(${target} NO_DEPLOY)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    add_dependencies(cudavm-conformance-host_linkage ${target})
    add_test(NAME conformance.host_linkage.host.${mode} COMMAND ${target})
    set_tests_properties(conformance.host_linkage.host.${mode} PROPERTIES
        LABELS "conformance;host_linkage;host;${mode};A01.headers;A01.exports;C01.link"
        ENVIRONMENT "CUDAVM_BACKENDS=" TIMEOUT 60)
endforeach()

# Exercise static compiler archive composition through the public APIs. This
# process needs neither shared compiler services nor an initialized Driver.
set(target cudavm-conformance-host_linkage-compiler-static)
add_executable(${target} "${CMAKE_CURRENT_LIST_DIR}/compiler_api.cpp")
target_compile_features(${target} PRIVATE cxx_std_17)
target_link_libraries(${target} PRIVATE cudavm::nvrtc_static cudavm::nvjitlink_static
    cudavm::nvptxcompiler_static cudavm::nvfatbin_static cudavm::nvvm_static cudavm::cuda_static)
cudavm_prepare_target(${target} NO_DEPLOY)
set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/host_linkage/static/$<CONFIG>")
add_dependencies(cudavm-conformance-host_linkage ${target})
add_test(NAME conformance.host_linkage.host.compiler_static COMMAND ${target})
set_tests_properties(conformance.host_linkage.host.compiler_static PROPERTIES
    LABELS "conformance;host_linkage;host;static;C01.link"
    ENVIRONMENT "CUDAVM_BACKENDS=" TIMEOUT 60)
