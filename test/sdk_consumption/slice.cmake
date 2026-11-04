# Installation witnesses are CMake-driven host tests, not GPU runner programs.
add_custom_target(cudavm-conformance-sdk_consumption)
add_dependencies(cudavm-conformance cudavm-conformance-sdk_consumption)
set(sdk_source_build "")
set(sdk_existing_prefix "")
get_target_property(sdk_imported cudavm::cudart_shared IMPORTED)
if(sdk_imported)
    get_filename_component(sdk_existing_prefix "${cudavm_DIR}/../../.." ABSOLUTE)
else()
    set(sdk_source_build "${CMAKE_BINARY_DIR}")
    foreach(component cuda cudart core nvrtc nvvm nvjitlink nvfatbin nvptxcompiler)
        foreach(mode shared static)
            if(TARGET cudavm::${component}_${mode})
                get_target_property(sdk_dependency cudavm::${component}_${mode} ALIASED_TARGET)
                add_dependencies(cudavm-conformance-sdk_consumption ${sdk_dependency})
            endif()
        endforeach()
    endforeach()
    foreach(tool cudavm-info cudavm-nvcc)
        if(TARGET ${tool})
            add_dependencies(cudavm-conformance-sdk_consumption ${tool})
        endif()
    endforeach()
endif()
configure_file("${CMAKE_CURRENT_LIST_DIR}/configuration.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/sdk_consumption/configuration.cmake" @ONLY)
foreach(mode shared static)
    if(NOT TARGET cudavm::cudart_${mode})
        continue()
    endif()
    add_test(NAME conformance.sdk_consumption.host.${mode}
        COMMAND "${CMAKE_COMMAND}" "-Dmode=${mode}" "-Dconfig=$<CONFIG>"
            "-Dsettings=${CMAKE_CURRENT_BINARY_DIR}/sdk_consumption/configuration.cmake"
            -P "${CMAKE_CURRENT_LIST_DIR}/run.cmake")
    set_tests_properties(conformance.sdk_consumption.host.${mode} PROPERTIES
        LABELS "conformance;sdk_consumption;host;${mode};A01.headers;A01.exports;C01.link"
        ENVIRONMENT "CUDAVM_BACKENDS=" TIMEOUT 300 RUN_SERIAL TRUE)
endforeach()
