if(NOT TARGET cudavm::nvcc)
    message(VERBOSE "Skipping public device_libraries conformance: SDK compiler tool is unavailable")
    return()
endif()

cudavm_add_conformance_slice(device_libraries API_ONLY CASES chain cycle FAMILY_CHECKS C01.link)

if(NOT TARGET cudavm_test_device_libraries)
    return()
endif()

foreach(backend IN LISTS conformance_backends)
    foreach(profile chain cycle)
        set_property(TEST "conformance.device_libraries.${backend}.${profile}" APPEND PROPERTY LABELS source_aot)
    endforeach()
endforeach()

set(archive_fixture_root "${CMAKE_CURRENT_BINARY_DIR}/device_libraries")
set(archive_fixture_sources "${CMAKE_CURRENT_LIST_DIR}")
set(archive_fixture_builder cudavm_test_device_libraries_builder)
add_executable(${archive_fixture_builder} "${archive_fixture_sources}/archive_builder.cpp")
target_compile_features(${archive_fixture_builder} PRIVATE cxx_std_20)

if(WIN32)
    set_property(TARGET ${archive_fixture_builder} PROPERTY MSVC_RUNTIME_LIBRARY MultiThreaded)
endif()

set(archive_compiler_dependencies "$<TARGET_FILE:cudavm::nvcc>" "$<TARGET_FILE:cudavm::core_shared>")
set(archive_fixture_images)

foreach(profile chain cycle)
    set(work "${archive_fixture_root}/${profile}")
    file(MAKE_DIRECTORY "${work}")
    set(cyclic 0)

    if(profile STREQUAL cycle)
        set(cyclic 1)
    endif()

    foreach(unit entry start leaf tail unused)
        add_custom_command(OUTPUT "${work}/${unit}.bc" "${work}/${unit}.host.cpp"
            COMMAND "$<TARGET_FILE:cudavm::nvcc>" compile-tu
                --input "${archive_fixture_sources}/${unit}.cu" --host "${work}/${unit}.host.cpp"
                --bitcode "${work}/${unit}.bc" --arch sm_75 --standard c++17 --define "ARCHIVE_CYCLE=${cyclic}"
            DEPENDS "${archive_fixture_sources}/${unit}.cu" ${archive_compiler_dependencies}
            COMMENT "Compiling ${profile} archive fixture ${unit}" VERBATIM)
    endforeach()

    add_custom_command(OUTPUT "${work}/first.a"
        COMMAND "$<TARGET_FILE:${archive_fixture_builder}>" --archive "${work}/first.a"
            "${work}/start.bc" "${work}/tail.bc" "${work}/unused.bc"
        DEPENDS ${archive_fixture_builder} "${work}/start.bc" "${work}/tail.bc" "${work}/unused.bc"
        COMMENT "Packaging ${profile} first device archive" VERBATIM)
    add_custom_command(OUTPUT "${work}/second.a"
        COMMAND "$<TARGET_FILE:${archive_fixture_builder}>" --archive "${work}/second.a" "${work}/leaf.bc"
        DEPENDS ${archive_fixture_builder} "${work}/leaf.bc"
        COMMENT "Packaging ${profile} second device archive" VERBATIM)

    foreach(order forward reverse)
        set(libraries "${work}/first.a" "${work}/second.a")

        if(order STREQUAL reverse)
            list(REVERSE libraries)
        endif()

        set(arguments --bitcode "${work}/entry.bc")

        foreach(library IN LISTS libraries)
            list(APPEND arguments --bitcode "${library}")
        endforeach()

        add_custom_command(OUTPUT "${work}/${order}.image" "${work}/${order}.bc" "${work}/${order}.json"
            COMMAND "$<TARGET_FILE:cudavm::nvcc>" link-device ${arguments}
                --portable "${work}/${order}.bc" --manifest "${work}/${order}.json"
                --target all --binary "${work}/${order}.image" --arch sm_75
            DEPENDS "${work}/entry.bc" ${libraries} ${archive_compiler_dependencies}
            COMMENT "Linking ${profile} device archives in ${order} order" VERBATIM)
        list(APPEND archive_fixture_images "${work}/${order}.image")
    endforeach()
endforeach()

set(archive_fixture_header "${archive_fixture_root}/device_library_images.hpp")
add_custom_command(OUTPUT "${archive_fixture_header}"
    COMMAND "$<TARGET_FILE:${archive_fixture_builder}>" --embed "${archive_fixture_header}" ${archive_fixture_images}
    DEPENDS ${archive_fixture_builder} ${archive_fixture_images}
    COMMENT "Embedding public device archive execution images" VERBATIM)
target_sources(cudavm_test_device_libraries PRIVATE "${archive_fixture_header}")
target_include_directories(cudavm_test_device_libraries PRIVATE "${archive_fixture_root}")
