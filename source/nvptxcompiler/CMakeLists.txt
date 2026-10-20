# Private optional PTX-to-shader service. This does not implement NVIDIA's
# public nvPTXCompiler binary-producing API.
add_library(cudavm_ptx_jit SHARED "${engine_dir}/ptx_import.cpp" "${engine_dir}/ptx_jit.cpp"
    "${engine_dir}/nvptx_emitter.cpp" "${engine_dir}/native_emission.cpp"
    "${engine_dir}/spirv_emitter.cpp" "${engine_dir}/shader_lowering.cpp"
    "${engine_dir}/dxil_emitter.cpp" "${engine_dir}/module_descriptor.cpp"
    "${engine_dir}/device_passes.cpp" "${engine_dir}/host_embedding.cpp" "${engine_dir}/compiler_io.cpp")
target_compile_features(cudavm_ptx_jit PRIVATE cxx_std_20)
target_include_directories(cudavm_ptx_jit PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_compile_definitions(cudavm_ptx_jit PRIVATE NDEBUG _ITERATOR_DEBUG_LEVEL=0 _CRT_SECURE_NO_WARNINGS)
cudavm_compiler_dependencies(cudavm_ptx_jit ${link_libraries})
set_target_properties(cudavm_ptx_jit PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(cudavm_ptx_jit PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${configuration} "${PROJECT_BINARY_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_${configuration} "${PROJECT_BINARY_DIR}")
endforeach()
install(TARGETS cudavm_ptx_jit RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
list(APPEND CUDAVM_SHADER_COMPILER_FILES "$<TARGET_FILE:cudavm_ptx_jit>")
set(CUDAVM_SHADER_COMPILER_FILES ${CUDAVM_SHADER_COMPILER_FILES} PARENT_SCOPE)
