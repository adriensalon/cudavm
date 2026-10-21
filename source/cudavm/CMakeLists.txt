# Private optional PTX-to-shader compiler service used by the Driver.
# The public NVIDIA nvPTXCompiler API belongs to source/nvptxcompiler.
add_library(cudavm_ptxjit SHARED ptx_import.cpp ptx_jit.cpp device_numeric_ir.cpp
    nvptx_emitter.cpp native_emission.cpp spirv_emitter.cpp shader_lowering.cpp
    dxil_emitter.cpp module_descriptor.cpp device_passes.cpp host_embedding.cpp compiler_io.cpp)
target_compile_features(cudavm_ptxjit PRIVATE cxx_std_20)
target_include_directories(cudavm_ptxjit PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_compile_definitions(cudavm_ptxjit PRIVATE NDEBUG _ITERATOR_DEBUG_LEVEL=0 _CRT_SECURE_NO_WARNINGS)
cudavm_compiler_dependencies(cudavm_ptxjit clangCodeGen ${source_libraries} ${link_libraries})
set_target_properties(cudavm_ptxjit PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(cudavm_ptxjit PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${configuration} "${PROJECT_BINARY_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_${configuration} "${PROJECT_BINARY_DIR}")
endforeach()
install(TARGETS cudavm_ptxjit RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
list(APPEND CUDAVM_SHADER_COMPILER_FILES "$<TARGET_FILE:cudavm_ptxjit>")
set(CUDAVM_SHADER_COMPILER_FILES ${CUDAVM_SHADER_COMPILER_FILES} PARENT_SCOPE)
