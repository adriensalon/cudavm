cudavm_add_component(nvjitlink linker_api.cpp "${engine_dir}/nvptx_emitter.cpp"
    "${engine_dir}/native_emission.cpp"
    "${engine_dir}/spirv_emitter.cpp"
    "${engine_dir}/shader_lowering.cpp"
    "${engine_dir}/ptx_import.cpp"
    "${engine_dir}/dxil_emitter.cpp"
    "${engine_dir}/module_descriptor.cpp"
    "${engine_dir}/device_passes.cpp" "${engine_dir}/host_embedding.cpp" "${engine_dir}/compiler_io.cpp")
set(cudavm_install_targets ${cudavm_install_targets} PARENT_SCOPE)
foreach(linkage shared static)
    if(TARGET cudavm_nvjitlink_${linkage})
        cudavm_compiler_dependencies(cudavm_nvjitlink_${linkage} ${link_libraries})
    endif()
endforeach()
