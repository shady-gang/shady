shady_add_component(NAME backend_spirv SOURCES
    emit_spv.c
    emit_spv_type.c
    emit_spv_value.c
    emit_spv_control_flow.c
    spirv_lower_entrypoint_args.c
    spirv_remove_bda_params.c
    spirv_callable_shaders.c
        spirv_specialize_explicit_layout.c
    spirv_builder.c
)

target_include_directories(backend_spirv PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_link_libraries(backend_spirv PRIVATE api)
target_link_libraries(backend_spirv PRIVATE shady)
target_link_libraries(backend_spirv PRIVATE common)

target_link_libraries(driver PUBLIC backend_spirv)
