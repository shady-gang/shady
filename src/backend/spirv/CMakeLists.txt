add_library(shady_spirv STATIC
    emit_spv.c
    emit_spv_type.c
    emit_spv_value.c
    emit_spv_control_flow.c
    spirv_lower_entrypoint_args.c
    spirv_remove_bda_params.c
    spirv_callable_shaders.c
    spirv_builder.c
)
set_property(TARGET shady_spirv PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(shady_spirv PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_link_libraries(shady_spirv PRIVATE "api")
target_link_libraries(shady_spirv INTERFACE "$<BUILD_INTERFACE:shady>")
target_link_libraries(shady_spirv PRIVATE "$<BUILD_INTERFACE:common>")

target_link_libraries(driver PUBLIC "$<BUILD_INTERFACE:shady_spirv>")
