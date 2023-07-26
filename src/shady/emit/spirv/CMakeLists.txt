add_library(shady_spirv OBJECT
    emit_spv.c
    emit_spv_type.c
    emit_spv_instructions.c
    spirv_builder.c
)
set_property(TARGET shady_spirv PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(shady_spirv PUBLIC "api")
target_link_libraries(shady_spirv PRIVATE "$<BUILD_INTERFACE:common>")
target_link_libraries(shady_spirv PRIVATE "$<BUILD_INTERFACE:SPIRV-Headers::SPIRV-Headers>")