add_library(shady_spirv STATIC
    emit.c
    emit_type.c
    emit_builtins.c
    spirv_builder.c
)

target_link_libraries(shady_spirv PUBLIC "api")
target_link_libraries(shady_spirv PRIVATE "$<BUILD_INTERFACE:common>")
target_link_libraries(shady_spirv PRIVATE "$<BUILD_INTERFACE:SPIRV-Headers::SPIRV-Headers>")