add_library(shady_c STATIC
    emit_c.c
    emit_c_instructions.c
    emit_c_signatures.c
)

target_link_libraries(shady_c PUBLIC "api")
target_link_libraries(shady_c PRIVATE "$<BUILD_INTERFACE:common>")
