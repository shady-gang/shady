add_library(shady_c OBJECT
    emit_c.c
    emit_c_instructions.c
    emit_c_signatures.c
    emit_c_builtins.c
)
set_property(TARGET shady_c PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(shady_c PUBLIC "api")
target_link_libraries(shady_c PRIVATE "$<BUILD_INTERFACE:common>")
