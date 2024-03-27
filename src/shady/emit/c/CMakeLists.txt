add_library(shady_c OBJECT
    emit_c.c
    emit_c_instructions.c
    emit_c_signatures.c
    emit_c_builtins.c
)
set_property(TARGET shady_c PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(shady_c PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(shady_c PUBLIC "api")
target_link_libraries(shady_c PRIVATE "$<BUILD_INTERFACE:common>")
target_link_libraries(shady_c PUBLIC "$<BUILD_INTERFACE:shady_generated>")

embed_file(string shady_cuda_prelude_src cuda_prelude.cu)
embed_file(string shady_cuda_builtins_src cuda_builtins.cu)
target_link_libraries(shady_c PRIVATE "$<BUILD_INTERFACE:shady_cuda_prelude_src>")
target_link_libraries(shady_c PRIVATE "$<BUILD_INTERFACE:shady_cuda_builtins_src>")