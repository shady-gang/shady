add_library(shady_c STATIC
    emit_c.c
    emit_c_value.c
    emit_c_type.c
    emit_c_builtin.c
    emit_c_control_flow.c
)
set_property(TARGET shady_c PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(shady_c PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_link_libraries(shady_c PRIVATE "api")
target_link_libraries(shady_c INTERFACE "$<BUILD_INTERFACE:shady>")
target_link_libraries(shady_c PRIVATE "$<BUILD_INTERFACE:common>")
target_link_libraries(shady_c PRIVATE "$<BUILD_INTERFACE:shady_generated>")

embed_file(string shady_cuda_prelude_src cuda_prelude.cu)
embed_file(string shady_cuda_builtins_src cuda_builtins.cu)
target_link_libraries(shady_c PRIVATE "$<BUILD_INTERFACE:shady_cuda_prelude_src>")
target_link_libraries(shady_c PRIVATE "$<BUILD_INTERFACE:shady_cuda_builtins_src>")

embed_file(string shady_glsl_120_polyfills_src glsl_120_polyfills.glsl)
target_link_libraries(shady_c PRIVATE "$<BUILD_INTERFACE:shady_glsl_120_polyfills_src>")

target_link_libraries(driver PUBLIC "$<BUILD_INTERFACE:shady_c>")
