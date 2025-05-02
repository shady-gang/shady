shady_add_component(NAME backend_c SOURCES
    emit_c.c
    emit_c_value.c
    emit_c_type.c
    emit_c_builtin.c
    emit_c_control_flow.c
)

target_include_directories(backend_c PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_link_libraries(backend_c PRIVATE api)
target_link_libraries(backend_c PRIVATE shady)
target_link_libraries(backend_c PRIVATE common)

embed_file(string shady_cuda_prelude_src prelude.cu)
embed_file(string shady_cuda_runtime_src runtime.cu)
target_link_libraries(backend_c PRIVATE "$<BUILD_INTERFACE:shady_cuda_prelude_src>")
target_link_libraries(backend_c PRIVATE "$<BUILD_INTERFACE:shady_cuda_runtime_src>")

embed_file(string shady_glsl_runtime_120_src runtime_120.glsl)
target_link_libraries(backend_c PRIVATE "$<BUILD_INTERFACE:shady_glsl_runtime_120_src>")

embed_file(string shady_ispc_runtime_src runtime.ispc)
target_link_libraries(backend_c PRIVATE "$<BUILD_INTERFACE:shady_ispc_runtime_src>")

target_link_libraries(driver PUBLIC backend_c)
