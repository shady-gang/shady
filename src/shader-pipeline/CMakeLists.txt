shady_add_component(NAME shader_pipeline SOURCES cf.c memory.c polyfill.c shader_pipeline.c)
target_link_libraries(shader_pipeline PRIVATE api shady slim_parser common)
