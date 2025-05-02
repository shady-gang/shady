shady_add_component(NAME shader_pipeline SOURCES cf.c memory.c polyfill.c scheduler.c shader_pipeline.c)
target_link_libraries(shader_pipeline PRIVATE api shady common frontend_slim)

embed_file(string shady_scheduler_src scheduler.slim)
target_link_libraries(shader_pipeline PRIVATE "$<BUILD_INTERFACE:shady_scheduler_src>")