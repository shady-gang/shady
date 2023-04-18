embed_file(string shady_scheduler_src scheduler.slim)
target_link_libraries(shady PRIVATE "$<BUILD_INTERFACE:shady_scheduler_src>")
