add_library(runtime runtime.c runtime_program.c runtime_cli.c)
target_link_libraries(runtime PUBLIC driver)
set_target_properties(runtime PROPERTIES OUTPUT_NAME "shady_runtime")

add_subdirectory(vulkan)
add_subdirectory(cuda)

add_executable(runtime_test runtime_test.c)
target_link_libraries(runtime_test runtime)

install(TARGETS runtime EXPORT shady_export_set ARCHIVE DESTINATION ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
