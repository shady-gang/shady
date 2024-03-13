add_library(runtime SHARED runtime.c runtime_program.c)
target_link_libraries(runtime PUBLIC shady)
target_link_libraries(runtime PUBLIC "$<BUILD_INTERFACE:driver>")
set_property(TARGET runtime PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(runtime PROPERTIES OUTPUT_NAME "shady_runtime")

add_subdirectory(vulkan)
add_subdirectory(cuda)

add_executable(runtime_test runtime_test.c)
target_link_libraries(runtime_test runtime)
