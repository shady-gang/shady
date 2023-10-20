add_library(runtime SHARED runtime.c runtime_program.c)
target_link_libraries(runtime PUBLIC api shady)
target_link_libraries(runtime PRIVATE common driver)
set_property(TARGET runtime PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(runtime PROPERTIES OUTPUT_NAME "shady_runtime")

add_subdirectory(vulkan)

add_executable(runtime_test runtime_test.c)
target_link_libraries(runtime_test common)
target_link_libraries(runtime_test shady)
target_link_libraries(runtime_test runtime)
