add_executable(runner_test runner_test.c runner_oracle.c)
target_link_libraries(runner_test PRIVATE common)
target_link_libraries(runner_test PRIVATE runner)
target_link_libraries(runner_test PRIVATE json-c::json-c)

add_test(NAME runner_hello_world COMMAND runner_test ${CMAKE_CURRENT_SOURCE_DIR}/hello_world_printf.slim)
add_test(NAME runner_buffer_copy COMMAND runner_test ${CMAKE_CURRENT_SOURCE_DIR}/buffer_copy.slim)
