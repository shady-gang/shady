add_executable(runner_test runner_test.c runner_oracle.c)
target_link_libraries(runner_test PRIVATE common)
target_link_libraries(runner_test PRIVATE runner)
target_link_libraries(runner_test PRIVATE json-c::json-c)
