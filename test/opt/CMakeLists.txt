add_executable(opt_oracle opt_oracle.c)
target_link_libraries(opt_oracle PRIVATE driver)

add_test(NAME "mem2reg1" COMMAND opt_oracle ${CMAKE_CURRENT_SOURCE_DIR}/mem2reg1.slim --no-dynamic-scheduling)
add_test(NAME "mem2reg2" COMMAND opt_oracle ${CMAKE_CURRENT_SOURCE_DIR}/mem2reg2.slim --no-dynamic-scheduling)
add_test(NAME "mem2reg3" COMMAND opt_oracle ${CMAKE_CURRENT_SOURCE_DIR}/mem2reg3.slim --no-dynamic-scheduling)

add_test(NAME "mem2reg_should_fail" COMMAND opt_oracle ${CMAKE_CURRENT_SOURCE_DIR}/mem2reg_should_fail.slim --no-dynamic-scheduling --expect-memops)
