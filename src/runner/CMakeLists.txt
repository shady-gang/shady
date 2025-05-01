option(SHADY_ENABLE_RUNNER "Framework for running compute applications. Some samples and tests depend on it." ON)

if (SHADY_ENABLE_RUNNER)
    shady_add_component(NAME runner SOURCES runner.c runner_program.c runner_cli.c)
    target_link_libraries(runner PUBLIC driver)

    add_subdirectory(vulkan)
    add_subdirectory(cuda)

    add_executable(runner_test runner_test.c)
    target_link_libraries(runner_test runner)
endif()
