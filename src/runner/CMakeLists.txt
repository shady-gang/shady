option(SHADY_ENABLE_RUNNER "Framework for running compute applications. Some samples and tests depend on it." ON)

if (SHADY_ENABLE_RUNNER)
    shady_add_component(NAME runner_base SOURCES runner.c runner_program.c runner_cli.c)
    target_link_libraries(runner_base PRIVATE common)
    target_link_libraries(runner_base PUBLIC api driver)

    shady_add_component(NAME runner SOURCES runner_init.c)
    target_link_libraries(runner PUBLIC runner_base)
    target_link_libraries(runner PRIVATE common)

    add_subdirectory(vulkan)
    add_subdirectory(cuda)
endif()
