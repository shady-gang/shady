option(SHADY_ENABLE_RUNTIME "Runtime support library for shady kernels" ON)

if (SHADY_ENABLE_RUNTIME)
    shady_add_component(NAME runtime SOURCES runtime.c)
    target_link_libraries(runtime PRIVATE driver common)

    add_subdirectory(vulkan)
endif ()
