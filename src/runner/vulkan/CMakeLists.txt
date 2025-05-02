find_package(Vulkan)

if (Vulkan_FOUND)
    option(SHADY_ENABLE_RUNNER_VULKAN_BACKEND "Vulkan support for the 'runner' component" ON)
else()
    message("Vulkan not found, runner component cannot be built.")
endif()

if (SHADY_ENABLE_RUNNER_VULKAN_BACKEND)
    shady_add_component(NAME runner_vk SOURCES vk_runner.c vk_runner_device.c vk_runner_program.c vk_runner_dispatch.c vk_runner_buffer.c vk_runner_interface.c)
    target_link_libraries(runner_vk PRIVATE api)
    target_link_libraries(runner_vk PRIVATE "$<BUILD_INTERFACE:common>")
    target_link_libraries(runner_vk PUBLIC runtime)
    target_link_libraries(runner_vk PUBLIC Vulkan::Headers Vulkan::Vulkan)

    target_compile_definitions(runner PUBLIC VK_BACKEND_PRESENT=1)
    target_link_libraries(runner PUBLIC runner_vk)
endif()
