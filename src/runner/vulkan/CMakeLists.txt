find_package(Vulkan)

if (Vulkan_FOUND)
    option(SHADY_ENABLE_RUNNER_VULKAN_BACKEND "Vulkan support for the 'runner' component" ON)
else()
    message("Vulkan not found, runner component cannot be built.")
endif()

if (SHADY_ENABLE_RUNNER_VULKAN_BACKEND)
    shady_add_component(NAME vk_runner SOURCES vk_runner.c vk_runner_device.c vk_runner_program.c vk_runner_dispatch.c vk_runner_buffer.c vk_runner_interface.c)
    target_link_libraries(vk_runner PRIVATE api)
    target_link_libraries(vk_runner PRIVATE "$<BUILD_INTERFACE:common>")
    target_link_libraries(vk_runner PRIVATE shd_runtime shd_vk_runtime)
    target_link_libraries(vk_runner PUBLIC Vulkan::Headers Vulkan::Vulkan)

    target_compile_definitions(runner PUBLIC VK_BACKEND_PRESENT=1)
    target_link_libraries(runner PRIVATE vk_runner)
endif()
