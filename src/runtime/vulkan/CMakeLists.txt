find_package(Vulkan)

if (Vulkan_FOUND)
    option(SHADY_ENABLE_RUNTIME_VULKAN_BACKEND "Vulkan support for the 'runner' component" ON)
else()
    message("Vulkan not found, runtime component cannot be built.")
endif()

if (SHADY_ENABLE_RUNTIME_VULKAN_BACKEND)
    shady_add_component(NAME runtime_vk SOURCES vk_runtime_module_interface.c vk_runtime_device_caps.c)
    target_link_libraries(runtime_vk PRIVATE common)
    target_link_libraries(runtime_vk PUBLIC Vulkan::Headers Vulkan::Vulkan api)
endif()
