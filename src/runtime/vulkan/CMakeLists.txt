find_package(Vulkan)

if (Vulkan_FOUND)
    option(SHADY_ENABLE_RUNTIME_VULKAN "Vulkan support for the 'runtime' component" ON)
else()
    message("Vulkan not found, runtime component cannot be built.")
endif()

if (SHADY_ENABLE_RUNTIME_VULKAN)
    add_library(vk_runtime STATIC vk_runtime.c vk_runtime_device.c vk_runtime_program.c vk_runtime_dispatch.c vk_runtime_buffer.c)
    target_link_libraries(vk_runtime PRIVATE api)
    target_link_libraries(vk_runtime PRIVATE "$<BUILD_INTERFACE:common>")
    target_link_libraries(vk_runtime PRIVATE "$<BUILD_INTERFACE:murmur3>")
    target_link_libraries(vk_runtime PRIVATE Vulkan::Headers Vulkan::Vulkan)

    target_compile_definitions(runtime PUBLIC VK_BACKEND_PRESENT=1)
    target_link_libraries(runtime PRIVATE "$<BUILD_INTERFACE:vk_runtime>")
endif()
