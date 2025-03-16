find_package(Vulkan)

if (Vulkan_FOUND)
    option(SHADY_ENABLE_RUNTIME_VULKAN_BACKEND "Vulkan support for the 'runner' component" ON)
else()
    message("Vulkan not found, runtime component cannot be built.")
endif()

if (SHADY_ENABLE_RUNTIME_VULKAN_BACKEND)
    add_library(shd_vk_runtime STATIC vk_runtime.c)
    target_link_libraries(shd_vk_runtime PRIVATE Vulkan::Headers api common)
endif()
