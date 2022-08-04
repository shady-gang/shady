find_package(Vulkan)

add_library(runtime runtime.c)
link_libraries(runtime Vulkan::Headers Vulkan::Registry)
target_include_directories(runtime PUBLIC ../../include)