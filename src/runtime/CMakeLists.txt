find_package(Vulkan)

add_library(runtime runtime.c)
target_link_libraries(runtime Vulkan::Headers)
target_include_directories(runtime PUBLIC ../../include)