find_package(Vulkan)

add_library(runtime runtime.c)
target_link_libraries(runtime shady common)
target_link_libraries(runtime Vulkan::Headers vulkan)
target_include_directories(runtime PUBLIC ../../include)

add_executable(runtime_test runtime_test.c)
target_link_libraries(runtime_test runtime)
