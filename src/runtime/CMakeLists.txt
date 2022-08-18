find_package(Vulkan)

add_library(runtime runtime.c runtime_device.c runtime_program.c runtime_dispatch.c)
target_link_libraries(runtime shady common murmur3)
target_link_libraries(runtime Vulkan::Headers Vulkan::Vulkan)
target_include_directories(runtime PUBLIC ../../include)

add_executable(runtime_test runtime_test.c)
target_link_libraries(runtime_test runtime)
