find_package(Vulkan)

add_library(runtime runtime.c runtime_device.c runtime_program.c runtime_dispatch.c runtime_buffer.c)
target_link_libraries(runtime PUBLIC api)
target_link_libraries(runtime PRIVATE shady)
target_link_libraries(runtime PRIVATE "$<BUILD_INTERFACE:common>")
target_link_libraries(runtime PRIVATE "$<BUILD_INTERFACE:murmur3>")
target_link_libraries(runtime INTERFACE Vulkan::Headers Vulkan::Vulkan)
target_include_directories(runtime PRIVATE ../../include)

add_executable(runtime_test runtime_test.c)
target_link_libraries(runtime_test runtime)
target_link_libraries(runtime_test common)
