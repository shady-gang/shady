add_subdirectory(spirv)
add_subdirectory(c)

target_link_libraries(shady PRIVATE "$<BUILD_INTERFACE:shady_spirv>")
target_link_libraries(shady PRIVATE "$<BUILD_INTERFACE:shady_c>")
