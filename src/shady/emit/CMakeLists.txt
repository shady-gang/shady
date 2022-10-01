add_subdirectory(spirv)

target_link_libraries(shady PRIVATE "$<BUILD_INTERFACE:shady_spirv>")
