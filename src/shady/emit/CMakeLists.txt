add_subdirectory(spirv)
add_subdirectory(c)

add_library(shady_emit_common OBJECT
    common.c
)

target_link_libraries(shady PRIVATE "$<BUILD_INTERFACE:shady_spirv>")
target_link_libraries(shady PRIVATE "$<BUILD_INTERFACE:shady_c>")
