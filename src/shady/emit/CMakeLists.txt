add_subdirectory(spirv)
add_subdirectory(c)

add_library(shady_emit_common OBJECT
    common.c
)
