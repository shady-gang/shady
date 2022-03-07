add_subdirectory(slim)
add_subdirectory(containers)

add_library(murmur3 ../murmur3/murmur3.c)

set(SHADY_SOURCES
    arena.c
    node.c
    type.c
    print.c
    emit/emit.c
    emit/spirv_builder.c)

add_library(shady ${SHADY_SOURCES})
target_link_libraries(shady murmur3 containers)
target_include_directories(shady PUBLIC ../include containers/ ../murmur3)
