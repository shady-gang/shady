add_library(murmur3 ../murmur3/murmur3.c)

set(SHADY_SOURCES
    arena.c
    implem.h
    type.c
    print.c
    containers/list.c
    emit/emit.c
    emit/spirv_builder.c)

add_library(shady ${SHADY_SOURCES})
target_link_libraries(shady murmur3)
target_include_directories(shady PUBLIC ../include)

add_subdirectory(slim)