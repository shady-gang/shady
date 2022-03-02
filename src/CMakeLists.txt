set(SHADY_SOURCES
    arena.c
    implem.h
    type.c
    print.c
    containers/list.c
    emit/emit.c
    emit/spirv_builder.c)

add_library(shady ${SHADY_SOURCES})
target_include_directories(shady PUBLIC ../include)

add_subdirectory(slim)