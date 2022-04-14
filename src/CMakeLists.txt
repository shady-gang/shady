add_subdirectory(slim)
add_subdirectory(containers)

add_library(murmur3 ../murmur3/murmur3.c)

set(SHADY_SOURCES
    arena.c
    node.c
    type.c
    rewrite.c
    print.c
    log.c
    transform/import.c
    analysis/scope.c
    analysis/scope.h
    passes/bind.c
    passes/infer.c
    emit/emit.c
    emit/spirv_builder.c)

add_library(shady ${SHADY_SOURCES})
target_link_libraries(shady murmur3 containers)
target_include_directories(shady PUBLIC ../include containers/ ../murmur3)
