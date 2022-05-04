add_subdirectory(slim)
add_subdirectory(containers)

add_library(murmur3 STATIC ../murmur3/murmur3.c)

set(SHADY_SOURCES
    arena.c
    node.c
    type.c
    rewrite.c
    visit.c
    print.c
    log.c
    util.c
    compile.c
    analysis/scope.c
    analysis/free_variables.c
    passes/import.c
    passes/memory_layout.c
    passes/bind.c
    passes/infer.c
    passes/lower_cf_instrs.c
    passes/lower_callc.c
    passes/lower_callf.c
    emit/emit.c
    emit/spirv_builder.c)

add_library(shady ${SHADY_SOURCES})
target_link_libraries(shady PRIVATE murmur3 containers)
target_include_directories(shady PUBLIC ../include containers/ ../murmur3)
