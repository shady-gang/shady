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

    transform/import.c
    transform/memory_layout.c
    transform/ir_gen_helpers.c

    passes/bind.c
    passes/infer.c
    passes/lower_cf_instrs.c
    passes/lower_callc.c
    passes/lower_callf.c
    passes/lower_stack.c
    emit/emit.c
    emit/spirv_builder.c)

add_library(shady ${SHADY_SOURCES})
target_link_libraries(shady PRIVATE murmur3 containers)
target_include_directories(shady PUBLIC ../include containers/ ../murmur3)
