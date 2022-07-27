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
    fold.c
    block_builder.c
    log.c
    util.c
    compile.c

    analysis/scope.c
    analysis/free_variables.c

    transform/import.c
    transform/memory_layout.c
    transform/ir_gen_helpers.c

    passes/bind.c
    passes/normalize.c
    passes/infer.c
    passes/lower_cf_instrs.c
    passes/lower_callc.c
    passes/lower_callf.c
    passes/lower_stack.c
    passes/lower_physical_ptrs.c
    passes/lower_tailcalls.c
    emit/emit.c
    emit/spirv_builder.c)

add_library(shady ${SHADY_SOURCES})
set_property(TARGET shady PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(shady PRIVATE murmur3 containers)
target_link_libraries(shady PRIVATE SPIRV-Headers::SPIRV-Headers)
target_include_directories(shady PUBLIC ../include containers/ ../murmur3)
