set(SHADY_SOURCES
    arena.c
    node.c
    type.c
    rewrite.c
    visit.c
    print.c
    fold.c
    block_builder.c
    compile.c
    annotation.c

    parser/parser.c
    parser/token.c

    analysis/scope.c
    analysis/free_variables.c
    analysis/verify.c
    analysis/callgraph.c

    transform/import.c
    transform/memory_layout.c
    transform/ir_gen_helpers.c
    transform/internal_constants.c

    passes/bind.c
    passes/normalize.c
    passes/infer.c
    passes/lower_cf_instrs.c
    passes/lower_continuations.c
    passes/lower_callf.c
    passes/lower_stack.c
    passes/lower_physical_ptrs.c
    passes/lower_tailcalls.c
    passes/lower_mask.c
    passes/setup_stack_frames.c
    passes/eliminate_constants.c
    passes/lower_subgroup_ops.c
    passes/remove_indirect_calls.c

    emit/emit.c
    emit/emit_type.c
    emit/emit_builtins.c
    emit/spirv_builder.c)

add_library(shady ${SHADY_SOURCES})
set_property(TARGET shady PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(builtin)

target_link_libraries(shady PUBLIC "api")
#target_link_libraries(shady PUBLIC "$<BUILD_INTERFACE:api>")
target_link_libraries(shady PRIVATE "$<BUILD_INTERFACE:builtin_code>")
target_link_libraries(shady PRIVATE "$<BUILD_INTERFACE:common>")
target_link_libraries(shady PRIVATE "$<BUILD_INTERFACE:murmur3>")
target_link_libraries(shady PRIVATE "$<BUILD_INTERFACE:SPIRV-Headers::SPIRV-Headers>")
