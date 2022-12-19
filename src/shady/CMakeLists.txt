set(SHADY_SOURCES
    ir.c
    node.c
    constructors.c
    type.c
    rewrite.c
    visit.c
    print.c
    fold.c
    body_builder.c
    compile.c
    annotation.c
    module.c

    parser/parser.c
    parser/token.c

    analysis/scope.c
    analysis/free_variables.c
    analysis/verify.c
    analysis/callgraph.c

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
    passes/lower_subgroup_vars.c
    passes/mark_leaf_functions.c
    passes/opt_simplify_cf.c
    passes/opt_stack.c
    passes/opt_restructurize.c
)

add_library(shady SHARED ${SHADY_SOURCES})
set_property(TARGET shady PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(builtin)
add_subdirectory(emit)

target_link_libraries(shady PUBLIC "api")
#target_link_libraries(shady PUBLIC "$<BUILD_INTERFACE:api>")
target_link_libraries(shady PRIVATE "$<BUILD_INTERFACE:builtin_code>")
target_link_libraries(shady PRIVATE "$<BUILD_INTERFACE:common>")
target_link_libraries(shady PRIVATE "$<BUILD_INTERFACE:murmur3>")
