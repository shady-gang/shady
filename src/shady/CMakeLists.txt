set(SHADY_SOURCES
    ir.c
    node.c
    constructors.c
    type.c
    type_helpers.c
    rewrite.c
    visit.c
    print.c
    fold.c
    body_builder.c
    compile.c
    annotation.c
    module.c
    cli.c

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
    passes/lower_lea.c
    passes/lower_physical_ptrs.c
    passes/lower_generic_ptrs.c
    passes/lower_memory_layout.c
    passes/lower_memcpy.c
    passes/lower_tailcalls.c
    passes/lower_mask.c
    passes/setup_stack_frames.c
    passes/eliminate_constants.c
    passes/lower_subgroup_ops.c
    passes/lower_subgroup_vars.c
    passes/lower_int64.c
    passes/mark_leaf_functions.c
    passes/opt_simplify_cf.c
    passes/opt_stack.c
    passes/opt_restructure.c
    passes/simt2d.c
    passes/specialize_for_entry_point.c

    passes/lower_entrypoint_args.c
    passes/spirv_map_entrypoint_args.c
    passes/spirv_lift_globals_ssbo.c
)

add_library(shady SHARED ${SHADY_SOURCES})
set_property(TARGET shady PROPERTY POSITION_INDEPENDENT_CODE ON)

if (WIN32)
    if (MSVC)
        target_link_options(shady PUBLIC /STACK:33554432)
    else()
        target_link_options(shady PUBLIC -Wl,--stack,33554432)
    endif()
endif()

add_subdirectory(builtin)
add_subdirectory(emit)

target_link_libraries(shady PUBLIC "api")
target_link_libraries(shady PRIVATE "$<BUILD_INTERFACE:builtin_code>")
target_link_libraries(shady PRIVATE "$<BUILD_INTERFACE:common>")
