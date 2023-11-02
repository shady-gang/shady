add_subdirectory(generator)

# :( see https://gitlab.kitware.com/cmake/cmake/-/issues/18399
# set_source_files_properties(${CMAKE_SOURCE_DIR}/include/shady/generated_grammar.h PROPERTIES GENERATED TRUE)
#set_property(SOURCE ${CMAKE_SOURCE_DIR}/include/shady/generated_grammar.h PROPERTY GENERATED TRUE)

add_library(api INTERFACE)
target_include_directories(api INTERFACE "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>" "$<INSTALL_INTERFACE:include>")

get_target_property(SPIRV_HEADERS_INCLUDE_DIRS SPIRV-Headers::SPIRV-Headers INTERFACE_INCLUDE_DIRECTORIES)
add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/include/shady/generated_grammar.h COMMENT "generating shady headers" COMMAND generator grammar-headers ${CMAKE_SOURCE_DIR}/include/shady/generated_grammar.h ${CMAKE_SOURCE_DIR}/include/shady/grammar.json ${SPIRV_HEADERS_INCLUDE_DIRS} DEPENDS generator ${CMAKE_SOURCE_DIR}/include/shady/grammar.json VERBATIM)
add_custom_target(generate_grammar DEPENDS ${CMAKE_SOURCE_DIR}/include/shady/generated_grammar.h)
add_dependencies(api INTERFACE generate_grammar)

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/node_generated.c COMMAND generator node-code ${CMAKE_CURRENT_SOURCE_DIR}/node_generated.c ${CMAKE_SOURCE_DIR}/include/shady/grammar.json ${SPIRV_HEADERS_INCLUDE_DIRS} DEPENDS generator ${CMAKE_SOURCE_DIR}/include/shady/grammar.json VERBATIM)
add_custom_target(node_generated DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/node_generated.c)

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/type_generated.h COMMAND generator type-headers ${CMAKE_CURRENT_SOURCE_DIR}/type_generated.h ${CMAKE_SOURCE_DIR}/include/shady/grammar.json ${SPIRV_HEADERS_INCLUDE_DIRS} DEPENDS generator ${CMAKE_SOURCE_DIR}/include/shady/grammar.json VERBATIM)
add_custom_target(type_generated DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/type_generated.h)

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/constructors_generated.c COMMAND generator constructors-code ${CMAKE_CURRENT_SOURCE_DIR}/constructors_generated.c ${CMAKE_SOURCE_DIR}/include/shady/grammar.json ${SPIRV_HEADERS_INCLUDE_DIRS} DEPENDS generator ${CMAKE_SOURCE_DIR}/include/shady/grammar.json VERBATIM)
add_custom_target(constructors_generated DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/constructors_generated.c)

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/visit_generated.c COMMAND generator visitor-code ${CMAKE_CURRENT_SOURCE_DIR}/visit_generated.c ${CMAKE_SOURCE_DIR}/include/shady/grammar.json ${SPIRV_HEADERS_INCLUDE_DIRS} DEPENDS generator ${CMAKE_SOURCE_DIR}/include/shady/grammar.json VERBATIM)
add_custom_target(visit_generated DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/visit_generated.c)

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/rewrite_generated.c COMMAND generator rewriter-code ${CMAKE_CURRENT_SOURCE_DIR}/rewrite_generated.c ${CMAKE_SOURCE_DIR}/include/shady/grammar.json ${SPIRV_HEADERS_INCLUDE_DIRS} DEPENDS generator ${CMAKE_SOURCE_DIR}/include/shady/grammar.json VERBATIM)
add_custom_target(rewrite_generated DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/rewrite_generated.c)

set(SHADY_SOURCES
    ir.c
    node.c
    constructors.c
    type.c
    type_helpers.c
    builtins.c
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
    analysis/uses.c
    analysis/looptree.c

    transform/memory_layout.c
    transform/ir_gen_helpers.c
    transform/internal_constants.c

    passes/import.c
    passes/bind.c
    passes/normalize.c
    passes/infer.c
    passes/lower_cf_instrs.c
    passes/lift_indirect_targets.c
    passes/lcssa.c
    passes/lower_callf.c
    passes/lower_stack.c
    passes/lower_lea.c
    passes/lower_physical_ptrs.c
    passes/lower_generic_ptrs.c
    passes/lower_memory_layout.c
    passes/lower_memcpy.c
    passes/lower_decay_ptrs.c
    passes/lower_tailcalls.c
    passes/lower_mask.c
    passes/lower_fill.c
    passes/lower_switch_btree.c
    passes/setup_stack_frames.c
    passes/eliminate_constants.c
    passes/normalize_builtins.c
    passes/lower_subgroup_ops.c
    passes/lower_subgroup_vars.c
    passes/lower_int64.c
    passes/lower_vec_arr.c
    passes/lower_workgroups.c
    passes/mark_leaf_functions.c
    passes/opt_inline.c
    passes/opt_stack.c
    passes/opt_restructure.c
    passes/reconvergence_heuristics.c
    passes/simt2d.c
    passes/specialize_entry_point.c
    passes/specialize_execution_model.c

    passes/lower_entrypoint_args.c
    passes/spirv_map_entrypoint_args.c
    passes/spirv_lift_globals_ssbo.c
)

add_library(shady SHARED ${SHADY_SOURCES})
set_property(TARGET shady PROPERTY POSITION_INDEPENDENT_CODE ON)

add_dependencies(shady node_generated type_generated constructors_generated visit_generated rewrite_generated)

if (WIN32)
    if (MSVC)
        target_link_options(shady PUBLIC /STACK:33554432)
    else()
        target_link_options(shady PUBLIC -Wl,--stack,33554432)
    endif()
endif()

add_subdirectory(internal)
add_subdirectory(emit)

target_link_libraries(shady PUBLIC "api")
target_link_libraries(shady PUBLIC "$<BUILD_INTERFACE:common>")
target_link_libraries(shady PRIVATE "$<BUILD_INTERFACE:SPIRV-Headers::SPIRV-Headers>")
