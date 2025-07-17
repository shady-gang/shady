list(APPEND VCC_SIMPLE_TESTS empty.c)
list(APPEND VCC_SIMPLE_TESTS address_spaces.c)
list(APPEND VCC_SIMPLE_TESTS alias.c)

set(VCC_TEST_ARGS --target none --vcc-include-path "${PROJECT_BINARY_DIR}/share/vcc/include/")

foreach(T IN LISTS VCC_SIMPLE_TESTS)
    add_test(NAME "${T}" COMMAND vcc ${PROJECT_SOURCE_DIR}/test/vcc/${T} ${VCC_TEST_ARGS})
endforeach()

shady_unit_test(COMPILER vcc NAME vcc_branch FILES branch.c ARGS ${VCC_TEST_ARGS})
shady_unit_test(COMPILER vcc NAME vcc_loop FILES loop.c ARGS ${VCC_TEST_ARGS})
# shady_unit_test(NAME test/vcc/loop_closed.c ARGS ${VCC_TEST_ARGS})
shady_unit_test(COMPILER vcc NAME vcc_goto FILES goto.c ARGS ${VCC_TEST_ARGS})
shady_unit_test(COMPILER vcc NAME vcc_ternary_expr FILES ternary.c ARGS ${VCC_TEST_ARGS})
shady_unit_test(COMPILER vcc NAME vcc_string FILES string.c ARGS ${VCC_TEST_ARGS})
shady_unit_test(COMPILER vcc NAME vcc_fn_addr FILES fn_addr.c ARGS ${VCC_TEST_ARGS})
shady_unit_test(COMPILER vcc NAME vcc_pointer FILES pointer.c ARGS ${VCC_TEST_ARGS})

shady_unit_test(COMPILER vcc NAME vcc_pass_struct_by_val FILES byval_struct.c ARGS ${VCC_TEST_ARGS} --entry-point main --execution-model Compute VALSPV SPV_VAL_ARGS ${SHADY_SPV_VAL_VK_ARGS})

shady_unit_test(COMPILER vcc NAME vcc_native_swizzle FILES native_vec_swizzle.c ARGS ${VCC_TEST_ARGS} --entry-point test --execution-model Fragment VALSPV SPV_VAL_ARGS ${SHADY_SPV_VAL_VK_ARGS})

shady_unit_test(COMPILER vcc NAME vcc_trivial_comp FILES empty.comp.c  ARGS ${VCC_TEST_ARGS} --entry-point main VALSPV SPV_VAL_ARGS ${SHADY_SPV_VAL_VK_ARGS})

shady_unit_test(COMPILER vcc NAME vcc_simple_frag FILES simple.frag.c ARGS ${VCC_TEST_ARGS} --entry-point main --execution-model Fragment VALSPV SPV_VAL_ARGS ${SHADY_SPV_VAL_VK_ARGS})
shady_unit_test(COMPILER vcc NAME vcc_flat_decoration_frag FILES flat_decoration.frag.c ARGS ${VCC_TEST_ARGS} --entry-point main --execution-model Fragment VALSPV SPV_VAL_ARGS ${SHADY_SPV_VAL_VK_ARGS})
shady_unit_test(COMPILER vcc NAME vcc_checker_frag FILES checkerboard.frag.c ARGS ${VCC_TEST_ARGS} --entry-point main --execution-model Fragment VALSPV SPV_VAL_ARGS ${SHADY_SPV_VAL_VK_ARGS})
shady_unit_test(COMPILER vcc NAME vcc_textured_frag FILES textured.frag.c ARGS ${VCC_TEST_ARGS} --entry-point main --execution-model Fragment VALSPV SPV_VAL_ARGS ${SHADY_SPV_VAL_VK_ARGS})

shady_unit_test(COMPILER vcc NAME vcc_linkage FILES linkage.frag.c lib.c ARGS ${VCC_TEST_ARGS} --entry-point main --execution-model Fragment VALSPV SPV_VAL_ARGS ${SHADY_SPV_VAL_VK_ARGS})
shady_unit_test(COMPILER vcc NAME vcc_linkage_type FILES linkage2.frag.c lib2.c ARGS ${VCC_TEST_ARGS} --entry-point main --execution-model Fragment VALSPV SPV_VAL_ARGS ${SHADY_SPV_VAL_VK_ARGS})

add_subdirectory(cpp)
