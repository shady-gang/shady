list(APPEND VCC_SIMPLE_TESTS empty.c)
list(APPEND VCC_SIMPLE_TESTS address_spaces.c)

set(VCC_TEST_ARGS --vcc-include-path "${PROJECT_BINARY_DIR}/share/vcc/include/")

foreach(T IN LISTS VCC_SIMPLE_TESTS)
    add_test(NAME "${T}" COMMAND vcc ${PROJECT_SOURCE_DIR}/test/vcc/${T} ${VCC_TEST_ARGS})
endforeach()

spv_outputting_test(COMPILER vcc NAME vcc_branch FILES test/vcc/branch.c ARGS ${VCC_TEST_ARGS})
spv_outputting_test(COMPILER vcc NAME vcc_loop FILES test/vcc/loop.c ARGS ${VCC_TEST_ARGS})
# spv_outputting_test(NAME test/vcc/loop_closed.c ARGS ${VCC_TEST_ARGS})
spv_outputting_test(COMPILER vcc NAME vcc_goto FILES test/vcc/goto.c ARGS ${VCC_TEST_ARGS})
spv_outputting_test(COMPILER vcc NAME vcc_ternary_expr FILES  test/vcc/ternary.c ARGS ${VCC_TEST_ARGS})
spv_outputting_test(COMPILER vcc NAME vcc_string FILES test/vcc/string.c ARGS ${VCC_TEST_ARGS})
spv_outputting_test(COMPILER vcc NAME vcc_pass_struct_by_val FILES test/vcc/byval_struct.c ARGS ${VCC_TEST_ARGS})

spv_outputting_test(COMPILER vcc NAME vcc_swizzle FILES test/vcc/vec_swizzle.c ARGS ${VCC_TEST_ARGS} --entry-point test --execution-model Fragment)

spv_outputting_test(COMPILER vcc NAME vcc_trivial_comp FILES test/vcc/empty.comp.c  ARGS ${VCC_TEST_ARGS} --entry-point main)

spv_outputting_test(COMPILER vcc NAME vcc_simple_frag FILES test/vcc/simple.frag.c ARGS ${VCC_TEST_ARGS} --entry-point main --execution-model Fragment)
spv_outputting_test(COMPILER vcc NAME vcc_checker_frag FILES test/vcc/checkerboard.frag.c ARGS ${VCC_TEST_ARGS} --entry-point main --execution-model Fragment)
spv_outputting_test(COMPILER vcc NAME vcc_textured_frag FILES test/vcc/textured.frag.c ARGS ${VCC_TEST_ARGS} --entry-point main --execution-model Fragment)

spv_outputting_test(COMPILER vcc NAME vcc_linkage FILES test/vcc/linkage.frag.c test/vcc/lib.c ARGS ${VCC_TEST_ARGS} --entry-point main --execution-model Fragment)
spv_outputting_test(COMPILER vcc NAME vcc_linkage_type FILES test/vcc/linkage2.frag.c test/vcc/lib2.c ARGS ${VCC_TEST_ARGS} --entry-point main --execution-model Fragment)

add_subdirectory(cpp)
