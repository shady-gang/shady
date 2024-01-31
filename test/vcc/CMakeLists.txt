list(APPEND VCC_SIMPLE_TESTS empty.c)
list(APPEND VCC_SIMPLE_TESTS address_spaces.c)

set(VCC_TEST_ARGS --vcc-include-path "${PROJECT_BINARY_DIR}/share/vcc/include/")

foreach(T IN LISTS VCC_SIMPLE_TESTS)
    add_test(NAME "test/vcc/${T}" COMMAND vcc ${PROJECT_SOURCE_DIR}/test/vcc/${T} ${VCC_TEST_ARGS})
endforeach()

spv_outputting_test(NAME test/vcc/branch.c COMPILER vcc EXTRA_ARGS ${VCC_TEST_ARGS})
spv_outputting_test(NAME test/vcc/loop.c COMPILER vcc EXTRA_ARGS ${VCC_TEST_ARGS})
spv_outputting_test(NAME test/vcc/goto.c COMPILER vcc EXTRA_ARGS ${VCC_TEST_ARGS})
spv_outputting_test(NAME test/vcc/ternary.c COMPILER vcc EXTRA_ARGS ${VCC_TEST_ARGS})

spv_outputting_test(NAME test/vcc/vec_swizzle.c COMPILER vcc EXTRA_ARGS ${VCC_TEST_ARGS} --entry-point test --execution-model Fragment)

spv_outputting_test(NAME test/vcc/empty.comp.c COMPILER vcc EXTRA_ARGS ${VCC_TEST_ARGS} --entry-point main)

spv_outputting_test(NAME test/vcc/simple.frag.c COMPILER vcc EXTRA_ARGS ${VCC_TEST_ARGS} --entry-point main --execution-model Fragment)
spv_outputting_test(NAME test/vcc/checkerboard.frag.c COMPILER vcc EXTRA_ARGS ${VCC_TEST_ARGS} --entry-point main --execution-model Fragment)
spv_outputting_test(NAME test/vcc/textured.frag.c COMPILER vcc EXTRA_ARGS ${VCC_TEST_ARGS} --entry-point main --execution-model Fragment)
