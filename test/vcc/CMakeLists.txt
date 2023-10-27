list(APPEND VCC_SIMPLE_TESTS empty.c)
list(APPEND VCC_SIMPLE_TESTS address_spaces.c)

foreach(T IN LISTS VCC_SIMPLE_TESTS)
    add_test(NAME "test/vcc/${T}" COMMAND vcc ${PROJECT_SOURCE_DIR}/test/vcc/${T})
endforeach()

spv_outputting_test(NAME test/vcc/branch.c COMPILER vcc EXTRA_ARGS)
spv_outputting_test(NAME test/vcc/loop.c COMPILER vcc EXTRA_ARGS)
spv_outputting_test(NAME test/vcc/goto.c COMPILER vcc EXTRA_ARGS)

spv_outputting_test(NAME test/vcc/empty.comp.c COMPILER vcc EXTRA_ARGS --entry-point main)

spv_outputting_test(NAME test/vcc/simple.frag.c COMPILER vcc EXTRA_ARGS --entry-point main --no-dynamic-scheduling --execution-model Fragment)
spv_outputting_test(NAME test/vcc/checkerboard.frag.c COMPILER vcc EXTRA_ARGS --entry-point main --no-dynamic-scheduling --execution-model Fragment)
