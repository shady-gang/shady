spv_outputting_test(NAME test/vcc/cpp/vec_swizzle.cpp COMPILER vcc EXTRA_ARGS ${VCC_TEST_ARGS} --entry-point test --execution-model Fragment --std=c++20)
spv_outputting_test(NAME test/vcc/cpp/textured.frag.cpp COMPILER vcc EXTRA_ARGS ${VCC_TEST_ARGS} --entry-point main --execution-model Fragment --std=c++20)
