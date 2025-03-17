spv_outputting_test(COMPILER vcc NAME vcc_swizzle_cpp FILES test/vcc/cpp/vec_swizzle.cpp ARGS ${VCC_TEST_ARGS} --entry-point test --execution-model Fragment --std=c++20)
spv_outputting_test(COMPILER vcc NAME vcc_textured_cpp FILES test/vcc/cpp/textured.frag.cpp ARGS ${VCC_TEST_ARGS} --entry-point main --execution-model Fragment --std=c++20)
