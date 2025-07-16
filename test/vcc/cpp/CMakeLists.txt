shady_unit_test(COMPILER vcc NAME vcc_textured_cpp FILES textured.frag.cpp ARGS ${VCC_TEST_ARGS} --entry-point main --execution-model Fragment --std=c++20 VALSPV SPV_VAL_ARGS ${SHADY_SPV_VAL_VK_ARGS})
shady_unit_test(COMPILER vcc NAME vcc_ternary_cpp FILES ternary.cpp ARGS ${VCC_TEST_ARGS} --entry-point main --execution-model Fragment --std=c++20 VALSPV SPV_VAL_ARGS ${SHADY_SPV_VAL_VK_ARGS})
shady_unit_test(COMPILER vcc NAME vcc_ternary_cpp_o3 FILES ternary.cpp ARGS ${VCC_TEST_ARGS} --entry-point main --execution-model Fragment --std=c++20 -O3 VALSPV SPV_VAL_ARGS ${SHADY_SPV_VAL_VK_ARGS})
