list(APPEND VCC_SIMPLE_TESTS empty.c)
list(APPEND VCC_SIMPLE_TESTS address_spaces.c)

foreach(T IN LISTS VCC_SIMPLE_TESTS)
    add_test(NAME "test/vcc/${T}" COMMAND vcc ${PROJECT_SOURCE_DIR}/test/vcc/${T})
endforeach()

list(APPEND VCC_COMPUTE_TESTS empty.comp.c)

foreach(T IN LISTS VCC_COMPUTE_TESTS)
    add_test(NAME "test/vcc/${T}" COMMAND vcc ${PROJECT_SOURCE_DIR}/test/vcc/${T} -o ${T}.spv --entry-point main)
endforeach()
