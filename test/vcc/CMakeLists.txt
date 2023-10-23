list(APPEND VCC_SIMPLE_TESTS empty.c)
list(APPEND VCC_SIMPLE_TESTS address_spaces.c)

foreach(T IN LISTS VCC_SIMPLE_TESTS)
    add_test(NAME "test/vcc/${T}" COMMAND vcc ${PROJECT_SOURCE_DIR}/test/vcc/${T})
endforeach()

function(spv_outputting_test)
    cmake_parse_arguments(PARSE_ARGV 0 F "" "SINGLE;NAME" "MULTI_VALUES;EXTRA_ARGS" )
    #add_test(NAME "test/vcc/${F_NAME}" COMMAND vcc ${PROJECT_SOURCE_DIR}/test/vcc/${F_NAME} ${F_EXTRA_ARGS} --entry-point main -o ${ARG_NAME}.spv)
    add_test(NAME "test/vcc/${F_NAME}" COMMAND ${CMAKE_COMMAND} -DVCC=${PROJECT_BINARY_DIR}/bin/vcc -DT=${F_NAME} "-DTARGS=${F_EXTRA_ARGS}" -DSRC=${CMAKE_CURRENT_SOURCE_DIR} -S ${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/vcc_and_val.cmake)
endfunction()

spv_outputting_test(NAME empty.comp.c EXTRA_ARGS --entry-point main)
