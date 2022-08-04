#file(READ scheduler.shady SCHEDULER_TXT)
#string(JOIN "\;" SCHEDULER_TXT ${SCHEDULER_TXT})
#string(REPLACE ";" "\\\;" SCHEDULER_TXT ${SCHEDULER_TXT})
#string(REPLACE "\"" "\\\"" SCHEDULER_TXT ${SCHEDULER_TXT})
#string(REPLACE "\n" "\"\n\"" SCHEDULER_TXT ${SCHEDULER_TXT})

#configure_file(builtin_code.c.in ${CMAKE_CURRENT_BINARY_DIR}/builtin_code.c @ONLY)

#set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/builtin_code.c PROPERTIES GENERATED TRUE)
add_library(builtin_code STATIC ${CMAKE_CURRENT_BINARY_DIR}/builtin_code.c)

add_custom_target(builtin_code_generate COMMAND ${CMAKE_COMMAND} -DBUILTIN_CODE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/embed_builtin_code.cmake)
add_dependencies(builtin_code builtin_code_generate)
