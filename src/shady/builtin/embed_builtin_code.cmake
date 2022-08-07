message("hi ${BUILTIN_CODE_DIR}")

file(READ ${BUILTIN_CODE_DIR}/scheduler.shady SCHEDULER_TXT)
string(JOIN "\;" SCHEDULER_TXT ${SCHEDULER_TXT})
string(REPLACE ";" "\\\;" SCHEDULER_TXT ${SCHEDULER_TXT})
string(REPLACE "\"" "\\\"" SCHEDULER_TXT ${SCHEDULER_TXT})
string(REPLACE "\n" "\\n" SCHEDULER_TXT ${SCHEDULER_TXT})

configure_file(${BUILTIN_CODE_DIR}/builtin_code.c.in ${CMAKE_CURRENT_BINARY_DIR}/builtin_code.c @ONLY)
