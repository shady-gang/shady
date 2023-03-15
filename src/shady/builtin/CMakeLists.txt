add_executable(embedder embed.c)

set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/builtin_code.c PROPERTIES GENERATED TRUE)
add_library(builtin_code OBJECT ${CMAKE_CURRENT_BINARY_DIR}/builtin_code.c)
target_include_directories(builtin_code INTERFACE .)

add_custom_target(builtin_code_generate COMMAND ${CMAKE_COMMAND} -DBUILTIN_CODE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/embed_builtin_code.cmake)
add_dependencies(builtin_code builtin_code_generate)
