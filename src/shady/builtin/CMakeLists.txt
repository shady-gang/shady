add_executable(embedder embed.c)

add_library(builtin_code INTERFACE)
target_include_directories(builtin_code INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

function(embed_file TYPE NAME SRC)
    add_custom_target(builtin_code_generate_${NAME} COMMAND embedder ${TYPE} ${NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${SRC} ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.h)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${SRC}.h PROPERTIES GENERATED TRUE)
    add_dependencies(builtin_code builtin_code_generate_${NAME})
endfunction()

embed_file(string shady_scheduler_src scheduler.slim)

