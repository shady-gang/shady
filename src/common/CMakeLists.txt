add_library(common STATIC list.c dict.c log.c portability.c util.c growy.c arena.c printer.c)
target_include_directories(common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(common PRIVATE "$<BUILD_INTERFACE:murmur3>")
set_property(TARGET common PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(embedder embed.c)

function(embed_file TYPE NAME SRC)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.h COMMAND embedder ${TYPE} ${NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${SRC} ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.h DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SRC} VERBATIM)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${NAME}.h PROPERTIES GENERATED TRUE)
	add_custom_target("${NAME}_h" DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.h)

    add_library(${NAME} INTERFACE)
	add_dependencies(${NAME} "${NAME}_h")
    target_include_directories(${NAME} INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
endfunction()
