add_library(common STATIC list.c dict.c log.c portability.c util.c growy.c arena.c printer.c)
target_link_libraries(common PRIVATE "$<BUILD_INTERFACE:murmur3>")
set_property(TARGET common PROPERTY POSITION_INDEPENDENT_CODE ON)

# We need to export 'common' because otherwise when using static libraries we will not be able to resolve those symbols
install(TARGETS common EXPORT shady_export_set)

# But we don't want projects outside this to be able to see these APIs and call into them
# (Also we couldn't since the header files live with the source anyways)
add_library(common_api INTERFACE)
target_include_directories(common_api INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(common INTERFACE "$<BUILD_INTERFACE:common_api>")

add_executable(embedder embed.c)

function(embed_file TYPE NAME SRC)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.h COMMAND embedder ${TYPE} ${NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${SRC} ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.h DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SRC} VERBATIM)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${NAME}.h PROPERTIES GENERATED TRUE)
	add_custom_target("${NAME}_h" DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.h)

    add_library(${NAME} INTERFACE)
	add_dependencies(${NAME} "${NAME}_h")
    target_include_directories(${NAME} INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

