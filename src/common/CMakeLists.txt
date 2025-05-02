shady_add_component(NAME common SOURCES list.c dict.c log.c portability.c util.c growy.c arena.c printer.c)

# But we don't want projects outside this to be able to see these APIs and call into them
# (Also we couldn't since the header files live with the source anyways)
add_library(common_api INTERFACE)
target_include_directories(common_api INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(common INTERFACE "$<BUILD_INTERFACE:common_api>")

function(embed_file TYPE NAME SRC)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.h COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/embed.py ${TYPE} ${NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${SRC} ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.h DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SRC} VERBATIM)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${NAME}.h PROPERTIES GENERATED TRUE)
	add_custom_target("${NAME}_h" DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.h)

    add_library(${NAME} INTERFACE)
	add_dependencies(${NAME} "${NAME}_h")
    target_include_directories(${NAME} INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

# This hacky job is required for being able to run built targets in-place when generating the code
# This is also required for the various drivers but since they're built in the same directory it will work for now
# TODO: review this on Windows
function(copy_windows_dlls TARGET)
    message("copying DLLs for target ${TARGET}")
    add_custom_target(copy_dlls
            DEPENDS ${TARGET}
            COMMAND ${CMAKE_COMMAND} -E copy -t "$<TARGET_FILE_DIR:${TARGET}>" "$<TARGET_RUNTIME_DLLS:${TARGET}>"
            COMMAND_EXPAND_LISTS
    )
endfunction()

if (BUILD_TESTING)
    add_executable(test_dict test_dict.c)
    target_link_libraries(test_dict PRIVATE common)
    add_test(NAME test_dict COMMAND test_dict)

    add_executable(test_util test_util.c)
    target_link_libraries(test_util PRIVATE common)
    add_test(NAME test_util COMMAND test_util)
endif ()