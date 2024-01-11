find_package(json-c REQUIRED)

if(WIN32)
	set(SHADY_C_JSON json-c::json-c-static)
else()
	set(SHADY_C_JSON json-c::json-c)
endif()

if(NOT TARGET ${SHADY_C_JSON})
	message(FATAL_ERROR "Found json-c module, but requested target ${SHADY_C_JSON} doesn't exist!")
endif()

add_library(generator_common generator.c generator_common.c json_apply.c)
target_link_libraries(generator_common PUBLIC common ${SHADY_C_JSON})
target_include_directories(generator_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

function(add_generated_file)
    cmake_parse_arguments(PARSE_ARGV 0 F "" "FILE_NAME;TARGET_NAME" "SOURCES" )
    set(GENERATOR_NAME generator_${F_FILE_NAME})
    add_executable(${GENERATOR_NAME} ${F_SOURCES})
    target_link_libraries(${GENERATOR_NAME} generator_common)

    get_target_property(SPIRV_HEADERS_INCLUDE_DIRS SPIRV-Headers::SPIRV-Headers INTERFACE_INCLUDE_DIRECTORIES)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME} COMMAND ${GENERATOR_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME} ${CMAKE_SOURCE_DIR}/include/shady/grammar.json ${CMAKE_SOURCE_DIR}/include/shady/primops.json ${SPIRV_HEADERS_INCLUDE_DIRS} DEPENDS ${GENERATOR_NAME} ${CMAKE_SOURCE_DIR}/include/shady/grammar.json ${CMAKE_SOURCE_DIR}/include/shady/primops.json VERBATIM)
    if ("${F_TARGET_NAME}" STREQUAL "")
        set(F_TARGET_NAME generate_${F_FILE_NAME})
    endif ()
    add_custom_target(${F_TARGET_NAME} DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME})
endfunction()
