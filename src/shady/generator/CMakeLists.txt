
if (NOT ${EXTERNAL_JSON_C})
	find_package(json-c REQUIRED)
    set(json_c_library json-c::json-c)
else ()
    set(json_c_library json-c)
endif ()

add_library(generator_common generator.c generator_common.c json_apply.c)
target_link_libraries(generator_common PUBLIC common ${json_c_library})
target_include_directories(generator_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (${EXTERNAL_JSON_C})
	target_include_directories(generator_common SYSTEM BEFORE PRIVATE ${EXTERNAL_JSON_C_INCLUDE})
endif ()

function(add_generated_file)
    cmake_parse_arguments(PARSE_ARGV 0 F "" "FILE_NAME;TARGET_NAME" "SOURCES" )
    set(GENERATOR_NAME generator_${F_FILE_NAME})
    add_executable(${GENERATOR_NAME} ${F_SOURCES})
    target_link_libraries(${GENERATOR_NAME} generator_common)

    get_target_property(SPIRV_HEADERS_INCLUDE_DIRS SPIRV-Headers::SPIRV-Headers INTERFACE_INCLUDE_DIRECTORIES)
	add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME} COMMAND ${GENERATOR_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME} ${SHADY_SOURCE_DIR}/include/shady/grammar.json ${SHADY_SOURCE_DIR}/include/shady/primops.json ${SPIRV_HEADERS_INCLUDE_DIRS} DEPENDS ${GENERATOR_NAME} ${SHADY_SOURCE_DIR}/include/shady/grammar.json ${SHADY_SOURCE_DIR}/include/shady/primops.json VERBATIM)
    if ("${F_TARGET_NAME}" STREQUAL "")
        set(F_TARGET_NAME generate_${F_FILE_NAME})
    endif ()
    add_custom_target(${F_TARGET_NAME} DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME})
endfunction()
