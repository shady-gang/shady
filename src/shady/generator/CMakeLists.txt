find_package(json-c REQUIRED)

add_executable(generator generator.c)
target_link_libraries(generator PRIVATE common json-c::json-c)

function(add_generated_file)
    cmake_parse_arguments(PARSE_ARGV 0 F "" "FILE_NAME;CONTENTS" "" )
    get_target_property(SPIRV_HEADERS_INCLUDE_DIRS SPIRV-Headers::SPIRV-Headers INTERFACE_INCLUDE_DIRECTORIES)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME} COMMAND generator ${F_CONTENTS} ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME} ${CMAKE_SOURCE_DIR}/include/shady/grammar.json ${CMAKE_SOURCE_DIR}/include/shady/primops.json ${SPIRV_HEADERS_INCLUDE_DIRS} DEPENDS generator ${CMAKE_SOURCE_DIR}/include/shady/grammar.json ${CMAKE_SOURCE_DIR}/include/shady/primops.json VERBATIM)
    add_custom_target(generate_${F_FILE_NAME} DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME})
    add_dependencies(api INTERFACE generate_${F_FILE_NAME})
endfunction()
