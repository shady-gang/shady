get_target_property(SPIRV_HEADERS_INCLUDE_DIRS SPIRV-Headers::SPIRV-Headers INTERFACE_INCLUDE_DIRECTORIES)

function(add_generated_file_python)
    cmake_parse_arguments(PARSE_ARGV 0 F "" "FILE_NAME;TARGET_NAME;GENERATOR" "")

    get_target_property(SPIRV_HEADERS_INCLUDE_DIRS SPIRV-Headers::SPIRV-Headers INTERFACE_INCLUDE_DIRECTORIES)

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} COMMAND ${Python_EXECUTABLE} -m ${F_GENERATOR} ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME}  ${PROJECT_SOURCE_DIR}/include/shady/grammar.json ${PROJECT_SOURCE_DIR}/include/shady/enum.json DEPENDS ${PROJECT_SOURCE_DIR}/include/shady/grammar.json ${PROJECT_SOURCE_DIR}/include/shady/enum.json VERBATIM)
    add_custom_target(${F_TARGET_NAME} DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME})
endfunction()