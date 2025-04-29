get_target_property(SPIRV_HEADERS_INCLUDE_DIRS SPIRV-Headers::SPIRV-Headers INTERFACE_INCLUDE_DIRECTORIES)

set(SHADY_IMPORTED_JSON_PATH ${CMAKE_CURRENT_BINARY_DIR}/imported.json CACHE INTERNAL "path to imported.json")

# ugh
# https://stackoverflow.com/a/4012677
add_custom_command(OUTPUT ${SHADY_IMPORTED_JSON_PATH} COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import_spv_defs.py ${SHADY_IMPORTED_JSON_PATH} ${PROJECT_SOURCE_DIR}/include/shady/spv_imports.json ${SPIRV_HEADERS_INCLUDE_DIRS} DEPENDS SPIRV-Headers::SPIRV-Headers ${PROJECT_SOURCE_DIR}/include/shady/spv_imports.json VERBATIM)
add_custom_target(shady_imported_json DEPENDS ${SHADY_IMPORTED_JSON_PATH})

function(add_generated_file_python)
    cmake_parse_arguments(PARSE_ARGV 0 F "" "FILE_NAME;TARGET_NAME;GENERATOR" "")

    get_target_property(SPIRV_HEADERS_INCLUDE_DIRS SPIRV-Headers::SPIRV-Headers INTERFACE_INCLUDE_DIRECTORIES)

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} COMMAND ${Python_EXECUTABLE} -m ${F_GENERATOR} ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME} "${SHADY_IMPORTED_JSON_PATH}" ${PROJECT_SOURCE_DIR}/include/shady/grammar.json ${PROJECT_SOURCE_DIR}/include/shady/enum.json DEPENDS shady_imported_json ${PROJECT_SOURCE_DIR}/include/shady/grammar.json ${PROJECT_SOURCE_DIR}/include/shady/enum.json VERBATIM)
    add_custom_target(${F_TARGET_NAME} DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME})
endfunction()