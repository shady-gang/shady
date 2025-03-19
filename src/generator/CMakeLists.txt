find_package(Python COMPONENTS Interpreter REQUIRED)

get_target_property(SPIRV_HEADERS_INCLUDE_DIRS SPIRV-Headers::SPIRV-Headers INTERFACE_INCLUDE_DIRECTORIES)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/imported.json COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/import_spv_defs.py ${CMAKE_CURRENT_BINARY_DIR}/imported.json ${PROJECT_SOURCE_DIR}/include/shady/spv_imports.json ${SPIRV_HEADERS_INCLUDE_DIRS} DEPENDS import_spv_defs SPIRV-Headers::SPIRV-Headers ${PROJECT_SOURCE_DIR}/include/shady/spv_imports.json VERBATIM)

# This hacky job is required for being able to run built targets in-place when generating the code
# This is also required for the various drivers but since they're built in the same directory it will work for now
# TODO: review this on Windows
# if (WIN32)
#     message("copying DLLs for generator targets")
#     add_custom_target(copy_dlls
#             DEPENDS import_spv_defs
#             COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:import_spv_defs> $<TARGET_RUNTIME_DLLS:import_spv_defs>
#             COMMAND_EXPAND_LISTS
#     )
# endif ()

set(SHADY_IMPORTED_JSON_PATH ${CMAKE_CURRENT_BINARY_DIR}/imported.json CACHE INTERNAL "path to imported.json")

function(add_generated_file_python)
    cmake_parse_arguments(PARSE_ARGV 0 F "" "FILE_NAME;TARGET_NAME;GENERATOR" "")

    get_target_property(SPIRV_HEADERS_INCLUDE_DIRS SPIRV-Headers::SPIRV-Headers INTERFACE_INCLUDE_DIRECTORIES)

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} COMMAND python3 -m ${F_GENERATOR} ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME} "${SHADY_IMPORTED_JSON_PATH}" ${PROJECT_SOURCE_DIR}/include/shady/grammar.json ${PROJECT_SOURCE_DIR}/include/shady/enum.json DEPENDS ${PROJECT_SOURCE_DIR}/include/shady/grammar.json ${PROJECT_SOURCE_DIR}/include/shady/enum.json VERBATIM)
    add_custom_target(${F_TARGET_NAME} DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME})
endfunction()