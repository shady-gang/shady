find_package(json-c REQUIRED)

add_library(generator_common generator.c generator_common.c json_apply.c)
target_link_libraries(generator_common PUBLIC common json-c::json-c)
target_include_directories(generator_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(import_spv_defs import_spv_defs.c)
target_link_libraries(import_spv_defs PUBLIC common generator_common)

# This hacky job is required for being able to run built targets in-place when generating the code
# This is also required for the various drivers but since they're built in the same directory it will work for now
if (WIN32)
    message("copying DLLs for generator targets")
    add_custom_command(TARGET import_spv_defs POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:import_spv_defs> $<TARGET_RUNTIME_DLLS:import_spv_defs}>
            COMMAND_EXPAND_LISTS
    )
endif ()

get_target_property(SPIRV_HEADERS_INCLUDE_DIRS SPIRV-Headers::SPIRV-Headers INTERFACE_INCLUDE_DIRECTORIES)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/imported.json COMMAND import_spv_defs ${CMAKE_CURRENT_BINARY_DIR}/imported.json ${CMAKE_SOURCE_DIR}/include/shady/spv_imports.json ${SPIRV_HEADERS_INCLUDE_DIRS} DEPENDS import_spv_defs SPIRV-Headers::SPIRV-Headers ${CMAKE_SOURCE_DIR}/include/shady/spv_imports.json VERBATIM)
add_custom_target(do_import_spv_defs DEPENDS import_spv_defs ${CMAKE_CURRENT_BINARY_DIR}/imported.json)

set(SHADY_IMPORTED_JSON_PATH ${CMAKE_CURRENT_BINARY_DIR}/imported.json CACHE INTERNAL "path to imported.json")

function(add_generated_file)
    cmake_parse_arguments(PARSE_ARGV 0 F "" "FILE_NAME;TARGET_NAME" "SOURCES" )
    set(GENERATOR_NAME generator_${F_FILE_NAME})
    add_executable(${GENERATOR_NAME} ${F_SOURCES} ${CMAKE_SOURCE_DIR}/src/shady/generator/generator_main.c)
    target_link_libraries(${GENERATOR_NAME} generator_common)

    if ("${F_TARGET_NAME}" STREQUAL "")
        set(F_TARGET_NAME generate_${F_FILE_NAME})
    endif ()

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME} COMMAND ${GENERATOR_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME} "${SHADY_IMPORTED_JSON_PATH}" ${CMAKE_SOURCE_DIR}/include/shady/grammar.json ${CMAKE_SOURCE_DIR}/include/shady/primops.json DEPENDS do_import_spv_defs ${GENERATOR_NAME} ${CMAKE_SOURCE_DIR}/include/shady/grammar.json ${CMAKE_SOURCE_DIR}/include/shady/primops.json VERBATIM)
    add_custom_target(${F_TARGET_NAME} DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${F_FILE_NAME})
endfunction()
