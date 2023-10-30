find_package(json-c REQUIRED)

add_executable(generator generator.c)
target_link_libraries(generator PRIVATE common json-c::json-c)

get_target_property(SPIRV_HEADERS_INCLUDE_DIRS SPIRV-Headers::SPIRV-Headers INTERFACE_INCLUDE_DIRECTORIES)
message("SPIR-V Include dirs are: ${SPIRV_HEADERS_INCLUDE_DIRS}")

add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/include/shady/generated_grammar.h COMMENT "generating shady headers" COMMAND generator grammar-headers ${CMAKE_SOURCE_DIR}/include/shady/generated_grammar.h ${CMAKE_SOURCE_DIR}/include/shady/grammar.json ${SPIRV_HEADERS_INCLUDE_DIRS} DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SRC} VERBATIM)
add_custom_target(generate_grammar DEPENDS ${CMAKE_SOURCE_DIR}/include/shady/generated_grammar.h)
# set_source_files_properties(${CMAKE_SOURCE_DIR}/include/shady/generated_grammar.h PROPERTIES GENERATED TRUE)