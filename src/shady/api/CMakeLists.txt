add_library(api INTERFACE)
target_include_directories(api INTERFACE "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>" "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>" "$<INSTALL_INTERFACE:include>")

get_target_property(SPIRV_HEADERS_INCLUDE_DIRS SPIRV-Headers::SPIRV-Headers INTERFACE_INCLUDE_DIRECTORIES)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/grammar_generated.h COMMAND generator grammar-headers ${CMAKE_CURRENT_BINARY_DIR}/grammar_generated.h ${CMAKE_SOURCE_DIR}/include/shady/grammar.json ${CMAKE_SOURCE_DIR}/include/shady/primops.json ${SPIRV_HEADERS_INCLUDE_DIRS} DEPENDS generator ${CMAKE_SOURCE_DIR}/include/shady/grammar.json ${CMAKE_SOURCE_DIR}/include/shady/primops.json VERBATIM)
add_custom_target(generate_grammar DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/grammar_generated.h)
add_dependencies(api INTERFACE generate_grammar)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grammar_generated.h DESTINATION include)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/primops_generated.h COMMAND generator primops-headers ${CMAKE_CURRENT_BINARY_DIR}/primops_generated.h ${CMAKE_SOURCE_DIR}/include/shady/grammar.json ${CMAKE_SOURCE_DIR}/include/shady/primops.json ${SPIRV_HEADERS_INCLUDE_DIRS} DEPENDS generator ${CMAKE_SOURCE_DIR}/include/shady/grammar.json ${CMAKE_SOURCE_DIR}/include/shady/primops.json VERBATIM)
add_custom_target(generate_primops DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/primops_generated.h)
add_dependencies(api INTERFACE generate_primops)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/primops_generated.h DESTINATION include)