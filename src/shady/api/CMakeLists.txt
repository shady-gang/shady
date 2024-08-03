add_library(api INTERFACE)
target_include_directories(api INTERFACE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>" "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>" "$<INSTALL_INTERFACE:include>")
target_include_directories(api INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")

get_target_property(SPIRV_HEADERS_INCLUDE_DIRS SPIRV-Headers::SPIRV-Headers INTERFACE_INCLUDE_DIRECTORIES)

add_generated_file(FILE_NAME grammar_generated.h TARGET_NAME generate-grammar-headers SOURCES generator_grammar.c)
add_dependencies(api INTERFACE generate-grammar-headers)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grammar_generated.h DESTINATION include)

add_generated_file(FILE_NAME primops_generated.h TARGET_NAME generate-primops-headers SOURCES generator_primops.c)
add_dependencies(api INTERFACE generate-primops-headers)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/primops_generated.h DESTINATION include)

add_generated_file(FILE_NAME type_generated.h TARGET_NAME generate-type-headers SOURCES generator_type.c)
add_dependencies(api INTERFACE generate-type-headers)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/type_generated.h DESTINATION include)

install(TARGETS api EXPORT shady_export_set)
