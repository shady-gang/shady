# :( see https://gitlab.kitware.com/cmake/cmake/-/issues/18399
# set_source_files_properties(${CMAKE_SOURCE_DIR}/include/shady/generated_grammar.h PROPERTIES GENERATED TRUE)
#set_property(SOURCE ${CMAKE_SOURCE_DIR}/include/shady/generated_grammar.h PROPERTY GENERATED TRUE)

add_library(api INTERFACE)
target_include_directories(api INTERFACE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>" "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>" "$<INSTALL_INTERFACE:include>")
# TODO: remove this once we are clean with private interfaces
target_include_directories(api INTERFACE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/shady/>")
target_link_libraries(api INTERFACE "$<BUILD_INTERFACE:SPIRV-Headers::SPIRV-Headers>")

get_target_property(SPIRV_HEADERS_INCLUDE_DIRS SPIRV-Headers::SPIRV-Headers INTERFACE_INCLUDE_DIRECTORIES)

add_generated_file(FILE_NAME grammar_generated.h TARGET_NAME generate-grammar-headers SOURCES generator_grammar.c)
add_dependencies(api INTERFACE generate-grammar-headers)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grammar_generated.h DESTINATION include)

add_generated_file(FILE_NAME enum_generated.h TARGET_NAME generate-enum-headers SOURCES generator_enum.c)
add_dependencies(api INTERFACE generate-enum-headers)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/enum_generated.h DESTINATION include)

add_generated_file(FILE_NAME type_generated.h TARGET_NAME generate-type-headers SOURCES generator_type.c)
add_dependencies(api INTERFACE generate-type-headers)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/type_generated.h DESTINATION include)

install(TARGETS api EXPORT shady_export_set)

find_package(Python COMPONENTS Interpreter REQUIRED)
function(generate_extinst_headers NAME SRC)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.h COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/SPIRV-Headers/tools/buildHeaders/bin/generate_language_headers.py --extinst-name=${NAME} --extinst-grammar=${CMAKE_CURRENT_SOURCE_DIR}/${SRC} --extinst-output-base=${NAME} DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SRC} VERBATIM)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${NAME}.h PROPERTIES GENERATED TRUE)
    add_custom_target("${NAME}_h" DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.h)

    add_library(${NAME} INTERFACE)
    add_dependencies(${NAME} "${NAME}_h")
    target_include_directories(${NAME} INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

get_target_property(SPIRV_HEADERS_INCLUDE_DIRS SPIRV-Headers::SPIRV-Headers INTERFACE_INCLUDE_DIRECTORIES)
