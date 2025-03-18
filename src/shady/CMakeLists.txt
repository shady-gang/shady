add_library(shady STATIC)

add_generated_file(FILE_NAME type_generated.c         TARGET_NAME type_generated         SOURCES generator_type.c)
add_generated_file(FILE_NAME node_generated.c         TARGET_NAME node_generated         SOURCES generator_node.c)
add_generated_file(FILE_NAME primops_generated.c      TARGET_NAME primops_generated      SOURCES generator_primops.c)
add_generated_file(FILE_NAME constructors_generated.c TARGET_NAME constructors_generated SOURCES generator_constructors.c)
add_generated_file(FILE_NAME visit_generated.c        TARGET_NAME visit_generated        SOURCES generator_visit.c)
add_generated_file(FILE_NAME rewrite_generated.c      TARGET_NAME rewrite_generated      SOURCES generator_rewrite.c)
add_generated_file(FILE_NAME print_generated.c        TARGET_NAME print_generated        SOURCES generator_print.c)

add_dependencies(shady node_generated primops_generated type_generated constructors_generated visit_generated rewrite_generated print_generated)
target_include_directories(shady PRIVATE "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")

target_sources(shady PRIVATE
    ir.c
    node.c
    check.c
    primops.c
    rewrite.c
    visit.c
    print.c
    fold.c
    body_builder.c
    compile.c
    config.c
    dict.c
)

add_subdirectory(analysis)
add_subdirectory(passes)
add_subdirectory(ir)
add_subdirectory(pipeline)

#target_include_directories(shady PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

if (WIN32)
    if (MSVC)
        target_link_options(shady PUBLIC /STACK:33554432)
    else()
        target_link_options(shady PUBLIC -Wl,--stack,33554432)
    endif()
endif()

add_subdirectory(internal)

target_link_libraries(shady PRIVATE "api")
target_link_libraries(shady PRIVATE "common")
target_link_libraries(shady PRIVATE "$<BUILD_INTERFACE:m>")

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/shady DESTINATION include)
#install(TARGETS shady EXPORT shady_export_set ARCHIVE DESTINATION ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
