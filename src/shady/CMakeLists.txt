shady_add_component(NAME shady SOURCES)

add_generated_file_python(FILE_NAME type_generated.c         TARGET_NAME type_generated         GENERATOR src.shady.generator_type)
add_generated_file_python(FILE_NAME node_generated.c         TARGET_NAME node_generated         GENERATOR src.shady.generator_node)
add_generated_file_python(FILE_NAME primops_generated.c      TARGET_NAME primops_generated      GENERATOR src.shady.generator_primops)
add_generated_file_python(FILE_NAME constructors_generated.c TARGET_NAME constructors_generated GENERATOR src.shady.generator_constructors)
add_generated_file_python(FILE_NAME visit_generated.c        TARGET_NAME visit_generated        GENERATOR src.shady.generator_visit)
add_generated_file_python(FILE_NAME rewrite_generated.c      TARGET_NAME rewrite_generated      GENERATOR src.shady.generator_rewrite)
add_generated_file_python(FILE_NAME print_generated.c        TARGET_NAME print_generated        GENERATOR src.shady.generator_print)

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
    pipeline.c
    cli.c
)

add_subdirectory(analysis)
add_subdirectory(passes)
add_subdirectory(ir)

#target_include_directories(shady PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

if (WIN32)
    if (MSVC)
        target_link_options(shady PUBLIC /STACK:33554432)
    else()
        target_link_options(shady PUBLIC -Wl,--stack,33554432)
    endif()
endif()

target_link_libraries(shady PRIVATE "api")
target_link_libraries(shady PRIVATE "common")
target_link_libraries(shady PRIVATE "$<BUILD_INTERFACE:m>")

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/shady DESTINATION include)
