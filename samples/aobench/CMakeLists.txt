if (TARGET vcc)
    add_executable(aobench_host ao_host.c ao_main.c)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ao.comp.c.ll COMMAND vcc ARGS ${CMAKE_CURRENT_SOURCE_DIR}/ao.comp.cpp --only-run-clang -O3 -o ${CMAKE_CURRENT_BINARY_DIR}/ao.comp.c.ll COMMENT ao.comp.c.ll DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ao.comp.cpp ${CMAKE_CURRENT_SOURCE_DIR}/ao.c)

    set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/ao_main.c APPEND PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ao.comp.c.ll)
    add_custom_command(TARGET aobench_host POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/ao.comp.c.ll
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ao.comp.c.ll)

    find_program(ISPC_EXE "ispc")
    if (ISPC_EXE)
        target_compile_definitions(aobench_host PUBLIC ENABLE_ISPC=1)
        add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/aobench.ispc COMMAND slim ARGS ${CMAKE_CURRENT_BINARY_DIR}/ao.comp.c.ll -o ${CMAKE_CURRENT_BINARY_DIR}/aobench.ispc --entry-point aobench_kernel --restructure-everything COMMENT generating aobench.ispc DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ao.comp.c.ll)
        add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/aobench.ispc.o COMMAND ispc ARGS ${CMAKE_CURRENT_BINARY_DIR}/aobench.ispc -o ${CMAKE_CURRENT_BINARY_DIR}/aobench.ispc.o --pic -g -O2 -woff COMMENT generating aobench.ispc.o DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/aobench.ispc)
        add_library(aobench_ispc OBJECT ${CMAKE_CURRENT_BINARY_DIR}/aobench.ispc.o)
        set_target_properties(aobench_ispc PROPERTIES LINKER_LANGUAGE C)
        target_link_libraries(aobench_host PRIVATE aobench_ispc ${CMAKE_CURRENT_BINARY_DIR}/aobench.ispc.o)
    endif()

    target_link_libraries(aobench_host PRIVATE m shady runtime common)
else()
    message("Vcc not enabled. Skipping aobench.")
endif()
