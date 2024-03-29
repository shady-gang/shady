find_program(CLANG_EXE "clang")
find_program(LLVM-SPIRV_EXE "llvm-spirv")

if (CLANG_EXE AND LLVM-SPIRV_EXE)
    add_executable(aobench_host ao_host.c ao_main.c)

    add_custom_command(OUTPUT ao.cl.ll COMMAND clang ARGS ${CMAKE_CURRENT_SOURCE_DIR}/ao.cl -std=clc++2021 -emit-llvm -o ${CMAKE_CURRENT_BINARY_DIR}/ao.cl.ll -c -target spir64-unknown-unknown -O2 DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ao.cl ${CMAKE_CURRENT_SOURCE_DIR}/ao.c)
    add_custom_command(OUTPUT ao.cl.spv COMMAND llvm-spirv ARGS ${CMAKE_CURRENT_BINARY_DIR}/ao.cl.ll -o ${CMAKE_CURRENT_BINARY_DIR}/ao.cl.spv DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ao.cl.ll)

    set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/ao_main.c APPEND PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ao.cl.spv)
    add_custom_command(TARGET aobench_host POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/ao.cl.spv
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ao.cl.spv)

    find_program(ISPC_EXE "ispc")
    if (ISPC_EXE)
        target_compile_definitions(aobench_host PUBLIC ENABLE_ISPC=1)
        add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/aobench.ispc COMMAND slim ARGS ao.cl.spv --output ${CMAKE_CURRENT_BINARY_DIR}/aobench.ispc --entry-point aobench_kernel COMMENT generating aobench.ispc DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ao.cl.spv)
        add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/aobench.ispc.o COMMAND ispc ARGS ${CMAKE_CURRENT_BINARY_DIR}/aobench.ispc -o ${CMAKE_CURRENT_BINARY_DIR}/aobench.ispc.o --pic -g -O2 -woff COMMENT generating aobench.ispc.o DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/aobench.ispc)
        add_library(aobench_ispc OBJECT ${CMAKE_CURRENT_BINARY_DIR}/aobench.ispc.o)
        set_target_properties(aobench_ispc PROPERTIES LINKER_LANGUAGE C)
        target_link_libraries(aobench_host PRIVATE aobench_ispc ${CMAKE_CURRENT_BINARY_DIR}/aobench.ispc.o)
    endif()

    target_link_libraries(aobench_host PRIVATE m shady runtime common)
else()
    message("Clang and/or llvm-spirv not found. Skipping aobench.")
endif()
