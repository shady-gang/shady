if (BUILD_TESTING)
    add_executable(test_math test_math.c)
    target_link_libraries(test_math driver)
    add_test(NAME test_math COMMAND test_math)

    add_executable(test_builder test_builder.c)
    target_link_libraries(test_builder driver)
    add_test(NAME test_builder COMMAND test_builder)

    set(SHADY_SPV_VAL_VK_ARGS "--target-env;vulkan1.3;--allow-linkage")

    function(shady_unit_test)
        cmake_parse_arguments(PARSE_ARGV 0 "ARG" "VALSPV" "NAME;COMPILER" "FILES;ARGS;SPV_VAL_ARGS" )
        list(TRANSFORM ARG_FILES PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)
        add_test(NAME ${ARG_NAME} COMMAND ${CMAKE_COMMAND} -DVALSPV=${ARG_VALSPV} "-DSPV_VAL_ARGS=${ARG_SPV_VAL_ARGS}" -DSPIRV_VALIDATOR=$<TARGET_FILE:spirv-val> -DCOMPILER=$<TARGET_FILE:${ARG_COMPILER}> -DNAME=${ARG_NAME} "-DFILES=${ARG_FILES}" "-DCOMP_ARGS=${ARG_ARGS}" -DSRC=${PROJECT_SOURCE_DIR} -DDST=${PROJECT_BINARY_DIR} -P ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/unit_test.cmake)
    endfunction()

    add_subdirectory(slim)
    add_subdirectory(shader-pipeline)
    add_subdirectory(reconvergence_heuristics)
    add_subdirectory(opt)

    if (TARGET vcc)
        add_subdirectory(vcc)
    endif()
endif()
