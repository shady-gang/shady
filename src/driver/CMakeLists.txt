add_library(driver driver.c cli.c)
target_link_libraries(driver PUBLIC "$<BUILD_INTERFACE:shady>" shady_spirv shady_c shady_emit_common)
set_property(TARGET driver PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(slim slim.c)
target_link_libraries(slim PRIVATE driver)
install(TARGETS slim EXPORT shady_export_set)

if (TARGET shady_s2s)
    target_compile_definitions(driver PUBLIC SPV_PARSER_PRESENT)
    target_link_libraries(driver PRIVATE shady_s2s)
endif()

if (TARGET shady_fe_llvm)
    target_link_libraries(driver PRIVATE shady_fe_llvm)
    target_compile_definitions(driver PUBLIC LLVM_PARSER_PRESENT)

    add_executable(vcc vcc.c)
    target_link_libraries(vcc PRIVATE driver api common)
    install(TARGETS vcc EXPORT shady_export_set)

    if (WIN32)
        add_custom_command(TARGET vcc POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:vcc> $<TARGET_RUNTIME_DLLS:vcc>
            COMMAND_EXPAND_LISTS
        )
    endif ()
endif ()
