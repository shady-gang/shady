add_library(driver driver.c cli.c)
target_link_libraries(driver PUBLIC "api")
target_link_libraries(driver PUBLIC "shady")
set_target_properties(driver PROPERTIES OUTPUT_NAME "shady_driver")
install(TARGETS driver EXPORT shady_export_set)

if (TARGET shady_s2s)
    target_compile_definitions(driver PUBLIC SPV_PARSER_PRESENT)
    target_link_libraries(driver PUBLIC "$<BUILD_INTERFACE:shady_s2s>")
endif()

if (TARGET shady_fe_llvm)
    target_compile_definitions(driver PUBLIC LLVM_PARSER_PRESENT)
    target_link_libraries(driver PUBLIC "$<BUILD_INTERFACE:shady_fe_llvm>")
endif ()

add_executable(slim slim.c)
target_link_libraries(slim PRIVATE driver)
install(TARGETS slim EXPORT shady_export_set)

if (TARGET shady_fe_llvm)
    add_executable(vcc vcc.c)
    target_link_libraries(vcc PRIVATE driver api)
    install(TARGETS vcc EXPORT shady_export_set)

    if (WIN32)
        add_custom_command(TARGET vcc POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:vcc> $<TARGET_RUNTIME_DLLS:vcc>
                COMMAND_EXPAND_LISTS
        )
    endif ()
endif ()
