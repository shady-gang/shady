if (NOT TARGET shady_fe_llvm)
    message("LLVM front-end unavailable. Skipping Vcc.")
else()
    option(SHADY_ENABLE_VCC "Allows compiling C and C++ code with Shady." ON)
endif()

if (SHADY_ENABLE_VCC)
    add_library(vcc_lib STATIC vcc_lib.c)
    target_link_libraries(vcc_lib PUBLIC driver api)

    add_executable(vcc vcc.c)
    target_link_libraries(vcc PRIVATE api vcc_lib)
    install(TARGETS vcc_lib vcc EXPORT shady_export_set)

    if (WIN32)
        add_custom_command(TARGET vcc POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:vcc> $<TARGET_RUNTIME_DLLS:vcc>
                COMMAND_EXPAND_LISTS
        )
    endif ()
    #message("Vcc will be built together with shady")
endif ()