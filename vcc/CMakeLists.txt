if (NOT TARGET shady_fe_llvm)
    message("LLVM front-end unavailable. Skipping Vcc.")
else()
    option(SHADY_ENABLE_VCC "Allows compiling C and C++ code with Shady." ON)
endif()

if (SHADY_ENABLE_VCC)
    set (VCC_CLANG_EXECUTABLE_NAME "clang" CACHE STRING "What 'clang' executable Vcc should call into")

    add_library(vcc_lib STATIC vcc_lib.c)
    target_link_libraries(vcc_lib PUBLIC driver api)

    add_executable(vcc vcc.c)
    add_executable(shady::vcc ALIAS vcc)
    target_compile_definitions(vcc_lib PRIVATE "VCC_CLANG_EXECUTABLE_NAME=${VCC_CLANG_EXECUTABLE_NAME}")
    target_link_libraries(vcc PRIVATE api vcc_lib)
    install(TARGETS vcc_lib vcc EXPORT shady_export_set)

    if (WIN32)
        add_custom_command(TARGET vcc POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:vcc> $<TARGET_RUNTIME_DLLS:vcc>
                COMMAND_EXPAND_LISTS
        )
    endif ()
    #message("Vcc will be built together with shady")

    add_custom_target(copy-vcc-files ALL COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/share/vcc/include/)
    add_dependencies(shady copy-vcc-files)
    install(DIRECTORY ${CMAKE_BINARY_DIR}/share/vcc/ DESTINATION share/vcc)

    # add_library(vcc-guest-headers INTERFACE)
    # set_target_properties(vcc-guest-headers PROPERTIES PUBLIC_HEADER include/shady.h)
    # install(TARGETS vcc-guest-headers)

endif ()