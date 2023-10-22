add_library(driver STATIC driver.c cli.c)
target_link_libraries(driver PUBLIC "$<BUILD_INTERFACE:shady>")
set_property(TARGET driver PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(slim slim.c)
target_link_libraries(slim PRIVATE driver)

if (TARGET shady_s2s)
    target_compile_definitions(driver PUBLIC SPV_PARSER_PRESENT)
    target_link_libraries(driver PRIVATE shady_s2s)
endif()

if (TARGET shady_fe_llvm)
    target_link_libraries(driver PRIVATE shady_fe_llvm)
    target_compile_definitions(driver PUBLIC LLVM_PARSER_PRESENT)

    add_executable(vcc vcc.c)
    target_link_libraries(vcc PRIVATE driver api common)
endif ()