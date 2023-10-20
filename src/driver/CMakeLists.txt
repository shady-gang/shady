add_library(driver STATIC driver.c cli.c)
target_link_libraries(driver PRIVATE shady common)
set_property(TARGET driver PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(slim slim.c)
target_link_libraries(slim PRIVATE driver api common)

if (TARGET shady_s2s)
    target_compile_definitions(driver PUBLIC SPV_PARSER_PRESENT)
    target_link_libraries(driver PRIVATE shady_s2s)
endif()

if (TARGET shady_fe_llvm)
    target_link_libraries(driver PRIVATE shady_fe_llvm)
    target_compile_definitions(driver PUBLIC LLVM_PARSER_PRESENT)
endif ()