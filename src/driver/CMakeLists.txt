add_library(driver STATIC driver.c cli.c)
target_link_libraries(driver PRIVATE shady common)
set_property(TARGET driver PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(slim slim.c)
target_link_libraries(slim PRIVATE driver api common)
