add_library(driver driver.c cli.c)
target_link_libraries(driver PUBLIC "api")
target_link_libraries(driver PUBLIC "shady")
set_target_properties(driver PROPERTIES OUTPUT_NAME "shady_driver")
install(TARGETS driver EXPORT shady_export_set)

add_executable(slim slim.c)
target_link_libraries(slim PRIVATE driver)
install(TARGETS slim EXPORT shady_export_set)

