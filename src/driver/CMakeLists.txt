shady_add_component(NAME driver SOURCES driver.c cli.c targets.c)
target_link_libraries(driver PUBLIC "api" common)
target_link_libraries(driver PUBLIC shady)
target_link_libraries(driver PRIVATE shader_pipeline)
set_target_properties(driver PROPERTIES OUTPUT_NAME "shady_driver")

add_executable(slim slim.c)
target_link_libraries(slim PRIVATE driver)
install(TARGETS slim EXPORT shady_export_set)

copy_windows_dlls(slim)