shady_add_component(NAME frontend_spirv SOURCES s2s.c)
target_link_libraries(frontend_spirv PRIVATE api common)
target_link_libraries(frontend_spirv PRIVATE "$<BUILD_INTERFACE:SPIRV-Headers::SPIRV-Headers>")

target_compile_definitions(driver PUBLIC SPV_PARSER_PRESENT)
target_link_libraries(driver PUBLIC frontend_spirv)
