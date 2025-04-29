shady_add_component(NAME shady_s2s SOURCES s2s.c)
target_link_libraries(shady_s2s PRIVATE api common)
target_link_libraries(shady_s2s PRIVATE "$<BUILD_INTERFACE:SPIRV-Headers::SPIRV-Headers>")

target_compile_definitions(driver PUBLIC SPV_PARSER_PRESENT)
target_link_libraries(driver PUBLIC shady_s2s)
