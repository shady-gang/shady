add_library(shady_s2s STATIC s2s.c)
target_link_libraries(shady_s2s PRIVATE api common)
target_link_libraries(shady_s2s PRIVATE "$<BUILD_INTERFACE:SPIRV-Headers::SPIRV-Headers>")
target_compile_definitions(shady PRIVATE SPV_PARSER_PRESENT)
set_property(TARGET shady_s2s PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(shady PRIVATE shady_s2s)
