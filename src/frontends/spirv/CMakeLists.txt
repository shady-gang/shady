add_library(shady_s2s STATIC s2s.c)
target_link_libraries(shady_s2s PRIVATE api common shady)
target_link_libraries(shady_s2s PRIVATE "$<BUILD_INTERFACE:SPIRV-Headers::SPIRV-Headers>")
set_property(TARGET shady_s2s PROPERTY POSITION_INDEPENDENT_CODE ON)
