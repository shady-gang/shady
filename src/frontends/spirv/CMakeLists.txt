add_library(shady_s2s STATIC s2s.c)
target_link_libraries(shady_s2s PRIVATE api common)
target_link_libraries(shady_s2s PRIVATE "$<BUILD_INTERFACE:SPIRV-Headers::SPIRV-Headers>")
