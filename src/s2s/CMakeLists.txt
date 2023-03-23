add_library(shady_s2s STATIC s2s.c)
target_link_libraries(shady_s2s api)
target_compile_definitions(shady PRIVATE SPV_PARSER_PRESENT)
target_link_libraries(shady PRIVATE shady_s2s)
