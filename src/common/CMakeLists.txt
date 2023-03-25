add_library(common STATIC list.c dict.c log.c portability.c util.c growy.c arena.c printer.c)
target_include_directories(common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(common PRIVATE "$<BUILD_INTERFACE:murmur3>")
set_property(TARGET common PROPERTY POSITION_INDEPENDENT_CODE ON)
