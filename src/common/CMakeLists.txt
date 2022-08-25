add_library(common OBJECT list.c dict.c log.c portability.c util.c)
target_include_directories(common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
