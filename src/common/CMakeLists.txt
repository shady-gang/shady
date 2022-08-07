add_library(common STATIC list.c dict.c log.c portability.c util.c)
target_include_directories(common INTERFACE .)