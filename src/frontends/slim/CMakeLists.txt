add_library(slim_parser STATIC parser.c token.c)
target_link_libraries(slim_parser PUBLIC common api)
if (shady STREQUAL STATIC_LIBRARY OR MINGW)
    target_link_libraries(slim_parser PRIVATE shady)
endif ()
target_include_directories(slim_parser PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_include_directories(slim_parser INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../>")
target_link_libraries(shady PUBLIC "$<BUILD_INTERFACE:slim_parser>")
