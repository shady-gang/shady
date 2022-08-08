set(SLIM_SOURCES
    main.c)

add_executable(slim ${SLIM_SOURCES})
target_link_libraries(slim shady common)
