set(SLIM_SOURCES
    main.c
    token.c
    parser.c token.h)

add_executable(slim ${SLIM_SOURCES})
target_link_libraries(slim shady)