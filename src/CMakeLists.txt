set(SHADY_SOURCES
        ../include/ir.h
    main.c
        arena.c implem.h type.c)

add_library(shady ${SHADY_SOURCES})
target_include_directories(shady PUBLIC ../include)

add_executable(shady-cli main.c)
target_link_libraries(shady-cli shady)