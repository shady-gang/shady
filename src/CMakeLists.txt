set(SHADY_SOURCES
    ../include/ir.h
    main.cpp)

add_library(shady ${SHADY_SOURCES})
target_include_directories(shady PUBLIC)

add_executable(shady-cli main.cpp)
target_link_libraries(shady-cli shady)