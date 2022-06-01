set(SHADY_SIMULATOR_SOURCES
    simulator.c)

add_executable(simulator ${SHADY_SIMULATOR_SOURCES})
target_link_libraries(simulator shady)