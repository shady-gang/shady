option(SHADY_ENABLE_SAMPLES "Demo applications and gpu programs" ON)

# TODO: this probably doesn't belong here
# find math lib; based on https://stackoverflow.com/a/74867749
find_library(MATH_LIBRARY m)
if (MATH_LIBRARY)
    add_library(m IMPORTED INTERFACE)
    target_link_libraries(m INTERFACE ${MATH_LIBRARY})
else()
    add_library(m INTERFACE)
endif()

if (SHADY_ENABLE_SAMPLES)
    add_subdirectory(checkerboard)
    add_subdirectory(aobench)

    if (BUILD_TESTING)
        shady_unit_test(COMPILER slim NAME fibonacci FILES fib.slim ARGS --entry-point fib_main VALSPV)
        shady_unit_test(COMPILER slim NAME hello_world FILES hello_world.slim ARGS --entry-point main VALSPV)
    endif ()
endif()
