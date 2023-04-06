add_executable(checkerboard checkerboard.c)
target_link_libraries(checkerboard PRIVATE runtime common)
target_include_directories(checkerboard PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
message(${CMAKE_CURRENT_BINARY_DIR})

# find math lib; based on https://stackoverflow.com/a/74867749
find_library(MATH_LIBRARY m)
if (MATH_LIBRARY)
    target_link_libraries(checkerboard PRIVATE ${MATH_LIBRARY})
endif()

embed_file(string checkerboard_kernel_src checkerboard_kernel.slim)
