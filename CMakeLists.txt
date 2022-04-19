cmake_minimum_required(VERSION 3.18)
project (shady C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(SPIRV-Headers REQUIRED)

add_subdirectory(src)

# todo: statically link those guys ?
export(TARGETS shady containers murmur3 FILE shady-targets.cmake)
configure_file(cmake/shady-config.cmake.in shady-config.cmake)
