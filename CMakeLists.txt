cmake_minimum_required(VERSION 3.20)
project (shady C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(SPIRV-Headers REQUIRED)

add_subdirectory(src)