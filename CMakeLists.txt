cmake_minimum_required(VERSION 3.20)
project (shady C)

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SPIRV-Headers REQUIRED)

add_subdirectory(src)