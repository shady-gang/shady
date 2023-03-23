add_library(murmur3 STATIC ../murmur3/murmur3.c)
target_include_directories(murmur3 INTERFACE ../murmur3)

add_library(api INTERFACE)
target_include_directories(api INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>" "$<INSTALL_INTERFACE:include>")

add_subdirectory(common)
add_subdirectory(shady)
add_subdirectory(runtime)
add_subdirectory(slim)
add_subdirectory(clang-ast)
add_subdirectory(s2s)
