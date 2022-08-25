cmake_minimum_required(VERSION 3.18)
project (shady C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(SPIRV-Headers REQUIRED)

add_subdirectory(src)

include(CMakePackageConfigHelpers)

install(TARGETS api EXPORT shady_export_set)
install(TARGETS shady EXPORT shady_export_set ARCHIVE DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/lib)
install(TARGETS runtime EXPORT shady_export_set ARCHIVE DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/shady DESTINATION include)
install(EXPORT shady_export_set DESTINATION share/cmake/shady/ NAMESPACE shady:: FILE shady-targets.cmake)

configure_file(cmake/shady-config.cmake.in shady-config.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/shady-config.cmake" DESTINATION share/cmake/shady)
#install(FILES "${CMAKE_CURRENT_BINARY_DIR}/shady-config.cmake" DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/shady)
