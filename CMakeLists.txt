cmake_minimum_required(VERSION 3.13)
project (shady C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if (MSVC)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

find_package(SPIRV-Headers REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# required for MSVC
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 1)

add_subdirectory(src)

include(CTest)
if (BUILD_TESTING)
  add_subdirectory(test)
endif()

set(BUILD_SAMPLES ON CACHE BOOL "Whether to build built-in demo applications")
if (BUILD_SAMPLES)
  add_subdirectory(samples)
endif()

include(CMakePackageConfigHelpers)

install(TARGETS api EXPORT shady_export_set)
install(TARGETS shady EXPORT shady_export_set ARCHIVE DESTINATION ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

if (TARGET runtime)
  install(TARGETS runtime EXPORT shady_export_set ARCHIVE DESTINATION ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/shady DESTINATION include)
install(EXPORT shady_export_set DESTINATION share/cmake/shady/ NAMESPACE shady:: FILE shady-targets.cmake)

configure_file(cmake/shady-config.cmake.in shady-config.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/shady-config.cmake" DESTINATION share/cmake/shady)
#install(FILES "${CMAKE_CURRENT_BINARY_DIR}/shady-config.cmake" DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/shady)
