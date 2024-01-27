cmake_minimum_required(VERSION 3.13)
project (shady C)

include(ExternalProject)
include(FetchContent)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if (MSVC)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()


FetchContent_Declare(
  SPIRV-Headers 
  GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers
  GIT_TAG main
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(SPIRV-Headers)

FetchContent_Declare(
  json-c
  GIT_REPOSITORY https://github.com/json-c/json-c
  GIT_TAG master
  OVERRIDE_FIND_PACKAGE
)

FetchContent_MakeAvailable(json-c)
add_library(json-c::json-c ALIAS json-c)
configure_file(${json-c_SOURCE_DIR}/json.h.cmakein ${json-c_BINARY_DIR}/json-c/json.h @ONLY)
target_include_directories(json-c PUBLIC $<BUILD_INTERFACE:${json-c_BINARY_DIR}>)

find_package(SPIRV-Headers REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# required for MSVC
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 1)

cmake_policy(SET CMP0118 NEW)
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

if (TARGET vcc)
  add_subdirectory(vcc-std)
endif ()

if (TARGET runtime)
  install(TARGETS runtime EXPORT shady_export_set ARCHIVE DESTINATION ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/shady DESTINATION include)
install(EXPORT shady_export_set DESTINATION share/cmake/shady/ NAMESPACE shady:: FILE shady-targets.cmake)

configure_file(cmake/shady-config.cmake.in shady-config.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/shady-config.cmake" DESTINATION share/cmake/shady)
#install(FILES "${CMAKE_CURRENT_BINARY_DIR}/shady-config.cmake" DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/shady)
