cmake_minimum_required(VERSION 3.24)
project (shady C)

include(ExternalProject)
include(FetchContent)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if (MSVC)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

add_subdirectory(SPIRV-Headers)

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

find_package(LLVM QUIET)
if(NOT ${LLVM_FOUND} AND WIN32)
    find_program(clang_exe "clang.exe")
    if(${clang_exe} STREQUAL "clang_exe-NOTFOUND")
        message(STATUS "Win32: Installed LLVM not found")
    else()
        execute_process(COMMAND ${clang_exe} --version OUTPUT_VARIABLE clang_status)
        string(REGEX MATCH "InstalledDir: (.*)[\r\n]" match ${clang_status})
        file(TO_CMAKE_PATH "${CMAKE_MATCH_1}/../" LLVM_DIR)
        cmake_path(ABSOLUTE_PATH LLVM_DIR NORMALIZE)
        string(REGEX MATCH "clang version ([0-9]+).([0-9]+).([0-9]+)" match2 ${clang_status})
        set(LLVM_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(LLVM_VERSION_MINOR ${CMAKE_MATCH_2})
        set(LLVM_VERSION_PATCH ${CMAKE_MATCH_3})
        set(LLVM_VERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
        add_library(LLVM-C SHARED IMPORTED)
        set_property(TARGET LLVM-C PROPERTY
                     IMPORTED_LOCATION "${LLVM_DIR}bin/LLVM-C.dll")
        set_property(TARGET LLVM-C PROPERTY
                     IMPORTED_IMPLIB "${LLVM_DIR}lib/LLVM-C.lib")
        target_include_directories(LLVM-C INTERFACE "vendor")
        target_compile_definitions(LLVM-C INTERFACE -DLLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR} -DLLVM_VERSION_MINOR=${LLVM_VERSION_MINOR} -DLLVM_VERSION_PATCH=${LLVM_VERSION_PATCH})
        message(STATUS "Win32: Installed LLVM ${LLVM_VERSION} found at ${LLVM_DIR}")
        set(LLVM_FOUND TRUE)
    endif()
else()
    message("LLVM ${LLVM_VERSION} found")
endif()

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
