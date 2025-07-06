cmake_minimum_required(VERSION 3.13)
project (shady C)

include(ExternalProject)
include(FetchContent)

find_package(Git)
find_package(Python COMPONENTS Interpreter REQUIRED)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
option(SHADY_USE_FETCHCONTENT "Use FetchContent to grab json-c" ON)
option(SHADY_WIN32_FIX_PARTIAL_LLVM_INSTALL "If you install LLVM on windows, it doesn't come with header files. This fixes it" ON)

if (MSVC)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (WIN32)
  set(BUILD_SHARED_LIBS OFF)
endif()

add_subdirectory(SPIRV-Headers)
set(SPIRV-Headers_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/SPIRV-Headers/")
set(SKIP_SPIRV_TOOLS_INSTALL TRUE)
set(SPIRV_SKIP_TESTS TRUE)
add_subdirectory(SPIRV-Tools)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24" AND ${SHADY_USE_FETCHCONTENT})
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
else ()
    message("CMake 3.24 or later is required to use FetchContent")
endif ()

find_package(LLVM QUIET)
if (LLVM_FOUND)
    message("LLVM ${LLVM_VERSION} found")
endif()
if(NOT ${LLVM_FOUND} AND WIN32 AND ${SHADY_WIN32_FIX_PARTIAL_LLVM_INSTALL})
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
        
        execute_process(
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND ${GIT_EXECUTABLE} clone -n --depth 1 --filter=tree:0 https://github.com/llvm/llvm-project/ --branch "llvmorg-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}"
        )
        execute_process(
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/llvm-project
            COMMAND ${GIT_EXECUTABLE} sparse-checkout set --no-cone llvm/include/llvm-c
        )
        execute_process(
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/llvm-project
            COMMAND ${GIT_EXECUTABLE} checkout
        )
        target_include_directories(LLVM-C INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/llvm-project/llvm/include)
        target_compile_definitions(LLVM-C INTERFACE -DLLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR} -DLLVM_VERSION_MINOR=${LLVM_VERSION_MINOR} -DLLVM_VERSION_PATCH=${LLVM_VERSION_PATCH})
        message(STATUS "Win32: Installed LLVM ${LLVM_VERSION} found at ${LLVM_DIR}")
        set(LLVM_FOUND TRUE)
    endif()
endif()

include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# required for MSVC
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 1)

cmake_policy(SET CMP0118 NEW)
add_subdirectory(src)
add_subdirectory(vcc)

add_subdirectory(zhady)

if (PROJECT_IS_TOP_LEVEL)
    include(CTest)
    if (BUILD_TESTING)
      add_subdirectory(test)
    endif()
    add_subdirectory(samples)
endif()

add_subdirectory(nasl)

include(CMakePackageConfigHelpers)
install(EXPORT shady_export_set DESTINATION share/cmake/shady/ NAMESPACE shady:: FILE shady-targets.cmake)

configure_file(cmake/shady-config.cmake.in shady-config.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/shady-config.cmake" DESTINATION share/cmake/shady)
