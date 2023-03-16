find_package(json-c CONFIG)

if (json-c_FOUND)
    add_executable(shady_clang main.c)
    target_link_libraries(shady_clang PRIVATE json-c::json-c)
    target_link_libraries(shady_clang PRIVATE shady common)
endif()
