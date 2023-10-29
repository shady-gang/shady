find_package(json-c REQUIRED)

add_executable(generator generator.c)
target_link_libraries(generator PRIVATE common json-c::json-c)