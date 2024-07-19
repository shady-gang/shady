add_custom_target(copy-vcc-files ALL COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/../share/vcc/include/)
add_dependencies(shady copy-vcc-files)

if (NOT MSVC)
    enable_language(CXX)
    add_executable(test_vcc_vec src/test_vec.cpp)
    target_include_directories(test_vcc_vec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    set_property(TARGET test_vcc_vec PROPERTY CXX_STANDARD 20)
    add_test(NAME test_vcc_vec COMMAND test_vcc_vec)
endif ()

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../share/vcc/ DESTINATION share/vcc)
