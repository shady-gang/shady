add_custom_target(copy-vcc-files ALL COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/../share/vcc/include/)
add_dependencies(shady copy-vcc-files)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../share/vcc/ DESTINATION share/vcc/include)
