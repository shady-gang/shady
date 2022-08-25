add_library(murmur3 OBJECT ../murmur3/murmur3.c)
target_include_directories(murmur3 INTERFACE ../murmur3)

add_library(api INTERFACE)
#target_sources(api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include/shady/runtime.h)
target_include_directories(api INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>" "$<INSTALL_INTERFACE:include>")
#target_sources(api INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>" "$<INSTALL_INTERFACE:${INSTALL_PREFIX}/include>")
#target_sources(api INTERFACE FILE_SET HEADERS BASE_DIR include)

#target_sources(api PUBLIC
#        FILE_SET HEADERS
#        BASE_DIRS ../include
#        FILES ../include/shady/ir.h
#        )

add_subdirectory(common)
add_subdirectory(shady)
add_subdirectory(runtime)
add_subdirectory(slim)
