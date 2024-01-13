
if (NOT ${EXTERNAL_MURMUR3})
	add_library(murmur3 STATIC ../murmur3/murmur3.c)
	target_include_directories(murmur3 INTERFACE ../murmur3)
	set_target_properties(murmur3 PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif ()

add_subdirectory(common)
add_subdirectory(shady)

if (${BUILD_RUNTIME})
	add_subdirectory(runtime)
endif ()

add_subdirectory(frontends)
add_subdirectory(driver)
