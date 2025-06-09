
add_subdirectory(common)
add_subdirectory(shady)

if (${BUILD_RUNTIME})
    add_subdirectory(runtime)
endif ()

add_subdirectory(driver)
add_subdirectory(frontend)
add_subdirectory(backend)
