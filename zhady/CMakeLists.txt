find_package(SWIG)
find_package(JNI)
find_package(Java)

if (SWIG_FOUND AND JNI_FOUND AND Java_FOUND)
    message("Enabling Java bindings")

    add_custom_target(zhady_dir COMMAND cmake -E <command> ${CMAKE_CURRENT_BINARY_DIR}/java_sources/de/unisaarland/zhady)

    include(UseSWIG)
    swig_add_library(zhady_shared_lib TYPE SHARED LANGUAGE java SOURCES shady.i OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/java_sources/de/unisaarland/zhady)
    set_property(TARGET zhady_shared_lib PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)
    set_property(TARGET zhady_shared_lib PROPERTY SWIG_COMPILE_OPTIONS -package de.unisaarland.zhady)
    target_link_libraries(zhady_shared_lib PUBLIC JNI::JNI api driver runtime)

    include(UseJava)
    get_property(zhady_jar_sources TARGET zhady_shared_lib PROPERTY SWIG_SUPPORT_FILES)
    set(CMAKE_JAVA_COMPILE_FLAGS ${CMAKE_CURRENT_BINARY_DIR}/java_sources/de/unisaarland/zhady/*.java)
    add_jar(zhady_jar SOURCES ${zhady_jar_sources})

    install(TARGETS zhady_shared_lib EXPORT shady_export_set)
endif ()
