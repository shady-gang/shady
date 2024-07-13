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

    if (TARGET vcc_lib)
        swig_add_library(vcc_swig_c TYPE OBJECT LANGUAGE java SOURCES vcc.i OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/java_sources/de/unisaarland/zhady)
        set_property(TARGET vcc_swig_c PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)
        set_property(TARGET vcc_swig_c PROPERTY SWIG_COMPILE_OPTIONS -package de.unisaarland.zhady)
        target_link_libraries(vcc_swig_c PUBLIC JNI::JNI api vcc_lib)
        target_link_libraries(zhady_shared_lib PRIVATE vcc_swig_c vcc_lib)
    endif ()

    include(UseJava)
    get_property(zhady_jar_sources TARGET zhady_shared_lib PROPERTY SWIG_SUPPORT_FILES)
    get_property(zhady_jar_sources_dir TARGET zhady_shared_lib PROPERTY SWIG_SUPPORT_FILES_DIRECTORY)
    #set(CMAKE_JAVA_COMPILE_FLAGS ${CMAKE_CURRENT_BINARY_DIR}/java_sources/de/unisaarland/zhady/*.java)
    #set(CMAKE_JAVA_COMPILE_FLAGS "-sourcepath" "${CMAKE_CURRENT_BINARY_DIR}/java_sources/")

    if (TARGET vcc_lib)
        get_property(vcc_java_sources TARGET vcc_swig_c PROPERTY SWIG_SUPPORT_FILES)
        list(APPEND zhady_jar_sources ${vcc_java_sources})
    endif ()

    # message("Zhady sources: ${zhady_jar_sources}")
    # message("Zhady sources dir: ${zhady_jar_sources_dir}")
    set(CMAKE_JAVA_INCLUDE_PATH ${CMAKE_CURRENT_BINARY_DIR}/java_sources/)
    add_jar(zhady_jar SOURCES ${zhady_jar_sources})

    install(TARGETS zhady_shared_lib EXPORT shady_export_set)
endif ()
