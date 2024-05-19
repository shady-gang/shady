find_package(SWIG)
find_package(JNI)
find_package(Java)

if (SWIG_FOUND AND JNI_FOUND AND Java_FOUND)
    message("Enabling Java bindings")

    include(UseSWIG)
    swig_add_library(zhady_shared_lib TYPE SHARED LANGUAGE java SOURCES shady.i)
    set_property(TARGET zhady_shared_lib PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)
    #set_property(TARGET zhady_shared_lib PROPERTY SWIG_COMPILE_OPTIONS -includeall)
    target_link_libraries(zhady_shared_lib PUBLIC JNI::JNI api shady common)

    include(UseJava)
    get_property(zhady_jar_sources TARGET zhady_shared_lib PROPERTY SWIG_SUPPORT_FILES)
    add_jar(zhady_jar SOURCES ${zhady_jar_sources})
endif ()