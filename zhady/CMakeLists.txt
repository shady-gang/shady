find_package(SWIG)
find_package(JNI)

if (SWIG_FOUND AND JNI_FOUND)
    message("Enabling Java bindings")

    include(UseSWIG)
    swig_add_library(zhady TYPE SHARED LANGUAGE java SOURCES shady.i)
    set_property(TARGET zhady PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)
    #set_property(TARGET zhady PROPERTY SWIG_COMPILE_OPTIONS -includeall)
    target_link_libraries(zhady PUBLIC JNI::JNI api shady common)
endif ()