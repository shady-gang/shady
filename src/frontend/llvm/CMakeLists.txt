if (NOT LLVM_FOUND)
    message("LLVM not found. Skipping LLVM front-end.")
else ()
    option (SHADY_ENABLE_LLVM_FRONTEND "Uses LLVM-C to parse and then convert LLVM IR into Shady IR" ON)
endif ()

if (LLVM_FOUND AND SHADY_ENABLE_LLVM_FRONTEND)
    add_generated_file_python(FILE_NAME l2s_generated.c GENERATOR src.frontend.llvm.generator_l2s)

    shady_add_component(NAME frontend_llvm SOURCES l2s.c l2s_type.c l2s_value.c l2s_instr.c shady_meta_intrinsics.c l2s_meta.c l2s_postprocess.c l2s_annotations.c l2s_promote_byval_params.c ${CMAKE_CURRENT_BINARY_DIR}/l2s_generated.c)

    target_include_directories(frontend_llvm PRIVATE ${LLVM_INCLUDE_DIRS})
    target_include_directories(frontend_llvm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}) # for l2s_generated.c
    target_include_directories(frontend_llvm PRIVATE ${PROJECT_SOURCE_DIR}/vcc/include) # for shady_meta.h
    separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
    add_definitions(${LLVM_DEFINITIONS_LIST})
    target_compile_definitions(frontend_llvm PRIVATE "LLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR}")

    if (TARGET LLVM-C)
        message("LLVM-C shared library target exists, major version = ${LLVM_VERSION_MAJOR}")
        target_link_libraries(frontend_llvm PRIVATE LLVM-C)
    elseif (TARGET LLVM)
        message("LLVM shared library target exists, major version = ${LLVM_VERSION_MAJOR}")
        target_link_libraries(frontend_llvm PRIVATE LLVM)
    else ()
        message(FATAL_ERROR "Failed to find LLVM-C target, but found LLVM module earlier")
    endif()

    target_link_libraries(frontend_llvm PRIVATE api common shady)

    target_compile_definitions(driver PUBLIC LLVM_PARSER_PRESENT)
    target_link_libraries(driver PUBLIC frontend_llvm)
endif ()
