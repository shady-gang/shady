add_generated_file(FILE_NAME l2s_generated.c SOURCES generator_l2s.c)

add_library(shady_fe_llvm STATIC l2s.c l2s_type.c l2s_value.c l2s_instr.c l2s_meta.c l2s_postprocess.c l2s_annotations.c ${CMAKE_CURRENT_BINARY_DIR}/l2s_generated.c)

target_include_directories(shady_fe_llvm PRIVATE ${LLVM_INCLUDE_DIRS})
target_include_directories(shady_fe_llvm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}) # for l2s_generated.c
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
target_compile_definitions(shady_fe_llvm PRIVATE "LLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR}")

if (TARGET LLVM)
    message("LLVM shared library target exists, major version = ${LLVM_VERSION_MAJOR}")
	target_link_libraries(shady_fe_llvm PRIVATE LLVM)
else ()
    llvm_map_components_to_libnames(LLVM support core)

    if(LLVM)
        target_link_libraries(shady_fe_llvm PRIVATE LLVM-C ${LLVM} LLVMMIRParser)
    else()
        message(FATAL_ERROR "Failed to resolve LLVM targets, but found LLVM module earlier!")
    endif()
endif()

target_link_libraries(shady_fe_llvm PRIVATE api common shady)
set_property(TARGET shady_fe_llvm PROPERTY POSITION_INDEPENDENT_CODE ON)
