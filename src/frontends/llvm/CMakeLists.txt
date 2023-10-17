add_library(shady_fe_llvm STATIC l2s.c l2s_type.c l2s_value.c l2s_instr.c l2s_meta.c l2s_postprocess.c l2s_annotations.c)

target_include_directories(shady_fe_llvm PRIVATE ${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

if (TARGET LLVM)
    message("LLVM shared library target exists")
    target_link_libraries(shady_fe_llvm PRIVATE LLVM)
else ()
    message(FATAL_ERROR "Failed to find LLVM target, but found LLVM module earlier")
endif()

target_link_libraries(shady_fe_llvm PRIVATE api common)
set_property(TARGET shady_fe_llvm PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(shady PRIVATE shady_fe_llvm)
target_compile_definitions(shady PRIVATE LLVM_PARSER_PRESENT)