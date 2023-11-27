get_target_property(SPIRV_HEADERS_INCLUDE_DIRS SPIRV-Headers::SPIRV-Headers INTERFACE_INCLUDE_DIRECTORIES)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/l2s_generated.c COMMENT "generating l2s code" COMMAND generator l2s-code ${CMAKE_CURRENT_BINARY_DIR}/l2s_generated.c ${CMAKE_SOURCE_DIR}/include/shady/grammar.json ${CMAKE_SOURCE_DIR}/include/shady/primops.json ${SPIRV_HEADERS_INCLUDE_DIRS} VERBATIM DEPENDS generator api)

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
    message(FATAL_ERROR "Failed to find LLVM target, but found LLVM module earlier")
endif()

target_link_libraries(shady_fe_llvm PRIVATE api common shady)
set_property(TARGET shady_fe_llvm PROPERTY POSITION_INDEPENDENT_CODE ON)
