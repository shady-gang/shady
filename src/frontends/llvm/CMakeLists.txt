add_library(shady_fe_llvm STATIC shady_llvm.c)

target_include_directories(shady_fe_llvm PRIVATE ${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

llvm_map_components_to_libnames(llvm_libs core irreader)
target_link_libraries(shady_fe_llvm PRIVATE ${llvm_libs})
#set_target_properties(shady_fe_llvm PROPERTIES LINKER_LANGUAGE "CXX")

target_link_libraries(shady_fe_llvm PRIVATE api common)
target_link_libraries(shady_fe_llvm PRIVATE stdc++)
set_property(TARGET shady_fe_llvm PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(shady PRIVATE shady_fe_llvm)
target_compile_definitions(shady PRIVATE LLVM_PARSER_PRESENT)