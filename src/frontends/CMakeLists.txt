add_subdirectory(slim)
add_subdirectory(spirv)

if(LLVM_FOUND)
    add_subdirectory(llvm)
endif()
