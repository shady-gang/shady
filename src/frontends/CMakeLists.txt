add_subdirectory(slim)
add_subdirectory(spirv)

find_package(LLVM)
if(LLVM_FOUND)
    message("LLVM ${LLVM_VERSION} found")
    add_subdirectory(llvm)
endif ()
