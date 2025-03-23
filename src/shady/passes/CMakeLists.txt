add_subdirectory(abi)
add_subdirectory(cf)
add_subdirectory(fncalls)
add_subdirectory(group)
add_subdirectory(io)
add_subdirectory(mem)
add_subdirectory(opt)
add_subdirectory(polyfill)
add_subdirectory(ptr)
add_subdirectory(stack)
add_subdirectory(structure)

target_sources(shady PRIVATE
    import.c
)
