find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message("CUDA toolkit found.")
    option(SHADY_ENABLE_RUNTIME_CUDA "CUDA support for the 'runtime' component" ON)
else()
    message("CUDA toolkit not found, CUDA runtime component cannot be built.")
endif ()

if (SHADY_ENABLE_RUNTIME_CUDA)
    add_library(cuda_runtime STATIC cuda_runtime.c cuda_runtime_buffer.c cuda_runtime_program.c)
    target_link_libraries(cuda_runtime PRIVATE api)
    target_link_libraries(cuda_runtime PRIVATE "$<BUILD_INTERFACE:common>")
    target_link_libraries(cuda_runtime PRIVATE "$<BUILD_INTERFACE:murmur3>")
    target_link_libraries(cuda_runtime PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)

    target_link_libraries(runtime PRIVATE "$<BUILD_INTERFACE:cuda_runtime>")
    target_compile_definitions(runtime PUBLIC CUDA_BACKEND_PRESENT=1)
endif()