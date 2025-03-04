find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message("CUDA toolkit found.")
    option(SHADY_ENABLE_RUNNER_CUDA_BACKEND "CUDA support for the 'runner' component" ON)
else()
    message("CUDA toolkit not found, CUDA runner component cannot be built.")
endif ()

if (SHADY_ENABLE_RUNNER_CUDA_BACKEND)
    add_library(cuda_runner STATIC cuda_runner.c cuda_runner_buffer.c cuda_runner_program.c)
    target_link_libraries(cuda_runner PRIVATE api)
    target_link_libraries(cuda_runner PRIVATE "$<BUILD_INTERFACE:common>")
    target_link_libraries(cuda_runner PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)

    target_link_libraries(runner PRIVATE "$<BUILD_INTERFACE:cuda_runner>")
    target_compile_definitions(runner PUBLIC CUDA_BACKEND_PRESENT=1)
endif()