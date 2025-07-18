find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message("CUDA toolkit found.")
    option(SHADY_ENABLE_RUNNER_CUDA_BACKEND "CUDA support for the 'runner' component" ON)
else()
    message("CUDA toolkit not found, CUDA runner component cannot be built.")
endif ()

if (SHADY_ENABLE_RUNNER_CUDA_BACKEND)
    shady_add_component(NAME runner_cuda SOURCES cuda_runner.c cuda_runner_buffer.c cuda_runner_program.c)
    target_link_libraries(runner_cuda PRIVATE common api shady)
    target_link_libraries(runner_cuda PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
    target_link_libraries(runner_cuda PRIVATE backend_c shader_pipeline)
    target_link_libraries(runner_cuda PRIVATE runner_base)

    target_compile_definitions(runner PUBLIC CUDA_BACKEND_PRESENT=1)
    target_link_libraries(runner PUBLIC runner_cuda)
endif()