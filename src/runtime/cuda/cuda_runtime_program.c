#include "cuda_runtime_private.h"

#include "log.h"
#include "portability.h"
#include "dict.h"

static CompilerConfig get_compiler_config_for_device(CudaDevice* device, const CompilerConfig* base_config) {
    CompilerConfig config = *base_config;
    config.specialization.subgroup_size = 32;

    return config;
}

static bool emit_cuda_c_code(CudaKernel* spec) {
    CompilerConfig config = get_compiler_config_for_device(spec->device, spec->key.base->base_config);
    config.specialization.entry_point = spec->key.entry_point;

    Module* dst_mod;
    CHECK(run_compiler_passes(&config, &dst_mod) == CompilationNoError, return false);

    CEmitterConfig emitter_config = {
        .dialect = CDialect_CUDA,
        .explicitly_sized_types = true,
        .allow_compound_literals = true,
    };
    Module* final_mod;
    emit_c(config, emitter_config, dst_mod, &spec->cuda_code_size, &spec->cuda_code, &final_mod);
    spec->final_module = final_mod;
    return true;
}

static bool cuda_c_to_ptx(CudaKernel* kernel) {
    nvrtcProgram program;
    CHECK_NVRTC(nvrtcCreateProgram(&program, kernel->cuda_code, kernel->key.entry_point, 0, NULL, NULL), return false);
    nvrtcResult compile_result = nvrtcCompileProgram(program, 0, false);
    if (compile_result != NVRTC_SUCCESS) {
        error_print("NVRTC compilation failed: %s\n", nvrtcGetErrorString(compile_result));
    }

    size_t log_size;
    CHECK_NVRTC(nvrtcGetProgramLogSize(program, &log_size), return false);
    char* log_buffer = calloc(log_size, 1);
    CHECK_NVRTC(nvrtcGetProgramLog(program, log_buffer), return false);
    log_string(compile_result == NVRTC_SUCCESS ? DEBUG : ERROR, "NVRTC compilation log: %s\n", log_buffer);
    free(log_buffer);

    CHECK_NVRTC(nvrtcGetPTXSize(program, &kernel->ptx_size), return false);
    kernel->ptx = calloc(kernel->ptx_size, 1);
    CHECK_NVRTC(nvrtcGetPTX(program, kernel->ptx), return false);
    CHECK_NVRTC(nvrtcDestroyProgram(&program), return false);

    return true;
}

static bool load_ptx_into_cuda_program(CudaKernel* kernel) {
    CHECK_CUDA(cuModuleLoadDataEx(&kernel->cuda_module, kernel->ptx, 0, NULL, NULL), return false);
    CHECK_CUDA(cuModuleGetFunction(&kernel->entry_point_function, kernel->cuda_module, kernel->key.entry_point), return false);
    return true;
}

static CudaKernel* create_specialized_program(CudaDevice* device, SpecProgramKey key) {
    CudaKernel* kernel = calloc(1, sizeof(CudaKernel));
    if (!kernel)
        return NULL;
    *kernel = (CudaKernel) {
        .key = key,
        .device = device,
    };

    CHECK(emit_cuda_c_code(kernel), return NULL);
    CHECK(cuda_c_to_ptx(kernel), return NULL);
    CHECK(load_ptx_into_cuda_program(kernel), return NULL);
    CHECK(shd_extract_parameters_info(&kernel->parameters, kernel->final_module), return false);

    return kernel;
}

CudaKernel* shd_cuda_get_specialized_program(CudaDevice* device, Program* program, String entry_point) {
    SpecProgramKey key = { .base = program, .entry_point = entry_point };
    CudaKernel** found = find_value_dict(SpecProgramKey, CudaKernel*, device->specialized_programs, key);
    if (found)
        return *found;
    CudaKernel* spec = create_specialized_program(device, key);
    assert(spec);
    insert_dict(SpecProgramKey, CudaKernel*, device->specialized_programs, key, spec);
    return spec;
}
