#include "cuda_runtime_private.h"

#include "shady/driver.h"

#include "log.h"
#include "portability.h"
#include "dict.h"
#include "util.h"

static CompilerConfig get_compiler_config_for_device(CudaDevice* device, const CompilerConfig* base_config) {
    CompilerConfig config = *base_config;
    config.specialization.subgroup_size = 32;

    return config;
}

static bool emit_cuda_c_code(CudaKernel* spec) {
    CompilerConfig config = get_compiler_config_for_device(spec->device, spec->key.base->base_config);
    config.specialization.entry_point = spec->key.entry_point;

    Module* dst_mod = spec->key.base->module;
    CHECK(run_compiler_passes(&config, &dst_mod) == CompilationNoError, return false);

    CEmitterConfig emitter_config = {
        .dialect = CDialect_CUDA,
        .explicitly_sized_types = false,
        .allow_compound_literals = false,
        .decay_unsized_arrays = true,
    };
    Module* final_mod;
    emit_c(&config, emitter_config, dst_mod, &spec->cuda_code_size, &spec->cuda_code, &final_mod);
    spec->final_module = final_mod;

    if (get_log_level() <= DEBUG)
        write_file("cuda_dump.cu", spec->cuda_code_size - 1, spec->cuda_code);

    return true;
}

static bool cuda_c_to_ptx(CudaKernel* kernel) {
    String override_file = getenv("SHADY_OVERRIDE_PTX");
    if (override_file) {
        read_file(override_file, &kernel->ptx_size, &kernel->ptx);
        return true;
    }

    nvrtcProgram program;
    CHECK_NVRTC(nvrtcCreateProgram(&program, kernel->cuda_code, kernel->key.entry_point, 0, NULL, NULL), return false);

    assert(kernel->device->cc_major < 10 && kernel->device->cc_minor < 10);

    char arch_flag[] = "-arch=compute_00";
    arch_flag[14] = '0' + kernel->device->cc_major;
    arch_flag[15] = '0' + kernel->device->cc_minor;

    const char* options[] = {
        arch_flag,
        "--use_fast_math"
    };

    nvrtcResult compile_result = nvrtcCompileProgram(program, sizeof(options)/sizeof(*options), options);
    if (compile_result != NVRTC_SUCCESS) {
        error_print("NVRTC compilation failed: %s\n", nvrtcGetErrorString(compile_result));
        debug_print("Dumping source:\n%s", kernel->cuda_code);
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

    if (get_log_level() <= DEBUG)
        write_file("cuda_dump.ptx", kernel->ptx_size - 1, kernel->ptx);

    return true;
}

static bool load_ptx_into_cuda_program(CudaKernel* kernel) {
    char info_log[10240] = {};
    char error_log[10240] = {};

    CUjit_option options[] = {
        CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_TARGET
    };

    void* option_values[]  = {
        info_log, (void*)(uintptr_t)sizeof(info_log),
        error_log, (void*)(uintptr_t)sizeof(error_log),
        (void*)(uintptr_t)(kernel->device->cc_major * 10 + kernel->device->cc_minor)
    };

    CUlinkState linker;
    CHECK_CUDA(cuLinkCreate(sizeof(options)/sizeof(options[0]), options, option_values, &linker), goto err_linker_create);
    CHECK_CUDA(cuLinkAddData(linker, CU_JIT_INPUT_PTX, kernel->ptx, kernel->ptx_size, NULL, 0U, NULL, NULL), goto err_post_linker_create);

    void* binary;
    size_t binary_size;
    CHECK_CUDA(cuLinkComplete(linker, &binary, &binary_size), goto err_post_linker_create);

    if (*info_log)
        info_print("CUDA JIT info: %s\n", info_log);

    if (get_log_level() <= DEBUG)
        write_file("cuda_dump.cubin", binary_size, binary);

    CHECK_CUDA(cuModuleLoadData(&kernel->cuda_module, binary), goto err_post_linker_create);
    CHECK_CUDA(cuModuleGetFunction(&kernel->entry_point_function, kernel->cuda_module, kernel->key.entry_point), goto err_post_module_load);

    cuLinkDestroy(linker);
    return true;

err_post_module_load:
    cuModuleUnload(kernel->cuda_module);
err_post_linker_create:
    cuLinkDestroy(linker);
    if (*info_log)
        info_print("CUDA JIT info: %s\n", info_log);
    if (*error_log)
        error_print("CUDA JIT failed: %s\n", error_log);
err_linker_create:
    return false;
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

bool shd_cuda_destroy_specialized_kernel(CudaKernel* kernel) {
    free(kernel->cuda_code);
    free(kernel->ptx);
    CHECK_CUDA(cuModuleUnload(kernel->cuda_module), return false);

    free(kernel);
    return true;
}
