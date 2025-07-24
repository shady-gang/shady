#include "runner_private.h"

#include "list.h"

Backend* shd_vkr_init(Runner* base);
Backend* shd_cur_init(Runner* base);

static void register_backends(Runner* runner) {
#if VK_BACKEND_PRESENT
    Backend* vk_backend = shd_vkr_init(runner);
    if (vk_backend)
        shd_list_append(Backend*, runner->backends, vk_backend);
#endif
#if CUDA_BACKEND_PRESENT
    Backend* cuda_backend = shd_cur_init(runner);
    if (cuda_backend)
        shd_list_append(Backend*, runner->backends, cuda_backend);
#endif
}

Runner* shd_rn_initialize(RunnerConfig config) {
    Runner* runner = shd_rn_initialize_base(config);
    register_backends(runner);
    return runner;
}
