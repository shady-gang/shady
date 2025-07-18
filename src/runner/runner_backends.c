#include "runner_private.h"

#include "list.h"

Backend* shd_vkr_init(Runner* base);
Backend* shd_cur_init(Runner* base);

void shd_rn_register_backends(Runner* runtime) {
#if VK_BACKEND_PRESENT
    Backend* vk_backend = shd_vkr_init(runtime);
    if (vk_backend)
        shd_list_append(Backend*, runtime->backends, vk_backend);
#endif
#if CUDA_BACKEND_PRESENT
    Backend* cuda_backend = shd_cur_init(runtime);
    if (cuda_backend)
        shd_list_append(Backend*, runtime->backends, cuda_backend);
#endif
}