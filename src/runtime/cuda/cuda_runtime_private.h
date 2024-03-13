#ifndef SHADY_CUDA_RUNTIME_PRIVATE_H
#define SHADY_CUDA_RUNTIME_PRIVATE_H

#include "../runtime_private.h"

#include <cuda_runtime_api.h>

typedef struct CudaBackend_ {
    Backend base;
} CudaBackend;

#endif
