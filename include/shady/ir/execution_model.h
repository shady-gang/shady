#ifndef SHADY_IR_EXECUTION_MODEL_H
#define SHADY_IR_EXECUTION_MODEL_H

#include "shady/ir/base.h"

#define EXECUTION_MODELS(EM) \
EM(Compute      ) \
EM(Fragment     ) \
EM(Vertex       ) \
EM(RayGeneration) \
EM(Callable     ) \

typedef enum {
    ShdExecutionModelNone,
#define EM(name) ShdExecutionModel##name,
    EXECUTION_MODELS(EM)
#undef EM
} ShdExecutionModel;

ShdExecutionModel shd_execution_model_from_string(const char*);
ShdExecutionModel shd_execution_model_from_entry_point(const Node* decl);

static inline bool shd_is_rt_execution_model(ShdExecutionModel em) {
    switch (em) {
        case ShdExecutionModelRayGeneration:
        case ShdExecutionModelCallable: return true;
        default: return false;
    }
}

#endif
