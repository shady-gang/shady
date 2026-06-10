#ifndef SHADY_IR_EXECUTION_MODEL_H
#define SHADY_IR_EXECUTION_MODEL_H

#include "shady/ir/base.h"

#define EXECUTION_MODELS(EM) \
EM(Compute      ) \
EM(Fragment     ) \
EM(Vertex       ) \
EM(RayGeneration) \
EM(Callable     ) \
EM(Mesh         ) \

typedef enum {
    ShdExecutionModelNone,
#define EM(name) ShdExecutionModel##name,
    EXECUTION_MODELS(EM)
#undef EM
} ShdExecutionModel;

ShdExecutionModel shd_execution_model_from_string(const char*);
ShdExecutionModel shd_execution_model_from_entry_point(const Node* decl);

typedef struct {
    ShdExecutionModel execution_model;
    union {
        struct {
            uint32_t workgroup_size[3];
        } grid_based;
    };
    uint32_t num_vertices;
    uint32_t num_primitives;
} ExecutionModelInfo;

ExecutionModelInfo shd_get_execution_model_info_from_entry_point(const Node* fn);

bool shd_get_workgroup_size(const ExecutionModelInfo*, uint32_t* out);
bool shd_get_num_vertices(const ExecutionModelInfo*, uint32_t* out);
bool shd_get_num_primitives(const ExecutionModelInfo*, uint32_t* out);
bool shd_get_num_subgroups_per_workgroups(const ExecutionModelInfo*, uint32_t subgroup_size, uint32_t* out);

/// If this execution model a stage that's part of a raytracing pipeline ?
static inline bool shd_is_execution_model_rt_stage(ShdExecutionModel em) {
    switch (em) {
        case ShdExecutionModelRayGeneration:
        case ShdExecutionModelCallable: return true;
        default: return false;
    }
}

/// Does this execution model feature a grid of workgroups ?
static inline bool shd_is_execution_model_workgroup_based(ShdExecutionModel em) {
    switch (em) {
        case ShdExecutionModelMesh: return true;
        case ShdExecutionModelCompute: return true;
        default: return false;
    }
}

#endif
