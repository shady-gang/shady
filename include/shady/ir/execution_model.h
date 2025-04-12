#ifndef SHADY_IR_EXECUTION_MODEL_H
#define SHADY_IR_EXECUTION_MODEL_H

#define EXECUTION_MODELS(EM) \
EM(Compute      ) \
EM(Fragment     ) \
EM(Vertex       ) \
EM(RayGeneration) \
EM(Callable     ) \

typedef enum {
    EmNone,
#define EM(name) Em##name,
    EXECUTION_MODELS(EM)
#undef EM
} ExecutionModel;

ExecutionModel shd_execution_model_from_string(const char*);
ExecutionModel shd_execution_model_from_entry_point(const Node* decl);

static inline bool shd_is_rt_execution_model(ExecutionModel em) {
    switch (em) {
        case EmRayGeneration:
        case EmCallable: return true;
        default: return false;
    }
}

#endif
