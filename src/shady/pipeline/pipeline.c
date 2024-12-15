#include "shady/pipeline/pipeline.h"

#include "arena.h"

#include <stdlib.h>

struct Step {
    ShdPipelineStepFn fn;
    void* payload;
    struct Step* next;
};

struct ShdPipeline_ {
    Arena* arena;
    struct Step* step;
};

ShdPipeline shd_create_empty_pipeline(void) {
    ShdPipeline pipeline = malloc(sizeof(struct ShdPipeline_));
    *pipeline = (struct ShdPipeline_) {
        .arena = shd_new_arena(),
        .step = NULL,
    };
    return pipeline;
}

void shd_destroy_pipeline(ShdPipeline pipeline) {
    shd_destroy_arena(pipeline->arena);
    free(pipeline);
}

/// Runs a given pipeline on a module
CompilationResult shd_pipeline_run(ShdPipeline pipeline, CompilerConfig* config, Module** pmod) {
    struct Step* s = pipeline->step;
    while (s) {
        s->fn(s->payload, config, pmod);
        s = s->next;
    }
    return CompilationNoError;
}

void shd_pipeline_add_step(ShdPipeline pipeline, ShdPipelineStepFn fn, void* payload, size_t payload_size) {
    void* stored_payload = NULL;
    if (payload_size > 0) {
        stored_payload = shd_arena_alloc(pipeline->arena, payload_size);
        memcpy(stored_payload, payload, payload_size);
    }
    struct Step* step = shd_arena_alloc(pipeline->arena, sizeof(struct Step));
    step->next = NULL;
    step->fn = fn;
    step->payload = stored_payload;
    struct Step** dst = &pipeline->step;
    while ((*dst)) {
        dst = &(*dst)->next;
    }
    *dst = step;
}