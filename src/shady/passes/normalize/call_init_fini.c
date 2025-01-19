#include "shady/ir/function.h"
#include "shady/ir/mem.h"

#include "shady/pass.h"

#include "shady/ir/annotation.h"
#include "shady/ir/decl.h"

#include "log.h"
#include "portability.h"

typedef struct {
    Rewriter rewriter;
} Context;

static const Node* process(Context* ctx, const Node* old) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    Module* m = r->dst_module;
    switch (old->tag) {
        case Function_TAG: {
            Function payload = old->payload.fun;
            const Node* entry_pt = shd_lookup_annotation(old, "EntryPoint");
            if (entry_pt) {
                // Nodes annotations = shd_filter_out_annotation(a, shd_rewrite_nodes(r, payload.annotations), "EntryPoint");
                Node* new = function_helper(m, shd_recreate_params(r, get_abstraction_params(old)), shd_rewrite_nodes(r, payload.return_types));
                shd_register_processed(r, old, new);
                shd_register_processed_list(r, get_abstraction_params(old), get_abstraction_params(new));

                shd_add_annotation_named(new, "Leaf");
                String exported_name = shd_get_exported_name(old);
                shd_remove_annotation_by_name(old, "Exported");
                shd_rewrite_annotations(r, old, new);
                shd_remove_annotation_by_name(new, "EntryPoint");

                shd_recreate_node_body(r, old, new);

                const Node* init_fn = shd_rewrite_node(r, shd_module_get_init_fn(r->src_module));
                const Node* fini_fn = shd_rewrite_node(r, shd_module_get_fini_fn(r->src_module));

                Nodes wrapper_params = shd_recreate_params(r, get_abstraction_params(old));
                Node* wrapper = function_helper(m, wrapper_params, shd_rewrite_nodes(r, payload.return_types));
                shd_rewrite_annotations(r, old, wrapper);
                shd_module_add_export(m, exported_name, wrapper);

                BodyBuilder* bld = shd_bld_begin(a, shd_get_abstraction_mem(wrapper));
                shd_bld_call(bld, fn_addr_helper(a, init_fn), shd_empty(a));
                Nodes results = shd_bld_call(bld, fn_addr_helper(a, new), wrapper_params);
                shd_bld_call(bld, fn_addr_helper(a, fini_fn), shd_empty(a));
                shd_set_abstraction_body(wrapper, shd_bld_return(bld, results));
                return new;
            }
        }
        default: break;
    }

    return shd_recreate_node(r, old);
}

static Module* run_pass(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
    };

    Rewriter* r = &ctx.rewriter;
    shd_rewrite_module(r);

    shd_destroy_rewriter(r);
    return dst;
}

#include "shady/pipeline/pipeline.h"

static void step_fn(SHADY_UNUSED void* unused, const CompilerConfig* config, Module** pmod) {
    RUN_PASS((RewritePass*) run_pass, config)
}

void shd_pipeline_add_init_fini(ShdPipeline pipeline) {
    shd_pipeline_add_step(pipeline, (ShdPipelineStepFn) step_fn, NULL, 0);
}
