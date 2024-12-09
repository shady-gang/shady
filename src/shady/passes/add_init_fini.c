#include <shady/ir/function.h>
#include <shady/ir/mem.h>

#include "shady/pass.h"

#include "shady/ir/annotation.h"
#include "shady/ir/decl.h"

#include "log.h"
#include "portability.h"

typedef struct {
    Rewriter rewriter;

    Node* init_fn;
    Node* fini_fn;
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
                Node* new = function(m, shd_recreate_params(r, get_abstraction_params(old)), shd_fmt_string_irarena(a, "%s_", payload.name), shd_empty(a), shd_rewrite_nodes(r, payload.return_types));
                shd_register_processed(r, old, new);
                shd_register_processed_list(r, get_abstraction_params(old), get_abstraction_params(new));
                shd_recreate_node_body(r, old, new);

                Nodes wrapper_params = shd_recreate_params(r, get_abstraction_params(old));
                Node* wrapper = function(m, wrapper_params, payload.name, shd_rewrite_nodes(r, payload.annotations), shd_rewrite_nodes(r, payload.return_types));
                BodyBuilder* bld = shd_bld_begin(a, shd_get_abstraction_mem(wrapper));
                shd_bld_call(bld, fn_addr_helper(a, ctx->init_fn), shd_empty(a));
                Nodes results = shd_bld_call(bld, fn_addr_helper(a, new), wrapper_params);
                shd_bld_call(bld, fn_addr_helper(a, ctx->fini_fn), shd_empty(a));
                shd_set_abstraction_body(wrapper, shd_bld_return(bld, results));
                return new;
            }
        }
        default: break;
    }

    return shd_recreate_node(r, old);
}

Module* shd_pass_add_init_fini(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
    };

    Node* fini_fn = function(dst, shd_nodes(a, 0, NULL), "generated_fini", mk_nodes(a, annotation_helper(a, "Generated"), annotation_helper(a, "Leaf"), annotation_helper(a, "NoInline")), shd_nodes(a, 0, NULL));
    shd_set_abstraction_body(fini_fn, fn_ret(a, (Return) { .args = shd_empty(a), .mem = shd_get_abstraction_mem(fini_fn) }));

    Node* init_fn = function(dst, shd_nodes(a, 0, NULL), "generated_init", mk_nodes(a, annotation_helper(a, "Generated"), annotation_helper(a, "Leaf"), annotation_helper(a, "NoInline")), shd_nodes(a, 0, NULL));
    shd_set_abstraction_body(init_fn, fn_ret(a, (Return) { .args = shd_empty(a), .mem = shd_get_abstraction_mem(init_fn) }));

    ctx.init_fn = init_fn;
    ctx.fini_fn = fini_fn;

    Rewriter* r = &ctx.rewriter;
    shd_rewrite_module(r);

    shd_destroy_rewriter(r);
    return dst;
}
