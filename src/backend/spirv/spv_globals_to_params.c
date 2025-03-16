#include "shady/pass.h"
#include "shady/ir/annotation.h"
#include "shady/ir/debug.h"
#include "shady/ir/function.h"
#include "shady/ir/mem.h"
#include "shady/ir/decl.h"
#include "shady/dict.h"

#include "portability.h"
#include "log.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
    BodyBuilder* bb;
    Node2Node lifted_globals;
    Nodes extra_params;
    Nodes extra_globals;
} Context;

static OpRewriteResult* process(Context* ctx, NodeClass use, String name, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (node->tag) {
        case Function_TAG: {
            if (!get_abstraction_body(node))
                break;
            Function payload = shd_rewrite_function_head_payload(r, node->payload.fun);
            Context fn_ctx = *ctx;
            shd_register_processed_list(&fn_ctx.rewriter, get_abstraction_params(node), payload.params);
            if (shd_lookup_annotation(node, "EntryPoint")) {
                payload.params = shd_concat_nodes(a, ctx->extra_params, payload.params);
            }
            Node* newfun = shd_function(r->dst_module, payload);
            shd_register_processed(r, node, newfun);
            shd_rewrite_annotations(r, node, newfun);
            fn_ctx.rewriter = shd_create_children_rewriter(r);
            fn_ctx.bb = shd_bld_begin(a, shd_get_abstraction_mem(newfun));
            if (shd_lookup_annotation(node, "EntryPoint")) {
                // copy the params
                for (size_t i = 0; i < ctx->extra_globals.count; i++) {
                    shd_bld_store(fn_ctx.bb, ctx->extra_globals.nodes[i], ctx->extra_params.nodes[i]);
                }
            }
            Node* post_prelude = basic_block_helper(a, shd_empty(a));
            shd_set_debug_name(post_prelude, "post-prelude");
            shd_register_processed(&fn_ctx.rewriter, shd_get_abstraction_mem(node), shd_get_abstraction_mem(post_prelude));
            shd_set_abstraction_body(post_prelude, shd_rewrite_op(&fn_ctx.rewriter, NcTerminator, "body", get_abstraction_body(node)));
            shd_set_abstraction_body(newfun, shd_bld_finish(fn_ctx.bb, jump_helper(a, shd_bld_mem(fn_ctx.bb), post_prelude,
                                                                                   shd_empty(a))));
            shd_destroy_rewriter(&fn_ctx.rewriter);
            return shd_new_rewrite_result(r, newfun);
        }
        case GlobalVariable_TAG: {
            if (node->payload.global_variable.address_space != AsGlobal)
                break;
            assert(ctx->bb && "this Global isn't appearing in an abstraction - we cannot replace it with a load!");
            const Node* ptr_addr = shd_node2node_find(ctx->lifted_globals, node);
            const Node* ptr = shd_bld_load(ctx->bb, ptr_addr);
            ptr = scope_cast_helper(a, shd_get_qualified_type_scope(node->type), ptr);
            OpRewriteResult* result = shd_new_rewrite_result_none(r);
            shd_rewrite_result_add_mask_rule(result, NcValue, ptr);
            return result;
        }
        default: break;
    }

    return shd_new_rewrite_result(r, shd_recreate_node(r, node));
}

static Rewriter* rewrite_globals_in_local_ctx(Rewriter* r, const Node* n) {
    if (n->tag == GlobalVariable_TAG && n->payload.global_variable.address_space == AsGlobal)
        return r;
    return shd_default_rewriter_selector(r, n);
}

Module* shd_spvbe_pass_lift_globals_ssbo(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
        .rewriter = shd_create_op_rewriter(src, dst, (RewriteOpFn) process),
        .config = config,
        .lifted_globals = shd_new_node2node(),
    };
    ctx.rewriter.select_rewriter_fn = rewrite_globals_in_local_ctx;

    Nodes oglobals = shd_module_collect_reachable_globals(src);
    for (size_t i = 0; i < oglobals.count; i++) {
        const Node* odecl = oglobals.nodes[i];
        if (odecl->payload.global_variable.address_space != AsGlobal)
            continue;

        const Type* t = shd_get_unqualified_type(shd_rewrite_op(&ctx.rewriter, NcType, "type", odecl->type));
        String name = shd_get_node_name_unsafe(odecl);

        const Node* g = shd_global_var(dst, (GlobalVariable) {
            .address_space = AsPrivate,
            .type = t,
            .is_ref = true,
        });
        const Node* p = param_helper(a, qualified_type_helper(a, ctx.config->target.scopes.constants, t));
        if (name) {
            shd_set_debug_name(p, name);
            shd_set_debug_name(g, name);
        }

        shd_node2node_insert(ctx.lifted_globals, odecl, g);
        ctx.extra_params = shd_nodes_append(a, ctx.extra_params, p);
        ctx.extra_globals = shd_nodes_append(a, ctx.extra_globals, g);
    }

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_node2node(ctx.lifted_globals);

    for (size_t i = 0; i < oglobals.count; i++) {
        const Node* odecl = oglobals.nodes[i];
        if (odecl->payload.global_variable.address_space != AsGlobal)
            continue;
        if (odecl->payload.global_variable.init) {
            shd_add_annotation(ctx.extra_params.nodes[i], annotation_values(a, (AnnotationValues) {
                    .name = "RuntimeProvideMem",
                    .values = mk_nodes(a, shd_rewrite_op(&ctx.rewriter, NcValue, "init", odecl->payload.global_variable.init))
            }));
        }
    }

    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
