#include "shady/pass.h"
#include "shady/ir/memory_layout.h"
#include "shady/ir/function.h"
#include "shady/ir/builtin.h"
#include "shady/ir/annotation.h"
#include "shady/ir/debug.h"
#include "shady/ir/decl.h"
#include "shady/dict.h"

#include "log.h"
#include "portability.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
    BodyBuilder* bb;
    Node2Node shared_backing;
} Context;

static OpRewriteResult* process(Context* ctx, SHADY_UNUSED NodeClass use_class, SHADY_UNUSED String name, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (node->tag) {
        case Function_TAG: {
            Node* newfun = shd_recreate_node_head(r, node);
            if (get_abstraction_body(node)) {
                Context fn_ctx = *ctx;
                fn_ctx.rewriter = shd_create_children_rewriter(r);
                shd_register_processed_list(&fn_ctx.rewriter, get_abstraction_params(node), get_abstraction_params(newfun));
                fn_ctx.bb = shd_bld_begin(a, shd_get_abstraction_mem(newfun));
                Node* post_prelude = basic_block_helper(a, shd_empty(a));
                shd_set_debug_name(post_prelude, "post-prelude");
                shd_register_processed(&fn_ctx.rewriter, shd_get_abstraction_mem(node), shd_get_abstraction_mem(post_prelude));
                shd_set_abstraction_body(post_prelude, shd_rewrite_op(&fn_ctx.rewriter, NcTerminator, "body", get_abstraction_body(node)));
                shd_set_abstraction_body(newfun, shd_bld_finish(fn_ctx.bb, jump_helper(a, shd_bld_mem(fn_ctx.bb), post_prelude, shd_empty(a))));
                shd_destroy_rewriter(&fn_ctx.rewriter);
            }
            return shd_new_rewrite_result(r, newfun);
        }
        case PtrType_TAG: {
            AddressSpace as = node->payload.ptr_type.address_space;
            if (as == AsSubgroup) {
                return shd_new_rewrite_result(r, ptr_type(a, (PtrType) {
                        .pointed_type = shd_rewrite_op(&ctx->rewriter, NcType, "pointed_type", node->payload.ptr_type.pointed_type),
                        .address_space = AsShared, .is_reference = node->payload.ptr_type.is_reference }));
            }
            break;
        }
        case GlobalVariable_TAG: {
            GlobalVariable payload = node->payload.global_variable;
            AddressSpace as = node->payload.global_variable.address_space;
            if (as == AsSubgroup) {
                const Node* backing_shared_alloc = shd_node2node_find(ctx->shared_backing, node);
                if (!backing_shared_alloc) {
                    assert(payload.is_ref && "All subgroup variables should be logical by now!");
                    payload = shd_rewrite_global_head_payload(r, payload);
                    payload.address_space = AsShared;
                    payload.type = arr_type(a, (ArrType) {
                        .element_type = payload.type,
                        .size = shd_rewrite_op(&ctx->rewriter, NcValue, "size", shd_module_get_exported(ctx->rewriter.src_module, "SUBGROUPS_PER_WG"))
                    });
                    Node* new = shd_global_var(r->dst_module, payload);
                    shd_rewrite_annotations(r, node, new);

                    if (node->payload.global_variable.init) {
                        new->payload.global_variable.init = fill(a, (Fill) {
                            .type = payload.type,
                            .value = shd_rewrite_op(&ctx->rewriter, NcValue, "init", node->payload.global_variable.init)
                        });
                    }
                    shd_node2node_insert(ctx->shared_backing, node, new);
                    backing_shared_alloc = new;
                }

                OpRewriteResult* result = shd_new_rewrite_result_none(r);
                if (ctx->bb) {
                    const Node* index = shd_bld_builtin_load(ctx->rewriter.dst_module, ctx->bb, BuiltinSubgroupId);
                    const Node* slice = lea_helper(a, backing_shared_alloc, shd_int32_literal(a, 0), mk_nodes(a, index));
                    shd_rewrite_result_add_mask_rule(result, NcValue, slice);
                }
                return result;
            }
            break;
        }
        default: break;
    }

    return shd_new_rewrite_result(r, shd_recreate_node(r, node));
}

static Rewriter* rewrite_globals_in_local_ctx(Rewriter* r, const Node* n) {
    if (n->tag == GlobalVariable_TAG && n->payload.global_variable.address_space == AsSubgroup)
        return r;
    return shd_default_rewriter_selector(r, n);
}

Module* shd_pass_lower_subgroup_vars(SHADY_UNUSED const CompilerConfig* config, SHADY_UNUSED const void* unused, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    aconfig.target.memory.address_spaces[AsSubgroup].allowed = false;
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_op_rewriter(src, dst, (RewriteOpFn) process),
        .shared_backing = shd_new_node2node(),
    };
    ctx.rewriter.select_rewriter_fn = rewrite_globals_in_local_ctx;
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_node2node(ctx.shared_backing);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
