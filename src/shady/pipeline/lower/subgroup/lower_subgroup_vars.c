#include "shady/pass.h"
#include "shady/ir/memory_layout.h"
#include "shady/ir/function.h"
#include "shady/ir/builtin.h"
#include "shady/ir/annotation.h"
#include "shady/ir/decl.h"

#include "log.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
    BodyBuilder* bb;
} Context;

static OpRewriteResult process(Context* ctx, NodeClass use_class, String name, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (node->tag) {
        case Function_TAG: {
            Node* newfun = shd_recreate_node_head(r, node);
            if (get_abstraction_body(node)) {
                Context fn_ctx = *ctx;
                fn_ctx.rewriter = shd_create_children_rewriter(shd_get_top_rewriter(r));
                shd_register_processed_list(&fn_ctx.rewriter, get_abstraction_params(node), get_abstraction_params(newfun));
                fn_ctx.bb = shd_bld_begin(a, shd_get_abstraction_mem(newfun));
                Node* post_prelude = basic_block(a, shd_empty(a), "post-prelude");
                shd_register_processed(&fn_ctx.rewriter, shd_get_abstraction_mem(node), shd_get_abstraction_mem(post_prelude));
                shd_set_abstraction_body(post_prelude, shd_rewrite_op(&fn_ctx.rewriter, NcTerminator, "body", get_abstraction_body(node)));
                shd_set_abstraction_body(newfun, shd_bld_finish(fn_ctx.bb, jump_helper(a, shd_bld_mem(fn_ctx.bb), post_prelude, shd_empty(a))));
                shd_destroy_rewriter(&fn_ctx.rewriter);
            }
            return (OpRewriteResult) { newfun, 0 };
        }
        case PtrType_TAG: {
            AddressSpace as = node->payload.ptr_type.address_space;
            if (as == AsSubgroup) {
                return (OpRewriteResult) { ptr_type(a, (PtrType) {
                        .pointed_type = shd_rewrite_op(&ctx->rewriter, NcType, "pointed_type", node->payload.ptr_type.pointed_type),
                        .address_space = AsShared, .is_reference = node->payload.ptr_type.is_reference }),
                    0 };
            }
            break;
        }
        case GlobalVariable_TAG: {
            AddressSpace as = node->payload.global_variable.address_space;
            if (as == AsSubgroup) {
                if (use_class == NcValue) {
                    const Node* ndecl = shd_rewrite_op(&ctx->rewriter, NcDeclaration, "", node);
                    assert(ctx->bb);
                    const Node* index = shd_bld_builtin_load(ctx->rewriter.dst_module, ctx->bb, BuiltinSubgroupId);
                    const Node* slice = lea_helper(a, ndecl, shd_int32_literal(a, 0), mk_nodes(a, index));
                    return (OpRewriteResult)  { slice, NcValue };
                } else {
                    const Type* ntype = shd_rewrite_op(&ctx->rewriter, NcType, "type", node->payload.global_variable.type);
                    const Type* atype = arr_type(a, (ArrType) {
                        .element_type = ntype,
                        .size = shd_rewrite_op(&ctx->rewriter, NcValue, "size", shd_module_get_declaration(ctx->rewriter.src_module, "SUBGROUPS_PER_WG"))
                    });

                    assert(shd_lookup_annotation(node, "Logical") && "All subgroup variables should be logical by now!");
                    Node* new = global_var(ctx->rewriter.dst_module, shd_rewrite_ops(&ctx->rewriter, NcAnnotation, "annotation", node->payload.global_variable.annotations), atype, node->payload.global_variable.name, AsShared);
                    shd_register_processed_mask(&ctx->rewriter, node, new, ~NcValue);

                    if (node->payload.global_variable.init) {
                        new->payload.global_variable.init = fill(a, (Fill) {
                            .type = atype,
                            .value = shd_rewrite_op(&ctx->rewriter, NcValue, "init", node->payload.global_variable.init)
                        });
                    }
                    return (OpRewriteResult) { new, ~NcValue };
                }
            }
            break;
        }
        default: break;
    }

    if (is_declaration(node)) {
        Context declctx = *ctx;
        declctx.bb = NULL;
        return (OpRewriteResult) { shd_recreate_node(&declctx.rewriter, node), 0};
    }

    return (OpRewriteResult) { shd_recreate_node(r, node), 0 };
}

Module* shd_pass_lower_subgroup_vars(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_op_rewriter(src, dst, (RewriteOpFn) process),
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
