#include "shady/pass.h"

#include "../type.h"
#include "../transform/ir_gen_helpers.h"

#include "dict.h"
#include "portability.h"
#include "log.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
    BodyBuilder* bb;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (node->tag) {
        case Function_TAG: {
            Node* newfun = shd_recreate_node_head(r, node);
            if (get_abstraction_body(node)) {
                Context functx = *ctx;
                functx.rewriter.map = shd_clone_dict(functx.rewriter.map);
                shd_dict_clear(functx.rewriter.map);
                shd_register_processed_list(&functx.rewriter, get_abstraction_params(node), get_abstraction_params(newfun));
                functx.bb = begin_body_with_mem(a, shd_get_abstraction_mem(newfun));
                Node* post_prelude = basic_block(a, shd_empty(a), "post-prelude");
                shd_register_processed(&functx.rewriter, shd_get_abstraction_mem(node), shd_get_abstraction_mem(post_prelude));
                shd_set_abstraction_body(post_prelude, shd_rewrite_node(&functx.rewriter, get_abstraction_body(node)));
                shd_set_abstraction_body(newfun, finish_body(functx.bb, jump_helper(a, bb_mem(functx.bb), post_prelude,
                                                                                    shd_empty(a))));
                shd_destroy_dict(functx.rewriter.map);
            }
            return newfun;
        }
        case PtrType_TAG: {
            AddressSpace as = node->payload.ptr_type.address_space;
            if (as == AsSubgroup) {
                return ptr_type(a, (PtrType) { .pointed_type = shd_rewrite_op(&ctx->rewriter, NcType, "pointed_type", node->payload.ptr_type.pointed_type), .address_space = AsShared, .is_reference = node->payload.ptr_type.is_reference });
            }
            break;
        }
        case RefDecl_TAG: {
            const Node* odecl = node->payload.ref_decl.decl;
            if (odecl->tag != GlobalVariable_TAG || odecl->payload.global_variable.address_space != AsSubgroup)
                break;
            const Node* ndecl = shd_rewrite_node(&ctx->rewriter, odecl);
            assert(ctx->bb);
            const Node* index = gen_builtin_load(ctx->rewriter.dst_module, ctx->bb, BuiltinSubgroupId);
            const Node* slice = gen_lea(ctx->bb, ref_decl_helper(a, ndecl), shd_int32_literal(a, 0), mk_nodes(a, index));
            return slice;
        }
        case GlobalVariable_TAG: {
            AddressSpace as = node->payload.global_variable.address_space;
            if (as == AsSubgroup) {
                const Type* ntype = shd_rewrite_node(&ctx->rewriter, node->payload.global_variable.type);
                const Type* atype = arr_type(a, (ArrType) {
                    .element_type = ntype,
                    .size = ref_decl_helper(a, shd_rewrite_node(&ctx->rewriter, shd_module_get_declaration(ctx->rewriter.src_module, "SUBGROUPS_PER_WG")))
                });

                assert(shd_lookup_annotation(node, "Logical") && "All subgroup variables should be logical by now!");
                Node* new = global_var(ctx->rewriter.dst_module, shd_rewrite_nodes(&ctx->rewriter, node->payload.global_variable.annotations), atype, node->payload.global_variable.name, AsShared);
                shd_register_processed(&ctx->rewriter, node, new);

                if (node->payload.global_variable.init) {
                    new->payload.global_variable.init = fill(a, (Fill) {
                        .type = atype,
                        .value = shd_rewrite_node(&ctx->rewriter, node->payload.global_variable.init)
                    });
                }
                return new;
            }
            break;
        }
        default: break;
    }

    if (is_declaration(node)) {
        Context declctx = *ctx;
        declctx.bb = NULL;
        return shd_recreate_node(&declctx.rewriter, node);
    }

    return shd_recreate_node(&ctx->rewriter, node);
}

Module* lower_subgroup_vars(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
