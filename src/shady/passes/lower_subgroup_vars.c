#include <dict.h>

#include "shady/pass.h"

#include "../type.h"
#include "../transform/ir_gen_helpers.h"

#include "portability.h"
#include "log.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
    BodyBuilder* bb;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    if (!node) return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (node->tag) {
        case Function_TAG: {
            Node* newfun = recreate_decl_header_identity(r, node);
            Context functx = *ctx;
            functx.rewriter.map = clone_dict(functx.rewriter.map);
            clear_dict(functx.rewriter.map);
            register_processed_list(&functx.rewriter, get_abstraction_params(node), get_abstraction_params(newfun));
            functx.bb = begin_body_with_mem(a, get_abstraction_mem(newfun));
            Node* post_prelude = basic_block(a, empty(a), "post-prelude");
            register_processed(&functx.rewriter, get_abstraction_mem(node), get_abstraction_mem(post_prelude));
            set_abstraction_body(post_prelude, rewrite_node(&functx.rewriter, get_abstraction_body(node)));
            set_abstraction_body(newfun, finish_body(functx.bb, jump_helper(a, post_prelude, empty(a), bb_mem(functx.bb))));
            destroy_dict(functx.rewriter.map);
            return newfun;
        }
        case PtrType_TAG: {
            AddressSpace as = node->payload.ptr_type.address_space;
            if (as == AsSubgroup) {
                return ptr_type(a, (PtrType) { .pointed_type = rewrite_op(&ctx->rewriter, NcType, "pointed_type", node->payload.ptr_type.pointed_type), .address_space = AsShared, .is_reference = node->payload.ptr_type.is_reference });
            }
            break;
        }
        case RefDecl_TAG: {
            const Node* odecl = node->payload.ref_decl.decl;
            if (odecl->tag != GlobalVariable_TAG || odecl->payload.global_variable.address_space != AsSubgroup)
                break;
            const Node* ndecl = rewrite_node(&ctx->rewriter, odecl);
            assert(ctx->bb);
            const Node* index = gen_builtin_load(ctx->rewriter.dst_module, ctx->bb, BuiltinSubgroupId);
            const Node* slice = gen_lea(ctx->bb, ref_decl_helper(a, ndecl), int32_literal(a, 0), mk_nodes(a, index));
            return slice;
        }
        case GlobalVariable_TAG: {
            AddressSpace as = node->payload.global_variable.address_space;
            if (as == AsSubgroup) {
                const Type* ntype = rewrite_node(&ctx->rewriter, node->payload.global_variable.type);
                const Type* atype = arr_type(a, (ArrType) {
                    .element_type = ntype,
                    .size = ref_decl_helper(a, rewrite_node(&ctx->rewriter, get_declaration(ctx->rewriter.src_module, "SUBGROUPS_PER_WG")))
                });

                assert(lookup_annotation(node, "Logical") && "All subgroup variables should be logical by now!");
                Node* new = global_var(ctx->rewriter.dst_module, rewrite_nodes(&ctx->rewriter, node->payload.global_variable.annotations), atype, node->payload.global_variable.name, AsShared);
                register_processed(&ctx->rewriter, node, new);

                if (node->payload.global_variable.init) {
                    new->payload.global_variable.init = fill(a, (Fill) {
                        .type = atype,
                        .value = rewrite_node(&ctx->rewriter, node->payload.global_variable.init)
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
        return recreate_node_identity(&declctx.rewriter, node);
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

Module* lower_subgroup_vars(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
