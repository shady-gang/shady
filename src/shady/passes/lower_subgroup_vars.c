#include "passes.h"

#include "portability.h"
#include "log.h"

#include "../rewrite.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    if (!node) return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    IrArena* a = ctx->rewriter.dst_arena;

    switch (node->tag) {
        case PtrType_TAG: {
            AddressSpace as = node->payload.ptr_type.address_space;
            if (as == AsSubgroupLogical) {
                return ptr_type(a, (PtrType) { .pointed_type = rewrite_node(&ctx->rewriter, node->payload.ptr_type.pointed_type), .address_space = AsSharedLogical });
            }
            break;
        }
        case RefDecl_TAG: {
            const Node* odecl = node->payload.ref_decl.decl;
            if (odecl->tag != GlobalVariable_TAG || odecl->payload.global_variable.address_space != AsSubgroupLogical)
                break;
            BodyBuilder* bb = begin_body(a);
            const Node* ndecl = rewrite_node(&ctx->rewriter, odecl);
            const Node* index = gen_builtin_load(ctx->rewriter.dst_module, bb, BuiltinSubgroupId);
            const Node* slice = gen_lea(bb, ref_decl_helper(a, ndecl), int32_literal(a, 0), mk_nodes(a, index));
            return anti_quote(a, (AntiQuote) { .instruction = yield_values_and_wrap_in_block(bb, singleton(slice)) });
        }
        case GlobalVariable_TAG: {
            AddressSpace as = node->payload.global_variable.address_space;
            if (as == AsSubgroupLogical) {
                const Type* ntype = rewrite_node(&ctx->rewriter, node->payload.global_variable.type);
                const Type* atype = arr_type(a, (ArrType) {
                    .element_type = ntype,
                    .size = access_decl(&ctx->rewriter, "SUBGROUPS_PER_WG")
                });

                Node* new = global_var(ctx->rewriter.dst_module, rewrite_nodes(&ctx->rewriter, node->payload.global_variable.annotations), atype, node->payload.global_variable.name, AsSharedLogical);
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
    return recreate_node_identity(&ctx->rewriter, node);
}

Module* lower_subgroup_vars(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .config = config
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
