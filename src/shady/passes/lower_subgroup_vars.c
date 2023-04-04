#include "passes.h"

#include "portability.h"
#include "log.h"

#include "../rewrite.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"

typedef struct {
    Rewriter rewriter;
    CompilerConfig* config;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    if (!node) return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    switch (node->tag) {
        case PtrType_TAG: {
            AddressSpace as = node->payload.ptr_type.address_space;
            if (as == AsSubgroupLogical) {
                return ptr_type(ctx->rewriter.dst_arena, (PtrType) { .pointed_type = rewrite_node(&ctx->rewriter, node->payload.ptr_type.pointed_type), .address_space = AsSharedLogical });
            }
            goto defolt;
        }
        case GlobalVariable_TAG: {
            // TODO: currently we just lower subgroup to shared. this is wrong and stupid!
            AddressSpace as = node->payload.global_variable.address_space;
            if (as == AsSubgroupLogical) {
                Node* new = global_var(ctx->rewriter.dst_module, rewrite_nodes(&ctx->rewriter, node->payload.global_variable.annotations),
                                  rewrite_node(&ctx->rewriter, node->payload.global_variable.type), node->payload.global_variable.name, AsSharedLogical);
                register_processed(&ctx->rewriter, node, new);
                new->payload.global_variable.init = rewrite_node(&ctx->rewriter, node->payload.global_variable.init);
                return new;
            }
            goto defolt;
        }
        defolt:
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

void lower_subgroup_vars(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .config = config
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
