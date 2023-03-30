#include "passes.h"

#include "../transform/memory_layout.h"
#include "../rewrite.h"

#include "log.h"
#include <assert.h>

typedef struct {
    Rewriter rewriter;
    CompilerConfig* config;
} Context;

static const Node* process(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

    IrArena* a = ctx->rewriter.dst_arena;

    switch (old->tag) {
        case PrimOp_TAG: {
            switch (old->payload.prim_op.op) {
                case size_of_op: {
                    const Type* t = rewrite_node(&ctx->rewriter, first(old->payload.prim_op.type_arguments));
                    TypeMemLayout layout = get_mem_layout(ctx->config, a, t);
                    const Node* byte_size = int_literal(a, (IntLiteral) { .width = a->config.memory.ptr_size, .is_signed = false, .value.u64 = layout.size_in_bytes });
                    return quote(a, singleton(byte_size));
                }
                case align_of_op: error("TODO");
                case offset_of_op: {
                    const Type* t = rewrite_node(&ctx->rewriter, first(old->payload.prim_op.type_arguments));
                    const Node* n = rewrite_node(&ctx->rewriter, first(old->payload.prim_op.operands));
                    const IntLiteral* literal = resolve_to_literal(n);
                    assert(literal);
                    const Node* offset_in_bytes = int_literal(a, (IntLiteral) { .width = a->config.memory.ptr_size, .is_signed = false, .value.u64 = literal->value.u64 });
                    return quote(a, singleton(offset_in_bytes));
                }
                default: break;
            }
            break;
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, old);
}

void lower_memory_layout(CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process)
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
