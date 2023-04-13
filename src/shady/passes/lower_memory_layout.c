#include "passes.h"

#include "../transform/memory_layout.h"
#include "../rewrite.h"
#include "../type.h"

#include "log.h"
#include "portability.h"
#include <assert.h>

typedef struct {
    Rewriter rewriter;
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
                    TypeMemLayout layout = get_mem_layout(a, t);
                    return quote(a, singleton(int_literal(a, (IntLiteral) {.width = a->config.memory.ptr_size, .is_signed = false, .value.u64 = layout.size_in_bytes})));
                }
                case align_of_op: {
                    const Type* t = rewrite_node(&ctx->rewriter, first(old->payload.prim_op.type_arguments));
                    TypeMemLayout layout = get_mem_layout(a, t);
                    return quote(a, singleton(int_literal(a, (IntLiteral) {.width = a->config.memory.ptr_size, .is_signed = false, .value.u64 = layout.alignment_in_bytes})));
                }
                case offset_of_op: {
                    const Type* t = rewrite_node(&ctx->rewriter, first(old->payload.prim_op.type_arguments));
                    const Node* n = rewrite_node(&ctx->rewriter, first(old->payload.prim_op.operands));
                    const IntLiteral* literal = resolve_to_literal(n);
                    assert(literal);
                    t = get_maybe_nominal_type_body(t);
                    uint64_t offset_in_bytes = (uint64_t) get_record_field_offset_in_bytes(a, t, literal->value.u64);
                    const Node* offset_literal = int_literal(a, (IntLiteral) { .width = a->config.memory.ptr_size, .is_signed = false, .value.u64 = offset_in_bytes });
                    return quote(a, singleton(offset_literal));
                }
                default: break;
            }
            break;
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, old);
}

void lower_memory_layout(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process)
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
