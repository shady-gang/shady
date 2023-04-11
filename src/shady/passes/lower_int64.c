#include "passes.h"

#include "log.h"
#include "portability.h"

#include "../ir_private.h"
#include "../type.h"
#include "../rewrite.h"
#include "../transform/ir_gen_helpers.h"

typedef struct {
    Rewriter rewriter;
    CompilerConfig* config;
} Context;

static bool should_convert(Context* ctx, const Type* t) {
    t = get_unqualified_type(t);
    return t->tag == Int_TAG && t->payload.int_type.width == IntTy64 && ctx->config->lower.int64;
}

static void extract_low_hi_halves(BodyBuilder* builder, const Node* src, const Node** lo, const Node** hi) {
    *lo = first(bind_instruction(builder, prim_op(builder->arena,
        (PrimOp) { .op = extract_op, .operands = mk_nodes(builder->arena, src, int32_literal(builder->arena, 0))})));
    *hi = first(bind_instruction(builder, prim_op(builder->arena,
        (PrimOp) { .op = extract_op, .operands = mk_nodes(builder->arena, src, int32_literal(builder->arena, 1))})));
}

static void extract_low_hi_halves_list(BodyBuilder* builder, Nodes src, const Node** lows, const Node** his) {
    for (size_t i = 0; i < src.count; i++) {
        extract_low_hi_halves(builder, src.nodes[i], lows, his);
        lows++;
        his++;
    }
}

static const Node* process(Context* ctx, const Node* node) {
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    IrArena* arena = ctx->rewriter.dst_arena;

    switch (node->tag) {
        case Int_TAG:
            if (node->payload.int_type.width == IntTy64 && ctx->config->lower.int64)
                return record_type(arena, (RecordType) {
                    .members = mk_nodes(arena, int32_type(arena), int32_type(arena))
                });
            break;
        case IntLiteral_TAG:
            if (node->payload.int_literal.width == IntTy64 && ctx->config->lower.int64) {
                uint64_t raw = node->payload.int_literal.value.u64;
                const Node* lower = uint32_literal(arena, (uint32_t) raw);
                const Node* upper = uint32_literal(arena, (uint32_t) (raw >> 32));
                return tuple(arena, mk_nodes(arena, lower, upper));
            }
            break;
        case PrimOp_TAG: {
            Op op = node->payload.prim_op.op;
            Nodes old_nodes = node->payload.prim_op.operands;
            LARRAY(const Node*, lows, old_nodes.count);
            LARRAY(const Node*, his, old_nodes.count);
            switch(op) {
                case add_op: if (should_convert(ctx, first(old_nodes)->type)) {
                    Nodes new_nodes = rewrite_nodes(&ctx->rewriter, old_nodes);
                    // TODO: convert into and then out of unsigned
                    BodyBuilder* builder = begin_body(arena);
                    extract_low_hi_halves_list(builder, new_nodes, lows, his);
                    Nodes low_and_carry = bind_instruction(builder, prim_op(arena, (PrimOp) { .op = add_carry_op, .operands = nodes(arena, 2, lows)}));
                    const Node* lo = first(low_and_carry);
                    // compute the high side, without forgetting the carry bit
                    const Node* hi = first(bind_instruction(builder, prim_op(arena, (PrimOp) { .op = add_op, .operands = nodes(arena, 2, his)})));
                                hi = first(bind_instruction(builder, prim_op(arena, (PrimOp) { .op = add_op, .operands = mk_nodes(arena, hi, low_and_carry.nodes[1])})));
                    return yield_values_and_wrap_in_block(builder, singleton(tuple(arena, mk_nodes(arena, lo, hi))));
                } break;
                default: break;
            }
            break;
        }
        default: break;
    }

    rebuild:
    return recreate_node_identity(&ctx->rewriter, node);
}

void lower_int(CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .config = config,
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
