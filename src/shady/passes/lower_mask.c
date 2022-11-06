#include "passes.h"

#include "log.h"
#include "portability.h"

#include "../ir_private.h"
#include "../rewrite.h"
#include "../transform/ir_gen_helpers.h"

typedef struct {
    Rewriter rewriter;
} Context;

/// Removes most instructions that deal with masks and lower them to bitwise operations on integers
const Node* process_let(Context* ctx, const Node* node) {
    IrArena* arena = ctx->rewriter.dst_arena;

    const Node* old_instruction = node->payload.let.instruction;
    const Node* instruction = NULL;
    const Node* tail = rewrite_node(&ctx->rewriter, node->payload.let.tail);

    if (old_instruction->tag == PrimOp_TAG) {
        Op op = old_instruction->payload.prim_op.op;
        Nodes old_nodes = old_instruction->payload.prim_op.operands;
        switch(op) {
            case empty_mask_op: {
                const Node* zero = int64_literal(arena, 0);
                instruction = quote(arena, zero);
                break;
            }
            case mask_is_thread_active_op: {
                BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
                const Node* mask = rewrite_node(&ctx->rewriter, old_nodes.nodes[0]);
                const Node* index = rewrite_node(&ctx->rewriter, old_nodes.nodes[1]);
                index = gen_reinterpret_cast(bb, int64_type(arena), index);
                const Node* acc = mask;
                // acc >>= index
                acc = gen_primop_ce(bb, rshift_logical_op, 2, (const Node* []) { acc, index });
                // acc &= 0x1
                acc = gen_primop_ce(bb, and_op, 2, (const Node* []) { acc, int64_literal(arena, 1) });
                // acc == 1
                acc = gen_primop_ce(bb, eq_op, 2, (const Node* []) { acc, int64_literal(arena, 1) });
                return finish_body(bb, let(arena, quote(arena, acc), tail));
            }
            case subgroup_active_mask_op:
                // this is just ballot(true), lower it to that
                old_nodes = nodes(ctx->rewriter.src_arena, 1, (const Node* []) { true_lit(ctx->rewriter.src_arena) });
                SHADY_FALLTHROUGH;
            case subgroup_ballot_op: {
                BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
                const Node* packed_result = gen_primop_e(bb, subgroup_ballot_op, empty(arena), rewrite_nodes(&ctx->rewriter, old_nodes));

                const Node* result = packed_result;
                // we need to extract the packed result ...
                if (arena->config.subgroup_mask_representation == SubgroupMaskSpvKHRBallot) {
                    // extract the 64 bits of mask we care about
                    const Node* lo = gen_primop_ce(bb, extract_op, 2, (const Node* []) {result, int32_literal(arena, 0) });
                    const Node* hi = gen_primop_ce(bb, extract_op, 2, (const Node* []) {result, int32_literal(arena, 1) });
                    result = gen_merge_i32s_i64(bb, lo, hi);
                }
                return finish_body(bb, let(arena, quote(arena, result), tail));
            }
            default: break;
        }
    }

    if (!instruction)
        instruction = rewrite_node(&ctx->rewriter, old_instruction);

    return let(arena, instruction, tail);
}

const Node* process(Context* ctx, const Node* node) {
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    if (node->tag == MaskType_TAG)
        return int64_type(ctx->rewriter.dst_arena);
    else if (node->tag == Let_TAG)
        return process_let(ctx, node);
    else return recreate_node_identity(&ctx->rewriter, node);
}

void lower_mask(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
