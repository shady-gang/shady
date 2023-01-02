#include "passes.h"

#include "log.h"
#include "portability.h"

#include "../ir_private.h"
#include "../type.h"
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

    const Type* mask_type = get_actual_mask_type(ctx->rewriter.dst_arena);
    assert(mask_type->tag == Int_TAG);

    const Node* zero = int_literal(arena, (IntLiteral) { .width = mask_type->payload.int_type.width, .value = 0 });
    const Node* one = int_literal(arena, (IntLiteral) { .width = mask_type->payload.int_type.width, .value = 1 });

    if (old_instruction->tag == PrimOp_TAG) {
        Op op = old_instruction->payload.prim_op.op;
        Nodes old_nodes = old_instruction->payload.prim_op.operands;
        switch(op) {
            case empty_mask_op: {
                instruction = quote_single(arena, zero);
                break;
            }
            case mask_is_thread_active_op: {
                BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
                const Node* mask = rewrite_node(&ctx->rewriter, old_nodes.nodes[0]);
                const Node* index = rewrite_node(&ctx->rewriter, old_nodes.nodes[1]);
                index = gen_reinterpret_cast(bb, get_actual_mask_type(ctx->rewriter.dst_arena), index);
                const Node* acc = mask;
                // acc >>= index
                acc = gen_primop_ce(bb, rshift_logical_op, 2, (const Node* []) { acc, index });
                // acc &= 0x1
                acc = gen_primop_ce(bb, and_op, 2, (const Node* []) { acc, one });
                // acc == 1
                acc = gen_primop_ce(bb, eq_op, 2, (const Node* []) { acc, one });
                return finish_body(bb, let(arena, quote_single(arena, acc), tail));
            }
            case subgroup_active_mask_op:
                // this is just ballot(true), lower it to that
                instruction = prim_op(arena, (PrimOp) { .op = subgroup_ballot_op, .type_arguments = empty(arena), .operands = singleton(true_lit(ctx->rewriter.dst_arena)) });
                break;
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
        return get_actual_mask_type(ctx->rewriter.dst_arena);
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
