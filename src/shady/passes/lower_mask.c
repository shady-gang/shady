#include "shady/ir.h"

#include "dict.h"
#include "log.h"
#include "portability.h"

#include "../arena.h"
#include "../rewrite.h"
#include "../transform/ir_gen_helpers.h"

typedef struct {
    Rewriter rewriter;
} Context;

/// Removes most instructions that deal with masks and lower them to bitwise operations on integers
const Node* process_block(Context* ctx, const Node* old_block) {
    IrArena* dst_arena = ctx->rewriter.dst_arena;
    BlockBuilder* bb = begin_block(dst_arena);

    Nodes old_instructions = old_block->payload.block.instructions;
    for (size_t i = 0; i < old_instructions.count; i++) {
        const Node* old_instruction = old_instructions.nodes[i];
        const Node* old_actual_instruction = old_instruction;
        if (old_instruction->tag == Let_TAG)
            old_actual_instruction = old_instruction->payload.let.instruction;

        if (old_actual_instruction->tag == PrimOp_TAG) {
            Op op = old_actual_instruction->payload.prim_op.op;
            Nodes old_nodes = old_actual_instruction->payload.prim_op.operands;
            switch(op) {
                case empty_mask_op: {
                    const Node* zero = int64_literal(dst_arena, 0);
                    register_processed(&ctx->rewriter, old_instruction->payload.let.variables.nodes[0], zero);
                    continue;
                }
                case mask_is_thread_active_op: {
                    const Node* mask = rewrite_node(&ctx->rewriter, old_nodes.nodes[0]);
                    const Node* index = rewrite_node(&ctx->rewriter, old_nodes.nodes[1]);
                    index = gen_primop_ce(bb, reinterpret_op, 2, (const Node* []) { int64_type(dst_arena), index });
                    const Node* acc = mask;
                    // acc >>= index
                    acc = gen_primop_ce(bb, rshift_logical_op, 2, (const Node* []) { acc, index });
                    // acc &= 0x1
                    acc = gen_primop_ce(bb, and_op, 2, (const Node* []) { acc, int64_literal(dst_arena, 1) });
                    // acc == 1
                    acc = gen_primop_ce(bb, eq_op, 2, (const Node* []) { acc, int64_literal(dst_arena, 1) });
                    register_processed(&ctx->rewriter, old_instruction->payload.let.variables.nodes[0], acc);
                    continue;
                }
                case subgroup_active_mask_op:
                    // this is just ballot(true), lower it to that
                    old_nodes = nodes(ctx->rewriter.src_arena, 1, (const Node* []) {true_lit(ctx->rewriter.src_arena)});
                    SHADY_FALLTHROUGH;
                case subgroup_ballot_op: {
                    if (old_actual_instruction == old_instruction)
                        continue; // This was a dead op anyways

                    const Node* packed_result = gen_primop_e(bb, subgroup_ballot_op, rewrite_nodes(&ctx->rewriter, old_nodes));

                    const Node* result = packed_result;
                    // we need to extract the packed result ...
                    if (dst_arena->config.subgroup_mask_representation == SubgroupMaskSpvKHRBallot) {
                        // extract the 64 bits of mask we care about
                        const Node* lo = gen_primop_ce(bb, extract_op, 2, (const Node* []) {result, int32_literal(dst_arena, 0) });
                        const Node* hi = gen_primop_ce(bb, extract_op, 2, (const Node* []) {result, int32_literal(dst_arena, 1) });
                        result = gen_merge_i32s_i64(bb, lo, hi);
                    }
                    register_processed(&ctx->rewriter, old_instruction->payload.let.variables.nodes[0], result);
                    continue;
                }
                default:
                    break;
            }
        }

        append_block(bb, rewrite_node(&ctx->rewriter, old_instruction));
    }

    return finish_block(bb, rewrite_node(&ctx->rewriter, old_block->payload.block.terminator));
}

const Node* process(Context* ctx, const Node* node) {
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    if (node->tag == MaskType_TAG)
        return int64_type(ctx->rewriter.dst_arena);
    else if (is_declaration(node->tag)) {
        Node* new = recreate_decl_header_identity(&ctx->rewriter, node);
        recreate_decl_body_identity(&ctx->rewriter, node, new);
        return new;
    } else if (node->tag == Block_TAG)
        return process_block(ctx, node);
    else return recreate_node_identity(&ctx->rewriter, node);
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

const Node* lower_mask(SHADY_UNUSED CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);

    Context context = {
        .rewriter = {
            .dst_arena = dst_arena,
            .src_arena = src_arena,
            .processed = done,
            .rewrite_fn = (RewriteFn) process,
        }
    };

    const Node* new = rewrite_node(&context.rewriter, src_program);

    destroy_dict(done);

    return new;
}
