#include "shady/ir.h"

#include "portability.h"
#include "log.h"

#include "../rewrite.h"
#include "../body_builder.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"
#include "../transform/memory_layout.h"

typedef struct {
    Rewriter rewriter;
    CompilerConfig* config;
} Context;

static const Node* process_body(Context* ctx, const Node* old_body) {
    assert(old_body->tag == Body_TAG);
    IrArena* arena = ctx->rewriter.dst_arena;
    BodyBuilder* builder = begin_body(arena);
    Nodes old_instructions = old_body->payload.body.instructions;
    for (size_t i = 0; i < old_instructions.count; i++) {
        const Node* old_instruction = old_instructions.nodes[i];
        const Node* old_actual_instr = old_instruction->tag == Let_TAG ? old_instruction->payload.let.instruction : old_instruction;
        if (old_actual_instr->tag == PrimOp_TAG) {
            PrimOp payload = old_actual_instr->payload.prim_op;
            switch (payload.op) {
                case subgroup_broadcast_first_op: {
                    assert(old_instruction != old_actual_instr);
                    const Node* operand = rewrite_node(&ctx->rewriter, payload.operands.nodes[0]);
                    const Type* operand_type = extract_operand_type(operand->type);

                    if (operand_type->tag == Int_TAG && operand_type->payload.int_type.width == IntTy32)
                        goto identity;
                    else if (!ctx->config->lower.emulate_subgroup_ops_extended_types)
                        goto identity;

                    const Type* local_arr_ty = arr_type(arena, (ArrType) { .element_type = int32_type(arena), .size = int32_literal(arena, 2) });
                    const Node* local_array = gen_primop_ce(builder, alloca_logical_op, 1, (const Node* []) { local_arr_ty });
                    gen_serialisation(builder, operand_type, local_array, int32_literal(arena, 0), operand);

                    for (int32_t j = 0; j < 2; j++) {
                        const Node* logical_addr = gen_lea(builder, local_array, NULL, nodes(arena, 1, (const Node* []) { int32_literal(arena, j) }));
                        const Node* input = gen_load(builder, logical_addr);
                        const Node* partial_result = gen_primop_ce(builder, subgroup_broadcast_first_op, 1, (const Node* []) { input });
                        gen_store(builder, logical_addr, partial_result);
                    }

                    const Node* result = gen_deserialisation(builder, operand_type, local_array, int32_literal(arena, 0));
                    register_processed(&ctx->rewriter, old_instruction->payload.let.variables.nodes[0], result);
                    continue;
                }
                default:
                    break;
            }
        }
        identity:
        append_body(builder, recreate_node_identity(&ctx->rewriter, old_instruction));
    }
    return finish_body(builder, recreate_node_identity(&ctx->rewriter, old_body->payload.body.terminator));
}

static const Node* process(Context* ctx, const Node* node) {
    if (!node) return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    switch (node->tag) {
        case Body_TAG: return process_body(ctx, node);
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

#include "dict.h"

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

const Node* lower_subgroup_ops(SHADY_UNUSED CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    assert(!config->lower.emulate_subgroup_ops && "TODO");

    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
    Context ctx = {
        .rewriter = {
            .src_arena = src_arena,
            .dst_arena = dst_arena,
            .rewrite_fn = (RewriteFn) process,
            .processed = done,
        },
        .config = config
    };

    const Node* rewritten = process(&ctx, src_program);

    destroy_dict(done);
    return rewritten;
}
