#include "passes.h"

#include "portability.h"
#include "log.h"

#include "../rewrite.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"
#include "../transform/memory_layout.h"

typedef struct {
    Rewriter rewriter;
    CompilerConfig* config;
} Context;

static const Node* process_let(Context* ctx, const Node* old) {
    assert(old->tag == Let_TAG);
    IrArena* arena = ctx->rewriter.dst_arena;
    const Node* tail = rewrite_node(&ctx->rewriter, old->payload.let.tail);
    const Node* old_instruction = old->payload.let.instruction;

    if (old_instruction->tag == PrimOp_TAG) {
        PrimOp payload = old_instruction->payload.prim_op;
        switch (payload.op) {
            case subgroup_broadcast_first_op: {
                BodyBuilder* builder = begin_body(arena);
                const Node* operand = rewrite_node(&ctx->rewriter, payload.operands.nodes[0]);
                const Type* operand_type = extract_operand_type(operand->type);

                if (operand_type->tag == Int_TAG && operand_type->payload.int_type.width == IntTy32)
                    break;
                else if (!ctx->config->lower.emulate_subgroup_ops_extended_types)
                    break;

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
                return finish_body(builder, let(arena, false, quote(arena, result), tail));
            }
            default: break;
        }
    }

    return let(arena, false, rewrite_node(&ctx->rewriter, old_instruction), tail);
}

static const Node* process(Context* ctx, const Node* node) {
    if (!node) return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    switch (node->tag) {
        case Let_TAG: return process_let(ctx, node);
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

void lower_subgroup_ops(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    assert(!config->lower.emulate_subgroup_ops && "TODO");
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .config = config
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
