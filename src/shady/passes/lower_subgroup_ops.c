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

static bool is_extended_type(SHADY_UNUSED IrArena* arena, const Type* t, bool allow_vectors) {
    switch (t->tag) {
        case Int_TAG: return true;
        // TODO allow 16-bit floats specifically !
        case Float_TAG: return true;
        case PackType_TAG:
            if (allow_vectors)
                return is_extended_type(arena, t->payload.pack_type.element_type, false);
            return false;
        default: return false;
    }
}

static const Node* process_let(Context* ctx, const Node* old) {
    assert(old->tag == Let_TAG);
    IrArena* arena = ctx->rewriter.dst_arena;
    const Node* tail = rewrite_node(&ctx->rewriter, old->payload.let.tail);
    const Node* old_instruction = old->payload.let.instruction;

    if (old_instruction->tag == PrimOp_TAG) {
        PrimOp payload = old_instruction->payload.prim_op;
        switch (payload.op) {
            case subgroup_broadcast_first_op: {
                BodyBuilder* builder = begin_body(ctx->rewriter.dst_module);
                const Node* varying_value = rewrite_node(&ctx->rewriter, payload.operands.nodes[0]);
                const Type* element_type = get_unqualified_type(varying_value->type);

                if (element_type->tag == Int_TAG && element_type->payload.int_type.width == IntTy32) {
                    cancel_body(builder);
                    break;
                } else if (is_extended_type(arena, element_type, true) && !ctx->config->lower.emulate_subgroup_ops_extended_types) {
                    cancel_body(builder);
                    break;
                }

                TypeMemLayout layout = get_mem_layout(ctx->config, arena, element_type);

                const Type* local_arr_ty = arr_type(arena, (ArrType) { .element_type = int32_type(arena), .size = int32_literal(arena, 2) });

                const Node* varying_top_of_stack = gen_primop_e(builder, get_stack_base_op, empty(arena), empty(arena));
                const Type* varying_raw_ptr_t = ptr_type(arena, (PtrType) { .address_space = AsPrivatePhysical, .pointed_type = local_arr_ty });
                const Node* varying_raw_ptr = gen_reinterpret_cast(builder, varying_raw_ptr_t, varying_top_of_stack);
                const Type* varying_typed_ptr_t = ptr_type(arena, (PtrType) { .address_space = AsPrivatePhysical, .pointed_type = element_type });
                const Node* varying_typed_ptr = gen_reinterpret_cast(builder, varying_typed_ptr_t, varying_top_of_stack);

                const Node* uniform_top_of_stack = gen_primop_e(builder, get_stack_base_uniform_op, empty(arena), empty(arena));
                const Type* uniform_raw_ptr_t = ptr_type(arena, (PtrType) { .address_space = AsSubgroupPhysical, .pointed_type = local_arr_ty });
                const Node* uniform_raw_ptr = gen_reinterpret_cast(builder, uniform_raw_ptr_t, uniform_top_of_stack);
                const Type* uniform_typed_ptr_t = ptr_type(arena, (PtrType) { .address_space = AsSubgroupPhysical, .pointed_type = element_type });
                const Node* uniform_typed_ptr = gen_reinterpret_cast(builder, uniform_typed_ptr_t, uniform_top_of_stack);

                gen_store(builder, varying_typed_ptr, varying_value);
                for (int32_t j = 0; j < bytes_to_i32_cells(layout.size_in_bytes); j++) {
                    const Node* varying_logical_addr = gen_lea(builder, varying_raw_ptr, int32_literal(arena, 0), nodes(arena, 1, (const Node* []) {int32_literal(arena, j) }));
                    const Node* input = gen_load(builder, varying_logical_addr);

                    const Node* partial_result = gen_primop_ce(builder, subgroup_broadcast_first_op, 1, (const Node* []) { input });

                    const Node* uniform_logical_addr = gen_lea(builder, uniform_raw_ptr, int32_literal(arena, 0), nodes(arena, 1, (const Node* []) {int32_literal(arena, j) }));
                    gen_store(builder, uniform_logical_addr, partial_result);
                }
                const Node* result = gen_load(builder, uniform_typed_ptr);
                return finish_body(builder, let(arena, quote_single(arena, result), tail));
            }
            default: break;
        }
    }

    return let(arena, rewrite_node(&ctx->rewriter, old_instruction), tail);
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
