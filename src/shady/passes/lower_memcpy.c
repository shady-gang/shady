#include "passes.h"

#include "../transform/ir_gen_helpers.h"
#include "../rewrite.h"
#include "../type.h"
#include "../ir_private.h"

#include "log.h"
#include "portability.h"

#include <assert.h>

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
} Context;

static const Node* size_t_literal(Context* ctx, uint64_t value) {
    IrArena* a = ctx->rewriter.dst_arena;
    return int_literal(a, (IntLiteral) { .width = a->config.memory.ptr_size, .is_signed = false, .value = value });
}

// TODO assumes alignment
static const Node* bytes_to_words(Context* ctx, BodyBuilder* bb, const Node* bytes) {
    IrArena* a = bb->arena;
    const Type* word_type = int_type(a, (Int) { .width = a->config.memory.word_size, .is_signed = false });
    size_t word_width = get_type_bitwidth(word_type);
    const Node* bytes_per_word = size_t_literal(ctx, word_width / 8);
    return gen_primop_e(bb, div_op, empty(a), mk_nodes(a, bytes, bytes_per_word));
}

static const Node* process(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

    IrArena* a = ctx->rewriter.dst_arena;
    Module* m = ctx->rewriter.dst_module;

    switch (old->tag) {
        case PrimOp_TAG: {
            switch (old->payload.prim_op.op) {
                case memcpy_op: {
                    const Type* word_type = int_type(a, (Int) { .is_signed = false, .width = a->config.memory.word_size });

                    BodyBuilder* bb = begin_body(a);
                    Nodes old_ops = old->payload.prim_op.operands;

                    const Node* dst_addr = rewrite_node(&ctx->rewriter, old_ops.nodes[0]);
                    const Type* dst_addr_type = dst_addr->type;
                    deconstruct_qualified_type(&dst_addr_type);
                    assert(dst_addr_type->tag == PtrType_TAG);
                    dst_addr_type = ptr_type(a, (PtrType) {
                        .address_space = dst_addr_type->payload.ptr_type.address_space,
                        .pointed_type = arr_type(a, (ArrType) { .element_type = word_type, .size = NULL }),
                    });
                    dst_addr = gen_reinterpret_cast(bb, dst_addr_type, dst_addr);

                    const Node* src_addr = rewrite_node(&ctx->rewriter, old_ops.nodes[1]);
                    const Type* src_addr_type = src_addr->type;
                    deconstruct_qualified_type(&src_addr_type);
                    assert(src_addr_type->tag == PtrType_TAG);
                    src_addr_type = ptr_type(a, (PtrType) {
                            .address_space = src_addr_type->payload.ptr_type.address_space,
                            .pointed_type = arr_type(a, (ArrType) { .element_type = word_type, .size = NULL }),
                    });
                    src_addr = gen_reinterpret_cast(bb, src_addr_type, src_addr);

                    const Node* num = rewrite_node(&ctx->rewriter, old_ops.nodes[2]);
                    const Node* num_in_bytes = gen_conversion(bb, uint32_type(a), bytes_to_words(ctx, bb, num));

                    const Node* index = var(a, qualified_type_helper(uint32_type(a), false), "memcpy_i");
                    BodyBuilder* loop_bb = begin_body(a);
                    const Node* loaded_word = gen_load(loop_bb, gen_lea(loop_bb, src_addr, index, singleton(uint32_literal(a, 0))));
                    gen_store(loop_bb, gen_lea(loop_bb, dst_addr, index, singleton(uint32_literal(a, 0))), loaded_word);
                    bind_instruction(loop_bb, if_instr(a, (If) {
                        .condition = gen_primop_e(loop_bb, lt_op, empty(a), mk_nodes(a, index, num_in_bytes)),
                        .yield_types = empty(a),
                        .if_true = case_(a, empty(a), merge_continue(a, (MergeContinue) {.args = singleton(gen_primop_e(loop_bb, add_op, empty(a), mk_nodes(a, index, uint32_literal(a, 1))))})),
                        .if_false = case_(a, empty(a), merge_break(a, (MergeBreak) {.args = empty(a)}))
                    }));

                    bind_instruction(bb, loop_instr(a, (Loop) {
                        .yield_types = empty(a),
                        .body = case_(a, singleton(index), finish_body(loop_bb, unreachable(a))),
                        .initial_args = singleton(uint32_literal(a, 0))
                    }));
                    return yield_values_and_wrap_in_block(bb, empty(a));
                }
                case memset_op: {
                    Nodes old_ops = old->payload.prim_op.operands;
                    const Node* src_value = rewrite_node(&ctx->rewriter, old_ops.nodes[1]);
                    const Type* src_type = src_value->type;
                    deconstruct_qualified_type(&src_type);
                    assert(src_type->tag == Int_TAG);
                    const Type* word_type = src_type;// int_type(a, (Int) { .is_signed = false, .width = a->config.memory.word_size });

                    BodyBuilder* bb = begin_body(a);

                    const Node* dst_addr = rewrite_node(&ctx->rewriter, old_ops.nodes[0]);
                    const Type* dst_addr_type = dst_addr->type;
                    deconstruct_qualified_type(&dst_addr_type);
                    assert(dst_addr_type->tag == PtrType_TAG);
                    dst_addr_type = ptr_type(a, (PtrType) {
                        .address_space = dst_addr_type->payload.ptr_type.address_space,
                        .pointed_type = arr_type(a, (ArrType) { .element_type = word_type, .size = NULL }),
                    });
                    dst_addr = gen_reinterpret_cast(bb, dst_addr_type, dst_addr);

                    const Node* num = rewrite_node(&ctx->rewriter, old_ops.nodes[2]);
                    const Node* num_in_bytes = gen_conversion(bb, uint32_type(a), bytes_to_words(ctx, bb, num));

                    const Node* index = var(a, qualified_type_helper(uint32_type(a), false), "memset_i");
                    BodyBuilder* loop_bb = begin_body(a);
                    gen_store(loop_bb, gen_lea(loop_bb, dst_addr, index, singleton(uint32_literal(a, 0))), src_value);
                    bind_instruction(loop_bb, if_instr(a, (If) {
                        .condition = gen_primop_e(loop_bb, lt_op, empty(a), mk_nodes(a, index, num_in_bytes)),
                        .yield_types = empty(a),
                        .if_true = case_(a, empty(a), merge_continue(a, (MergeContinue) {.args = singleton(gen_primop_e(loop_bb, add_op, empty(a), mk_nodes(a, index, uint32_literal(a, 1))))})),
                        .if_false = case_(a, empty(a), merge_break(a, (MergeBreak) {.args = empty(a)}))
                    }));

                    bind_instruction(bb, loop_instr(a, (Loop) {
                        .yield_types = empty(a),
                        .body = case_(a, singleton(index), finish_body(loop_bb, unreachable(a))),
                        .initial_args = singleton(uint32_literal(a, 0))
                    }));
                    return yield_values_and_wrap_in_block(bb, empty(a));
                }
                default: break;
            }
            break;
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, old);
}

Module* lower_memcpy(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));

    Context ctx = {
            .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process)
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
