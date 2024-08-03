#include "pass.h"

#include "../transform/ir_gen_helpers.h"
#include "../transform/memory_layout.h"
#include "../type.h"
#include "../ir_private.h"

#include "log.h"
#include "portability.h"

#include <assert.h>

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
} Context;

static const Node* process(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

    IrArena* a = ctx->rewriter.dst_arena;
    Module* m = ctx->rewriter.dst_module;

    switch (old->tag) {
        case CopyBytes_TAG: {
            CopyBytes payload = old->payload.copy_bytes;
            const Type* word_type = int_type(a, (Int) { .is_signed = false, .width = a->config.memory.word_size });

            BodyBuilder* bb = begin_body(a);

            const Node* dst_addr = rewrite_node(&ctx->rewriter, payload.dst);
            const Type* dst_addr_type = dst_addr->type;
            deconstruct_qualified_type(&dst_addr_type);
            assert(dst_addr_type->tag == PtrType_TAG);
            dst_addr_type = ptr_type(a, (PtrType) {
                    .address_space = dst_addr_type->payload.ptr_type.address_space,
                    .pointed_type = word_type,
            });
            dst_addr = gen_reinterpret_cast(bb, dst_addr_type, dst_addr);

            const Node* src_addr = rewrite_node(&ctx->rewriter, payload.src);
            const Type* src_addr_type = src_addr->type;
            deconstruct_qualified_type(&src_addr_type);
            assert(src_addr_type->tag == PtrType_TAG);
            src_addr_type = ptr_type(a, (PtrType) {
                    .address_space = src_addr_type->payload.ptr_type.address_space,
                    .pointed_type = word_type,
            });
            src_addr = gen_reinterpret_cast(bb, src_addr_type, src_addr);

            const Node* num_in_bytes = convert_int_extend_according_to_dst_t(bb, size_t_type(a), rewrite_node(&ctx->rewriter, payload.count));
            const Node* num_in_words = gen_conversion(bb, uint32_type(a), bytes_to_words(bb, num_in_bytes));

            const Node* index = param(a, qualified_type_helper(uint32_type(a), false), "memcpy_i");
            BodyBuilder* loop_bb = begin_body(a);
            const Node* loaded_word = gen_load(loop_bb, gen_lea(loop_bb, src_addr, index, empty(a)));
            gen_store(loop_bb, gen_lea(loop_bb, dst_addr, index, empty(a)), loaded_word);
            const Node* next_index = gen_primop_e(loop_bb, add_op, empty(a), mk_nodes(a, index, uint32_literal(a, 1)));
            Node* true_case = case_(a, empty(a));
            set_abstraction_body(true_case, merge_continue(a, (MergeContinue) {.args = singleton(next_index)}));
            Node* false_case = case_(a, empty(a));
            set_abstraction_body(false_case, merge_break(a, (MergeBreak) {.args = empty(a)}));
            gen_if(loop_bb, empty(a), gen_primop_e(loop_bb, lt_op, empty(a), mk_nodes(a, next_index, num_in_words)), true_case, false_case);
            Node* loop_case = case_(a, singleton(index));
            set_abstraction_body(loop_case, finish_body(loop_bb, unreachable(a)));
            gen_loop(bb, empty(a), singleton(uint32_literal(a, 0)), loop_case);
            return yield_values_and_wrap_in_block(bb, empty(a));
        }
        case FillBytes_TAG: {
            FillBytes payload = old->payload.fill_bytes;
            const Node* src_value = rewrite_node(&ctx->rewriter, payload.src);
            const Type* src_type = src_value->type;
            deconstruct_qualified_type(&src_type);
            assert(src_type->tag == Int_TAG);
            const Type* word_type = src_type;// int_type(a, (Int) { .is_signed = false, .width = a->config.memory.word_size });

            BodyBuilder* bb = begin_body(a);

            const Node* dst_addr = rewrite_node(&ctx->rewriter, payload.dst);
            const Type* dst_addr_type = dst_addr->type;
            deconstruct_qualified_type(&dst_addr_type);
            assert(dst_addr_type->tag == PtrType_TAG);
            dst_addr_type = ptr_type(a, (PtrType) {
                    .address_space = dst_addr_type->payload.ptr_type.address_space,
                    .pointed_type = word_type,
            });
            dst_addr = gen_reinterpret_cast(bb, dst_addr_type, dst_addr);

            const Node* num = rewrite_node(&ctx->rewriter, payload.count);
            const Node* num_in_bytes = gen_conversion(bb, uint32_type(a), bytes_to_words(bb, num));

            const Node* index = param(a, qualified_type_helper(uint32_type(a), false), "memset_i");
            BodyBuilder* loop_bb = begin_body(a);
            gen_store(loop_bb, gen_lea(loop_bb, dst_addr, index, empty(a)), src_value);
            const Node* next_index = gen_primop_e(loop_bb, add_op, empty(a), mk_nodes(a, index, uint32_literal(a, 1)));
            Node* true_case = case_(a, empty(a));
            set_abstraction_body(true_case, merge_continue(a, (MergeContinue) {.args = singleton(next_index)}));
            Node* false_case = case_(a, empty(a));
            set_abstraction_body(false_case, merge_break(a, (MergeBreak) {.args = empty(a)}));
            gen_if(loop_bb, empty(a), gen_primop_e(loop_bb, lt_op, empty(a), mk_nodes(a, next_index, num_in_bytes)), true_case, false_case);
            Node* loop_case = case_(a, singleton(index));
            set_abstraction_body(loop_case, finish_body(loop_bb, unreachable(a)));
            gen_loop(bb, empty(a), singleton(uint32_literal(a, 0)), loop_case);
            return yield_values_and_wrap_in_block(bb, empty(a));
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, old);
}

Module* lower_memcpy(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));

    Context ctx = {
            .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process)
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
