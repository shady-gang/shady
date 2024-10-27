#include "shady/pass.h"
#include "shady/ir/cast.h"
#include "shady/ir/memory_layout.h"

#include "ir_private.h"

#include "log.h"
#include "portability.h"

#include <assert.h>

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
} Context;

static const Node* process(Context* ctx, const Node* old) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    Module* m = r->dst_module;

    switch (old->tag) {
        case CopyBytes_TAG: {
            CopyBytes payload = old->payload.copy_bytes;
            const Type* word_type = int_type(a, (Int) { .is_signed = false, .width = a->config.memory.word_size });

            BodyBuilder* bb = shd_bld_begin_pseudo_instr(a, shd_rewrite_node(r, payload.mem));

            const Node* dst_addr = shd_rewrite_node(&ctx->rewriter, payload.dst);
            const Type* dst_addr_type = dst_addr->type;
            shd_deconstruct_qualified_type(&dst_addr_type);
            assert(dst_addr_type->tag == PtrType_TAG);
            dst_addr_type = ptr_type(a, (PtrType) {
                .address_space = dst_addr_type->payload.ptr_type.address_space,
                .pointed_type = word_type,
            });
            dst_addr = shd_bld_reinterpret_cast(bb, dst_addr_type, dst_addr);

            const Node* src_addr = shd_rewrite_node(&ctx->rewriter, payload.src);
            const Type* src_addr_type = src_addr->type;
            shd_deconstruct_qualified_type(&src_addr_type);
            assert(src_addr_type->tag == PtrType_TAG);
            src_addr_type = ptr_type(a, (PtrType) {
                .address_space = src_addr_type->payload.ptr_type.address_space,
                .pointed_type = word_type,
            });
            src_addr = shd_bld_reinterpret_cast(bb, src_addr_type, src_addr);

            const Node* num_in_bytes = shd_bld_convert_int_extend_according_to_dst_t(bb, size_t_type(a), shd_rewrite_node(&ctx->rewriter, payload.count));
            const Node* num_in_words = shd_bld_conversion(bb, shd_uint32_type(a), shd_bytes_to_words(bb, num_in_bytes));

            begin_loop_helper_t l = shd_bld_begin_loop_helper(bb, shd_empty(a), shd_singleton(shd_uint32_type(a)), shd_singleton(shd_uint32_literal(a, 0)));

            const Node* index = shd_first(l.params);
            shd_set_value_name(index, "memcpy_i");
            Node* loop_case = l.loop_body;
            BodyBuilder* loop_bb = shd_bld_begin(a, shd_get_abstraction_mem(loop_case));
            const Node* loaded_word = shd_bld_load(loop_bb, lea_helper(a, src_addr, index, shd_empty(a)));
            shd_bld_store(loop_bb, lea_helper(a, dst_addr, index, shd_empty(a)), loaded_word);
            const Node* next_index = prim_op_helper(a, add_op, shd_empty(a), mk_nodes(a, index, shd_uint32_literal(a, 1)));

            Node* true_case = case_(a, shd_empty(a));
            shd_set_abstraction_body(true_case, join(a, (Join) { .join_point = l.continue_jp, .mem = shd_get_abstraction_mem(true_case), .args = shd_singleton(next_index) }));
            Node* false_case = case_(a, shd_empty(a));
            shd_set_abstraction_body(false_case, join(a, (Join) { .join_point = l.break_jp, .mem = shd_get_abstraction_mem(false_case), .args = shd_empty(a) }));

            shd_set_abstraction_body(loop_case, shd_bld_finish(loop_bb, branch(a, (Branch) {
                .mem = shd_bb_mem(loop_bb),
                .condition = prim_op_helper(a, lt_op, shd_empty(a), mk_nodes(a, next_index, num_in_words)),
                .true_jump = jump_helper(a, shd_bb_mem(loop_bb), true_case, shd_empty(a)),
                .false_jump = jump_helper(a, shd_bb_mem(loop_bb), false_case, shd_empty(a)),
            })));

            return shd_bld_to_instr_yield_values(bb, shd_empty(a));
        }
        case FillBytes_TAG: {
            FillBytes payload = old->payload.fill_bytes;
            const Node* src_value = shd_rewrite_node(&ctx->rewriter, payload.src);
            const Type* src_type = src_value->type;
            shd_deconstruct_qualified_type(&src_type);
            assert(src_type->tag == Int_TAG);
            const Type* word_type = src_type;// int_type(a, (Int) { .is_signed = false, .width = a->config.memory.word_size });

            BodyBuilder* bb = shd_bld_begin_pseudo_instr(a, shd_rewrite_node(r, payload.mem));

            const Node* dst_addr = shd_rewrite_node(&ctx->rewriter, payload.dst);
            const Type* dst_addr_type = dst_addr->type;
            shd_deconstruct_qualified_type(&dst_addr_type);
            assert(dst_addr_type->tag == PtrType_TAG);
            dst_addr_type = ptr_type(a, (PtrType) {
                .address_space = dst_addr_type->payload.ptr_type.address_space,
                .pointed_type = word_type,
            });
            dst_addr = shd_bld_reinterpret_cast(bb, dst_addr_type, dst_addr);

            const Node* num = shd_rewrite_node(&ctx->rewriter, payload.count);
            const Node* num_in_words = shd_bld_conversion(bb, shd_uint32_type(a), shd_bytes_to_words(bb, num));

            begin_loop_helper_t l = shd_bld_begin_loop_helper(bb, shd_empty(a), shd_singleton(shd_uint32_type(a)), shd_singleton(shd_uint32_literal(a, 0)));

            const Node* index = shd_first(l.params);
            shd_set_value_name(index, "memset_i");
            Node* loop_case = l.loop_body;
            BodyBuilder* loop_bb = shd_bld_begin(a, shd_get_abstraction_mem(loop_case));
            shd_bld_store(loop_bb, lea_helper(a, dst_addr, index, shd_empty(a)), src_value);
            const Node* next_index = prim_op_helper(a, add_op, shd_empty(a), mk_nodes(a, index, shd_uint32_literal(a, 1)));

            Node* true_case = case_(a, shd_empty(a));
            shd_set_abstraction_body(true_case, join(a, (Join) { .join_point = l.continue_jp, .mem = shd_get_abstraction_mem(true_case), .args = shd_singleton(next_index) }));
            Node* false_case = case_(a, shd_empty(a));
            shd_set_abstraction_body(false_case, join(a, (Join) { .join_point = l.break_jp, .mem = shd_get_abstraction_mem(false_case), .args = shd_empty(a) }));

            shd_set_abstraction_body(loop_case, shd_bld_finish(loop_bb, branch(a, (Branch) {
                .mem = shd_bb_mem(loop_bb),
                .condition = prim_op_helper(a, lt_op, shd_empty(a), mk_nodes(a, next_index, num_in_words)),
                .true_jump = jump_helper(a, shd_bb_mem(loop_bb), true_case, shd_empty(a)),
                .false_jump = jump_helper(a, shd_bb_mem(loop_bb), false_case, shd_empty(a)),
            })));
            return shd_bld_to_instr_yield_values(bb, shd_empty(a));
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, old);
}

Module* shd_pass_lower_memcpy(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
            .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process)
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
