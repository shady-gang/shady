#include "shady/pass.h"
#include "shady/memory_layout.h"

#include "../type.h"

#include "log.h"
#include "portability.h"

#include <assert.h>

typedef struct {
    Rewriter rewriter;
} Context;

static const Node* process(Context* ctx, const Node* old) {
    IrArena* a = ctx->rewriter.dst_arena;

    switch (old->tag) {
        case PrimOp_TAG: {
            switch (old->payload.prim_op.op) {
                case size_of_op: {
                    const Type* t = shd_rewrite_node(&ctx->rewriter, shd_first(old->payload.prim_op.type_arguments));
                    TypeMemLayout layout = shd_get_mem_layout(a, t);
                    return int_literal(a, (IntLiteral) {.width = shd_get_arena_config(a)->memory.ptr_size, .is_signed = false, .value = layout.size_in_bytes});
                }
                case align_of_op: {
                    const Type* t = shd_rewrite_node(&ctx->rewriter, shd_first(old->payload.prim_op.type_arguments));
                    TypeMemLayout layout = shd_get_mem_layout(a, t);
                    return int_literal(a, (IntLiteral) {.width = shd_get_arena_config(a)->memory.ptr_size, .is_signed = false, .value = layout.alignment_in_bytes});
                }
                case offset_of_op: {
                    const Type* t = shd_rewrite_node(&ctx->rewriter, shd_first(old->payload.prim_op.type_arguments));
                    const Node* n = shd_rewrite_node(&ctx->rewriter, shd_first(old->payload.prim_op.operands));
                    const IntLiteral* literal = resolve_to_int_literal(n);
                    assert(literal);
                    t = get_maybe_nominal_type_body(t);
                    uint64_t offset_in_bytes = (uint64_t) shd_get_record_field_offset_in_bytes(a, t, literal->value);
                    const Node* offset_literal = int_literal(a, (IntLiteral) { .width = shd_get_arena_config(a)->memory.ptr_size, .is_signed = false, .value = offset_in_bytes });
                    return offset_literal;
                }
                default: break;
            }
            break;
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, old);
}

Module* lower_memory_layout(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(get_module_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process)
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
