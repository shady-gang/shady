#include "shady/pass.h"
#include "shady/ir/memory_layout.h"
#include "shady/ir/type.h"

#include "log.h"
#include "portability.h"

#include <assert.h>

typedef struct {
    Rewriter rewriter;
} Context;

static const Node* process(Context* ctx, const Node* old) {
    IrArena* a = ctx->rewriter.dst_arena;

    switch (old->tag) {
        case SizeOf_TAG: {
            SizeOf payload = old->payload.size_of;
            const Type* t = shd_rewrite_node(&ctx->rewriter, payload.type);
            TypeMemLayout layout = shd_get_mem_layout(a, t);
            return int_literal(a, (IntLiteral) {.width = shd_get_arena_config(a)->target.memory.ptr_size, .is_signed = false, .value = layout.size_in_bytes});
        }
        case AlignOf_TAG: {
            AlignOf payload = old->payload.align_of;
            const Type* t = shd_rewrite_node(&ctx->rewriter, payload.type);
            TypeMemLayout layout = shd_get_mem_layout(a, t);
            return int_literal(a, (IntLiteral) {.width = shd_get_arena_config(a)->target.memory.ptr_size, .is_signed = false, .value = layout.alignment_in_bytes});
        }
        case OffsetOf_TAG: {
            OffsetOf payload = old->payload.offset_of;
            const Type* t = shd_rewrite_node(&ctx->rewriter, payload.type);
            const Node* n = shd_rewrite_node(&ctx->rewriter, payload.idx);
            const IntLiteral* literal = shd_resolve_to_int_literal(n);
            assert(literal);
            t = shd_get_maybe_nominal_type_body(t);
            uint64_t offset_in_bytes = (uint64_t) shd_get_record_field_offset_in_bytes(a, t, literal->value);
            const Node* offset_literal = int_literal(a, (IntLiteral) { .width = shd_get_arena_config(a)->target.memory.ptr_size, .is_signed = false, .value = offset_in_bytes });
            return offset_literal;
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, old);
}

Module* shd_pass_lower_memory_layout(SHADY_UNUSED const CompilerConfig* config, SHADY_UNUSED const void* unused, Module* src) {
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
