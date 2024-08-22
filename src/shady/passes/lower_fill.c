#include "shady/pass.h"

#include "../ir_private.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"

#include "log.h"
#include "portability.h"

typedef struct {
    Rewriter rewriter;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    switch (node->tag) {
        case Fill_TAG: {
            const Type* composite_t = rewrite_node(r, node->payload.fill.type);
            size_t actual_size = get_int_literal_value(*resolve_to_int_literal(get_fill_type_size(composite_t)), false);
            const Node* value = rewrite_node(r, node->payload.fill.value);
            LARRAY(const Node*, copies, actual_size);
            for (size_t i = 0; i < actual_size; i++) {
                copies[i] = value;
            }
            return composite_helper(a, composite_t, nodes(a, actual_size, copies));
        }
        default: break;
    }

    return recreate_node_identity(r, node);
}

Module* lower_fill(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
