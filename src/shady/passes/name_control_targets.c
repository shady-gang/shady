#include "passes.h"

#include "log.h"
#include "portability.h"
#include "../type.h"
#include "../rewrite.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    bool disable_lowering;
    Node* current_fn;
} Context;

static const Node* process_node(Context* ctx, const Node* node);

static const Node* process_let(Context* ctx, const Node* node) {
    assert(node->tag == Let_TAG);
    IrArena* arena = ctx->rewriter.dst_arena;

    const Node* old_instruction = node->payload.let.instruction;
    const Node* new_instruction = NULL;
    const Node* old_tail = node->payload.let.tail;
    const Node* new_tail = NULL;

    switch (old_instruction->tag) {
        default:
            new_instruction = rewrite_node(&ctx->rewriter, old_instruction);
            break;
    }

    if (new_instruction->tag == Control_TAG || new_instruction->tag == IndirectCall_TAG) {
        Nodes oparams = get_abstraction_params(old_tail);
        Nodes nparams = recreate_variables(&ctx->rewriter, oparams);
        register_processed_list(&ctx->rewriter, oparams, nparams);
        Node* new_tail = basic_block(ctx->rewriter.dst_arena, ctx->current_fn, nparams, unique_name(ctx->rewriter.dst_arena, "control_join"));
        new_tail->payload.basic_block.body = rewrite_node(&ctx->rewriter, get_abstraction_body(old_tail));
        return let_into(arena, new_instruction, new_tail);
    }

    if (!new_tail)
        new_tail = rewrite_node(&ctx->rewriter, old_tail);

    assert(new_instruction && new_tail);
    return let(arena, new_instruction, new_tail);
}

static const Node* process_node(Context* ctx, const Node* node) {
    const Node* already_done = search_processed(&ctx->rewriter, node);
    if (already_done)
        return already_done;

    IrArena* dst_arena = ctx->rewriter.dst_arena;

    if (node->tag == Function_TAG) {
        Node* fun = recreate_decl_header_identity(&ctx->rewriter, node);
        Context sub_ctx = *ctx;
        sub_ctx.disable_lowering = lookup_annotation_with_string_payload(fun, "DisablePass", "name_control_targets");
        sub_ctx.current_fn = fun;
        fun->payload.fun.body = rewrite_node(&sub_ctx.rewriter, node->payload.fun.body);
        return fun;
    }

    if (ctx->disable_lowering)
        return recreate_node_identity(&ctx->rewriter, node);

    switch (node->tag) {
        case Let_TAG: return process_let(ctx, node);
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

void name_control_targets(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process_node),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
