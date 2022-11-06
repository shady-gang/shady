#include "passes.h"

#include "../rewrite.h"
#include "../visit.h"
#include "../type.h"
#include "log.h"
#include "portability.h"

#include "../transform/ir_gen_helpers.h"

#include "list.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    bool disable_lowering;

    const Node* entry_sp_val;
} Context;

typedef struct {
    Visitor visitor;
    Context* context;
    BodyBuilder* builder;
} VContext;

static void collect_allocas(VContext* vctx, const Node* node) {
    IrArena* arena = vctx->context->rewriter.dst_arena;
    if (node->tag == PrimOp_TAG && node->payload.prim_op.op == alloca_op) {
        // Lower to a slot
        const Type* elem_type = rewrite_node(&vctx->context->rewriter, node->payload.prim_op.type_arguments.nodes[0]);
        const Node* slot = bind_instruction(vctx->builder, prim_op(arena, (PrimOp) {
            .op = alloca_slot_op,
            .type_arguments = nodes(arena, 1, (const Node* []) { elem_type }),
            .operands = nodes(arena, 1, (const Node* []) { vctx->context->entry_sp_val }) })).nodes[0];
        debug_node(node);
        debug_print("%zu \n", node);
        // make it so that we will rewrite the `alloca` to the slot
        register_processed(&vctx->context->rewriter, node, quote(arena, slot));
        return;
    }

    visit_children(&vctx->visitor, node);
}

static const Node* process(Context* ctx, const Node* node) {
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    IrArena* arena = ctx->rewriter.dst_arena;
    switch (node->tag) {
        case Function_TAG: {
            Node* fun = recreate_decl_header_identity(&ctx->rewriter, node);
            Context ctx2 = *ctx;
            ctx2.disable_lowering = lookup_annotation_with_string_payload(node, "DisablePass", "setup_stack_frames");

            BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
            ctx2.entry_sp_val = gen_primop_ce(bb, get_stack_pointer_op, 0, NULL);
            VContext vctx = {
                .visitor = {
                    .visit_fn = (VisitFn) collect_allocas,
                    .visit_fn_scope_rpo = true,
                },
                .context = &ctx2,
                .builder = bb,
            };
            visit_children(&vctx.visitor, node->payload.fun.body);
            fun->payload.fun.body = finish_body(bb, rewrite_node(&ctx2.rewriter, node->payload.fun.body));
            return fun;
        }
        case Return_TAG: {
            assert(ctx->entry_sp_val);
            BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
            // Restore SP before calling exit
            bind_instruction(bb, prim_op(arena, (PrimOp) {
                .op = set_stack_pointer_op,
                .operands = nodes(arena, 1, (const Node* []) { ctx->entry_sp_val })
            }));
            return finish_body(bb, recreate_node_identity(&ctx->rewriter, node));
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

void  setup_stack_frames(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
