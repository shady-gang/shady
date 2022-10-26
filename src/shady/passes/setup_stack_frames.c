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
    if (node->tag == PrimOp_TAG && node->payload.prim_op.op == alloca_op) {
        // Lower to a slot
        const Type* elem_type = rewrite_node(&vctx->context->rewriter, node->payload.prim_op.operands.nodes[0]);
        const Node* slot = gen_primop_ce(vctx->builder, alloca_slot_op, 2, (const Node* []) { elem_type, vctx->context->entry_sp_val });
        debug_node(node);
        debug_print("%zu \n", node);
        // make it so that we will rewrite the `alloca` to the slot
        register_processed(&vctx->context->rewriter, node, quote(vctx->context->rewriter.dst_arena, slot));
        return;
    }

    visit_children(&vctx->visitor, node);
}

static const Node* process(Context* ctx, const Node* node) {
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    IrArena* arena = ctx->rewriter.dst_arena;
    switch (node->tag) {
        case Lambda_TAG: {
            Node* fun = recreate_decl_header_identity(&ctx->rewriter, node);
            Context ctx2 = *ctx;
            if (node->payload.lam.tier == FnTier_Function) {
                ctx2.disable_lowering = lookup_annotation_with_string_payload(node, "DisablePass", "setup_stack_frames");

                BodyBuilder* bb = begin_body(arena);
                ctx->entry_sp_val = gen_primop_ce(bb, get_stack_pointer_op, 0, NULL);
                VContext vctx = {
                    .visitor = {
                        .visit_fn = (VisitFn) collect_allocas,
                        .visit_fn_scope_rpo = true,
                    },
                    .context = ctx,
                    .builder = bb,
                };
                visit_children(&vctx.visitor, node->payload.lam.body);

                fun->payload.lam.body = finish_body(bb, rewrite_node(&ctx->rewriter, node->payload.lam.body));
            } else
                fun->payload.lam.body = process(&ctx2, node->payload.lam.body);
            return fun;
        }
        case Return_TAG: {
            assert(ctx->entry_sp_val);
            BodyBuilder* bb = begin_body(arena);
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
