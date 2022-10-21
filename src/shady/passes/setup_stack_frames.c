#include "shady/ir.h"

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

    const Node* old_entry_body;
    const Node* entry_sp_val;

    struct List* new_decls;
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
                ctx2.old_entry_body = node->payload.lam.body;
                ctx2.entry_sp_val = NULL;
                fun->payload.lam.body = process(&ctx2, node->payload.lam.body);
            }
            return fun;
        }
        case Let_TAG: {
            if (ctx->disable_lowering)
                return recreate_node_identity(&ctx->rewriter, node);

            // If we are the entry block to a function, we need to visit the entire thing
            // and handle all the allocas inside it
            if (node == ctx->old_entry_body) {
                BodyBuilder* bb = begin_body(arena);
                assert(!ctx->entry_sp_val);
                ctx->entry_sp_val = gen_primop_ce(bb, get_stack_pointer_op, 0, NULL);

                VContext vctx = {
                    .visitor = {
                        .visit_fn = (VisitFn) collect_allocas,
                        .visit_fn_scope_rpo = true,
                    },
                    .context = ctx,
                    .builder = bb,
                };

                visit_children(&vctx.visitor, node);
                return finish_body(bb, recreate_node_identity(&ctx->rewriter, node));
            }
            assert(ctx->entry_sp_val);
            return recreate_node_identity(&ctx->rewriter, node);
        }
        case Return_TAG: {
            // Restore SP before calling exit
            const Node* restore_sp = prim_op(arena, (PrimOp) {
                .op = set_stack_pointer_op,
                .operands = nodes(arena, 1, (const Node* []) { ctx->entry_sp_val })
            });
            return let(arena, false, restore_sp, recreate_node_identity(&ctx->rewriter, node));
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

const Node* setup_stack_frames(SHADY_UNUSED CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct List* new_decls_list = new_list(const Node*);

    Context ctx = {
        .rewriter = create_rewriter(src_arena, dst_arena, (RewriteFn) process),
        .new_decls = new_decls_list,
    };

    const Node* rewritten = recreate_node_identity(&ctx.rewriter, src_program);
    Nodes new_decls = rewritten->payload.root.declarations;
    for (size_t i = 0; i < entries_count_list(new_decls_list); i++) {
        new_decls = append_nodes(dst_arena, new_decls, read_list(const Node*, new_decls_list)[i]);
    }
    rewritten = root(dst_arena, (Root) {
        .declarations = new_decls
    });

    destroy_list(new_decls_list);
    destroy_rewriter(&ctx.rewriter);
    return rewritten;
}
