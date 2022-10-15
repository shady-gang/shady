#include "shady/ir.h"

#include "../rewrite.h"
#include "../visit.h"
#include "../type.h"
#include "log.h"
#include "portability.h"

#include "../transform/ir_gen_helpers.h"

#include "list.h"
#include "dict.h"

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

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

static void collect_allocas(VContext* vctx, const Node* node) {
    if (is_instruction(node)) {
        const Node* actual_old_instr = node;
        if (actual_old_instr->tag == Let_TAG)
            actual_old_instr = actual_old_instr->payload.let.instruction;
        if (actual_old_instr->tag == PrimOp_TAG && actual_old_instr->payload.prim_op.op == alloca_op) {
            // Ignore non-let bound allocas
            if (actual_old_instr == node)
                goto no_op;

            // Lower to a slot
            const Type* elem_type = rewrite_node(&vctx->context->rewriter, actual_old_instr->payload.prim_op.operands.nodes[0]);
            const Node* slot = gen_primop_ce(vctx->builder, alloca_slot_op, 2, (const Node* []) { elem_type, vctx->context->entry_sp_val });
            debug_node(node);
            debug_print("%zu \n", node);
            register_processed(&vctx->context->rewriter, node->payload.let.variables.nodes[0], slot);
            return;
        }
    }

    no_op:
    visit_children(&vctx->visitor, node);
}

static const Node* process(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

    IrArena* dst_arena = ctx->rewriter.dst_arena;
    switch (old->tag) {
        case Lambda_TAG: {
            Node* fun = recreate_decl_header_identity(&ctx->rewriter, old);
            Context ctx2 = *ctx;
            ctx2.disable_lowering = lookup_annotation_with_string_payload(old, "DisablePass", "setup_stack_frames");
            ctx2.old_entry_body = old->payload.lam.body;
            ctx2.entry_sp_val = NULL;
            fun->payload.lam.body = process(&ctx2, old->payload.lam.body);
            return fun;
        }
        case Body_TAG: {
            if (ctx->disable_lowering)
                return recreate_node_identity(&ctx->rewriter, old);

            // this may miss call instructions...
            BodyBuilder* instructions = begin_body(dst_arena);

            // We are the entry block for a FN !
            if (old == ctx->old_entry_body) {
                assert(!ctx->entry_sp_val);
                ctx->entry_sp_val = gen_primop_ce(instructions, get_stack_pointer_op, 0, NULL);
            }
            assert(ctx->entry_sp_val);

            if (old == ctx->old_entry_body) {
                VContext vctx = {
                    .visitor = {
                        .visit_fn = (VisitFn) collect_allocas,
                        .visit_fn_scope_rpo = true,
                    },
                    .context = ctx,
                    .builder = instructions,
                };

                visit_children(&vctx.visitor, old);
            }

            for (size_t i = 0; i < old->payload.body.instructions.count; i++) {
                const Node* old_instr = old->payload.body.instructions.nodes[i];

                const Node* actual_old_instr = old_instr;
                if (actual_old_instr->tag == Let_TAG)
                    actual_old_instr = actual_old_instr->payload.let.instruction;
                if (actual_old_instr->tag == PrimOp_TAG && actual_old_instr->payload.prim_op.op == alloca_op) {
                    // Ignore non-let bound allocas
                    if (actual_old_instr == old_instr)
                        continue;

                    // Check it was lowered properly before
                    assert(find_processed(&ctx->rewriter, old_instr->payload.let.variables.nodes[0]));
                    continue;
                }
                append_body(instructions, rewrite_node(&ctx->rewriter, old_instr));
            }

            const Node* terminator = old->payload.body.terminator;
            switch (terminator->tag) {
                case Return_TAG: {
                    // Restore SP before calling exit
                    gen_primop_c(instructions, set_stack_pointer_op, 1, (const Node* []) { ctx->entry_sp_val });
                    SHADY_FALLTHROUGH
                }
                default: terminator = process(ctx, terminator); break;
            }
            return finish_body(instructions, terminator);
        }
        default: return recreate_node_identity(&ctx->rewriter, old);
    }
}

const Node* setup_stack_frames(SHADY_UNUSED CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct List* new_decls_list = new_list(const Node*);
    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);

    Context ctx = {
        .rewriter = {
            .dst_arena = dst_arena,
            .src_arena = src_arena,
            .rewrite_fn = (RewriteFn) process,
            .processed = done,
        },
        .disable_lowering = false,

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

    destroy_dict(done);
    return rewritten;
}
