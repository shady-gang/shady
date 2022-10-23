#include "shady/ir.h"

#include "log.h"
#include "portability.h"
#include "../type.h"
#include "../rewrite.h"

#include <assert.h>

typedef struct {
    const Node* join_point_selection_merge;
    const Node* join_point_switch_merge;
    const Node* join_point_loop_break;
    const Node* join_point_loop_continue;
} JoinPoints;

typedef struct Context_ {
    Rewriter rewriter;
    bool disable_lowering;
    JoinPoints join_points;
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
        case If_TAG: {
            bool has_false_branch = old_instruction->payload.if_instr.if_false;
            Nodes yield_types = rewrite_nodes(&ctx->rewriter, old_instruction->payload.if_instr.yield_types);

            const Type* jp_type = qualified_type(arena, (QualifiedType) {
                .type = join_point_type(arena, (JoinPointType) { .yield_types = yield_types }),
                .is_uniform = true,
            });
            const Node* join_point = var(arena, jp_type, "if_join");
            Context join_context = *ctx;
            join_context.join_points.join_point_selection_merge = join_point;

            Node* true_block = basic_block(arena, nodes(arena, 0, NULL), unique_name(arena, "if_true"));
            true_block->payload.lam.body = rewrite_node(&join_context.rewriter,  old_instruction->payload.if_instr.if_true->payload.lam.body);

            Node* flse_block = basic_block(arena, nodes(arena, 0, NULL), unique_name(arena, "if_false"));
            if (has_false_branch)
                flse_block->payload.lam.body = rewrite_node(&join_context.rewriter,  old_instruction->payload.if_instr.if_false->payload.lam.body);
            else {
                assert(yield_types.count == 0);
                flse_block->payload.lam.body = join(arena, (Join) { .join_point = join_point, .args = nodes(arena, 0, NULL) });
            }

            const Node* branch_t = branch(arena, (Branch) {
                .branch_mode = BrIfElse,
                .branch_condition = rewrite_node(&ctx->rewriter, old_instruction->payload.if_instr.condition),
                .true_target = true_block,
                .false_target = flse_block,
                .args = nodes(arena, 0, NULL),
            });

            Node* if_body = lambda(arena, nodes(arena, 1, (const Node*[]) { join_point }));
            if_body->payload.lam.body = branch_t;
            new_instruction = control(arena, (Control) { .yield_types = yield_types, .inside = if_body });
            break;
        }
        case Loop_TAG: {
            const Node* old_loop_body = old_instruction->payload.loop_instr.body;

            Nodes yield_types = rewrite_nodes(&ctx->rewriter, old_instruction->payload.loop_instr.yield_types);
            Nodes param_types = rewrite_nodes(&ctx->rewriter, extract_variable_types(arena, &old_loop_body->payload.lam.params));

            const Type* break_jp_type = qualified_type(arena, (QualifiedType) {
                .type = join_point_type(arena, (JoinPointType) { .yield_types = yield_types }),
                .is_uniform = true,
            });
            const Type* continue_jp_type = qualified_type(arena, (QualifiedType) {
                .type = join_point_type(arena, (JoinPointType) { .yield_types = param_types }),
                .is_uniform = true,
            });
            const Node* break_point = var(arena, break_jp_type, "loop_break_point");
            const Node* continue_point = var(arena, continue_jp_type, "loop_continue_point");
            Context join_context = *ctx;
            join_context.join_points.join_point_loop_break = break_point;
            join_context.join_points.join_point_loop_continue = continue_point;

            Nodes new_params = recreate_variables(&ctx->rewriter, old_loop_body->payload.lam.params);
            Node* loop_body = basic_block(arena, new_params, unique_name(arena, "loop_body"));
            register_processed_list(&join_context.rewriter, old_loop_body->payload.lam.params, loop_body->payload.lam.params);

            Node* actual_body = lambda(arena, nodes(arena, 1, (const Node*[]) { continue_point }));
            actual_body->payload.lam.body = rewrite_node(&join_context.rewriter, old_loop_body->payload.lam.body);

            const Node* inner_control = control(arena, (Control) {
                .yield_types = param_types,
                .inside = actual_body,
            });

            loop_body->payload.lam.body = let(arena, false, inner_control, loop_body);

            Node* outer_body = lambda(arena, nodes(arena, 1, (const Node*[]) { break_point }));
            const Node* initial_jump = branch(arena, (Branch) {
                .branch_mode = BrJump,
                .target = loop_body,
                .args = rewrite_nodes(&ctx->rewriter, old_instruction->payload.loop_instr.initial_args),
            });
            outer_body->payload.lam.body = initial_jump;
            new_instruction = control(arena, (Control) { .yield_types = yield_types, .inside = outer_body });
            break;
        }
        default:
            new_instruction = rewrite_node(&ctx->rewriter, old_instruction);
            break;
    }

    if (!new_tail)
        new_tail = rewrite_node(&ctx->rewriter, old_tail);

    assert(new_instruction && new_tail);
    return let(arena, false, new_instruction, new_tail);
}

static const Node* process_node(Context* ctx, const Node* node) {
    const Node* already_done = search_processed(&ctx->rewriter, node);
    if (already_done)
        return already_done;

    IrArena* dst_arena = ctx->rewriter.dst_arena;

    switch (node->tag) {
        case Lambda_TAG: {
            Node* fun = recreate_decl_header_identity(&ctx->rewriter, node);
            Context sub_ctx = *ctx;
            if (node->payload.lam.tier == FnTier_Function) {
                sub_ctx.disable_lowering = lookup_annotation_with_string_payload(fun, "DisablePass", "lower_cf_instrs");
                sub_ctx.join_points = (JoinPoints) {
                    .join_point_selection_merge = NULL,
                    .join_point_switch_merge = NULL,
                    .join_point_loop_break = NULL,
                    .join_point_loop_continue = NULL,
                };
            }
            fun->payload.lam.body = rewrite_node(&sub_ctx.rewriter, node->payload.lam.body);
            return fun;
        }
        case Let_TAG: return process_let(ctx, node);
        case MergeConstruct_TAG: {
            const Node* jp = NULL;
            switch (node->payload.merge_construct.construct) {
                case Selection: jp = ctx->join_points.join_point_selection_merge; break;
                case Continue:  jp = ctx->join_points.join_point_loop_continue; break;
                case Break:     jp = ctx->join_points.join_point_loop_break; break;
            }
            assert(jp);
            return join(dst_arena, (Join) {
                .join_point = jp,
                .args = rewrite_nodes(&ctx->rewriter, node->payload.merge_construct.args),
            });
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

const Node* lower_cf_instrs(SHADY_UNUSED CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    Context ctx = {
        .rewriter = create_rewriter(src_arena, dst_arena, (RewriteFn) process_node),
    };
    const Node* rewritten = recreate_node_identity(&ctx.rewriter, src_program);
    destroy_rewriter(&ctx.rewriter);
    return rewritten;
}
