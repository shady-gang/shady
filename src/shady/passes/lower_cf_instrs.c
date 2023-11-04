#include "passes.h"

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
    Node* current_fn;
} Context;

static const Node* process_node(Context* ctx, const Node* node);

static const Node* process_let(Context* ctx, const Node* node) {
    assert(node->tag == Let_TAG);
    IrArena* a = ctx->rewriter.dst_arena;

    const Node* old_instruction = node->payload.let.instruction;
    const Node* new_instruction = NULL;
    const Node* old_tail = node->payload.let.tail;
    const Node* new_tail = NULL;

    switch (old_instruction->tag) {
        case If_TAG: {
            bool has_false_branch = old_instruction->payload.if_instr.if_false;
            Nodes yield_types = rewrite_nodes(&ctx->rewriter, old_instruction->payload.if_instr.yield_types);

            const Type* jp_type = qualified_type(a, (QualifiedType) {
                .type = join_point_type(a, (JoinPointType) { .yield_types = yield_types }),
                .is_uniform = true,
            });
            const Node* join_point = var(a, jp_type, "if_join");
            Context join_context = *ctx;
            join_context.join_points.join_point_selection_merge = join_point;

            Node* true_block = basic_block(a, ctx->current_fn, nodes(a, 0, NULL), unique_name(a, "if_true"));
            true_block->payload.basic_block.body = rewrite_node(&join_context.rewriter, old_instruction->payload.if_instr.if_true->payload.case_.body);

            Node* flse_block = basic_block(a, ctx->current_fn, nodes(a, 0, NULL), unique_name(a, "if_false"));
            if (has_false_branch)
                flse_block->payload.basic_block.body = rewrite_node(&join_context.rewriter, old_instruction->payload.if_instr.if_false->payload.case_.body);
            else {
                assert(yield_types.count == 0);
                flse_block->payload.basic_block.body = join(a, (Join) { .join_point = join_point, .args = nodes(a, 0, NULL) });
            }

            const Node* control_body = branch(a, (Branch) {
                .branch_condition = rewrite_node(&ctx->rewriter, old_instruction->payload.if_instr.condition),
                .true_jump = jump_helper(a, true_block, empty(a)),
                .false_jump = jump_helper(a, flse_block, empty(a)),
            });

            const Node* control_lam = case_(a, nodes(a, 1, (const Node* []) {join_point}), control_body);
            new_instruction = control(a, (Control) { .yield_types = yield_types, .inside = control_lam });
            break;
        }
        case Loop_TAG: {
            const Node* old_loop_body = old_instruction->payload.loop_instr.body;
            assert(is_case(old_loop_body));

            Nodes yield_types = rewrite_nodes(&ctx->rewriter, old_instruction->payload.loop_instr.yield_types);
            Nodes param_types = rewrite_nodes(&ctx->rewriter, get_variables_types(a, old_loop_body->payload.case_.params));
            param_types = strip_qualifiers(a, param_types);

            const Type* break_jp_type = qualified_type(a, (QualifiedType) {
                .type = join_point_type(a, (JoinPointType) { .yield_types = yield_types }),
                .is_uniform = true,
            });
            const Type* continue_jp_type = qualified_type(a, (QualifiedType) {
                .type = join_point_type(a, (JoinPointType) { .yield_types = param_types }),
                .is_uniform = true,
            });
            const Node* break_point = var(a, break_jp_type, "loop_break_point");
            const Node* continue_point = var(a, continue_jp_type, "loop_continue_point");
            Context join_context = *ctx;
            join_context.join_points.join_point_loop_break = break_point;
            join_context.join_points.join_point_loop_continue = continue_point;

            Nodes new_params = recreate_variables(&ctx->rewriter, old_loop_body->payload.case_.params);
            Node* loop_body = basic_block(a, ctx->current_fn, new_params, unique_name(a, "loop_body"));
            register_processed_list(&join_context.rewriter, old_loop_body->payload.case_.params, loop_body->payload.basic_block.params);

            const Node* inner_control_body = rewrite_node(&join_context.rewriter, old_loop_body->payload.case_.body);
            const Node* inner_control_lam = case_(a, nodes(a, 1, (const Node* []) {continue_point}), inner_control_body);

            BodyBuilder* bb = begin_body(a);
            const Node* inner_control = control(a, (Control) {
                .yield_types = param_types,
                .inside = inner_control_lam,
            });
            Nodes args = bind_instruction(bb, inner_control);

            // TODO let_in_block or use a Jump !
            loop_body->payload.basic_block.body = finish_body(bb, jump(a, (Jump) { .target = loop_body, .args = args }));

            const Node* initial_jump = jump(a, (Jump) {
                .target = loop_body,
                .args = rewrite_nodes(&ctx->rewriter, old_instruction->payload.loop_instr.initial_args),
            });
            const Node* outer_body = case_(a, nodes(a, 1, (const Node* []) {break_point}), initial_jump);
            new_instruction = control(a, (Control) { .yield_types = yield_types, .inside = outer_body });
            break;
        }
        default:
            new_instruction = rewrite_node(&ctx->rewriter, old_instruction);
            break;
    }

    if (!new_tail)
        new_tail = rewrite_node(&ctx->rewriter, old_tail);

    assert(new_instruction && new_tail);
    return let(a, new_instruction, new_tail);
}

static const Node* process_node(Context* ctx, const Node* node) {
    const Node* already_done = search_processed(&ctx->rewriter, node);
    if (already_done)
        return already_done;

    IrArena* a = ctx->rewriter.dst_arena;

    if (node->tag == Function_TAG) {
        Node* fun = recreate_decl_header_identity(&ctx->rewriter, node);
        Context sub_ctx = *ctx;
        sub_ctx.disable_lowering = lookup_annotation(fun, "Structured");
        sub_ctx.current_fn = fun;
        sub_ctx.join_points = (JoinPoints) {
            .join_point_selection_merge = NULL,
            .join_point_switch_merge = NULL,
            .join_point_loop_break = NULL,
            .join_point_loop_continue = NULL,
        };
        fun->payload.fun.body = rewrite_node(&sub_ctx.rewriter, node->payload.fun.body);
        return fun;
    }

    if (ctx->disable_lowering)
        return recreate_node_identity(&ctx->rewriter, node);

    switch (node->tag) {
        case Let_TAG: return process_let(ctx, node);
        case Yield_TAG: {
            const Node* jp = ctx->join_points.join_point_selection_merge;
            assert(jp);
            return join(a, (Join) {
                .join_point = jp,
                .args = rewrite_nodes(&ctx->rewriter, node->payload.yield.args),
            });
        }
        case MergeContinue_TAG: {
            const Node* jp = ctx->join_points.join_point_loop_continue;
            assert(jp);
            return join(a, (Join) {
                .join_point = jp,
                .args = rewrite_nodes(&ctx->rewriter, node->payload.merge_continue.args),
            });
        }
        case MergeBreak_TAG: {
            const Node* jp = ctx->join_points.join_point_loop_break;
            assert(jp);
            return join(a, (Join) {
                .join_point = jp,
                .args = rewrite_nodes(&ctx->rewriter, node->payload.merge_break.args),
            });
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

Module* lower_cf_instrs(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process_node),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}

