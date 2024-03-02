#include "passes.h"

#include "log.h"
#include "portability.h"
#include "dict.h"

#include "../type.h"
#include "../rewrite.h"
#include "../analysis/scope.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    bool disable_lowering;
    Node* current_fn;

    struct Dict* structured_join_tokens;
    Scope* scope;
    const Node* abs;
} Context;

static const Node* process_node(Context* ctx, const Node* node) {
    const Node* already_done = search_processed(&ctx->rewriter, node);
    if (already_done)
        return already_done;

    IrArena* a = ctx->rewriter.dst_arena;

    Context sub_ctx = *ctx;
    if (node->tag == Function_TAG) {
        Node* fun = recreate_decl_header_identity(&ctx->rewriter, node);
        sub_ctx.disable_lowering = lookup_annotation(fun, "Structured");
        sub_ctx.current_fn = fun;
        sub_ctx.scope = new_scope(node);
        sub_ctx.abs = node;
        fun->payload.fun.body = rewrite_node(&sub_ctx.rewriter, node->payload.fun.body);
        destroy_scope(sub_ctx.scope);
        return fun;
    } else if (node->tag == Constant_TAG) {
        sub_ctx.scope = NULL;
        sub_ctx.abs = NULL;
        sub_ctx.current_fn = NULL;
        ctx = &sub_ctx;
    }

    if (is_abstraction(node)) {
        sub_ctx.abs = node;
        ctx = &sub_ctx;
    }

    if (ctx->disable_lowering)
        return recreate_node_identity(&ctx->rewriter, node);

    CFNode* cfnode = ctx->scope ? scope_lookup(ctx->scope, ctx->abs) : NULL;
    switch (node->tag) {
        case If_TAG: {
            bool has_false_branch = node->payload.structured_if.if_false;
            Nodes yield_types = rewrite_nodes(&ctx->rewriter, node->payload.structured_if.yield_types);

            const Type* jp_type = qualified_type(a, (QualifiedType) {
                .type = join_point_type(a, (JoinPointType) { .yield_types = yield_types }),
                .is_uniform = false,
            });
            const Node* join_point = var(a, jp_type, "if_join");
            Context join_context = *ctx;
            Nodes jps = singleton(join_point);
            insert_dict(const Node*, Nodes, ctx->structured_join_tokens, node, jps);

            Node* true_block = basic_block(a, ctx->current_fn, nodes(a, 0, NULL), unique_name(a, "if_true"));
            join_context.abs = node->payload.structured_if.if_true;
            true_block->payload.basic_block.body = rewrite_node(&join_context.rewriter, node->payload.structured_if.if_true->payload.case_.body);

            Node* flse_block = basic_block(a, ctx->current_fn, nodes(a, 0, NULL), unique_name(a, "if_false"));
            if (has_false_branch) {
                join_context.abs = node->payload.structured_if.if_false;
                flse_block->payload.basic_block.body = rewrite_node(&join_context.rewriter, node->payload.structured_if.if_false->payload.case_.body);
            } else {
                assert(yield_types.count == 0);
                flse_block->payload.basic_block.body = join(a, (Join) { .join_point = join_point, .args = nodes(a, 0, NULL) });
            }

            const Node* control_body = branch(a, (Branch) {
                .branch_condition = rewrite_node(&ctx->rewriter, node->payload.structured_if.condition),
                .true_destination = jump_helper(a, true_block, empty(a)),
                .false_destination = jump_helper(a, flse_block, empty(a)),
            });

            const Node* control_lam = case_(a, nodes(a, 1, (const Node* []) {join_point}), control_body);
            join_context.abs = node->payload.structured_if.tail;
            const Node* ntail = rewrite_node(&join_context.rewriter, node->payload.structured_if.tail);
            return control(a, (Control) { .yield_types = yield_types, .inside = control_lam, .tail = ntail });
        }
        case Loop_TAG: {
            const Node* old_loop_body = node->payload.structured_loop.body;
            assert(is_case(old_loop_body));

            Nodes yield_types = rewrite_nodes(&ctx->rewriter, node->payload.structured_loop.yield_types);
            Nodes param_types = rewrite_nodes(&ctx->rewriter, get_variables_types(a, old_loop_body->payload.case_.params));
            param_types = strip_qualifiers(a, param_types);

            const Type* break_jp_type = qualified_type(a, (QualifiedType) {
                    .type = join_point_type(a, (JoinPointType) { .yield_types = yield_types }),
                    .is_uniform = false,
            });
            const Type* continue_jp_type = qualified_type(a, (QualifiedType) {
                    .type = join_point_type(a, (JoinPointType) { .yield_types = param_types }),
                    .is_uniform = false,
            });
            const Node* break_point = var(a, break_jp_type, "loop_break_point");
            const Node* continue_point = var(a, continue_jp_type, "loop_continue_point");
            Context join_context = *ctx;
            Nodes jps = mk_nodes(a, break_point, continue_point);
            insert_dict(const Node*, Nodes, ctx->structured_join_tokens, node, jps);

            Nodes new_params = recreate_variables(&ctx->rewriter, old_loop_body->payload.case_.params);
            Node* loop_body = basic_block(a, ctx->current_fn, new_params, unique_name(a, "loop_body"));
            register_processed_list(&join_context.rewriter, old_loop_body->payload.case_.params, loop_body->payload.basic_block.params);

            join_context.abs = old_loop_body;
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
                    .args = rewrite_nodes(&ctx->rewriter, node->payload.structured_loop.initial_args),
            });
            const Node* outer_body = case_(a, nodes(a, 1, (const Node* []) {break_point}), initial_jump);
            join_context.abs = node->payload.structured_loop.tail;
            const Node* ntail = rewrite_node(&join_context.rewriter, node->payload.structured_loop.tail);
            return control(a, (Control) { .yield_types = yield_types, .inside = outer_body, .tail = ntail });
        }
        case Yield_TAG: {
            if (!cfnode)
                break;
            CFNode* dom = cfnode->idom;
            const Node* selection_instr = NULL;
            while (dom) {
                const Node* body = get_abstraction_body(dom->node);
                assert(body->tag == Body_TAG);
                const Node* terminator = body->payload.body.terminator;
                if (terminator->tag == If_TAG || terminator->tag == Match_TAG) {
                    selection_instr = terminator;
                    break;
                }

                dom = dom->idom;
            }

            if (!selection_instr) {
                error_print("Scoping error: Failed to find a dominating selection construct for ");
                log_node(ERROR, node);
                error_print(".\n");
                error_die();
            }

            Nodes* jps = find_value_dict(const Node*, Nodes, ctx->structured_join_tokens, selection_instr);
            assert(jps && jps->count == 1);
            const Node* jp = first(*jps);
            assert(jp);
            return join(a, (Join) {
                .join_point = jp,
                .args = rewrite_nodes(&ctx->rewriter, node->payload.yield.args),
            });
        }
        case MergeContinue_TAG: {
            assert(cfnode);
            CFNode* dom = cfnode->idom;
            const Node* selection_instr = NULL;
            while (dom) {
                const Node* body = get_abstraction_body(dom->node);
                assert(body->tag == Body_TAG);
                const Node* terminator = body->payload.body.terminator;
                if (terminator->tag == Loop_TAG) {
                    selection_instr = terminator;
                    break;
                }
                dom = dom->idom;
            }

            if (!selection_instr) {
                error_print("Scoping error: Failed to find a dominating selection construct for ");
                log_node(ERROR, node);
                error_print(".\n");
                error_die();
            }

            Nodes* jps = find_value_dict(const Node*, Nodes, ctx->structured_join_tokens, selection_instr);
            assert(jps && jps->count == 2);
            const Node* jp = jps->nodes[1];
            assert(jp);
            return join(a, (Join) {
                .join_point = jp,
                .args = rewrite_nodes(&ctx->rewriter, node->payload.merge_continue.args),
            });
        }
        case MergeBreak_TAG: {
            assert(cfnode);
            CFNode* dom = cfnode->idom;
            const Node* selection_instr = NULL;
            while (dom) {
                const Node* body = get_abstraction_body(dom->node);
                assert(body->tag == Body_TAG);
                const Node* terminator = body->payload.body.terminator;
                if (terminator->tag == Loop_TAG) {
                    selection_instr = terminator;
                    break;
                }
                dom = dom->idom;
            }

            if (!selection_instr) {
                error_print("Scoping error: Failed to find a dominating selection construct for ");
                log_node(ERROR, node);
                error_print(".\n");
                error_die();
            }

            Nodes* jps = find_value_dict(const Node*, Nodes, ctx->structured_join_tokens, selection_instr);
            assert(jps && jps->count == 2);
            const Node* jp = first(*jps);
            assert(jp);
            return join(a, (Join) {
                .join_point = jp,
                .args = rewrite_nodes(&ctx->rewriter, node->payload.merge_break.args),
            });
        }
        default: break;
    }
    return recreate_node_identity(&ctx->rewriter, node);
}

KeyHash hash_node(const Node**);
bool compare_node(const Node**, const Node**);

Module* lower_cf_instrs(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process_node),
        .structured_join_tokens = new_dict(const Node*, Nodes, (HashFn) hash_node, (CmpFn) compare_node),
    };
    ctx.rewriter.config.fold_quote = false;
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    destroy_dict(ctx.structured_join_tokens);
    return dst;
}

