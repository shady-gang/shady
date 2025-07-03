#include "shady/pass.h"
#include "shady/ir/annotation.h"
#include "shady/ir/function.h"
#include "shady/ir/debug.h"

#include "log.h"
#include "portability.h"
#include "dict.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;

    const Node* nearest_if_join;
    const Node* nearest_continue_join;
    const Node* nearest_break_join;
} Context;

static const Node* process_node(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    if (node->tag == Function_TAG) {
        Node* fun = shd_recreate_node_head(r, node);
        Context sub_ctx = *ctx;
        sub_ctx.rewriter = shd_create_children_rewriter(r);
        sub_ctx.nearest_if_join = NULL;
        sub_ctx.nearest_continue_join = NULL;
        sub_ctx.nearest_break_join = NULL;
        shd_set_abstraction_body(fun, shd_rewrite_node(&sub_ctx.rewriter, node->payload.fun.body));
        shd_destroy_rewriter(&sub_ctx.rewriter);
        return fun;
    }

    switch (node->tag) {
        case If_TAG: {
            If payload = node->payload.if_instr;
            bool has_false_branch = payload.if_false;
            Nodes yield_types = shd_rewrite_nodes(r, node->payload.if_instr.yield_types);
            const Node* nmem = shd_rewrite_node(r, node->payload.if_instr.mem);

            const Type* jp_type = qualified_type(a, (QualifiedType) {
                .type = join_point_type(a, (JoinPointType) { .yield_types = yield_types }),
                .scope = shd_get_arena_config(a)->target.scopes.gang,
            });
            const Node* jp = param_helper(a, jp_type);
            shd_set_debug_name(jp, "if_join");

            Context sub_ctx = *ctx;
            sub_ctx.rewriter = shd_create_children_rewriter(r);
            sub_ctx.nearest_if_join = jp;

            const Node* true_block = shd_rewrite_node(&sub_ctx.rewriter, payload.if_true);

            const Node* false_block;
            if (has_false_branch) {
                false_block = shd_rewrite_node(&sub_ctx.rewriter, payload.if_false);
            } else {
                assert(yield_types.count == 0);
                false_block = basic_block_helper(a, shd_nodes(a, 0, NULL));
                shd_set_debug_name(false_block, "if_false");
                shd_set_abstraction_body((Node*) false_block, join(a, (Join) { .join_point = jp, .args = shd_nodes(a, 0, NULL), .mem = shd_get_abstraction_mem(false_block) }));
            }

            shd_destroy_rewriter(&sub_ctx.rewriter);

            Node* control_case = basic_block_helper(a, shd_singleton(jp));
            const Node* control_body = branch(a, (Branch) {
                .condition = shd_rewrite_node(r, node->payload.if_instr.condition),
                .true_jump = jump_helper(a, shd_get_abstraction_mem(control_case), true_block, shd_empty(a)),
                .false_jump = jump_helper(a, shd_get_abstraction_mem(control_case), false_block, shd_empty(a)),
                .mem = shd_get_abstraction_mem(control_case),
            });
            shd_set_abstraction_body(control_case, control_body);

            BodyBuilder* bb = shd_bld_begin(a, nmem);
            Nodes results = shd_bld_control(bb, yield_types, control_case);
            return shd_bld_finish(bb, jump_helper(a, shd_bld_mem(bb), shd_rewrite_node(r, payload.tail), results));
        }
        // TODO: match
        case Loop_TAG: {
            Loop payload = node->payload.loop_instr;
            const Node* old_loop_block = payload.body;

            Nodes yield_types = shd_rewrite_nodes(r, node->payload.loop_instr.yield_types);
            Nodes param_types = shd_rewrite_nodes(r, shd_get_param_types(a, get_abstraction_params(old_loop_block)));
            param_types = shd_strip_qualifiers(a, param_types);

            const Type* break_jp_type = qualified_type(a, (QualifiedType) {
                .type = join_point_type(a, (JoinPointType) { .yield_types = yield_types }),
                .scope = shd_get_arena_config(a)->target.scopes.gang,
            });
            const Type* continue_jp_type = qualified_type(a, (QualifiedType) {
                .type = join_point_type(a, (JoinPointType) { .yield_types = param_types }),
                .scope = shd_get_arena_config(a)->target.scopes.gang,
            });
            const Node* break_point = param_helper(a, break_jp_type);
            shd_set_debug_name(break_point, "loop_break_point");
            const Node* continue_point = param_helper(a, continue_jp_type);
            shd_set_debug_name(continue_jp_type, "loop_continue_point");

            Context sub_ctx = *ctx;
            sub_ctx.rewriter = shd_create_children_rewriter(r);
            sub_ctx.nearest_if_join = NULL;
            sub_ctx.nearest_continue_join = continue_point;
            sub_ctx.nearest_break_join = break_point;

            Nodes new_params = shd_recreate_params(r, get_abstraction_params(old_loop_block));
            Node* loop_header_block = basic_block_helper(a, new_params);
            shd_set_debug_name(loop_header_block, "loop_header");

            BodyBuilder* inner_bb = shd_bld_begin(a, shd_get_abstraction_mem(loop_header_block));
            Node* inner_control_case = basic_block_helper(a, shd_singleton(continue_point));
            shd_set_debug_name(inner_control_case, "inner_control");
            shd_set_abstraction_body(inner_control_case, jump_helper(a, shd_get_abstraction_mem(inner_control_case), shd_rewrite_node(&sub_ctx.rewriter, old_loop_block), new_params));
            Nodes args = shd_bld_control(inner_bb, param_types, inner_control_case);

            shd_destroy_rewriter(&sub_ctx.rewriter);

            shd_set_abstraction_body(loop_header_block, shd_bld_finish(inner_bb, jump(a, (Jump) { .target = loop_header_block, .args = args, .mem = shd_bld_mem(inner_bb) })));

            Node* outer_control_case = basic_block_helper(a, shd_singleton(break_point));
            shd_set_debug_name(outer_control_case, "outer_control");
            const Node* first_iteration_jump = jump(a, (Jump) {
                .target = loop_header_block,
                .args = shd_rewrite_nodes(r, payload.initial_args),
                .mem = shd_get_abstraction_mem(outer_control_case),
            });
            shd_set_abstraction_body(outer_control_case, first_iteration_jump);

            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            Nodes results = shd_bld_control(bb, yield_types, outer_control_case);
            return shd_bld_finish(bb, jump_helper(a, shd_bld_mem(bb), shd_rewrite_node(r, payload.tail), results));
        }
        case MergeSelection_TAG: {
            MergeSelection payload = node->payload.merge_selection;
            const Node* jp = ctx->nearest_if_join;
            assert(jp);
            const Node* nmem = shd_rewrite_node(r, payload.mem);
            return join(a, (Join) {
                .join_point = jp,
                .args = shd_rewrite_nodes(r, payload.args),
                .mem = nmem
            });
        }
        case MergeContinue_TAG: {
            MergeContinue payload = node->payload.merge_continue;
            const Node* jp = ctx->nearest_continue_join;
            assert(jp);
            const Node* nmem = shd_rewrite_node(r, payload.mem);
            return join(a, (Join) {
                .join_point = jp,
                .args = shd_rewrite_nodes(r, payload.args),
                .mem = nmem,
            });
        }
        case MergeBreak_TAG: {
            MergeBreak payload = node->payload.merge_break;
            const Node* jp = ctx->nearest_break_join;
            assert(jp);
            const Node* nmem = shd_rewrite_node(r, payload.mem);
            return join(a, (Join) {
                .join_point = jp,
                .args = shd_rewrite_nodes(r, payload.args),
                .mem = nmem,
            });
        }
        default: break;
    }
    return shd_recreate_node(r, node);
}

KeyHash shd_hash_node(const Node**);
bool shd_compare_node(const Node**, const Node**);

Module* shd_pass_lower_cf_instrs(SHADY_UNUSED const CompilerConfig* config, SHADY_UNUSED const void* unused, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process_node),
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}

