#include "shady/ir.h"

#include "list.h"
#include "dict.h"
#include "log.h"
#include "portability.h"

#include "../type.h"
#include "../rewrite.h"

#include "../analysis/scope.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
} Context;

static const Node* process_node(Context* ctx, const Node* node) {
    if (node == NULL) return NULL;

    const Node* already_done = search_processed(&ctx->rewriter, node);
    if (already_done)
        return already_done;

    if (is_abstraction(node)) {
        const Node* terminator = get_abstraction_body(node);
        if (terminator) {
            assert(is_terminator(terminator));
            if (terminator->tag == Branch_TAG) {
                Scope* branched_scope = new_scope_flipped(node);
                CFNode* cfnode = scope_lookup(branched_scope, node);
                CFNode* idom = cfnode->idom;

                Rewriter * rewriter = &ctx->rewriter;
                IrArena * arena = rewriter->dst_arena;
                Node* result = NULL;

                Node* fn = NULL;
                Node* new_bb = NULL;
                if (is_function(node)) {
                    fn = recreate_decl_header_identity(rewriter, node);
                } else if (is_basic_block(node)) {
                    fn = find_processed(rewriter, node->payload.basic_block.fn);

                    new_bb = basic_block(arena, fn, empty(arena), node->payload.basic_block.name);
                    register_processed(rewriter, node, new_bb);
                }

                Nodes yield_types;
                Nodes exit_args;
                Nodes lambda_args;
                const Nodes old_params = idom->node->payload.basic_block.params;
                if (old_params.count == 0) {
                    yield_types = empty(arena);
                    exit_args = empty(arena);
                    lambda_args = empty(arena);
                } else {
                    const Node* types[old_params.count];
                    const Node* inner_args[old_params.count];
                    const Node* outer_args[old_params.count];

                    for (size_t j = 0; j < old_params.count; j++) {
                        const Node* qualified_type = rewrite_node(rewriter, old_params.nodes[j]->payload.var.type);
                        types[j] = qualified_type;

                        if (contains_qualified_type(types[j]))
                            types[j] = get_unqualified_type(types[j]);

                        inner_args[j] = var(arena, qualified_type, old_params.nodes[j]->payload.var.name);
                        outer_args[j] = var(arena, qualified_type, old_params.nodes[j]->payload.var.name);
                    }

                    yield_types = nodes(arena, old_params.count, types);
                    exit_args = nodes(arena, old_params.count, inner_args);
                    lambda_args = nodes(arena, old_params.count, outer_args);
                }

                const Node* join_token = var(arena, join_point_type(arena, (JoinPointType) {
                            .yield_types = yield_types
                            }), "jp");

                Node* pre_join = basic_block(arena, fn, exit_args, "exit");
                pre_join->payload.basic_block.body = join(arena, (Join) {
                        .join_point = join_token,
                        .args = exit_args
                        });

                const Node* cached = search_processed(rewriter, idom->node);
                if (cached)
                    remove_dict(const Node*, rewriter->processed, idom->node);
                register_processed(rewriter, idom->node, pre_join);

                Node* body = basic_block(arena, fn, empty(arena), "inner");
                body->payload.basic_block.body = recreate_node_identity(rewriter, terminator);

                remove_dict(const Node*, rewriter->processed, idom->node);
                if (cached)
                    register_processed(rewriter, idom->node, cached);

                const Node* inner_terminator = jump(arena, (Jump) {
                        .target = body,
                        .args = empty(arena)
                        });

                const Node* control_inner = lambda(rewriter->dst_module, singleton(join_token), inner_terminator);
                const Node* new_target = control (arena, (Control) {
                        .inside = control_inner,
                        .yield_types = yield_types
                        });

                const Node* recreated_join = rewrite_node(rewriter, idom->node);
                const Node* outer_terminator = jump(arena, (Jump) {
                        .target = recreated_join,
                        .args = lambda_args
                        });

                const Node* anon_lam = lambda(rewriter->dst_module, lambda_args, outer_terminator);
                const Node* empty_let = let(arena, new_target, anon_lam);

                if (is_function(node)) {
                    fn->payload.fun.body = empty_let;
                    result = fn;
                } else if (is_basic_block(node)) {
                    new_bb->payload.basic_block.body = empty_let;
                    result = new_bb;
                } else {
                    assert(false);
                }

                destroy_scope(branched_scope);

                if (result)
                    return result;
            }
        }
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

void reconvergence_heuristics(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process_node),
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
