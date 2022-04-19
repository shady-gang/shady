#include "passes.h"

#include "list.h"

#include "../log.h"
#include "../local_array.h"

#include <assert.h>
#include <string.h>

struct BindEntry {
    const char* name;
    const Node* bound_node;
};

struct BindRewriter {
    Rewriter rewriter;
    struct List* bound_variables;
    const Node* current_function;
};

static const Node* resolve(struct BindRewriter* ctx, const char* name) {
    for (size_t i = 0; i < entries_count_list(ctx->bound_variables); i++) {
        const struct BindEntry* entry = &read_list(const struct BindEntry, ctx->bound_variables)[i];
        if (strcmp(entry->name, name) == 0) {
            return entry->bound_node;
        }
    }
    error("could not resolve variable %s", name)
}

static const Node* bind_node(struct BindRewriter* ctx, const Node* node);

static Node* rewrite_fn_head(struct BindRewriter* ctx, const Node* node) {
    assert(node != NULL && node->tag == Function_TAG);
    Rewriter* rewriter = &ctx->rewriter;
    IrArena* dst_arena = rewriter->dst_arena;

    // rebuild the parameters and shove them in the list
    size_t params_count = node->payload.fn.params.count;
    LARRAY(const Node*, nparams, params_count);
    for (size_t i = 0; i < params_count; i++) {
        const Variable* old_param = &node->payload.fn.params.nodes[i]->payload.var;
        const Node* new_param = var(dst_arena, rewrite_node(rewriter, old_param->type), string(dst_arena, old_param->name));
        nparams[i] = new_param;
    }

    return fn(dst_arena, node->payload.fn.is_continuation, string(dst_arena, node->payload.fn.name), nodes(dst_arena, params_count, nparams), rewrite_nodes(rewriter, node->payload.fn.return_types));
}

static void rewrite_fn_body(struct BindRewriter* ctx, const Node* node, Node* target) {
    assert(node != NULL && node->tag == Function_TAG);
    Rewriter* rewriter = &ctx->rewriter;
    IrArena* dst_arena = rewriter->dst_arena;

    size_t old_bound_variables_size = entries_count_list(ctx->bound_variables);

    // bind the rebuilt parameters for rewriting the body
    for (size_t i = 0; i < node->payload.fn.params.count; i++) {
        const Node* param = target->payload.fn.params.nodes[i];
        struct BindEntry entry = {
            .name = string(dst_arena, param->payload.var.name),
            .bound_node = param
        };
        append_list(struct BindEntry, ctx->bound_variables, entry);
        printf("Bound param %s\n", entry.name);
    }

    struct BindRewriter sub_ctx = *ctx;
    if (!node->payload.fn.is_continuation) {
        assert(ctx->current_function == NULL);
        sub_ctx.current_function = target;
    } else {
        // maybe not beneficial/relevant
        assert(sub_ctx.current_function != NULL);
    }

    target->payload.fn.block = bind_node(&sub_ctx, node->payload.fn.block);

    // cleanup
    while (entries_count_list(ctx->bound_variables) > old_bound_variables_size)
        remove_last_list(struct BindEntry, ctx->bound_variables);
}

static const Node* bind_node(struct BindRewriter* ctx, const Node* node) {
    if (node == NULL)
        return NULL;

    Rewriter* rewriter = &ctx->rewriter;
    IrArena* dst_arena = rewriter->dst_arena;
    switch (node->tag) {
        case Root_TAG: {
            const Root* src_root = &node->payload.root;
            const size_t count = src_root->variables.count;

            LARRAY(const Node*, new_variables, count);
            LARRAY(const Node*, new_definitions, count);

            for (size_t i = 0; i < count; i++) {
                const Node* old_variable = src_root->variables.nodes[i];

                const Node* new_variable = var(rewriter->dst_arena, rewrite_node(rewriter, old_variable->payload.var.type), string(rewriter->dst_arena, old_variable->payload.var.name));

                struct BindEntry entry = {
                    .name = old_variable->payload.var.name,
                    .bound_node = new_variable
                };

                append_list(struct BindEntry, ctx->bound_variables, entry);
                printf("Bound root def %s\n", entry.name);
                new_variables[i] = new_variable;
            }

            for (size_t i = 0; i < count; i++) {
                const Node* old_definition = src_root->definitions.nodes[i];
                new_definitions[i] = bind_node(ctx, old_definition);
            }

            return root(rewriter->dst_arena, (Root) {
                .variables = nodes(rewriter->dst_arena, count, new_variables),
                .definitions = nodes(rewriter->dst_arena, count, new_definitions)
            });
        }
        case Variable_TAG: error("the binders should be handled such that this node is never reached");
        case Unbound_TAG: {
            return resolve(ctx, node->payload.unbound.name);
        }
        case Let_TAG: {
            size_t outputs_count = node->payload.let.variables.count;
            LARRAY(const Node*, noutputs, outputs_count);
            for (size_t p = 0; p < outputs_count; p++) {
                const Variable* old_var = &node->payload.let.variables.nodes[p]->payload.var;
                const Node* new_binding = var(rewriter->dst_arena, rewrite_node(rewriter, old_var->type), string(rewriter->dst_arena, old_var->name));
                noutputs[p] = new_binding;
                struct BindEntry entry = {
                    .name = string(ctx->rewriter.dst_arena, old_var->name),
                    .bound_node = new_binding
                };
                append_list(struct BindEntry, ctx->bound_variables, entry);
                printf("Bound primop result %s\n", entry.name);
            }

            return let(rewriter->dst_arena, (Let) {
                .variables = nodes(rewriter->dst_arena, outputs_count, noutputs),
                .op = node->payload.let.op,
                .args = rewrite_nodes(rewriter, node->payload.let.args)
            });
        }
        case ParsedBlock_TAG: {
            const ParsedBlock* pblock = &node->payload.parsed_block;
            size_t old_bound_variables_size = entries_count_list(ctx->bound_variables);

            size_t inner_conts_count = pblock->continuations_vars.count;
            LARRAY(Node*, new_conts, inner_conts_count);

            // First create stubs and inline that crap
            for (size_t i = 0; i < inner_conts_count; i++) {
                Node* new_cont = rewrite_fn_head(ctx, pblock->continuations.nodes[i]);
                new_conts[i] = new_cont;
                struct BindEntry entry = {
                    .name = string(ctx->rewriter.dst_arena, pblock->continuations_vars.nodes[i]->payload.var.name),
                    .bound_node = new_cont
                };
                append_list(struct BindEntry, ctx->bound_variables, entry);
                printf("Bound (stub) continuation %s\n", entry.name);
            }

            const Node* new_block = block(rewriter->dst_arena, (Block) {
                .instructions = rewrite_nodes(rewriter, pblock->instructions),
                .terminator = bind_node(ctx, pblock->terminator)
            });

            // Rebuild the actual continuations now
            for (size_t i = 0; i < inner_conts_count; i++) {
                rewrite_fn_body(ctx, pblock->continuations.nodes[i], new_conts[i]);
                printf("Processed (full) continuation %s\n", new_conts[i]->payload.fn.name);
            }

            while (entries_count_list(ctx->bound_variables) > old_bound_variables_size)
                remove_last_list(struct BindEntry, ctx->bound_variables);

            return new_block;
        }
        case Block_TAG: {
            return block(rewriter->dst_arena, (Block) {
                .instructions = rewrite_nodes(rewriter, node->payload.block.instructions),
            });
        }
        case Return_TAG: {
            assert(ctx->current_function);
            return fn_ret(dst_arena, (Return) {
                .fn = ctx->current_function,
                .values = rewrite_nodes(rewriter, node->payload.fn_ret.values)
            });
        }
        case Function_TAG: {
            Node* new_fn = rewrite_fn_head(ctx, node);
            rewrite_fn_body(ctx, node, new_fn);
            return new_fn;
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

const Node* bind_program(IrArena* src_arena, IrArena* dst_arena, const Node* source) {
    struct List* bound_variables = new_list(struct BindEntry);
    struct BindRewriter ctx = {
        .rewriter = {
            .src_arena = src_arena,
            .dst_arena = dst_arena,
            .rewrite_fn = (RewriteFn) bind_node,
        },
        .bound_variables = bound_variables
    };

    const Node* rewritten = rewrite_node(&ctx.rewriter, source);

    destroy_list(bound_variables);
    return rewritten;
}