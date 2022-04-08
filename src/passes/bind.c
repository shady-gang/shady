#include "passes.h"

#include "list.h"

#include "../implem.h"

#include <assert.h>
#include <string.h>

struct BindEntry {
    const char* name;
    const Node* bound_node;
};

struct BindRewriter {
    Rewriter rewriter;
    struct List* bound_variables;
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

const Node* bind_node(struct BindRewriter* ctx, const Node* node) {
    if (node == NULL)
        return NULL;

    Rewriter* rewriter = &ctx->rewriter;
    switch (node->tag) {
        case Root_TAG: {
            const Root* src_root = &node->payload.root;
            const size_t count = src_root->variables.count;

            LARRAY(const Node*, new_variables, count);
            LARRAY(const Node*, new_definitions, count);

            for (size_t i = 0; i < count; i++) {
                const Node* variable = src_root->variables.nodes[i];

                const Node* new_variable = var(rewriter->dst_arena, rewrite_node(rewriter, variable->payload.var.type), string(rewriter->dst_arena, variable->payload.var.name));

                struct BindEntry entry = {
                    .name = variable->payload.var.name,
                    .bound_node = new_variable
                };
                append_list(struct BindEntry, ctx->bound_variables, entry);
                printf("Bound root def %s\n", entry.name);
                new_variables[i] = new_variable;
            }

            for (size_t i = 0; i < count; i++)
                new_definitions[i] = bind_node(ctx, src_root->definitions.nodes[i]);

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
        case Block_TAG: {
            size_t old_bound_variables_size = entries_count_list(ctx->bound_variables);

            size_t inner_conts_count = node->payload.block.continuations_vars.count;

            LARRAY(const Node*, nvars, inner_conts_count);
            LARRAY(const Node*, nconts, inner_conts_count);

            for (size_t p = 0; p < inner_conts_count; p++) {
                const Variable* old_var = &node->payload.block.continuations_vars.nodes[p]->payload.var;

                const Node* new_var = var(rewriter->dst_arena, rewrite_node(rewriter, old_var->type), string(rewriter->dst_arena, old_var->name));
                nvars[p] = new_var;
                struct BindEntry entry = {
                    .name = string(ctx->rewriter.dst_arena, old_var->name),
                    .bound_node = new_var
                };
                append_list(struct BindEntry, ctx->bound_variables, entry);
                printf("Bound continuation %s\n", entry.name);
            }

            const Node* new_block = block(rewriter->dst_arena, (Block) {
                .continuations_vars = nodes(rewriter->dst_arena, inner_conts_count, nvars),
                .instructions = rewrite_nodes(rewriter, node->payload.block.instructions),
                .continuations = rewrite_nodes(rewriter, node->payload.block.continuations),
            });

            while (entries_count_list(ctx->bound_variables) > old_bound_variables_size)
                pop_list(struct BindEntry, ctx->bound_variables);

            return new_block;
        }
        case Function_TAG: {
            size_t old_bound_variables_size = entries_count_list(ctx->bound_variables);

            size_t params_count = node->payload.fn.params.count;
            LARRAY(const Node*, nparams, params_count);
            for (size_t p = 0; p < params_count; p++) {
                const Variable* old_param = &node->payload.fn.params.nodes[p]->payload.var;
                const Node* new_param = var(rewriter->dst_arena, rewrite_node(rewriter, old_param->type), string(rewriter->dst_arena, old_param->name));
                nparams[p] = new_param;
                struct BindEntry entry = {
                    .name = string(ctx->rewriter.dst_arena, old_param->name),
                    .bound_node = new_param
                };
                append_list(struct BindEntry, ctx->bound_variables, entry);
                printf("Bound param %s\n", entry.name);
            }

            const Node* new_fn = fn(rewriter->dst_arena, (Function) {
                .is_continuation = node->payload.fn.is_continuation,
                .return_types = rewrite_nodes(rewriter, node->payload.fn.return_types),
                .block = bind_node(ctx, node->payload.fn.block),
                .params = nodes(rewriter->dst_arena, params_count, nparams),
            });

            while (entries_count_list(ctx->bound_variables) > old_bound_variables_size)
                pop_list(struct BindEntry, ctx->bound_variables);

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