#include "passes.h"

#include "list.h"

#include "../implem.h"

#include <assert.h>
#include <string.h>

struct BindEntry {
    const char* id;
    const struct Node* bound_node;
};

struct BindRewriter {
    struct Rewriter rewriter;
    struct List* bound_variables;
};

static const struct Node* resolve(struct BindRewriter* ctx, const char* id) {
    for (size_t i = 0; i < entries_count_list(ctx->bound_variables); i++) {
        const struct BindEntry* entry = &read_list(const struct BindEntry, ctx->bound_variables)[i];
        if (strcmp(entry->id, id) == 0) {
            return entry->bound_node;
        }
    }
    error("could not resolve variable %s", id)
}

const struct Node* bind_node(struct BindRewriter* ctx, const struct Node* node) {
    if (node == NULL)
        return NULL;

    struct Rewriter* rewriter = &ctx->rewriter;
    switch (node->tag) {
        case Root_TAG: {
            const struct Root* src_root = &node->payload.root;
            const size_t count = src_root->variables.count;

            const struct Node* new_variables[count];
            const struct Node* new_definitions[count];

            for (size_t i = 0; i < count; i++) {
                const struct Node* variable = src_root->variables.nodes[i];

                const struct Node* new_variable = var(rewriter->dst_arena, (struct Variable) {
                    .name = string(rewriter->dst_arena, variable->payload.var.name),
                    .type = rewrite_node(rewriter, variable->payload.var.type)
                });

                struct BindEntry entry = {
                    .id = variable->payload.var.name,
                    .bound_node = new_variable
                };
                append_list(struct BindEntry, ctx->bound_variables, entry);
                new_variables[i] = new_variable;
            }

            for (size_t i = 0; i < count; i++)
                new_definitions[i] = bind_node(ctx, src_root->definitions.nodes[i]);

            return root(rewriter->dst_arena, (struct Root) {
                .variables = nodes(rewriter->dst_arena, count, new_variables),
                .definitions = nodes(rewriter->dst_arena, count, new_definitions)
            });
        }
        case Variable_TAG: {
            assert(node->payload.var.type == NULL);
            return resolve(ctx, node->payload.var.name);
        }
        case Let_TAG: {
            size_t outputs_count = node->payload.let.variables.count;
            const struct Node* noutputs[outputs_count];
            for (size_t p = 0; p < outputs_count; p++) {
                const struct Variable* old_var = &node->payload.let.variables.nodes[p]->payload.var;
                const struct Node* new_binding = var(rewriter->dst_arena, (struct Variable) {
                    .name = string(rewriter->dst_arena, old_var->name),
                    .type = rewrite_node(rewriter, old_var->type)
                });
                noutputs[p] = new_binding;
                struct BindEntry entry = {
                    .id = string(ctx->rewriter.dst_arena, old_var->name),
                    .bound_node = new_binding
                };
                append_list(struct BindEntry, ctx->bound_variables, entry);
                printf("Bound %s\n", entry.id);
            }

            return let(rewriter->dst_arena, (struct Let) {
                .variables = nodes(rewriter->dst_arena, outputs_count, noutputs),
                .target = rewrite_node(rewriter, node->payload.let.target)
            });
        }
        case Function_TAG: {
            size_t old_bound_variables_size = entries_count_list(ctx->bound_variables);

            size_t params_count = node->payload.fn.params.count;
            const struct Node* nparams[params_count];
            for (size_t p = 0; p < params_count; p++) {
                const struct Variable* old_param = &node->payload.fn.params.nodes[p]->payload.var;
                const struct Node* new_param = var(rewriter->dst_arena, (struct Variable) {
                    .name = string(rewriter->dst_arena, old_param->name),
                    .type = rewrite_node(rewriter, old_param->type)
                });
                nparams[p] = new_param;
                struct BindEntry entry = {
                    .id = string(ctx->rewriter.dst_arena, old_param->name),
                    .bound_node = new_param
                };
                append_list(struct BindEntry, ctx->bound_variables, entry);
                printf("Bound %s\n", entry.id);
            }

            const struct Node* new_fn = fn(rewriter->dst_arena, (struct Function) {
                .return_type = rewrite_node(rewriter, node->payload.fn.return_type),
                .instructions = rewrite_nodes(rewriter, node->payload.fn.instructions),
                .params = nodes(rewriter->dst_arena, params_count, nparams),
            });

            while (entries_count_list(ctx->bound_variables) > old_bound_variables_size)
                pop_list(struct BindEntry, ctx->bound_variables);

            return new_fn;
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

const struct Node* bind_program(struct IrArena* src_arena, struct IrArena* dst_arena, const struct Node* source) {
    struct List* bound_variables = new_list(struct BindEntry);
    struct BindRewriter ctx = {
        .rewriter = {
            .src_arena = src_arena,
            .dst_arena = dst_arena,
            .rewrite_fn = (RewriteFn) bind_node,
        },
        .bound_variables = bound_variables
    };

    const struct Node* rewritten = rewrite_node(&ctx.rewriter, source);

    destroy_list(bound_variables);
    return rewritten;
}