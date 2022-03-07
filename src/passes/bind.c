#include "passes.h"

#include "list.h"

#include "../implem.h"

#include <assert.h>
#include <string.h>

struct BindEntry {
    const char* id;
    // const struct Node* old_node;
    const struct Node* new_node;
};

struct BindRewriter {
    struct Rewriter rewriter;
    struct List* bound_variables;
};

const struct Type* derive_fn_type(struct IrArena* arena, struct Function fn);

const struct Node* resolve(struct BindRewriter* ctx, const char* id) {
    for (size_t i = 0; i < entries_count_list(ctx->bound_variables); i++) {
        const struct BindEntry* entry = &read_list(const struct BindEntry, ctx->bound_variables)[i];
        if (strcmp(entry->id, id) == 0) {
            /*if (entry->new_node == NULL) {
                switch (entry->old_node) {

                }
            }*/
            return entry->new_node;
        }
    }
    error("could not resolve variable %s", id)
}

const struct Node* bind_node(struct BindRewriter* ctx, const struct Node* node) {
    if (node == NULL)
        return NULL;

    struct Rewriter* rewriter = &ctx->rewriter;
    switch (node->tag) {
        case Variable_TAG: {
            assert(node->payload.var.type == NULL);
            return resolve(ctx, node->payload.var.name);
        }
        case VariableDecl_TAG: {
            const struct Node* new_variable = var(rewriter->dst_arena, (struct Variable) {
                .name = string(rewriter->dst_arena, node->payload.var_decl.variable->payload.var.name),
                .type = rewriter->rewrite_type(rewriter, node->payload.var_decl.variable->payload.var.type)
            });
            struct BindEntry entry = {
                .id = string(ctx->rewriter.dst_arena, node->payload.var_decl.variable->payload.var.name),
                .new_node = new_variable
            };
            append_list(struct BindEntry, ctx->bound_variables, entry);
            return var_decl(ctx->rewriter.dst_arena, (struct VariableDecl){
                .variable = new_variable,
                .address_space = node->payload.var_decl.address_space,
                .init = rewriter->rewrite_node(rewriter, node->payload.var_decl.init)
            });
        }
        case Function_TAG: {
            size_t old_bound_variables_size = entries_count_list(ctx->bound_variables);

            const struct Node* new_fn = fn(rewriter->dst_arena, (struct Function) {
                .name = string(rewriter->dst_arena, node->payload.fn.name),
                .return_type = rewriter->rewrite_type(rewriter, node->payload.fn.return_type),
                .instructions = rewrite_nodes(rewriter, node->payload.fn.params),
                .params = rewrite_nodes(rewriter, node->payload.fn.params),
            });

            while (entries_count_list(ctx->bound_variables) > old_bound_variables_size)
                pop_list(struct BindEntry, ctx->bound_variables);

            return new_fn;
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

struct Program bind_program(struct IrArena* src_arena, struct IrArena* dst_arena, struct Program* src_program) {
    const struct Node* new_top_level[src_program->declarations_and_definitions.count];

    struct List* bound_variables = new_list(struct BindEntry);
    struct BindRewriter ctx = {
        .rewriter = {
            .src_arena = src_arena,
            .dst_arena = dst_arena,
            .rewrite_node = (NodeRewriteFn) bind_node,
            .rewrite_type = (TypeRewriteFn) recreate_type_identity
        },
        .bound_variables = bound_variables
    };

    for (size_t i = 0; i < src_program->declarations_and_definitions.count; i++) {
        const struct Node* decl = src_program->declarations_and_definitions.nodes[i];
        switch (decl->tag) {
            case VariableDecl_TAG: {
                struct BindEntry entry = {
                    .id = decl->payload.var_decl.variable->payload.var.name,
                    .new_node = var(dst_arena, (struct Variable) {
                        .name = string(dst_arena, decl->payload.var_decl.variable->payload.var.name),
                        .type = ctx.rewriter.rewrite_type(&ctx.rewriter, decl->payload.var_decl.variable->payload.var.type)
                    })
                };
                append_list(struct BindEntry, bound_variables, entry);
                break;
            }
            case Function_TAG: {
                struct BindEntry entry = {
                    .id = string(dst_arena, decl->payload.fn.name),
                    .new_node = var(dst_arena, (struct Variable) {
                        .name = string(dst_arena, decl->payload.fn.name),
                        .type = ctx.rewriter.rewrite_type(&ctx.rewriter, derive_fn_type(src_arena, decl->payload.fn))
                    })
                };
                append_list(struct BindEntry, bound_variables, entry);
                break;
            }
            default: error("not a decl");
        }
    }

    for (size_t i = 0; i < src_program->declarations_and_definitions.count; i++) {
        new_top_level[i] = bind_node(&ctx, src_program->declarations_and_definitions.nodes[i]);
    }

    destroy_list(bound_variables);

    return (struct Program) { nodes(dst_arena, src_program->declarations_and_definitions.count, new_top_level) };
}