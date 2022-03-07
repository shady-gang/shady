#include "passes.h"

#include "list.h"

#include "../implem.h"

#include <assert.h>
#include <string.h>

struct BindEntry {
    const char* id;
    const struct Node* old_node;
    const struct Node* new_node;
};

struct BindRewriter {
    struct Rewriter rewriter;
    struct List* bound_variables;
};

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
    switch (node->tag) {
        case Variable_TAG: {
            assert(node->payload.var.type == NULL);
            return resolve(ctx, node->payload.var.name);
        }
        case VariableDecl_TAG: {
            error("todo")
        }
        case Function_TAG: {
            // TODO create extra new context for function
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

struct Program bind_program(struct IrArena* src_arena, struct IrArena* dst_arena, struct Program* src_program) {
    const struct Node* new_top_level[src_program->declarations_and_definitions.count];

    struct List* bound_variables = new_list(struct BindEntry);
    struct BindRewriter rewriter = {
        .rewriter = {
            .src_arena = src_arena,
            .dst_arena = dst_arena,
            .rewrite_node = (NodeRewriteFn) bind_node,
            .rewrite_type = (TypeRewriteFn) recreate_node_identity
        }
    };

    for (size_t i = 0; i < src_program->declarations_and_definitions.count; i++) {
        const struct Node* decl = src_program->declarations_and_definitions.nodes[i];
        switch (decl->tag) {
            case VariableDecl_TAG: {
                struct BindEntry entry = {
                    .id = decl->payload.var_decl.variable->payload.var.name,
                    .old_node = decl,
                    .new_node = NULL
                };
                append_list(struct BindEntry, bound_variables, entry);
                break;
            }
            case Function_TAG: {
                struct BindEntry entry = {
                    .id = decl->payload.fn.name,
                    .old_node = decl,
                    .new_node = NULL
                };
                append_list(struct BindEntry, bound_variables, entry);
                break;
            }
            default: error("not a decl");
        }
    }

    for (size_t i = 0; i < src_program->declarations_and_definitions.count; i++) {
        new_top_level[i] = bind_node(&rewriter, src_program->declarations_and_definitions.nodes[i]);
    }

    destroy_list(bound_variables);

    return (struct Program) { nodes(dst_arena, src_program->declarations_and_definitions.count, new_top_level) };
}