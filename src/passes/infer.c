#include "passes.h"

#include "../implem.h"
#include "../type.h"

#include "list.h"

#include <assert.h>
#include <string.h>

struct BindEntry {
    const char* id;
    const struct Node* typed;
};

struct TypeRewriter {
    struct Rewriter rewriter;
    struct List* typed_variables;
    const struct Type* const expected_type;
};

static const struct Node* resolve(struct TypeRewriter* ctx, const char* id) {
    for (size_t i = 0; i < entries_count_list(ctx->typed_variables); i++) {
        const struct BindEntry* entry = &read_list(const struct BindEntry, ctx->typed_variables)[i];
        if (strcmp(entry->id, id) == 0) {
            return entry->typed;
        }
    }
    error("could not resolve variable %s", id)
}

struct TypeRewriter set_expected_type(struct TypeRewriter* octx, struct Type* expected_type) {
    return (struct TypeRewriter) { octx->rewriter, octx->typed_variables, expected_type };
}

const struct Node* type_node(struct TypeRewriter* ctx, const struct Node* node) {
    if (node == NULL)
        return NULL;

    /*switch (node->tag) {
        case Root_TAG: {
            size_t count = node->payload.root.variables.count;
            const struct Node* new_variables[count];
            const struct Node* new_definitions[count];

            for (size_t i = 0; i < count; i++) {
                new_variables[i] = rewrite_node(rewriter, node->payload.root.variables.nodes[i]);
                new_definitions[i] = rewrite_node(rewriter, node->payload.root.definitions.nodes[i]);
            }

            return root(rewriter->dst_arena, (struct Root) {
                .variables = nodes(rewriter->dst_arena, count, new_variables),
                .definitions = nodes(rewriter->dst_arena, count, new_definitions)
            });
        }
        case Function_TAG: {
            struct TypeRewriter nctx = { ctx->rewriter, ctx->typed_variables, NULL };
            struct Nodes nparams = rewrite_nodes(&nctx.rewriter, node->payload.fn.params);
            return fn(rewriter->dst_arena, (struct Function) {
               .return_type = rewrite_type_(rewriter, node->payload.fn.return_type),
               .instructions = rewrite_nodes(rewriter, node->payload.fn.instructions),
               .params = nparams,
            });
        }
        default: {
            struct TypeRewriter nctx = { ctx->rewriter, ctx->typed_variables, ctx->expected_type };
            const struct Node* typed = recreate_node_identity(&nctx.rewriter, node);
            assert(typed->type);
            if (ctx->expected_type)
                assert(is_subtype(typed->type, ctx->expected_type));
            return typed;
        }
    }*/
}

const struct Node* type_program(struct IrArena* src_arena, struct IrArena* dst_arena, const struct Node* src_program) {
    SHADY_NOT_IMPLEM
}
