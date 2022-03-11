#include "passes.h"

#include "../implem.h"
#include "../type.h"

#include "list.h"

#include <assert.h>
#include <string.h>

struct BindEntry {
    const char* id;
    const Node* typed;
};

struct TypeRewriter {
    Rewriter rewriter;
    struct List* typed_variables;
    const Type* expected_type;
};

static const Node* resolve(struct TypeRewriter* ctx, const char* id) {
    for (size_t i = 0; i < entries_count_list(ctx->typed_variables); i++) {
        const struct BindEntry* entry = &read_list(const struct BindEntry, ctx->typed_variables)[i];
        if (strcmp(entry->id, id) == 0) {
            return entry->typed;
        }
    }
    error("could not resolve variable %s", id)
}

const Node* type_node(const struct TypeRewriter* ctx, const Node* node) {
    if (node == NULL)
        return NULL;

    IrArena* dst_arena = ctx->rewriter.dst_arena;

    /*switch (node->tag) {
        case Root_TAG: {
            size_t count = node->payload.root.variables.count;
            const Node* new_variables[count];
            const Node* new_definitions[count];

            for (size_t i = 0; i < count; i++) {
                struct TypeRewriter nctx = *ctx;
                nctx.expected_type = NULL;
                new_variables[i] = rewrite_node(&nctx.rewriter, node->payload.root.variables.nodes[i]);

                nctx.expected_type = new_variables[i]->type;
                new_definitions[i] = rewrite_node(&nctx.rewriter, node->payload.root.definitions.nodes[i]);
            }

            return root(dst_arena, (struct Root) {
                .variables = nodes(dst_arena, count, new_variables),
                .definitions = nodes(dst_arena, count, new_definitions)
            });
        }
        case Let_TAG: {
            struct TypeRewriter vars_infer_ctx = *ctx;
            vars_infer_ctx.expected_type = NULL;

            struct Nodes nvars = rewrite_nodes(&vars_infer_ctx.rewriter, node->payload.let.variables);
        }
        case Function_TAG: {
            struct TypeRewriter param_infer_ctx = *ctx;
            param_infer_ctx.expected_type = NULL;
            struct Nodes nparams = rewrite_nodes(&param_infer_ctx.rewriter, node->payload.fn.params);

            size_t icount = node->payload.fn.instructions.count;
            Node* ninstructions[icount];
            for (size_t i = 0; i < icount; i++) {
                ninstructions[i] = rewrite_node(rewriter, node->payload.fn.instructions),
            }

            return fn(dst_arena, (struct Function) {
               .return_type = rewrite_type(rewriter, node->payload.fn.return_type),
               .instructions = rewrite_nodes(rewriter, node->payload.fn.instructions),
               .params = nparams,
            });
        }
        default: {
            struct TypeRewriter nctx = { ctx->rewriter, ctx->typed_variables, ctx->expected_type };
            const Node* typed = recreate_node_identity(&nctx.rewriter, node);
            assert(typed->type);
            if (ctx->expected_type)
                assert(is_subtype(typed->type, ctx->expected_type));
            return typed;
        }
    }*/
}

const Node* type_program(IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    SHADY_NOT_IMPLEM
}
