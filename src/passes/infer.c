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

enum Expectation {
    None,
    YieldType,
    YieldNTypes
};

struct TypeRewriter {
    Rewriter rewriter;
    struct List* typed_variables;
    enum Expectation expectation;
    union {
        const Node* expected_type;
        const Nodes** expected_types;
    } expectation_payload;
    const Nodes* current_fn_expected_return_types;
};

static const Node* resolve(const struct TypeRewriter* ctx, const char* id) {
    for (size_t i = 0; i < entries_count_list(ctx->typed_variables); i++) {
        const struct BindEntry* entry = &read_list(const struct BindEntry, ctx->typed_variables)[i];
        if (strcmp(entry->id, id) == 0) {
            return entry->typed;
        }
    }
    error("could not resolve variable %s", id)
}

const Node* handle_binder(struct TypeRewriter* ctx, const Node* node) {
    assert(ctx->expectation == None && "you don't infer the type of a binder");
    const Node* fresh = recreate_node_identity(&ctx->rewriter, node);
    struct BindEntry entry = {
        .id = node->payload.var.name,
        .typed = fresh
    };
    append_list(struct BindEntry, ctx->typed_variables, entry);
    return fresh;
}

const Node* type_node(struct TypeRewriter* ctx, const Node* node) {
    if (node == NULL)
        return NULL;

    IrArena* dst_arena = ctx->rewriter.dst_arena;

    switch (node->tag) {
        case Variable_TAG:
            return resolve(ctx, node->payload.var.name);
        case Root_TAG: {
            assert(ctx->expectation == None);
            assert(ctx->current_fn_expected_return_types == NULL);
            size_t count = node->payload.root.variables.count;
            const Node* new_variables[count];
            const Node* new_definitions[count];

            for (size_t i = 0; i < count; i++) {
                struct TypeRewriter nctx = *ctx;
                nctx.expectation = None;
                new_variables[i] = handle_binder(ctx, node->payload.root.variables.nodes[i]);
            }
            for (size_t i = 0; i < count; i++) {
                struct TypeRewriter nctx = *ctx;
                nctx.expectation = YieldType;
                nctx.expectation_payload.expected_type = new_variables[i]->type;
                new_definitions[i] = rewrite_node(&nctx.rewriter, node->payload.root.definitions.nodes[i]);
            }

            return root(dst_arena, (Root) {
                .variables = nodes(dst_arena, count, new_variables),
                .definitions = nodes(dst_arena, count, new_definitions)
            });
        }
        case UntypedNumber_TAG: {
            assert(ctx->expectation == YieldType);
            assert(ctx->expectation_payload.expected_type == int_type(dst_arena));
            long v = strtol(node->payload.untyped_number.plaintext, NULL, 10);
            return int_literal(dst_arena, (IntLiteral) { .value = (int) v });
        }
        case Let_TAG: {
            struct TypeRewriter vars_infer_ctx = *ctx;
            vars_infer_ctx.expectation = None;
            const Node* nvars[node->payload.let.variables.count];
            for (size_t i = 0; i < node->payload.let.variables.count; i++)
                nvars[i] = handle_binder(ctx, node->payload.let.variables.nodes[i]);

            const Node* rewritten_rhs = rewrite_node(&ctx->rewriter, node->payload.let.target);
            return let(dst_arena, (Let) {
                .variables = nodes(dst_arena, node->payload.let.variables.count, nvars),
                .target = rewritten_rhs,
            });
        }
        case Function_TAG: {
            assert(ctx->current_fn_expected_return_types == NULL);

            const Node* nparams[node->payload.fn.params.count];
            struct TypeRewriter param_infer_ctx = *ctx;
            param_infer_ctx.expectation = None;
            for (size_t i = 0; i < node->payload.fn.params.count; i++)
               nparams[i] = handle_binder(ctx, node->payload.fn.params.nodes[i]);

            struct TypeRewriter rtypes_infer_ctx = *ctx;
            rtypes_infer_ctx.expectation = None;
            Nodes nret_types = rewrite_nodes(&rtypes_infer_ctx.rewriter, node->payload.fn.return_types);

            struct TypeRewriter instrs_infer_ctx = *ctx;
            instrs_infer_ctx.expectation = None;
            instrs_infer_ctx.current_fn_expected_return_types = &nret_types;

            size_t icount = node->payload.fn.instructions.count;
            const Node* ninstructions[icount];
            for (size_t i = 0; i < icount; i++) {
                ninstructions[i] = rewrite_node(&instrs_infer_ctx.rewriter, node->payload.fn.instructions.nodes[i]);
            }

            return fn(dst_arena, (Function) {
               .return_types = nret_types,
               .instructions = nodes(dst_arena, icount, ninstructions),
               .params = nodes(dst_arena, node->payload.fn.params.count, nparams),
            });
        }
        default: {
            struct TypeRewriter nctx = *ctx;
            const Node* typed = recreate_node_identity(&nctx.rewriter, node);
            assert(typed->type || is_type(typed));
            if (ctx->expectation == YieldType)
                assert(is_subtype(typed->type, ctx->expectation_payload.expected_type));
            return typed;
        }
    }
}

const Node* type_program(IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct List* bound_variables = new_list(struct BindEntry);
    struct TypeRewriter ctx = {
            .rewriter = {
                    .src_arena = src_arena,
                    .dst_arena = dst_arena,
                    .rewrite_fn = (RewriteFn) type_node,
            },
            .typed_variables = bound_variables,
            .expectation = None
    };

    const Node* rewritten = rewrite_node(&ctx.rewriter, src_program);

    destroy_list(bound_variables);
    return rewritten;
}
