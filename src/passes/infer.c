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
    // Rewriter rewriter;
    IrArena* dst_arena;
    struct List* typed_variables;
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

const Node* handle_binder(struct TypeRewriter* ctx, const Node* node, const Type* inferred_ty) {
    const char* name = node->payload.var.name;

    const Node* fresh = var(ctx->dst_arena, (Variable) {
        .name = name,
        .type = inferred_ty
    });
    struct BindEntry entry = {
        .id = name,
        .typed = fresh
    };
    append_list(struct BindEntry, ctx->typed_variables, entry);
    return fresh;
}

const Node* type_value(struct TypeRewriter* ctx, const Node* node, const Node* expected_type) {
    IrArena* dst_arena = ctx->dst_arena;
    switch (node->tag) {
        case Variable_TAG:
            return resolve(ctx, node->payload.var.name);
        case UntypedNumber_TAG: {
            assert(expected_type == int_type(dst_arena));
            long v = strtol(node->payload.untyped_number.plaintext, NULL, 10);
            return int_literal(dst_arena, (IntLiteral) { .value = (int) v });
        }
        default: error("not a value");
    }
}

const Node* type_primop_or_call(struct TypeRewriter* ctx, const Node* node, const Type* expected_types[], const Type* inferred_param_tys[]) {
    IrArena* dst_arena = ctx->dst_arena;

    switch (node->tag) {
        case Call_TAG: SHADY_NOT_IMPLEM
        case PrimOp_TAG: {
            Nodes param_tys = op_params(dst_arena, node->payload.primop.op);
            const size_t argsc = node->payload.primop.args.count;
            assert(argsc == param_tys.count);
            const Node* nargs[argsc];
            for (size_t i = 0; i < argsc; i++)
                nargs[i] = type_value(ctx, node->payload.primop.args.nodes[i], expected_types[i]);

            // Nodes yield_tys = op_params(dst_arena, node->payload.primop.op);
            for (size_t i = 0; i < argsc; i++)
                inferred_param_tys[i] = node->type;

            return primop(dst_arena, (PrimOp) {
                .op = node->payload.primop.op,
                .args = nodes(dst_arena, argsc, nargs)
            });
        }
        default: error("not a primop or a call");
    }
}

const Node* type_instruction(struct TypeRewriter* ctx, const Node* node) {
    switch (node->tag) {
        case Let_TAG: {
            const size_t varsc = node->payload.let.variables.count;

            // first import the type annotations (if set)
            const Type* expected_types[varsc];
            for (size_t i = 0; i < varsc; i++)
                expected_types[i] = import_node(ctx->dst_arena, node->payload.let.variables.nodes[i]->type);

            const Type* actual_types[varsc];
            const Node* rewritten_rhs = type_primop_or_call(ctx, node->payload.let.target, expected_types, actual_types);

            struct TypeRewriter vars_infer_ctx = *ctx;
            const Node* nvars[varsc];
            for (size_t i = 0; i < varsc; i++)
                nvars[i] = handle_binder(&vars_infer_ctx, node->payload.let.variables.nodes[i], actual_types[i]);

            return let(ctx->dst_arena, (Let) {
                .variables = nodes(ctx->dst_arena, varsc, nvars),
                .target = rewritten_rhs,
            });
        }
        default: error("not an instruction");
    }
}

const Node* type_node(struct TypeRewriter* ctx, const Node* node) {
    if (node == NULL)
        return NULL;

    IrArena* dst_arena = ctx->rewriter.dst_arena;

    switch (node->tag) {
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

        case Function_TAG: {
            assert(ctx->current_fn_expected_return_types == NULL);

            const Node* nparams[node->payload.fn.params.count];
            struct TypeRewriter param_infer_ctx = *ctx;
            param_infer_ctx.expectation = None;
            for (size_t i = 0; i < node->payload.fn.params.count; i++)
               nparams[i] = handle_binder(&param_infer_ctx, node->payload.fn.params.nodes[i]);

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
