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

const Node* new_binder(struct TypeRewriter* ctx, const char* oldname, const Type* inferred_ty) {
    const char* name = string(ctx->dst_arena, oldname);
    const Node* fresh = var(ctx->dst_arena, inferred_ty, name);
    struct BindEntry entry = {
        .id = name,
        .typed = fresh
    };
    append_list(struct BindEntry, ctx->typed_variables, entry);
    return fresh;
}

const Node* type_instruction(struct TypeRewriter* ctx, const Node* node);

Nodes type_block(struct TypeRewriter* ctx, const Nodes* block) {
    const Node* ninstructions[block->count];
    for (size_t i = 0; i < block->count; i++)
        ninstructions[i] = type_instruction(ctx, block->nodes[i]);
    return nodes(ctx->dst_arena, block->count, ninstructions);
}

static const Node* type_value_impl(struct TypeRewriter* ctx, const Node* node, const Node* expected_type) {
    IrArena* dst_arena = ctx->dst_arena;
    switch (node->tag) {
        case Variable_TAG:
            return resolve(ctx, node->payload.var.name);
        case UntypedNumber_TAG: {
            // TODO handle different prim types
            assert(expected_type == int_type(dst_arena));
            long v = strtol(node->payload.untyped_number.plaintext, NULL, 10);
            return int_literal(dst_arena, (IntLiteral) { .value = (int) v });
        }
        case Function_TAG: {
            size_t old_typed_variables_size = entries_count_list(ctx->typed_variables);

            // TODO handle expected_type
            const Node* nparams[node->payload.fn.params.count];
            for (size_t i = 0; i < node->payload.fn.params.count; i++)
               nparams[i] = new_binder(ctx, node->payload.fn.params.nodes[i]->payload.var.name, import_node(dst_arena, node->payload.fn.params.nodes[i]->payload.var.type));

            Nodes nret_types = import_nodes(ctx->dst_arena, node->payload.fn.return_types);

            // Handle the insides of the function
            struct TypeRewriter instrs_infer_ctx = *ctx;
            instrs_infer_ctx.current_fn_expected_return_types = &nret_types;
            Nodes ninstructions = type_block(&instrs_infer_ctx, &node->payload.fn.instructions);

            const Node* fun = fn(dst_arena, (Function) {
               .return_types = nret_types,
               .instructions = ninstructions,
               .params = nodes(dst_arena, node->payload.fn.params.count, nparams),
            });

            while (entries_count_list(ctx->typed_variables) > old_typed_variables_size)
                pop_list(struct BindEntry, ctx->typed_variables);

            return fun;
        }
        default: error("not a value");
    }
}

const Node* type_value(struct TypeRewriter* ctx, const Node* node, const Node* expected_type) {
    const Node* typed = type_value_impl(ctx, node, expected_type);
    assert(typed->yields.count == 1);
    return typed;
}

const Node* type_primop_or_call(struct TypeRewriter* ctx, const Node* node, size_t expected_yield_count, const Type* expected_yield_types[], const Type* actual_yield_types[]) {
    IrArena* dst_arena = ctx->dst_arena;

    switch (node->tag) {
        case Call_TAG: SHADY_NOT_IMPLEM
        case PrimOp_TAG: {
            Nodes param_tys = op_params(dst_arena, node->payload.primop.op);
            const size_t argsc = node->payload.primop.args.count;
            assert(argsc == param_tys.count);
            const Node* nargs[argsc];
            for (size_t i = 0; i < argsc; i++)
                nargs[i] = type_value(ctx, node->payload.primop.args.nodes[i], param_tys.nodes[i]);

            const Node* new = primop(dst_arena, (PrimOp) {
                .op = node->payload.primop.op,
                .args = nodes(dst_arena, argsc, nargs)
            });

            const Nodes yield_tys = new->yields;
            assert(expected_yield_count == yield_tys.count);
            for (size_t i = 0; i < yield_tys.count; i++)
                actual_yield_types[i] = yield_tys.nodes[i];

            return new;
        }
        default: error("not a primop or a call");
    }
}

const Node* type_instruction(struct TypeRewriter* ctx, const Node* node) {
    switch (node->tag) {
        case Let_TAG: {
            const size_t count = node->payload.let.variables.count;

            // first import the type annotations (if set)
            const Type* expected_types[count];
            for (size_t i = 0; i < count; i++)
                expected_types[i] = import_node(ctx->dst_arena, node->payload.let.variables.nodes[i]->payload.var.type);

            const Type* actual_types[count];
            const Node* rewritten_rhs = type_primop_or_call(ctx, node->payload.let.target, count, expected_types, actual_types);

            struct TypeRewriter vars_infer_ctx = *ctx;
            const Node* nvars[count];
            for (size_t i = 0; i < count; i++)
                nvars[i] = new_binder(&vars_infer_ctx, node->payload.let.variables.nodes[i]->payload.var.name, actual_types[i]);

            return let(ctx->dst_arena, (Let) {
                .variables = nodes(ctx->dst_arena, count, nvars),
                .target = rewritten_rhs,
            });
        }
        case Return_TAG: {
            const Nodes* old_values = &node->payload.fn_ret.values;
            const Node* nvalues[old_values->count];
            for (size_t i = 0; i < old_values->count; i++)
                nvalues[i] = type_value(ctx, old_values->nodes[i], NULL);
            return fn_ret(ctx->dst_arena, (Return) {
                .values = nodes(ctx->dst_arena, old_values->count, nvalues)
            });
        }
        default: error("not an instruction");
    }
}

const Node* type_root(struct TypeRewriter* ctx, const Node* node) {
    if (node == NULL)
        return NULL;

    switch (node->tag) {
        case Root_TAG: {
            assert(ctx->current_fn_expected_return_types == NULL);
            size_t count = node->payload.root.variables.count;
            const Node* new_variables[count];
            const Node* new_definitions[count];

            for (size_t i = 0; i < count; i++) {
                const Variable* oldvar = &node->payload.root.variables.nodes[i]->payload.var;
                const Type* imported_ty = import_node(ctx->dst_arena, oldvar->type);

                // Some top-level stuff does not have a definition
                if (node->payload.root.definitions.nodes[i] == NULL) {
                    new_variables[i] = new_binder(ctx, oldvar->name, imported_ty);
                    new_definitions[i] = NULL;
                } else {
                    new_definitions[i] = type_value(ctx, node->payload.root.definitions.nodes[i], imported_ty);
                }
            }

            for (size_t i = 0; i < count; i++) {
                if (node->payload.root.definitions.nodes[i] == NULL) continue;

                const Variable* oldvar = &node->payload.root.variables.nodes[i]->payload.var;
                assert(new_definitions[i]->yields.count == 1);
                new_variables[i] = new_binder(ctx, oldvar->name, new_definitions[i]->yields.nodes[0]);
            }

            return root(ctx->dst_arena, (Root) {
                .variables = nodes(ctx->dst_arena, count, new_variables),
                .definitions = nodes(ctx->dst_arena, count, new_definitions)
            });
        }
        /*default: {
            struct TypeRewriter nctx = *ctx;
            const Node* typed = recreate_node_identity(&nctx.rewriter, node);
            assert(typed->type || is_type(typed));
            if (ctx->expectation == YieldType)
                assert(is_subtype(typed->type, ctx->expectation_payload.expected_type));
            return typed;
        }*/
        default: error("not a root node");
    }
}

const Node* type_program(IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct List* bound_variables = new_list(struct BindEntry);
    struct TypeRewriter ctx = {
            .dst_arena = dst_arena,
            .typed_variables = bound_variables,
            .current_fn_expected_return_types = NULL,
    };

    const Node* rewritten = type_root(&ctx, src_program);

    destroy_list(bound_variables);
    return rewritten;
}
