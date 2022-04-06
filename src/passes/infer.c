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
const Node* type_value(struct TypeRewriter* ctx, const Node* node, const Node* expected_type);

const Node* type_block(struct TypeRewriter* ctx, const Node* node) {
    // const Block* block = &node->payload.block;
    size_t old_typed_variables_size = entries_count_list(ctx->typed_variables);

    size_t count = node->payload.block.continuations_vars.count;
    LARRAY(const Node*, ninstructions, node->payload.block.instructions.count);
    LARRAY(const Node*, new_variables, count);
    LARRAY(const Node*, new_definitions, count);

    for (size_t i = 0; i < count; i++) {
        const Variable* oldvar = &node->payload.block.continuations_vars.nodes[i]->payload.var;
        const Type* imported_ty = import_node(ctx->dst_arena, oldvar->type);
        new_variables[i] = new_binder(ctx, oldvar->name, imported_ty);
    }

    for (size_t i = 0; i < count; i++) {
        const Variable* oldvar = &node->payload.block.continuations_vars.nodes[i]->payload.var;
        const Type* imported_ty = import_node(ctx->dst_arena, oldvar->type);

        // Some top-level stuff does not have a definition
        new_definitions[i] = type_value(ctx, node->payload.block.continuations.nodes[i], imported_ty);
    }

    for (size_t i = 0; i < node->payload.block.instructions.count; i++)
        ninstructions[i] = type_instruction(ctx, node->payload.block.instructions.nodes[i]);

    Nodes typed_instructions = nodes(ctx->dst_arena, node->payload.block.instructions.count, ninstructions);
    Nodes typed_conts_vars = nodes(ctx->dst_arena, count, new_variables);
    Nodes typed_conts = nodes(ctx->dst_arena, count, new_definitions);

    while (entries_count_list(ctx->typed_variables) > old_typed_variables_size)
        pop_list(struct BindEntry, ctx->typed_variables);

    return block(ctx->dst_arena, (Block) {
        .instructions = typed_instructions,
        .continuations_vars = typed_conts_vars,
        .continuations = typed_conts
    });
}

static const Node* type_value_impl(struct TypeRewriter* ctx, const Node* node, const Node* expected_type) {
    IrArena* dst_arena = ctx->dst_arena;
    switch (node->tag) {
        case Variable_TAG:
            return resolve(ctx, node->payload.var.name);
        case UntypedNumber_TAG: {
            // TODO handle different prim types
            assert(without_qualifier(expected_type) == int_type(dst_arena));
            long v = strtol(node->payload.untyped_number.plaintext, NULL, 10);
            return int_literal(dst_arena, (IntLiteral) { .value = (int) v });
        }
        case Continuation_TAG: {
            size_t old_typed_variables_size = entries_count_list(ctx->typed_variables);

            // TODO handle expected_type
            LARRAY(const Node*, nparams, node->payload.cont.params.count);
            for (size_t i = 0; i < node->payload.cont.params.count; i++)
               nparams[i] = new_binder(ctx, node->payload.cont.params.nodes[i]->payload.var.name, import_node(dst_arena, node->payload.cont.params.nodes[i]->payload.var.type));

            // Handle the insides of the function
            struct TypeRewriter instrs_infer_ctx = *ctx;
            const Node* nblock = type_block(&instrs_infer_ctx, node->payload.cont.block);

            const Node* continuation = cont(dst_arena, (Continuation) {
               .block = nblock,
               .params = nodes(dst_arena, node->payload.cont.params.count, nparams),
            });

            while (entries_count_list(ctx->typed_variables) > old_typed_variables_size)
                pop_list(struct BindEntry, ctx->typed_variables);

            return continuation;
        }
        case Function_TAG: {
            size_t old_typed_variables_size = entries_count_list(ctx->typed_variables);

            // TODO handle expected_type
            LARRAY(const Node*, nparams, node->payload.fn.params.count);
            for (size_t i = 0; i < node->payload.fn.params.count; i++)
               nparams[i] = new_binder(ctx, node->payload.fn.params.nodes[i]->payload.var.name, import_node(dst_arena, node->payload.fn.params.nodes[i]->payload.var.type));

            Nodes nret_types = import_nodes(ctx->dst_arena, node->payload.fn.return_types);

            // Handle the insides of the function
            struct TypeRewriter instrs_infer_ctx = *ctx;
            instrs_infer_ctx.current_fn_expected_return_types = &nret_types;
            const Node* nblock = type_block(&instrs_infer_ctx, node->payload.fn.block);

            const Node* fun = fn(dst_arena, (Function) {
               .return_types = nret_types,
               .block = nblock,
               .params = nodes(dst_arena, node->payload.fn.params.count, nparams),
            });

            while (entries_count_list(ctx->typed_variables) > old_typed_variables_size)
                pop_list(struct BindEntry, ctx->typed_variables);

            return fun;
        }
        case True_TAG: return true_lit(dst_arena);
        case False_TAG: return false_lit(dst_arena);
        default: error("not a value");
    }
}

const Node* type_value(struct TypeRewriter* ctx, const Node* node, const Node* expected_type) {
    const Node* typed = type_value_impl(ctx, node, expected_type);
    return typed;
}

Nodes type_primop_or_call(struct TypeRewriter* ctx, Op op, Nodes oldargs, size_t expected_count, const Type* actual_yield_types[]) {
    IrArena* dst_arena = ctx->dst_arena;

    Nodes param_tys = op_params(dst_arena, op, oldargs);

    const size_t argsc = oldargs.count;
    assert(argsc == param_tys.count);
    LARRAY(const Node*, nargs, argsc);
    for (size_t i = 0; i < argsc; i++)
        nargs[i] = type_value(ctx, oldargs.nodes[i], param_tys.nodes[i]);

    Nodes typed_args = nodes(dst_arena, argsc, nargs);
    Nodes yield_tys = op_yields(dst_arena, op, typed_args);

    assert(expected_count == yield_tys.count);
    for (size_t i = 0; i < yield_tys.count; i++)
        actual_yield_types[i] = yield_tys.nodes[i];

    return typed_args;
}

Nodes type_values(struct TypeRewriter* ctx, Nodes old) {
    LARRAY(const Node*, tmp, old.count);
    for (size_t i = 0; i < old.count; i++)
        tmp[i] = type_value(ctx, old.nodes[i], NULL);
    return nodes(ctx->dst_arena, old.count, tmp);
}

const Node* type_instruction(struct TypeRewriter* ctx, const Node* node) {
    switch (node->tag) {
        case Let_TAG: {
            const size_t count = node->payload.let.variables.count;

            LARRAY(const Type*, actual_types, count);
            Nodes rewritten_args = type_primop_or_call(ctx, node->payload.let.op, node->payload.let.args, count, actual_types);

            struct TypeRewriter vars_infer_ctx = *ctx;
            LARRAY(const Node*, nvars, count);
            for (size_t i = 0; i < count; i++)
                nvars[i] = new_binder(&vars_infer_ctx, node->payload.let.variables.nodes[i]->payload.var.name, actual_types[i]);

            return let(ctx->dst_arena, (Let) {
                .variables = nodes(ctx->dst_arena, count, nvars),
                .op = node->payload.let.op,
                .args = rewritten_args
            });
        }
        case StructuredSelection_TAG: {
            const Node* condition = type_value(ctx, node->payload.selection.condition, bool_type(ctx->dst_arena));

            struct TypeRewriter instrs_infer_ctx = *ctx;
            const Node* ifTrue = type_block(&instrs_infer_ctx, node->payload.selection.ifTrue);
            const Node* ifFalse = type_block(&instrs_infer_ctx, node->payload.selection.ifFalse);
            return selection(ctx->dst_arena, (StructuredSelection) {
                .condition = condition,
                .ifTrue = ifTrue,
                .ifFalse = ifFalse
            });
        }
        case Return_TAG: {
            const Nodes* old_values = &node->payload.fn_ret.values;
            LARRAY(const Node*, nvalues, old_values->count);
            for (size_t i = 0; i < old_values->count; i++)
                nvalues[i] = type_value(ctx, old_values->nodes[i], ctx->current_fn_expected_return_types->nodes[i]);
            return fn_ret(ctx->dst_arena, (Return) {
                .values = nodes(ctx->dst_arena, old_values->count, nvalues)
            });
        }
        case Jump_TAG: {
            const Node* ntarget = type_value(ctx, node->payload.jump.target, NULL);

            assert(get_qualifier(ntarget->type) == Uniform);
            assert(without_qualifier(ntarget->type)->tag == ContType_TAG);
            const ContType* tgt_type = &without_qualifier(ntarget->type)->payload.cont_type;

            LARRAY(const Node*, tmp, node->payload.jump.args.count);
            for (size_t i = 0; i < node->payload.jump.args.count; i++)
                tmp[i] = type_value(ctx, node->payload.jump.args.nodes[i], tgt_type->param_types.nodes[i]);

            Nodes new_args = nodes(ctx->dst_arena, node->payload.jump.args.count, tmp);

            return jump(ctx->dst_arena, (Jump) {
                .target = ntarget,
                .args = new_args
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
            LARRAY(const Node*, new_variables, count);
            LARRAY(const Node*, new_definitions, count);

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
                new_variables[i] = new_binder(ctx, oldvar->name, new_definitions[i]->type);
            }

            return root(ctx->dst_arena, (Root) {
                .variables = nodes(ctx->dst_arena, count, new_variables),
                .definitions = nodes(ctx->dst_arena, count, new_definitions)
            });
        }
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
