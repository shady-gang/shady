#include "passes.h"

#include "../implem.h"
#include "../type.h"

#include "list.h"
#include "dict.h"

#include <assert.h>
#include <string.h>

struct BindEntry {
    VarId id;
    const Node* typed;
};

struct TypeRewriter {
    IrArena* dst_arena;
    struct List* typed_variables;
    struct Dict* rewritten_fns;
    // const Nodes* current_fn_expected_return_types;
};

static const Node* resolve(const struct TypeRewriter* ctx, VarId id) {
    for (size_t i = 0; i < entries_count_list(ctx->typed_variables); i++) {
        const struct BindEntry* entry = &read_list(const struct BindEntry, ctx->typed_variables)[i];
        if (entry->id == id) {
            return entry->typed;
        }
    }
    error("could not resolve variable %d", id)
}

static const Node* new_binder(struct TypeRewriter* ctx, const char* old_name, const Type* inferred_ty, const VarId old_id) {
    const char* name = string(ctx->dst_arena, old_name);
    const Node* fresh = var(ctx->dst_arena, inferred_ty, name);
    struct BindEntry entry = {
        .id = old_id,
        .typed = fresh
    };
    append_list(struct BindEntry, ctx->typed_variables, entry);
    return fresh;
}

static const Node* type_instruction(struct TypeRewriter* ctx, const Node* node);
static const Node* type_value(struct TypeRewriter* ctx, const Node* node, const Node* expected_type);

static const Node* type_block(struct TypeRewriter* ctx, const Node* node) {
    size_t old_typed_variables_size = entries_count_list(ctx->typed_variables);

    LARRAY(const Node*, ninstructions, node->payload.block.instructions.count);

    for (size_t i = 0; i < node->payload.block.instructions.count; i++)
        ninstructions[i] = type_instruction(ctx, node->payload.block.instructions.nodes[i]);

    Nodes typed_instructions = nodes(ctx->dst_arena, node->payload.block.instructions.count, ninstructions);

    while (entries_count_list(ctx->typed_variables) > old_typed_variables_size)
        pop_list(struct BindEntry, ctx->typed_variables);

    return block(ctx->dst_arena, (Block) {
        .instructions = typed_instructions
    });
}

static const Node* type_fn(struct TypeRewriter* ctx, const Node* node) {
    IrArena* dst_arena = ctx->dst_arena;
    assert(node->tag == Function_TAG);

    Node** already_done = find_value_dict(const Node*, Node*, ctx->rewritten_fns, node);
    if (already_done)
        return *already_done;

    size_t old_typed_variables_size = entries_count_list(ctx->typed_variables);

    LARRAY(const Node*, nparams, node->payload.fn.params.count);
    for (size_t i = 0; i < node->payload.fn.params.count; i++) {
        const Variable* old_param = &node->payload.fn.params.nodes[i]->payload.var;
        const Type* imported_param_type = import_node(dst_arena, node->payload.fn.params.nodes[i]->payload.var.type);
        nparams[i] = new_binder(ctx, old_param->name, imported_param_type, old_param->id);
    }

    Nodes nret_types = import_nodes(ctx->dst_arena, node->payload.fn.return_types);

    Node* fun = (Node*) fn(dst_arena, (Function) {
        .name = string(dst_arena, node->payload.fn.name),
        .is_continuation = node->payload.fn.is_continuation,
        .params = nodes(dst_arena, node->payload.fn.params.count, nparams),
        .return_types = nret_types,
    });
    bool r = insert_dict_and_get_result(const Node*, Node*, ctx->rewritten_fns, node, fun);
    assert(r && "insertion of fun failed - the dict isn't working as it should");

    // Handle the insides of the function - if this is a function indeed
    struct TypeRewriter instrs_infer_ctx = *ctx;
    //if (!node->payload.fn.is_continuation)
    //    instrs_infer_ctx.current_fn_expected_return_types = &nret_types;
    const Node* nblock = type_block(&instrs_infer_ctx, node->payload.fn.block);

    fun->payload.fn.block = nblock;

    while (entries_count_list(ctx->typed_variables) > old_typed_variables_size)
        pop_list(struct BindEntry, ctx->typed_variables);

    return fun;
}

static const Node* type_value(struct TypeRewriter* ctx, const Node* node, const Node* expected_type) {
    IrArena* dst_arena = ctx->dst_arena;
    switch (node->tag) {
        case Variable_TAG:
            return resolve(ctx, node->payload.var.id);
        case UntypedNumber_TAG: {
            // TODO handle different prim types
            assert(without_qualifier(expected_type) == int_type(dst_arena));
            long v = strtol(node->payload.untyped_number.plaintext, NULL, 10);
            return int_literal(dst_arena, (IntLiteral) { .value = (int) v });
        }
        case True_TAG: return true_lit(dst_arena);
        case False_TAG: return false_lit(dst_arena);
        default: error("not a value");
    }
}

static const Node* type_value_or_def(struct TypeRewriter* ctx, const Node* node, const Node* expected_type) {
    const Node* typed = node->tag == Function_TAG ? type_fn(ctx, node) : type_value(ctx, node, expected_type);
    return typed;
}

static Nodes type_primop_or_call(struct TypeRewriter* ctx, Op op, Nodes oldargs, size_t expected_count, const Type* actual_yield_types[]) {
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

static Nodes type_values(struct TypeRewriter* ctx, Nodes old) {
    LARRAY(const Node*, tmp, old.count);
    for (size_t i = 0; i < old.count; i++)
        tmp[i] = type_value(ctx, old.nodes[i], NULL);
    return nodes(ctx->dst_arena, old.count, tmp);
}

static const Node* type_instruction(struct TypeRewriter* ctx, const Node* node) {
    switch (node->tag) {
        case Let_TAG: {
            const size_t count = node->payload.let.variables.count;

            LARRAY(const Type*, output_types, count);
            Nodes rewritten_args = type_primop_or_call(ctx, node->payload.let.op, node->payload.let.args, count, output_types);

            struct TypeRewriter vars_infer_ctx = *ctx;
            LARRAY(const Node*, nvars, count);
            for (size_t i = 0; i < count; i++) {
                const Variable* old_output = &node->payload.let.variables.nodes[i]->payload.var;
                nvars[i] = new_binder(&vars_infer_ctx, old_output->name, output_types[i], old_output->id);
            }

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
            Nodes return_types = node->payload.fn_ret.fn->payload.fn.return_types;

            const Nodes* old_values = &node->payload.fn_ret.values;
            LARRAY(const Node*, nvalues, old_values->count);
            for (size_t i = 0; i < old_values->count; i++)
                nvalues[i] = type_value(ctx, old_values->nodes[i], return_types.nodes[i]);
            return fn_ret(ctx->dst_arena, (Return) {
                .values = nodes(ctx->dst_arena, old_values->count, nvalues)
            });
        }
        case Jump_TAG: {
            const Node* ntarget = type_value(ctx, node->payload.jump.target, NULL);

            assert(get_qualifier(ntarget->type) == Uniform);
            assert(without_qualifier(ntarget->type)->tag == FnType_TAG);
            const FnType* tgt_type = &without_qualifier(ntarget->type)->payload.fn_type;
            assert(tgt_type->is_continuation);

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

static const Node* type_root(struct TypeRewriter* ctx, const Node* node) {
    if (node == NULL)
        return NULL;

    switch (node->tag) {
        case Root_TAG: {
            // assert(ctx->current_fn_expected_return_types == NULL);
            size_t count = node->payload.root.variables.count;
            LARRAY(const Node*, new_variables, count);
            LARRAY(const Node*, new_definitions, count);

            for (size_t i = 0; i < count; i++) {
                const Variable* oldvar = &node->payload.root.variables.nodes[i]->payload.var;
                const Type* imported_ty = import_node(ctx->dst_arena, oldvar->type);

                // Some top-level stuff does not have a definition
                if (node->payload.root.definitions.nodes[i] == NULL) {
                    new_variables[i] = new_binder(ctx, oldvar->name, imported_ty, oldvar->id);
                    new_definitions[i] = NULL;
                } else {
                    new_definitions[i] = type_value_or_def(ctx, node->payload.root.definitions.nodes[i], imported_ty);
                }
            }

            for (size_t i = 0; i < count; i++) {
                if (node->payload.root.definitions.nodes[i] == NULL) continue;

                const Variable* oldvar = &node->payload.root.variables.nodes[i]->payload.var;
                new_variables[i] = new_binder(ctx, oldvar->name, new_definitions[i]->type, oldvar->id);
            }

            return root(ctx->dst_arena, (Root) {
                .variables = nodes(ctx->dst_arena, count, new_variables),
                .definitions = nodes(ctx->dst_arena, count, new_definitions)
            });
        }
        default: error("not a root node");
    }
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

const Node* type_program(IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct List* bound_variables = new_list(struct BindEntry);
    struct Dict* rewritten_fns = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
    struct TypeRewriter ctx = {
            .dst_arena = dst_arena,
            .typed_variables = bound_variables,
            .rewritten_fns = rewritten_fns,
            // .current_fn_expected_return_types = NULL,
    };

    const Node* rewritten = type_root(&ctx, src_program);

    destroy_list(bound_variables);
    destroy_dict(rewritten_fns);
    return rewritten;
}
