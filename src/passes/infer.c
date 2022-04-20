#include "passes.h"

#include "../log.h"
#include "../local_array.h"
#include "../type.h"

#include "list.h"
#include "dict.h"

#include <assert.h>

struct BindEntry {
    VarId id;
    const Node* typed;
};

struct TypeRewriter {
    IrArena* src_arena;
    IrArena* dst_arena;
    struct List* typed_variables;
    struct Dict* rewritten_fns;
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
static const Node* type_terminator(struct TypeRewriter* ctx, const Node* node);
static const Node* type_value(struct TypeRewriter* ctx, const Node* node, const Node* expected_type);

static const Node* type_block(struct TypeRewriter* ctx, const Node* node) {
    if (node == NULL) return NULL;
    size_t old_typed_variables_size = entries_count_list(ctx->typed_variables);

    LARRAY(const Node*, ninstructions, node->payload.block.instructions.count);

    for (size_t i = 0; i < node->payload.block.instructions.count; i++)
        ninstructions[i] = type_instruction(ctx, node->payload.block.instructions.nodes[i]);

    Nodes typed_instructions = nodes(ctx->dst_arena, node->payload.block.instructions.count, ninstructions);
    const Node* typed_term = type_terminator(ctx, node->payload.block.terminator);

    while (entries_count_list(ctx->typed_variables) > old_typed_variables_size)
        remove_last_list(struct BindEntry, ctx->typed_variables);

    return block(ctx->dst_arena, (Block) {
        .instructions = typed_instructions,
        .terminator = typed_term,
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

    Node* fun = fn(dst_arena, node->payload.fn.is_continuation, string(dst_arena, node->payload.fn.name), nodes(dst_arena, node->payload.fn.params.count, nparams), nret_types);
    bool r = insert_dict_and_get_result(const Node*, Node*, ctx->rewritten_fns, node, fun);
    assert(r && "insertion of fun failed - the dict isn't working as it should");

    const Node* nblock = type_block(ctx, node->payload.fn.block);
    fun->payload.fn.block = nblock;

    while (entries_count_list(ctx->typed_variables) > old_typed_variables_size)
        remove_last_list(struct BindEntry, ctx->typed_variables);

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

static Nodes type_primop_or_call(struct TypeRewriter* ctx, Op op, Nodes old_inputs, size_t outputs_count, const Type* output_types[]) {
    IrArena* dst_arena = ctx->dst_arena;

    LARRAY(const Node*, new_inputs_scratch, old_inputs.count);
    Nodes yield_tys;

    if (op == call_op) {
        assert(old_inputs.count >= 1);
        size_t fn_args_count = old_inputs.count - 1;

        const Node* new_callee = type_value(ctx, old_inputs.nodes[0], NULL);
        new_inputs_scratch[0] = new_callee;

        const Type* callee_type = without_qualifier(new_callee->type);
        if (callee_type->tag != FnType_TAG)
            error("Callees must have a function type");
        if (callee_type->payload.fn_type.param_types.count != fn_args_count)
            error("Mismatched argument counts");
        for (size_t i = 0; i < fn_args_count; i++) {
            const Node* arg = old_inputs.nodes[1 + i];
            assert(arg);
            new_inputs_scratch[1 + i] = type_value(ctx, old_inputs.nodes[1 + i], callee_type->payload.fn_type.param_types.nodes[i]);
        }
    } else {
        Nodes input_types;
        switch (op) {
            case add_op:
            case sub_op: input_types = nodes(dst_arena, 2, (const Type*[]){ int_type(dst_arena), int_type(dst_arena) }); break;
            case push_stack_op:
            case push_stack_uniform_op: {
                assert(old_inputs.count == 2);
                const Type* element_type = import_node(dst_arena, old_inputs.nodes[0]);
                assert(get_qualifier(element_type) == Unknown);
                new_inputs_scratch[0] = element_type;
                new_inputs_scratch[1] = type_value(ctx, old_inputs.nodes[1], element_type);
                goto skip_input_types;
            }
            case pop_stack_op:
            case pop_stack_uniform_op: {
                assert(old_inputs.count == 1);
                const Type* element_type = import_node(dst_arena, old_inputs.nodes[0]);
                assert(get_qualifier(element_type) == Unknown);
                new_inputs_scratch[0] = element_type;
                goto skip_input_types;
            }
            default: error("unhandled op params");
        }

        assert(input_types.count == old_inputs.count);
        for (size_t i = 0; i < input_types.count; i++)
            new_inputs_scratch[i] = type_value(ctx, old_inputs.nodes[i], input_types.nodes[i]);
    }

    skip_input_types:
    yield_tys = typecheck_operation(dst_arena, op, nodes(dst_arena, old_inputs.count, new_inputs_scratch));

    assert(outputs_count == yield_tys.count);
    for (size_t i = 0; i < yield_tys.count; i++) {
        output_types[i] = yield_tys.nodes[i];
        assert(get_qualifier(output_types[i]) != Unknown);
    }

    return nodes(dst_arena, old_inputs.count, new_inputs_scratch);
}

static const Node* type_instruction(struct TypeRewriter* ctx, const Node* node) {
    switch (node->tag) {
        case Let_TAG: {
            const size_t count = node->payload.let.variables.count;

            LARRAY(const Type*, output_types, count);
            Nodes new_ops = type_primop_or_call(ctx, node->payload.let.op, node->payload.let.args, count, output_types);

            // extract the outputs
            LARRAY(const Node*, noutputs, count);
            for (size_t i = 0; i < count; i++) {
                const Variable* old_output = &node->payload.let.variables.nodes[i]->payload.var;
                noutputs[i] = new_binder(ctx, old_output->name, output_types[i], old_output->id);
            }

            return let(ctx->dst_arena, (Let) {
                .variables = nodes(ctx->dst_arena, count, noutputs),
                .op = node->payload.let.op,
                .args = new_ops
            });
        }
        case IfInstr_TAG: {
            const Node* condition = type_value(ctx, node->payload.if_instr.condition, bool_type(ctx->dst_arena));

            struct TypeRewriter instrs_infer_ctx = *ctx;
            const Node* ifTrue = type_block(&instrs_infer_ctx, node->payload.if_instr.if_true);
            const Node* ifFalse = type_block(&instrs_infer_ctx, node->payload.if_instr.if_false);
            return if_instr(ctx->dst_arena, (IfInstr) {
                .condition = condition,
                .if_true = ifTrue,
                .if_false = ifFalse
            });
        }
        default: error("not an instruction");
    }
}

static const Node* type_terminator(struct TypeRewriter* ctx, const Node* node) {
    switch (node->tag) {
        case Return_TAG: {
            const Node* imported_fn = type_fn(ctx, node->payload.fn_ret.fn);
            Nodes return_types = imported_fn->payload.fn.return_types;

            const Nodes* old_values = &node->payload.fn_ret.values;
            LARRAY(const Node*, nvalues, old_values->count);
            for (size_t i = 0; i < old_values->count; i++)
                nvalues[i] = type_value(ctx, old_values->nodes[i], return_types.nodes[i]);
            return fn_ret(ctx->dst_arena, (Return) {
                .values = nodes(ctx->dst_arena, old_values->count, nvalues),
                .fn = imported_fn
            });
        }
        case Jump_TAG: {
            const Node* ntarget = type_fn(ctx, node->payload.jump.target);

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
        case Join_TAG: return join(ctx->dst_arena);
        case Unreachable_TAG: return unreachable(ctx->dst_arena);
        default: error("not a terminator");
    }
}

static const Node* type_root(struct TypeRewriter* ctx, const Node* node) {
    if (node == NULL)
        return NULL;

    switch (node->tag) {
        case Root_TAG: {
            size_t count = node->payload.root.variables.count;
            LARRAY(const Node*, new_variables, count);
            LARRAY(const Node*, new_definitions, count);

            for (size_t i = 0; i < count; i++) {
                // we rely on this being zeroed
                new_definitions[i] = NULL;

                const Variable* old_var = &node->payload.root.variables.nodes[i]->payload.var;
                const Node* old_definition = node->payload.root.definitions.nodes[i];

                const Type* imported_ty = import_node(ctx->dst_arena, old_var->type);;
                if (old_definition) {
                    // in the case of literals, we allow importing the definition first since it cannot depend on neighbouring defs
                    if (old_definition->tag != Function_TAG) {
                        new_definitions[i] = type_value_or_def(ctx, old_definition, imported_ty);
                        imported_ty = new_definitions[i]->type;
                    } else {
                        const Node* derived_fn_type = derive_fn_type(ctx->src_arena, &old_definition->payload.fn);
                        // we lack the ability to refine a function type currently, either a fully qualified type is given, or it's fully inferred.
                        assert(!imported_ty || imported_ty == derived_fn_type);
                        imported_ty = qualified_type(ctx->dst_arena, (QualifiedType) {
                            .type = import_node(ctx->dst_arena, derived_fn_type),
                            .is_uniform = true
                        });
                    }
                }

                assert(get_qualifier(imported_ty) != Unknown);

                new_variables[i] = new_binder(ctx, old_var->name, imported_ty, old_var->id);
            }

            for (size_t i = 0; i < count; i++) {
                assert(new_variables[i]->type);
                const Node* old_definition = node->payload.root.definitions.nodes[i];

                // if there was no definition, or if it was already imported, bail
                if (old_definition == NULL || new_definitions[i] != NULL) continue;

                new_definitions[i] = type_value_or_def(ctx, old_definition, new_variables[i]->type);
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
        .src_arena = src_arena,
        .dst_arena = dst_arena,
        .typed_variables = bound_variables,
        .rewritten_fns = rewritten_fns,
    };

    const Node* rewritten = type_root(&ctx, src_program);

    destroy_list(bound_variables);
    destroy_dict(rewritten_fns);
    return rewritten;
}
