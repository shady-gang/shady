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

typedef struct Context_ {
    IrArena* src_arena;
    IrArena* dst_arena;
    struct List* typed_variables;
    struct Dict* done;

    const Nodes* join_types;
    const Nodes* break_types;
    const Nodes* continue_types;
} Context;

static const Node* resolve(const Context* ctx, VarId id) {
    for (size_t i = 0; i < entries_count_list(ctx->typed_variables); i++) {
        const struct BindEntry* entry = &read_list(const struct BindEntry, ctx->typed_variables)[i];
        if (entry->id == id) {
            return entry->typed;
        }
    }
    error("could not resolve variable %d", id)
}

static const Node* new_binder(Context* ctx, const char* old_name, const Type* inferred_ty, const VarId old_id) {
    const char* name = string(ctx->dst_arena, old_name);
    const Node* fresh = var(ctx->dst_arena, inferred_ty, name);
    struct BindEntry entry = {
        .id = old_id,
        .typed = fresh
    };
    append_list(struct BindEntry, ctx->typed_variables, entry);
    return fresh;
}

static const Node* infer(Context* ctx, const Node* node);
static const Node* infer_terminator(Context* ctx, const Node* node);
static const Node* infer_value(Context* ctx, const Node* node, const Node* expected_type);

static const Node* infer_block(Context* ctx, const Node* node) {
    if (node == NULL) return NULL;
    size_t old_typed_variables_size = entries_count_list(ctx->typed_variables);

    LARRAY(const Node*, ninstructions, node->payload.block.instructions.count);

    for (size_t i = 0; i < node->payload.block.instructions.count; i++)
        ninstructions[i] = infer(ctx, node->payload.block.instructions.nodes[i]);

    Nodes typed_instructions = nodes(ctx->dst_arena, node->payload.block.instructions.count, ninstructions);
    const Node* typed_term = infer_terminator(ctx, node->payload.block.terminator);

    while (entries_count_list(ctx->typed_variables) > old_typed_variables_size)
        remove_last_list(struct BindEntry, ctx->typed_variables);

    return block(ctx->dst_arena, (Block) {
        .instructions = typed_instructions,
        .terminator = typed_term,
    });
}

static const Node* infer_constant(Context* ctx, const Node* node) {
    assert(node->tag == Constant_TAG);
    Node** already_done = find_value_dict(const Node*, Node*, ctx->done, node);
    if (already_done)
        return *already_done;

    const Constant* oconstant = &node->payload.constant;
    Node* nconstant = constant(ctx->dst_arena, oconstant->name);
    insert_dict(const Node*, Node*, ctx->done, node, nconstant);

    const Type* imported_hint = import_node(ctx->dst_arena, oconstant->type_hint);
    const Node* typed_value = infer_value(ctx, oconstant->value, imported_hint);
    nconstant->payload.constant.value = typed_value;
    nconstant->type = typed_value->type;

    return nconstant;
}

static const Node* infer_fn(Context* ctx, const Node* node) {
    IrArena* dst_arena = ctx->dst_arena;
    assert(node->tag == Function_TAG);

    Node** already_done = find_value_dict(const Node*, Node*, ctx->done, node);
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

    Node* fun = fn(dst_arena, node->payload.fn.atttributes, string(dst_arena, node->payload.fn.name), nodes(dst_arena, node->payload.fn.params.count, nparams), nret_types);
    bool r = insert_dict_and_get_result(const Node*, Node*, ctx->done, node, fun);
    assert(r && "insertion of fun failed - the dict isn't working as it should");

    const Node* nblock = infer_block(ctx, node->payload.fn.block);
    fun->payload.fn.block = nblock;

    while (entries_count_list(ctx->typed_variables) > old_typed_variables_size)
        remove_last_list(struct BindEntry, ctx->typed_variables);

    return fun;
}

static const Node* infer_value(Context* ctx, const Node* node, const Node* expected_type) {
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

static const Node* infer_value_or_cont(Context* ctx, const Node* node, const Node* expected_type) {
    const Node* typed = node->tag == Function_TAG ? infer_fn(ctx, node) : infer_value(ctx, node, expected_type);
    return typed;
}

static const Node* infer_primop(Context* ctx, const Node* node) {
    assert(node->tag == PrimOp_TAG);
    IrArena* dst_arena = ctx->dst_arena;
    Nodes old_inputs = node->payload.prim_op.operands;

    LARRAY(const Node*, new_inputs_scratch, old_inputs.count);
    Nodes input_types;
    switch (node->payload.prim_op.op) {
        case add_op:
        case sub_op: input_types = nodes(dst_arena, 2, (const Type*[]){ int_type(dst_arena), int_type(dst_arena) }); break;
        case push_stack_op:
        case push_stack_uniform_op: {
            assert(old_inputs.count == 2);
            const Type* element_type = import_node(dst_arena, old_inputs.nodes[0]);
            assert(get_qualifier(element_type) == Unknown);
            new_inputs_scratch[0] = element_type;
            new_inputs_scratch[1] = infer_value(ctx, old_inputs.nodes[1], element_type);
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
        case load_op: {
            assert(old_inputs.count == 1);
            new_inputs_scratch[0] = infer_value(ctx, old_inputs.nodes[0], NULL);
            goto skip_input_types;
        }
        case store_op: {
            assert(old_inputs.count == 2);
            new_inputs_scratch[0] = infer_value(ctx, old_inputs.nodes[0], NULL);
            const Type* op0_type = without_qualifier(new_inputs_scratch[0]->type);
            assert(op0_type->tag == PtrType_TAG);
            const PtrType* ptr_type = &op0_type->payload.ptr_type;
            new_inputs_scratch[1] = infer_value(ctx, old_inputs.nodes[1], ptr_type->pointed_type);
            goto skip_input_types;
        }
        case alloca_op: {
            assert(old_inputs.count == 1);
            new_inputs_scratch[0] = import_node(ctx->dst_arena, old_inputs.nodes[0]);
            assert(is_type(new_inputs_scratch[0]));
            assert(get_qualifier(new_inputs_scratch[0]) == Unknown);
            goto skip_input_types;
        }
        default: error("unhandled op params");
    }

    assert(input_types.count == old_inputs.count);
    for (size_t i = 0; i < input_types.count; i++)
        new_inputs_scratch[i] = infer_value(ctx, old_inputs.nodes[i], input_types.nodes[i]);

    skip_input_types:
    return prim_op(dst_arena, (PrimOp) {
        .op = node->payload.prim_op.op,
        .operands = nodes(dst_arena, old_inputs.count, new_inputs_scratch)
    });
}

static const Node* infer_call(Context* ctx, const Node* node) {
    assert(node->tag == Call_TAG);

    const Node* new_callee = infer_value_or_cont(ctx, node->payload.call_instr.callee, NULL);
    LARRAY(const Node*, new_args, node->payload.call_instr.args.count);

    const Type* callee_type = without_qualifier(new_callee->type);
    if (callee_type->tag != FnType_TAG)
        error("Callees must have a function type");
    if (callee_type->payload.fn_type.param_types.count != node->payload.call_instr.args.count)
        error("Mismatched argument counts");
    for (size_t i = 0; i < node->payload.call_instr.args.count; i++) {
        const Node* arg = node->payload.call_instr.args.nodes[i];
        assert(arg);
        new_args[i] = infer_value(ctx, node->payload.call_instr.args.nodes[i], callee_type->payload.fn_type.param_types.nodes[i]);
    }

    return call_instr(ctx->dst_arena, (Call) {
        .callee = new_callee,
        .args = nodes(ctx->dst_arena, node->payload.call_instr.args.count, new_args)
    });
}

static const Node* infer_if(Context* ctx, const Node* node) {
    assert(node->tag == If_TAG);
    const Node* condition = infer_value(ctx, node->payload.if_instr.condition, bool_type(ctx->dst_arena));

    Context instrs_infer_ctx = *ctx;
    const Node* ifTrue = infer_block(&instrs_infer_ctx, node->payload.if_instr.if_true);
    const Node* ifFalse = infer_block(&instrs_infer_ctx, node->payload.if_instr.if_false);
    return if_instr(ctx->dst_arena, (If) {
        // TODO handle !!!
        .yield_types = import_nodes(ctx->dst_arena, node->payload.if_instr.yield_types),
        .condition = condition,
        .if_true = ifTrue,
        .if_false = ifFalse
    });
}

static const Node* infer_instruction(Context* ctx, const Node* node) {
    switch (node->tag) {
        case PrimOp_TAG: return infer_primop(ctx, node);
        case Call_TAG:   return infer_call(ctx, node);
        case If_TAG:     return infer_if(ctx, node);
        default: error("not an instruction");
    }
    SHADY_UNREACHABLE;
}

static const Node* infer(Context* ctx, const Node* node) {
    assert(node->tag == Let_TAG);
    const size_t count = node->payload.let.variables.count;

    const Node* new_instruction = infer_instruction(ctx, node->payload.let.instruction);
    Nodes output_types = typecheck_instruction(ctx->dst_arena, new_instruction);

    assert(output_types.count == count);
    
    // extract the outputs
    LARRAY(const Node*, noutputs, count);
    for (size_t i = 0; i < count; i++) {
        const Variable* old_output = &node->payload.let.variables.nodes[i]->payload.var;
        noutputs[i] = new_binder(ctx, old_output->name, output_types.nodes[i], old_output->id);
    }

    return let(ctx->dst_arena, (Let) {
        .variables = nodes(ctx->dst_arena, count, noutputs),
        .instruction = new_instruction
    });
}

static const Node* infer_terminator(Context* ctx, const Node* node) {
    switch (node->tag) {
        case Return_TAG: {
            const Node* imported_fn = infer_fn(ctx, node->payload.fn_ret.fn);
            Nodes return_types = imported_fn->payload.fn.return_types;

            const Nodes* old_values = &node->payload.fn_ret.values;
            LARRAY(const Node*, nvalues, old_values->count);
            for (size_t i = 0; i < old_values->count; i++)
                nvalues[i] = infer_value(ctx, old_values->nodes[i], return_types.nodes[i]);
            return fn_ret(ctx->dst_arena, (Return) {
                .values = nodes(ctx->dst_arena, old_values->count, nvalues),
                .fn = NULL
            });
        }
        case Jump_TAG: {
            const Node* ntarget = infer_fn(ctx, node->payload.jump.target);

            assert(get_qualifier(ntarget->type) == Uniform);
            assert(without_qualifier(ntarget->type)->tag == FnType_TAG);
            const FnType* tgt_type = &without_qualifier(ntarget->type)->payload.fn_type;
            assert(tgt_type->is_continuation);

            LARRAY(const Node*, tmp, node->payload.jump.args.count);
            for (size_t i = 0; i < node->payload.jump.args.count; i++)
                tmp[i] = infer_value(ctx, node->payload.jump.args.nodes[i], tgt_type->param_types.nodes[i]);

            Nodes new_args = nodes(ctx->dst_arena, node->payload.jump.args.count, tmp);

            return jump(ctx->dst_arena, (Jump) {
                .target = ntarget,
                .args = new_args
            });
        }
        case Merge_TAG: {
            const Nodes* expected_types = NULL;
            switch (node->payload.merge.what) {
                case Join: expected_types = ctx->join_types; break;
                case Continue: expected_types = ctx->continue_types; break;
                case Break: expected_types = ctx->break_types; break;
                default: error("we don't know this sort of merge");
            }
            assert(expected_types && "Merge terminator found but we're not within a suitable if/loop instruction !");
            const Nodes* old_args = &node->payload.merge.args;
            assert(expected_types->count == old_args->count);
            LARRAY(const Node*, new_args, old_args->count);
            for (size_t i = 0; i < old_args->count; i++)
                new_args[i] = infer_value(ctx, old_args->nodes[i], (*expected_types).nodes[i]);
            return merge(ctx->dst_arena, (Merge) {
                .what = node->payload.merge.what,
                .args = nodes(ctx->dst_arena, old_args->count, new_args)
            });
        }
        // TODO break, continue
        case Unreachable_TAG: return unreachable(ctx->dst_arena);
        default: error("not a terminator");
    }
}

static const Node* type_root(Context* ctx, const Node* node) {
    if (node == NULL)
        return NULL;

    switch (node->tag) {
        case Root_TAG: {
            size_t count = node->payload.root.declarations.count;
            LARRAY(const Node*, new_decls, count);

            // First type and bind global variables
            for (size_t i = 0; i < count; i++) {
                const Node* odecl = node->payload.root.declarations.nodes[i];

                switch (odecl->tag) {
                    case Variable_TAG: {
                        const Variable* old_var = &odecl->payload.var;
                        const Type* imported_ty = import_node(ctx->dst_arena, old_var->type);;
                        new_decls[i] = new_binder(ctx, old_var->name, imported_ty, old_var->id);
                        break;
                    }
                    case Function_TAG:
                    case Constant_TAG: continue;
                    default: error("not a decl");
                }
            }

            // Then process the rest
            for (size_t i = 0; i < count; i++) {
                const Node *odecl = node->payload.root.declarations.nodes[i];

                switch (odecl->tag) {
                    case Variable_TAG: continue;
                    case Function_TAG: {
                        new_decls[i] = infer_fn(ctx, odecl);
                        break;
                    }
                    case Constant_TAG: {
                        new_decls[i] = infer_constant(ctx, odecl);
                        break;
                    }
                    default: error("not a decl");
                }
            }

            return root(ctx->dst_arena, (Root) {
                .declarations = nodes(ctx->dst_arena, count, new_decls),
            });
        }
        default: error("not a root node");
    }
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

const Node* type_program(IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct List* bound_variables = new_list(struct BindEntry);
    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
    Context ctx = {
        .src_arena = src_arena,
        .dst_arena = dst_arena,
        .typed_variables = bound_variables,
        .done = done,
    };

    const Node* rewritten = type_root(&ctx, src_program);

    destroy_list(bound_variables);
    destroy_dict(done);
    return rewritten;
}
