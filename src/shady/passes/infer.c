#include "passes.h"

#include "log.h"
#include "portability.h"

#include "../type.h"
#include "../rewrite.h"

#include <assert.h>

static Nodes annotate_all_types(IrArena* arena, Nodes types, bool uniform_by_default) {
    LARRAY(const Type*, ntypes, types.count);
    for (size_t i = 0; i < types.count; i++) {
        if (!contains_qualified_type(types.nodes[i]))
            ntypes[i] = qualified_type(arena, (QualifiedType) {
                .type = types.nodes[i],
                .is_uniform = uniform_by_default,
            });
        else
            ntypes[i] = types.nodes[i];
    }
    return nodes(arena, types.count, ntypes);
}

typedef struct {
    Rewriter rewriter;

    const Nodes* join_types;
    const Nodes* break_types;
    const Nodes* continue_types;
} Context;

static const Node* infer_type(Context* ctx, const Type* type);

static Nodes infer_types(Context* ctx, Nodes types) {
    LARRAY(const Type*, new, types.count);
    for (size_t i = 0; i < types.count; i++)
        new[i] = infer_type(ctx, types.nodes[i]);
    return nodes(ctx->rewriter.dst_arena, types.count, new);
}

static const Node* infer_let(Context* ctx, const Node* node);
static const Node* infer_terminator(Context* ctx, const Node* node);
static const Node* infer_value(Context* ctx, const Node* node, const Node* expected_type);

static Nodes infer_values(Context* ctx, Nodes values) {
    LARRAY(const Node*, inferred, values.count);
    for (size_t i = 0; i < values.count; i++)
        inferred[i] = infer_value(ctx, values.nodes[i], NULL);
    return nodes(ctx->rewriter.dst_arena, values.count, inferred);
}

static Nodes infer_annotations(Context* ctx, Nodes annotations) {
    LARRAY(const Node*, arr, annotations.count);
    for (size_t i = 0; i < annotations.count; i++) {
        Annotation old = annotations.nodes[i]->payload.annotation;
        Annotation payload = {
            .payload_type = old.payload_type,
            .name = string(ctx->rewriter.dst_arena, old.name)
        };
        switch (old.payload_type) {
            case AnPayloadNone: break;
            case AnPayloadValue:
                payload.value = infer_value(ctx, old.value, NULL);
                break;
            case AnPayloadMap:
                payload.labels = import_strings(ctx->rewriter.dst_arena, old.labels);
                SHADY_FALLTHROUGH
            case AnPayloadValues:
                payload.values = infer_values(ctx, old.values);
                break;
            default: error("TODO");
        }
        arr[i] = annotation(ctx->rewriter.dst_arena, payload);
    }
    return nodes(ctx->rewriter.dst_arena, annotations.count, arr);
}

static const Node* infer_type(Context* ctx, const Type* type) {
    switch (type->tag) {
        case ArrType_TAG: {
            const Node* size = infer_value(ctx, type->payload.arr_type.size, int32_type(ctx->rewriter.dst_arena));
            return arr_type(ctx->rewriter.dst_arena, (ArrType) {
                .size = size,
                .element_type = infer_type(ctx, type->payload.arr_type.element_type)
            });
        }
        default: return import_node(ctx->rewriter.dst_arena, type);
    }
}

static const Node* infer_decl(Context* ctx, const Node* node) {
    assert(is_declaration(node));
    const Node* already_done = search_processed(&ctx->rewriter, node);
    if (already_done)
        return already_done;

    IrArena* dst_arena = ctx->rewriter.dst_arena;
    switch (node->tag) {
        case Lambda_TAG: {
            Context body_context = *ctx;

            LARRAY(const Node*, nparams, node->payload.lam.params.count);
            for (size_t i = 0; i < node->payload.lam.params.count; i++) {
                const Variable* old_param = &node->payload.lam.params.nodes[i]->payload.var;
                const Type* imported_param_type = infer_type(ctx, old_param->type);
                nparams[i] = var(body_context.rewriter.dst_arena, imported_param_type, old_param->name);
                register_processed(&body_context.rewriter, node->payload.lam.params.nodes[i], nparams[i]);
            }

            Node* fun = NULL;
            switch (node->payload.lam.tier) {
                case FnTier_Lambda: assert(false);
                case FnTier_BasicBlock:
                    fun = basic_block(dst_arena, nodes(dst_arena, node->payload.lam.params.count, nparams), string(dst_arena, node->payload.lam.name));
                    break;
                case FnTier_Function: {
                    Nodes nret_types = annotate_all_types(dst_arena, infer_types(ctx, node->payload.lam.return_types), false);
                    fun = function(dst_arena, nodes(dst_arena, node->payload.lam.params.count, nparams), string(dst_arena, node->payload.lam.name), infer_annotations(ctx, node->payload.lam.annotations), nret_types);
                    break;
                }
            }
            assert(fun);
            register_processed(&ctx->rewriter, node, fun);

            fun->payload.lam.body = infer_terminator(&body_context, node->payload.lam.body);

            return fun;
        }
        case Constant_TAG: {
            const Constant* oconstant = &node->payload.constant;
            Node* nconstant = constant(ctx->rewriter.dst_arena, infer_annotations(ctx, oconstant->annotations), oconstant->name);
            register_processed(&ctx->rewriter, node, nconstant);

            const Type* imported_hint = import_node(ctx->rewriter.dst_arena, oconstant->type_hint);
            const Node* typed_value = infer_value(ctx, oconstant->value, imported_hint);
            nconstant->payload.constant.type_hint = NULL;
            nconstant->payload.constant.value = typed_value;
            nconstant->type = typed_value->type;

            return nconstant;
        }
        case GlobalVariable_TAG: {
             const GlobalVariable* old_gvar = &node->payload.global_variable;
             const Type* imported_ty = infer_type(ctx, old_gvar->type);
             Node* ngvar = global_var(ctx->rewriter.dst_arena, infer_annotations(ctx, old_gvar->annotations), imported_ty, old_gvar->name, old_gvar->address_space);
             register_processed(&ctx->rewriter, node, ngvar);

             ngvar->payload.global_variable.init = infer_value(ctx, old_gvar->init, NULL);
             return ngvar;
        }
        default: SHADY_UNREACHABLE;
    }
}

/// Like extract_operand_type but won't error out if type wasn't qualified to begin with
static const Type* remove_uniformity_qualifier(const Node* type) {
    if (contains_qualified_type(type))
        return extract_operand_type(type);
    return type;
}

static const Node* infer_value(Context* ctx, const Node* node, const Type* expected_type) {
    if (!node) return NULL;

    IrArena* dst_arena = ctx->rewriter.dst_arena;
    switch (node->tag) {
        case Variable_TAG: return find_processed(&ctx->rewriter, node);
        case IntLiteral_TAG:
        case UntypedNumber_TAG: {
            expected_type = expected_type ? expected_type : int32_type(ctx->rewriter.dst_arena);
            expected_type = remove_uniformity_qualifier(expected_type);
            assert(expected_type->tag == Int_TAG);
            if (node->tag == UntypedNumber_TAG) {
                int64_t v;
                if (sizeof(long) == sizeof(int64_t))
                    v = strtol(node->payload.untyped_number.plaintext, NULL, 10);
                else if (sizeof(long long) == sizeof(int64_t))
                    v = strtoll(node->payload.untyped_number.plaintext, NULL, 10);
                else
                    assert(false);
                // TODO chop off extra bits based on width ?
                return int_literal(dst_arena, (IntLiteral) { .value_i64 = v, .width = expected_type->payload.int_type.width });
            } else {
                assert(expected_type->payload.int_type.width == node->payload.int_literal.width);
                return int_literal(dst_arena, (IntLiteral) { .width = node->payload.int_literal.width, .value_u64 = node->payload.int_literal.value_u64 });
            }
        }
        case True_TAG: return true_lit(dst_arena);
        case False_TAG: return false_lit(dst_arena);
        case StringLiteral_TAG: return string_lit(dst_arena, (StringLiteral) { .string = string(dst_arena, node->payload.string_lit.string )});
        case GlobalVariable_TAG: return ref_decl(dst_arena, (RefDecl) { .decl = infer_decl(ctx, node) }); // TODO check types match
        case Constant_TAG: return ref_decl(dst_arena, (RefDecl) { .decl = infer_decl(ctx, node) }); // TODO check types match
        case Lambda_TAG: return fn_addr(dst_arena, (FnAddr) { .fn = infer_decl(ctx, node) }); // TODO check types match
        default: error("not a value");
    }
}

static const Node* infer_tail(Context* ctx, const Node* tail, Nodes inferred_arg_type) {
    assert(tail->tag == Lambda_TAG);
    assert(inferred_arg_type.count == tail->payload.lam.params.count);
    IrArena* arena = ctx->rewriter.dst_arena;

    if (is_anonymous_lambda(tail)) {
        Context body_context = *ctx;
        LARRAY(const Node*, nparams, tail->payload.lam.params.count);
        for (size_t i = 0; i < tail->payload.lam.params.count; i++) {
            const Variable* old_param = &tail->payload.lam.params.nodes[i]->payload.var;
            // for the param type: use the inferred one if none is already provided
            // if one is provided, check the inferred argument type is a subtype of the param type
            const Type* param_type = old_param->type;
            param_type = param_type ? param_type : inferred_arg_type.nodes[i];
            assert(is_subtype(param_type, inferred_arg_type.nodes[i]));
            nparams[i] = var(body_context.rewriter.dst_arena, param_type, old_param->name);
            register_processed(&body_context.rewriter, tail->payload.lam.params.nodes[i], nparams[i]);
        }

        Node* fun = lambda(arena, nodes(arena, tail->payload.lam.params.count, nparams));
        assert(fun);
        register_processed(&ctx->rewriter, tail, fun);

        fun->payload.lam.body = infer_terminator(&body_context, tail->payload.lam.body);
    } else return infer_decl(ctx, tail);
}

static const Node* infer_primop(Context* ctx, const Node* node) {
    assert(node->tag == PrimOp_TAG);
    IrArena* dst_arena = ctx->rewriter.dst_arena;
    Nodes old_inputs = node->payload.prim_op.operands;

    LARRAY(const Node*, new_inputs_scratch, old_inputs.count);
    Nodes input_types;
    switch (node->payload.prim_op.op) {
        case neg_op:
            input_types = nodes(dst_arena, 1, (const Type*[]){ int32_type(dst_arena) }); break;
        case rshift_arithm_op:
        case rshift_logical_op:
        case lshift_op:
        case add_op:
        case sub_op:
        case mul_op:
        case div_op:
        case mod_op:
        case lt_op:
        case lte_op:
        case eq_op:
        case neq_op:
        case gt_op:
        case gte_op:
            input_types = nodes(dst_arena, 2, (const Type*[]){ int32_type(dst_arena), int32_type(dst_arena) }); break;
        case push_stack_op:
        case push_stack_uniform_op: {
            assert(old_inputs.count == 2);
            const Type* element_type = import_node(dst_arena, old_inputs.nodes[0]);
            assert(!contains_qualified_type(element_type));
            new_inputs_scratch[0] = element_type;
            new_inputs_scratch[1] = infer_value(ctx, old_inputs.nodes[1], element_type);
            goto skip_input_types;
        }
        case pop_stack_op:
        case pop_stack_uniform_op: {
            assert(old_inputs.count == 1);
            const Type* element_type = import_node(dst_arena, old_inputs.nodes[0]);
            assert(!contains_qualified_type(element_type));
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
            const Type* op0_type = extract_operand_type(new_inputs_scratch[0]->type);
            assert(op0_type->tag == PtrType_TAG);
            const PtrType* ptr_type = &op0_type->payload.ptr_type;
            new_inputs_scratch[1] = infer_value(ctx, old_inputs.nodes[1], ptr_type->pointed_type);
            goto skip_input_types;
        }
        case alloca_op: {
            assert(old_inputs.count == 1);
            new_inputs_scratch[0] = import_node(ctx->rewriter.dst_arena, old_inputs.nodes[0]);
            const Type* element_type = new_inputs_scratch[0];
            assert(is_type(new_inputs_scratch[0]));
            assert(!contains_qualified_type(element_type));
            goto skip_input_types;
        }
        case lea_op: {
            assert(old_inputs.count >= 2);
            new_inputs_scratch[0] = infer_value(ctx, old_inputs.nodes[0], NULL);
            for (size_t i = 1; i < old_inputs.count; i++) {
                new_inputs_scratch[i] = old_inputs.nodes[i] ? infer_value(ctx, old_inputs.nodes[i], int32_type(dst_arena)) : NULL;
            }
            goto skip_input_types;
        }
        case empty_mask_op:
        case subgroup_active_mask_op:
        case subgroup_local_id_op:
        case subgroup_elect_first_op:
            input_types = nodes(dst_arena, 0, NULL);
            break;
        case subgroup_broadcast_first_op:
            new_inputs_scratch[0] = infer_value(ctx, old_inputs.nodes[0], NULL);
            goto skip_input_types;
        case subgroup_ballot_op:
            input_types = nodes(dst_arena, 1, (const Type* []) { bool_type(dst_arena) });
            break;
        case mask_is_thread_active_op: {
            input_types = nodes(dst_arena, 2, (const Type* []) { mask_type(dst_arena), int32_type(dst_arena) });
            break;
        }
        default: {
            for (size_t i = 0; i < old_inputs.count; i++) {
                new_inputs_scratch[i] = old_inputs.nodes[i] ? infer_value(ctx, old_inputs.nodes[i], int32_type(dst_arena)) : NULL;
            }
            goto skip_input_types;
        }
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

    const Node* new_callee = infer_value(ctx, node->payload.call_instr.callee, NULL);
    LARRAY(const Node*, new_args, node->payload.call_instr.args.count);

    const Type* callee_type = extract_operand_type(new_callee->type);
    if (node->payload.call_instr.is_indirect) {
        if (callee_type->tag != PtrType_TAG)
            error("functions are called through function pointers");
        callee_type = callee_type->payload.ptr_type.pointed_type;
    }

    if (callee_type->tag != FnType_TAG)
        error("Callees must have a function type");
    if (callee_type->payload.fn_type.param_types.count != node->payload.call_instr.args.count)
        error("Mismatched argument counts");
    for (size_t i = 0; i < node->payload.call_instr.args.count; i++) {
        const Node* arg = node->payload.call_instr.args.nodes[i];
        assert(arg);
        new_args[i] = infer_value(ctx, node->payload.call_instr.args.nodes[i], callee_type->payload.fn_type.param_types.nodes[i]);
        assert(new_args[i]->type);
    }

    return call_instr(ctx->rewriter.dst_arena, (Call) {
        .is_indirect = node->payload.call_instr.is_indirect,
        .callee = new_callee,
        .args = nodes(ctx->rewriter.dst_arena, node->payload.call_instr.args.count, new_args)
    });
}

static const Node* infer_if(Context* ctx, const Node* node) {
    assert(node->tag == If_TAG);
    const Node* condition = infer_value(ctx, node->payload.if_instr.condition, bool_type(ctx->rewriter.dst_arena));

    Nodes join_types = infer_types(ctx, node->payload.if_instr.yield_types);
    // The type annotation on `if` may not include divergence/convergence info, we default that stuff to divergent
    join_types = annotate_all_types(ctx->rewriter.dst_arena, join_types, false);
    Context joinable_ctx = *ctx;
    joinable_ctx.join_types = &join_types;

    const Node* true_body = infer_tail(&joinable_ctx, node->payload.if_instr.if_true, nodes(ctx->rewriter.dst_arena, 0, NULL));
    // don't allow seeing the variables made available in the true branch
    joinable_ctx.rewriter = ctx->rewriter;
    const Node* false_body = infer_tail(&joinable_ctx, node->payload.if_instr.if_false, nodes(ctx->rewriter.dst_arena, 0, NULL));

    return if_instr(ctx->rewriter.dst_arena, (If) {
        .yield_types = join_types,
        .condition = condition,
        .if_true = true_body,
        .if_false = false_body,
    });
}

static const Node* infer_loop(Context* ctx, const Node* node) {
    assert(node->tag == Loop_TAG);
    Context loop_body_ctx = *ctx;
    const Node* old_body = node->payload.loop_instr.body;
    Nodes old_params = old_body->payload.lam.params;
    Nodes old_initial_args = node->payload.loop_instr.initial_args;
    assert(old_params.count == old_initial_args.count);
    LARRAY(const Type*, new_params_types, old_params.count);
    //LARRAY(const Node*, new_params, old_params.count);
    LARRAY(const Node*, new_initial_args, old_params.count);
    for (size_t i = 0; i < old_params.count; i++) {
        const Variable* old_param = &old_params.nodes[i]->payload.var;
        new_params_types[i] = import_node(ctx->rewriter.dst_arena, old_param->type);
        new_initial_args[i] = infer_value(ctx, old_initial_args.nodes[i], new_params_types[i]);
        //new_params[i] = var(loop_body_ctx.rewriter.dst_arena, new_params_types[i], old_param->name);
        //register_processed(&loop_body_ctx.rewriter, old_params.nodes[i], new_params[i]);
    }

    Nodes loop_yield_types = infer_types(ctx, node->payload.loop_instr.yield_types);
    loop_yield_types = annotate_all_types(ctx->rewriter.dst_arena, loop_yield_types, false);

    loop_body_ctx.join_types = NULL;
    loop_body_ctx.break_types = &loop_yield_types;
    Nodes param_types = nodes(ctx->rewriter.dst_arena, old_params.count, new_params_types);
    loop_body_ctx.continue_types = &param_types;

    return loop_instr(ctx->rewriter.dst_arena, (Loop) {
        .yield_types = loop_yield_types,
        .initial_args = nodes(ctx->rewriter.dst_arena, old_params.count, new_initial_args),
        .body = infer_tail(&loop_body_ctx, node->payload.loop_instr.body, param_types)
    });
}

static const Node* infer_instruction(Context* ctx, const Node* node, const Type* expected_type) {
    switch (is_instruction(node)) {
        case PrimOp_TAG:  return infer_primop(ctx, node);
        case Call_TAG:    return infer_call  (ctx, node);
        case If_TAG:      return infer_if    (ctx, node);
        case Loop_TAG:    return infer_loop  (ctx, node);
        case Match_TAG:   error("TODO")
        case Control_TAG: error("TODO")
        case NotAnInstruction: error("not an instruction");
    }
    SHADY_UNREACHABLE;
}

static const Node* infer_terminator(Context* ctx, const Node* node) {
    IrArena* arena = ctx->rewriter.dst_arena;
    switch (node->tag) {
        case Let_TAG: {
            assert(!node->payload.let.is_mutable);
            const Node* otail = node->payload.let.tail;
            assert(otail->tag == Lambda_TAG);
            Nodes annotated_types = extract_variable_types(arena, &otail->payload.lam.params);
            bool valid_annotations = true;
            for (size_t i = 0; i < annotated_types.count; i++) {
                valid_annotations &= (annotated_types.nodes[i] != NULL);
            }
            const Node* inferred_instruction = infer_instruction(ctx, node->payload.let.instruction, wrap_multiple_yield_types(arena, annotated_types));
            Nodes inferred_yield_types = unwrap_multiple_yield_types(arena, inferred_instruction->type);
            const Node* inferred_tail = infer_tail(ctx, node->payload.let.tail, inferred_yield_types);
            return let(arena, false, inferred_instruction, inferred_tail);
        }
        case Return_TAG: {
            const Node* imported_fn = infer_decl(ctx, node->payload.fn_ret.fn);
            Nodes return_types = imported_fn->payload.lam.return_types;

            const Nodes* old_values = &node->payload.fn_ret.values;
            LARRAY(const Node*, nvalues, old_values->count);
            for (size_t i = 0; i < old_values->count; i++)
                nvalues[i] = infer_value(ctx, old_values->nodes[i], return_types.nodes[i]);
            return fn_ret(ctx->rewriter.dst_arena, (Return) {
                .values = nodes(ctx->rewriter.dst_arena, old_values->count, nvalues),
                .fn = NULL
            });
        }
        case Branch_TAG: {
            switch (node->payload.branch.branch_mode) {
                case BrJump: {
                    const Node* ntarget = infer_decl(ctx, node->payload.branch.target);
                    const Type* ntarget_type;
                    bool ntarget_is_uniform;
                    deconstruct_operand_type(ntarget->type, &ntarget_type, &ntarget_is_uniform);
                    assert(ntarget_is_uniform);
                    assert(ntarget_type->tag == FnType_TAG);
                    const FnType* tgt_type = &ntarget_type->payload.fn_type;
                    assert(tgt_type->tier == FnTier_BasicBlock);

                    LARRAY(const Node*, tmp, node->payload.branch.args.count);
                    for (size_t i = 0; i < node->payload.branch.args.count; i++)
                        tmp[i] = infer_value(ctx, node->payload.branch.args.nodes[i], tgt_type->param_types.nodes[i]);

                    Nodes new_args = nodes(ctx->rewriter.dst_arena, node->payload.branch.args.count, tmp);

                    return branch(ctx->rewriter.dst_arena, (Branch) {
                        .branch_mode = node->payload.branch.branch_mode,
                        .target = ntarget,
                        .args = new_args
                    });
                }
                case BrIfElse: {
                    const Node* ncond = infer_value(ctx, node->payload.branch.branch_condition, bool_type(ctx->rewriter.dst_arena));

                    const Node* t_target = infer_decl(ctx, node->payload.branch.true_target);
                    const Node* f_target = infer_decl(ctx, node->payload.branch.false_target);

                    assert(is_operand_uniform(t_target->type));
                    assert(extract_operand_type(t_target->type)->tag == FnType_TAG);
                    const FnType* t_tgt_type = &extract_operand_type(t_target->type)->payload.fn_type;
                    assert(t_tgt_type->tier == FnTier_BasicBlock);

                    assert(is_operand_uniform(f_target->type));
                    assert(extract_operand_type(f_target->type)->tag == FnType_TAG);
                    const FnType* f_tgt_type = &extract_operand_type(f_target->type)->payload.fn_type;
                    assert(f_tgt_type->tier == FnTier_BasicBlock);

                    // TODO: unify the two target types

                    LARRAY(const Node*, tmp, node->payload.branch.args.count);
                    for (size_t i = 0; i < node->payload.branch.args.count; i++)
                        tmp[i] = infer_value(ctx, node->payload.branch.args.nodes[i], t_tgt_type->param_types.nodes[i]);

                    Nodes new_args = nodes(ctx->rewriter.dst_arena, node->payload.branch.args.count, tmp);

                    return branch(ctx->rewriter.dst_arena, (Branch) {
                        .branch_mode = node->payload.branch.branch_mode,
                        .branch_condition = ncond,
                        .true_target = t_target,
                        .false_target = f_target,
                        .args = new_args
                    });
                }
                default: error("TODO")
            }
        }
        case MergeConstruct_TAG: {
            const Nodes* expected_types = NULL;
            switch (node->payload.merge_construct.construct) {
                case Selection: expected_types = ctx->join_types; break;
                case Continue: expected_types = ctx->continue_types; break;
                case Break: expected_types = ctx->break_types; break;
                default: error("we don't know this sort of merge");
            }
            assert(expected_types && "Merge terminator found but we're not within a suitable if/loop instruction !");
            const Nodes* old_args = &node->payload.merge_construct.args;
            assert(expected_types->count == old_args->count);
            LARRAY(const Node*, new_args, old_args->count);
            for (size_t i = 0; i < old_args->count; i++)
                new_args[i] = infer_value(ctx, old_args->nodes[i], (*expected_types).nodes[i]);
            return merge_construct(ctx->rewriter.dst_arena, (MergeConstruct) {
                .construct = node->payload.merge_construct.construct,
                .args = nodes(ctx->rewriter.dst_arena, old_args->count, new_args)
            });
        }
        // TODO break, continue
        case Unreachable_TAG: return unreachable(ctx->rewriter.dst_arena);
        default: error("not a terminator");
    }
}

static const Node* infer_root(Context* ctx, const Node* node) {
    if (node == NULL)
        return NULL;

    switch (node->tag) {
        case Root_TAG: {
            size_t count = node->payload.root.declarations.count;
            LARRAY(const Node*, new_decls, count);

            for (size_t i = 0; i < count; i++) {
                const Node* odecl = node->payload.root.declarations.nodes[i];
                new_decls[i] = infer_decl(ctx, odecl);
            }

            return root(ctx->rewriter.dst_arena, (Root) {
                .declarations = nodes(ctx->rewriter.dst_arena, count, new_decls),
            });
        }
        default: error("not a root node");
    }
}

#include "dict.h"

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

const Node* infer_program(SHADY_UNUSED CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
    Context ctx = {
        .rewriter = {
            .src_arena = src_arena,
            .dst_arena = dst_arena,
            .rewrite_fn = NULL, // we do all the rewriting ourselves
            .processed = done,
        },
    };

    const Node* rewritten = infer_root(&ctx, src_program);

    destroy_dict(done);
    return rewritten;
}
