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

    const Type* expected_type;

    const Nodes* join_types;
    const Nodes* break_types;
    const Nodes* continue_types;
} Context;

static const Node* infer(Context* ctx, const Node* node, const Type* expect) {
    Context ctx2 = *ctx;
    ctx2.expected_type = expect;
    return rewrite_node(&ctx2.rewriter, node);
}

static Nodes infer_nodes(Context* ctx, Nodes nodes) {
    Context ctx2 = *ctx;
    ctx2.expected_type = NULL;
    return rewrite_nodes(&ctx->rewriter, nodes);
}

#define rewrite_node error("don't use this directly, use the 'infer' and 'infer_node' helpers")
#define rewrite_nodes rewrite_node

static const Node* _infer_annotation(Context* ctx, const Node* node) {
    assert(node->tag == Annotation_TAG);
    Annotation old = node->payload.annotation;
    Annotation payload = {
        .payload_type = old.payload_type,
        .name = string(ctx->rewriter.dst_arena, old.name)
    };
    switch (old.payload_type) {
        case AnPayloadNone: break;
        case AnPayloadValue:
            payload.value = infer(ctx, old.value, NULL);
            break;
        case AnPayloadMap:
            payload.labels = import_strings(ctx->rewriter.dst_arena, old.labels);
            SHADY_FALLTHROUGH
        case AnPayloadValues:
            payload.values = infer_nodes(ctx, old.values);
            break;
        default: error("TODO");
    }
    return annotation(ctx->rewriter.dst_arena, payload);
}

static const Node* _infer_type(Context* ctx, const Type* type) {
    switch (type->tag) {
        case ArrType_TAG: {
            const Node* size = infer(ctx, type->payload.arr_type.size, int32_type(ctx->rewriter.dst_arena));
            return arr_type(ctx->rewriter.dst_arena, (ArrType) {
                .size = size,
                .element_type = infer(ctx, type->payload.arr_type.element_type, NULL)
            });
        }
        default: return recreate_node_identity(&ctx->rewriter, type);
    }
}

static const Node* _infer_decl(Context* ctx, const Node* node) {
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
                const Type* imported_param_type = infer(ctx, old_param->type, NULL);
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
                    Nodes nret_types = annotate_all_types(dst_arena, infer_nodes(ctx, node->payload.lam.return_types), false);
                    fun = function(ctx->rewriter.dst_module, nodes(dst_arena, node->payload.lam.params.count, nparams), string(dst_arena, node->payload.lam.name), infer_nodes(ctx, node->payload.lam.annotations), nret_types);
                    break;
                }
            }
            assert(fun);
            register_processed(&ctx->rewriter, node, fun);

            fun->payload.lam.body = infer(&body_context, node->payload.lam.body, NULL);

            return fun;
        }
        case Constant_TAG: {
            const Constant* oconstant = &node->payload.constant;
            Node* nconstant = constant(ctx->rewriter.dst_module, infer_nodes(ctx, oconstant->annotations), oconstant->name);
            register_processed(&ctx->rewriter, node, nconstant);

            const Type* imported_hint = infer(ctx, oconstant->type_hint, NULL);
            const Node* typed_value = infer(ctx, oconstant->value, imported_hint);
            nconstant->payload.constant.type_hint = NULL;
            if (is_declaration(typed_value))
                typed_value = ref_decl(dst_arena, (RefDecl) { .decl = typed_value });
            nconstant->payload.constant.value = typed_value;
            nconstant->type = extract_operand_type(typed_value->type);

            return nconstant;
        }
        case GlobalVariable_TAG: {
             const GlobalVariable* old_gvar = &node->payload.global_variable;
             const Type* imported_ty = infer(ctx, old_gvar->type, NULL);
             Node* ngvar = global_var(ctx->rewriter.dst_module, infer_nodes(ctx, old_gvar->annotations), imported_ty, old_gvar->name, old_gvar->address_space);
             register_processed(&ctx->rewriter, node, ngvar);

             ngvar->payload.global_variable.init = infer(ctx, old_gvar->init, NULL);
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

static const Node* _infer_value(Context* ctx, const Node* node, const Type* expected_type) {
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
        case Lambda_TAG: return fn_addr(dst_arena, (FnAddr) { .fn = infer(ctx, node, NULL) }); // TODO check types match
        case RefDecl_TAG:
        case FnAddr_TAG: return recreate_node_identity(&ctx->rewriter, node);
        default: error("not a value");
    }
}

static const Node* _infer_anonymous_lambda(Context* ctx, const Node* node, const Node* expected) {
    assert(is_anonymous_lambda(node));
    assert(expected);
    Nodes inferred_arg_type = unwrap_multiple_yield_types(ctx->rewriter.dst_arena, expected);
    assert(inferred_arg_type.count == node->payload.lam.params.count);
    IrArena* arena = ctx->rewriter.dst_arena;

    Context body_context = *ctx;
    LARRAY(const Node*, nparams, node->payload.lam.params.count);
    for (size_t i = 0; i < node->payload.lam.params.count; i++) {
        const Variable* old_param = &node->payload.lam.params.nodes[i]->payload.var;
        // for the param type: use the inferred one if none is already provided
        // if one is provided, check the inferred argument type is a subtype of the param type
        const Type* param_type = infer(ctx, old_param->type, NULL);
        param_type = param_type ? param_type : inferred_arg_type.nodes[i];
        assert(is_subtype(param_type, inferred_arg_type.nodes[i]));
        nparams[i] = var(body_context.rewriter.dst_arena, param_type, old_param->name);
        register_processed(&body_context.rewriter, node->payload.lam.params.nodes[i], nparams[i]);
    }

    Node* lam = lambda(arena, nodes(arena, node->payload.lam.params.count, nparams));
    assert(lam);
    register_processed(&ctx->rewriter, node, lam);

    lam->payload.lam.body = infer(&body_context, node->payload.lam.body, NULL);
    return lam;
}

static const Node* _infer_primop(Context* ctx, const Node* node, const Type* expected_type) {
    assert(node->tag == PrimOp_TAG);
    IrArena* dst_arena = ctx->rewriter.dst_arena;

    for (size_t i = 0; i < node->payload.prim_op.type_arguments.count; i++)
        assert(node->payload.prim_op.type_arguments.nodes[i] && is_type(node->payload.prim_op.type_arguments.nodes[i]));
    for (size_t i = 0; i < node->payload.prim_op.operands.count; i++)
        assert(node->payload.prim_op.operands.nodes[i] && is_value(node->payload.prim_op.operands.nodes[i]));

    Nodes type_args = infer_nodes(ctx, node->payload.prim_op.type_arguments);
    Nodes old_operands = node->payload.prim_op.operands;

    LARRAY(const Node*, new_inputs_scratch, old_operands.count);
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
            assert(old_operands.count == 1);
            assert(type_args.count == 1);
            const Type* element_type = type_args.nodes[0];
            assert(!contains_qualified_type(element_type));
            new_inputs_scratch[0] = infer(ctx, old_operands.nodes[0], element_type);
            goto skip_input_types;
        }
        case pop_stack_op:
        case pop_stack_uniform_op: {
            assert(old_operands.count == 0);
            assert(type_args.count == 1);
            const Type* element_type = type_args.nodes[0];
            assert(!contains_qualified_type(element_type));
            new_inputs_scratch[0] = element_type;
            goto skip_input_types;
        }
        case load_op: {
            assert(old_operands.count == 1);
            new_inputs_scratch[0] = infer(ctx, old_operands.nodes[0], NULL);
            goto skip_input_types;
        }
        case store_op: {
            assert(old_operands.count == 2);
            new_inputs_scratch[0] = infer(ctx, old_operands.nodes[0], NULL);
            const Type* ptr_type = extract_operand_type(new_inputs_scratch[0]->type);
            assert(ptr_type->tag == PtrType_TAG);
            new_inputs_scratch[1] = infer(ctx, old_operands.nodes[1], (&ptr_type->payload.ptr_type)->pointed_type);
            goto skip_input_types;
        }
        case alloca_op: {
            assert(type_args.count == 1);
            assert(old_operands.count == 0);
            const Type* element_type = type_args.nodes[0];
            assert(is_type(element_type));
            assert(!contains_qualified_type(element_type));
            goto skip_input_types;
        }
        case lea_op: {
            assert(old_operands.count >= 2);
            new_inputs_scratch[0] = infer(ctx, old_operands.nodes[0], NULL);
            for (size_t i = 1; i < old_operands.count; i++) {
                new_inputs_scratch[i] = infer(ctx, old_operands.nodes[i], int32_type(dst_arena));
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
            new_inputs_scratch[0] = infer(ctx, old_operands.nodes[0], NULL);
            goto skip_input_types;
        case subgroup_ballot_op:
            input_types = nodes(dst_arena, 1, (const Type* []) { bool_type(dst_arena) });
            break;
        case mask_is_thread_active_op: {
            input_types = nodes(dst_arena, 2, (const Type* []) { mask_type(dst_arena), int32_type(dst_arena) });
            break;
        }
        default: {
            for (size_t i = 0; i < old_operands.count; i++) {
                new_inputs_scratch[i] = old_operands.nodes[i] ? infer(ctx, old_operands.nodes[i], int32_type(dst_arena)) : NULL;
            }
            goto skip_input_types;
        }
    }

    assert(input_types.count == old_operands.count);
    for (size_t i = 0; i < input_types.count; i++)
        new_inputs_scratch[i] = infer(ctx, old_operands.nodes[i], input_types.nodes[i]);

    skip_input_types:
    return prim_op(dst_arena, (PrimOp) {
        .op = node->payload.prim_op.op,
        .type_arguments = type_args,
        .operands = nodes(dst_arena, old_operands.count, new_inputs_scratch)
    });
}

static const Node* _infer_call(Context* ctx, const Node* node, const Type* expected_type) {
    assert(node->tag == Call_TAG);

    const Node* new_callee = infer(ctx, node->payload.call_instr.callee, NULL);
    LARRAY(const Node*, new_args, node->payload.call_instr.args.count);

    const Type* callee_type = node->payload.call_instr.is_indirect ? extract_operand_type(new_callee->type) : new_callee->type;
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
        new_args[i] = infer(ctx, node->payload.call_instr.args.nodes[i], callee_type->payload.fn_type.param_types.nodes[i]);
        assert(new_args[i]->type);
    }

    return call_instr(ctx->rewriter.dst_arena, (Call) {
        .is_indirect = node->payload.call_instr.is_indirect,
        .callee = new_callee,
        .args = nodes(ctx->rewriter.dst_arena, node->payload.call_instr.args.count, new_args)
    });
}

static const Node* _infer_if(Context* ctx, const Node* node, const Type* expected_type) {
    assert(node->tag == If_TAG);
    IrArena* arena = ctx->rewriter.dst_arena;
    const Node* condition = infer(ctx, node->payload.if_instr.condition, bool_type(ctx->rewriter.dst_arena));

    Nodes join_types = infer_nodes(ctx, node->payload.if_instr.yield_types);
    // The type annotation on `if` may not include divergence/convergence info, we default that stuff to divergent
    join_types = annotate_all_types(ctx->rewriter.dst_arena, join_types, false);
    Context joinable_ctx = *ctx;
    joinable_ctx.join_types = &join_types;

    const Node* true_body = infer(&joinable_ctx, node->payload.if_instr.if_true, wrap_multiple_yield_types(arena, nodes(ctx->rewriter.dst_arena, 0, NULL)));
    // don't allow seeing the variables made available in the true branch
    joinable_ctx.rewriter = ctx->rewriter;
    const Node* false_body = node->payload.if_instr.if_false ? infer(&joinable_ctx, node->payload.if_instr.if_false, wrap_multiple_yield_types(arena, nodes(ctx->rewriter.dst_arena, 0, NULL))) : NULL;

    return if_instr(ctx->rewriter.dst_arena, (If) {
        .yield_types = join_types,
        .condition = condition,
        .if_true = true_body,
        .if_false = false_body,
    });
}

static const Node* _infer_loop(Context* ctx, const Node* node, const Type* expected_type) {
    assert(node->tag == Loop_TAG);
    IrArena* arena = ctx->rewriter.dst_arena;
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
        new_params_types[i] = infer(ctx, old_param->type, NULL);
        new_initial_args[i] = infer(ctx, old_initial_args.nodes[i], new_params_types[i]);
        //new_params[i] = var(loop_body_ctx.rewriter.dst_arena, new_params_types[i], old_param->name);
        //register_processed(&loop_body_ctx.rewriter, old_params.nodes[i], new_params[i]);
    }

    Nodes loop_yield_types = infer_nodes(ctx, node->payload.loop_instr.yield_types);
    loop_yield_types = annotate_all_types(ctx->rewriter.dst_arena, loop_yield_types, false);

    loop_body_ctx.join_types = NULL;
    loop_body_ctx.break_types = &loop_yield_types;
    Nodes param_types = nodes(ctx->rewriter.dst_arena, old_params.count, new_params_types);
    loop_body_ctx.continue_types = &param_types;

    return loop_instr(ctx->rewriter.dst_arena, (Loop) {
        .yield_types = loop_yield_types,
        .initial_args = nodes(ctx->rewriter.dst_arena, old_params.count, new_initial_args),
        .body = infer(&loop_body_ctx, node->payload.loop_instr.body, wrap_multiple_yield_types(arena, param_types))
    });
}

static const Node* _infer_instruction(Context* ctx, const Node* node, const Type* expected_type) {
    switch (is_instruction(node)) {
        case PrimOp_TAG:  return _infer_primop(ctx, node, expected_type);
        case Call_TAG:    return _infer_call  (ctx, node, expected_type);
        case If_TAG:      return _infer_if    (ctx, node, expected_type);
        case Loop_TAG:    return _infer_loop  (ctx, node, expected_type);
        case Match_TAG:   error("TODO")
        case Control_TAG: error("TODO")
        case NotAnInstruction: error("not an instruction");
    }
    SHADY_UNREACHABLE;
}

static const Node* _infer_terminator(Context* ctx, const Node* node) {
    IrArena* arena = ctx->rewriter.dst_arena;
    switch (node->tag) {
        case Let_TAG: {
            assert(!node->payload.let.is_mutable);
            const Node* otail = node->payload.let.tail;
            assert(otail->tag == Lambda_TAG);
            Nodes annotated_types = extract_variable_types(arena, &otail->payload.lam.params);
            const Node* inferred_instruction = infer(ctx, node->payload.let.instruction, wrap_multiple_yield_types(arena, annotated_types));
            Nodes inferred_yield_types = unwrap_multiple_yield_types(arena, inferred_instruction->type);
            const Node* inferred_tail = infer(ctx, node->payload.let.tail, wrap_multiple_yield_types(arena, inferred_yield_types));
            return let(arena, false, inferred_instruction, inferred_tail);
        }
        case Return_TAG: {
            const Node* imported_fn = infer(ctx, node->payload.fn_ret.fn, NULL);
            Nodes return_types = imported_fn->payload.lam.return_types;

            const Nodes* old_values = &node->payload.fn_ret.values;
            LARRAY(const Node*, nvalues, old_values->count);
            for (size_t i = 0; i < old_values->count; i++)
                nvalues[i] = infer(ctx, old_values->nodes[i], return_types.nodes[i]);
            return fn_ret(ctx->rewriter.dst_arena, (Return) {
                .values = nodes(ctx->rewriter.dst_arena, old_values->count, nvalues),
                .fn = NULL
            });
        }
        case Branch_TAG: {
            switch (node->payload.branch.branch_mode) {
                case BrJump: {
                    const Node* ntarget = infer(ctx, node->payload.branch.target, NULL);
                    const Type* ntarget_type;
                    bool ntarget_is_uniform;
                    deconstruct_operand_type(ntarget->type, &ntarget_type, &ntarget_is_uniform);
                    assert(ntarget_is_uniform);
                    assert(ntarget_type->tag == FnType_TAG);
                    const FnType* tgt_type = &ntarget_type->payload.fn_type;
                    assert(tgt_type->tier == FnTier_BasicBlock);

                    LARRAY(const Node*, tmp, node->payload.branch.args.count);
                    for (size_t i = 0; i < node->payload.branch.args.count; i++)
                        tmp[i] = infer(ctx, node->payload.branch.args.nodes[i], tgt_type->param_types.nodes[i]);

                    Nodes new_args = nodes(ctx->rewriter.dst_arena, node->payload.branch.args.count, tmp);

                    return branch(ctx->rewriter.dst_arena, (Branch) {
                        .branch_mode = node->payload.branch.branch_mode,
                        .target = ntarget,
                        .args = new_args
                    });
                }
                case BrIfElse: {
                    const Node* ncond = infer(ctx, node->payload.branch.branch_condition, bool_type(ctx->rewriter.dst_arena));

                    const Node* t_target = infer(ctx, node->payload.branch.true_target, NULL);
                    const Node* f_target = infer(ctx, node->payload.branch.false_target, NULL);

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
                        tmp[i] = infer(ctx, node->payload.branch.args.nodes[i], t_tgt_type->param_types.nodes[i]);

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
                new_args[i] = infer(ctx, old_args->nodes[i], (*expected_types).nodes[i]);
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

static const Node* process(Context* src_ctx, const Node* node) {
    const Type* expect = src_ctx->expected_type;
    Context ctx = *src_ctx;
    ctx.expected_type = NULL;

    const Node* found = search_processed(&src_ctx->rewriter, node);
    if (found) {
        //if (expect)
        //    assert(is_subtype(expect, found->type));
        return found;
    }

    if (is_type(node)) {
        assert(expect == NULL);
        return _infer_type(&ctx, node);
    } else if (is_value(node)) {
        const Node* value = _infer_value(&ctx, node, expect);
        assert(contains_qualified_type(value->type));
        return value;
    }else if (is_instruction(node))
        return _infer_instruction(&ctx, node, expect);
    else if (is_terminator(node)) {
        assert(expect == NULL);
        return _infer_terminator(&ctx, node);
    } else if (is_declaration(node)) {
        return _infer_decl(&ctx, node);
    } else if (node->tag == Annotation_TAG) {
        assert(expect == NULL);
        return _infer_annotation(&ctx, node);
    } else if (is_anonymous_lambda(node)) {
        assert(expect != NULL);
        return _infer_anonymous_lambda(&ctx, node, expect);
    }
    assert(false);
}

void infer_program(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
