#include "passes.h"

#include "log.h"
#include "portability.h"

#include "../type.h"
#include "../rewrite.h"

#include <assert.h>
#include <string.h>

#pragma GCC diagnostic error "-Wswitch"

static Nodes annotate_all_types(IrArena* arena, Nodes types, bool uniform_by_default) {
    LARRAY(const Type*, ntypes, types.count);
    for (size_t i = 0; i < types.count; i++) {
        if (is_data_type(types.nodes[i]))
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

    const Nodes* merge_types;
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
    assert(is_annotation(node));
    switch (node->tag) {
        case Annotation_TAG: return annotation(ctx->rewriter.dst_arena, (Annotation) { .name = node->payload.annotation.name });
        case AnnotationValue_TAG: return annotation_value(ctx->rewriter.dst_arena, (AnnotationValue) { .name = node->payload.annotation_value.name, .value = infer(ctx, node->payload.annotation_value.value, NULL) });
        case AnnotationValues_TAG: return annotation_values(ctx->rewriter.dst_arena, (AnnotationValues) { .name = node->payload.annotation_values.name, .values = infer_nodes(ctx, node->payload.annotation_values.values) });
        case AnnotationCompound_TAG: return annotations_compound(ctx->rewriter.dst_arena, (AnnotationCompound) { .name = node->payload.annotations_compound.name, .entries = infer_nodes(ctx, node->payload.annotations_compound.entries) });
        default: error("Not an annotation");
    }
}

static const Node* _infer_type(Context* ctx, const Type* type) {
    switch (type->tag) {
        case ArrType_TAG: {
            const Node* size = infer(ctx, type->payload.arr_type.size, NULL);
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
    switch (is_declaration(node)) {
        case Function_TAG: {
            Context body_context = *ctx;

            LARRAY(const Node*, nparams, node->payload.fun.params.count);
            for (size_t i = 0; i < node->payload.fun.params.count; i++) {
                const Variable* old_param = &node->payload.fun.params.nodes[i]->payload.var;
                const Type* imported_param_type = infer(ctx, old_param->type, NULL);
                nparams[i] = var(body_context.rewriter.dst_arena, imported_param_type, old_param->name);
                register_processed(&body_context.rewriter, node->payload.fun.params.nodes[i], nparams[i]);
            }

            Nodes nret_types = annotate_all_types(dst_arena, infer_nodes(ctx, node->payload.fun.return_types), false);
            Node* fun = function(ctx->rewriter.dst_module, nodes(dst_arena, node->payload.fun.params.count, nparams), string(dst_arena, node->payload.fun.name), infer_nodes(ctx, node->payload.fun.annotations), nret_types);
            register_processed(&ctx->rewriter, node, fun);
            fun->payload.fun.body = infer(&body_context, node->payload.fun.body, NULL);
            return fun;
        }
        case Constant_TAG: {
            const Constant* oconstant = &node->payload.constant;
            const Type* imported_hint = infer(ctx, oconstant->type_hint, NULL);
            assert(is_data_type(imported_hint));
            const Node* typed_value = infer(ctx, oconstant->value, qualified_type_helper(imported_hint, true));
            assert(is_value(typed_value));
            imported_hint = get_unqualified_type(typed_value->type);

            Node* nconstant = constant(ctx->rewriter.dst_module, infer_nodes(ctx, oconstant->annotations), imported_hint, oconstant->name);
            register_processed(&ctx->rewriter, node, nconstant);
            nconstant->payload.constant.value = typed_value;

            return nconstant;
        }
        case GlobalVariable_TAG: {
             const GlobalVariable* old_gvar = &node->payload.global_variable;
             const Type* imported_ty = infer(ctx, old_gvar->type, NULL);
             Node* ngvar = global_var(ctx->rewriter.dst_module, infer_nodes(ctx, old_gvar->annotations), imported_ty, old_gvar->name, old_gvar->address_space);
             register_processed(&ctx->rewriter, node, ngvar);

             ngvar->payload.global_variable.init = infer(ctx, old_gvar->init, qualified_type_helper(imported_ty, true));
             return ngvar;
        }
        case Decl_NominalType_TAG: {
            const NominalType* onom_type = &node->payload.nom_type;
            Node* nnominal_type = nominal_type(ctx->rewriter.dst_module, infer_nodes(ctx, onom_type->annotations), onom_type->name);
            register_processed(&ctx->rewriter, node, nnominal_type);
            nnominal_type->payload.nom_type.body = infer(ctx, onom_type->body, NULL);
            return nnominal_type;
        }
        case NotADecl: error("not a decl");
    }
}

/// Like get_unqualified_type but won't error out if type wasn't qualified to begin with
static const Type* remove_uniformity_qualifier(const Node* type) {
    if (is_value_type(type))
        return get_unqualified_type(type);
    return type;
}

static const Node* _infer_value(Context* ctx, const Node* node, const Type* expected_type) {
    if (!node) return NULL;

    if (expected_type) {
        assert(is_value_type(expected_type));
    }

    IrArena* dst_arena = ctx->rewriter.dst_arena;
    switch (is_value(node)) {
        case NotAValue: error("");
        case Variable_TAG: return find_processed(&ctx->rewriter, node);
        case Value_ConstrainedValue_TAG: {
            const Type* type = infer(ctx, node->payload.constrained.type, NULL);
            bool expect_uniform = false;
            if (expected_type) {
                expect_uniform = deconstruct_qualified_type(&expected_type);
                assert(is_subtype(expected_type, type));
            }
            return infer(ctx, node->payload.constrained.value, qualified_type_helper(type, expect_uniform));
        }
        case IntLiteral_TAG: {
            if (expected_type) {
                expected_type = remove_uniformity_qualifier(expected_type);
                assert(expected_type->tag == Int_TAG);
                assert(expected_type->payload.int_type.width == node->payload.int_literal.width);
            }
            return int_literal(dst_arena, (IntLiteral) {
                .width = node->payload.int_literal.width,
                .is_signed = node->payload.int_literal.is_signed,
                .value.u64 = node->payload.int_literal.value.u64});
        }
        case UntypedNumber_TAG: {
            char* endptr;
            int64_t i = strtoll(node->payload.untyped_number.plaintext, &endptr, 10);
            if (!expected_type) {
                bool valid_int = *endptr == '\0';
                expected_type = valid_int ? int32_type(dst_arena) : fp32_type(dst_arena);
            }
            expected_type = remove_uniformity_qualifier(expected_type);
            if (expected_type->tag == Int_TAG) {
                // TODO chop off extra bits based on width ?
                return int_literal(dst_arena, (IntLiteral) {
                    .width = expected_type->payload.int_type.width,
                    .is_signed = expected_type->payload.int_literal.is_signed,
                    .value.i64 = i
                });
            } else if (expected_type->tag == Float_TAG) {
                FloatLiteralValue v;
                switch (expected_type->payload.float_type.width) {
                    case FloatTy16:
                        error("TODO: implement fp16 parsing");
                    case FloatTy32:
                        assert(sizeof(float) == sizeof(uint32_t));
                        float f = strtof(node->payload.untyped_number.plaintext, NULL);
                        memcpy(&v.b32, &f, sizeof(uint32_t));
                        break;
                    case FloatTy64:
                        assert(sizeof(double) == sizeof(uint64_t));
                        double d = strtod(node->payload.untyped_number.plaintext, NULL);
                        memcpy(&v.b64, &d, sizeof(uint64_t));
                        break;
                }
                return float_literal(dst_arena, (FloatLiteral) {.value = v, .width = expected_type->payload.float_type.width});
            }
        }
        case FloatLiteral_TAG: {
            if (expected_type) {
                expected_type = remove_uniformity_qualifier(expected_type);
                assert(expected_type->tag == Float_TAG);
                assert(expected_type->payload.float_type.width == node->payload.float_literal.width);
            }
            return float_literal(dst_arena, (FloatLiteral) { .width = node->payload.float_literal.width, .value = node->payload.float_literal.value });
        }
        case True_TAG: return true_lit(dst_arena);
        case False_TAG: return false_lit(dst_arena);
        case StringLiteral_TAG: return string_lit(dst_arena, (StringLiteral) { .string = string(dst_arena, node->payload.string_lit.string )});
        case RefDecl_TAG:
        case FnAddr_TAG: return recreate_node_identity(&ctx->rewriter, node);
        case Value_Undef_TAG: return recreate_node_identity(&ctx->rewriter, node);
        case Value_Composite_TAG: {
            const Node* elem_type = infer(ctx, node->payload.composite.type, NULL);
            bool uniform = false;
            if (elem_type && expected_type) {
                assert(is_subtype(get_unqualified_type(expected_type), elem_type));
            } else if (expected_type) {
                uniform = deconstruct_qualified_type(&elem_type);
                elem_type = expected_type;
            }

            Nodes omembers = node->payload.composite.contents;
            LARRAY(const Node*, inferred, omembers.count);
            if (elem_type) {
                Nodes expected_members = get_composite_type_element_types(elem_type);
                for (size_t i = 0; i < omembers.count; i++)
                    inferred[i] = infer(ctx, omembers.nodes[i], qualified_type(dst_arena, (QualifiedType) { .is_uniform = uniform, .type = expected_members.nodes[i] }));
            } else {
                for (size_t i = 0; i < omembers.count; i++)
                    inferred[i] = infer(ctx, omembers.nodes[i], NULL);
            }
            Nodes nmembers = nodes(dst_arena, omembers.count, inferred);

            // Composites are tuples by default
            if (!elem_type)
                elem_type = record_type(dst_arena, (RecordType) { .members = strip_qualifiers(dst_arena, get_values_types(dst_arena, nmembers)) });

            return composite(dst_arena, elem_type, nmembers);
        }
        case Value_Fill_TAG: {
            const Node* composite_t = infer(ctx, node->payload.fill.type, NULL);
            assert(composite_t);
            bool uniform = false;
            if (composite_t && expected_type) {
                assert(is_subtype(get_unqualified_type(expected_type), composite_t));
            } else if (expected_type) {
                uniform = deconstruct_qualified_type(&composite_t);
                composite_t = expected_type;
            }
            assert(composite_t);
            const Node* element_t = get_fill_type_element_type(composite_t);
            const Node* value = infer(ctx, node->payload.fill.value, qualified_type(dst_arena, (QualifiedType) { .is_uniform = uniform, .type = element_t }));
            return fill(dst_arena, (Fill) { .type = composite_t, .value = value });
        }
        case Value_AntiQuote_TAG: error("TODO");
    }
}

static const Node* _infer_anonymous_lambda(Context* ctx, const Node* node, const Node* expected) {
    assert(is_anonymous_lambda(node));
    assert(expected);
    Nodes inferred_arg_type = unwrap_multiple_yield_types(ctx->rewriter.dst_arena, expected);
    assert(inferred_arg_type.count == node->payload.anon_lam.params.count || node->payload.anon_lam.params.count == 0);
    IrArena* arena = ctx->rewriter.dst_arena;

    Context body_context = *ctx;
    LARRAY(const Node*, nparams, inferred_arg_type.count);
    for (size_t i = 0; i < inferred_arg_type.count; i++) {
        if (node->payload.anon_lam.params.count == 0) {
            // syntax sugar: make up a parameter if there was none
            nparams[i] = var(body_context.rewriter.dst_arena, inferred_arg_type.nodes[i], unique_name(arena, "_"));
        } else {
            const Variable* old_param = &node->payload.anon_lam.params.nodes[i]->payload.var;
            // for the param type: use the inferred one if none is already provided
            // if one is provided, check the inferred argument type is a subtype of the param type
            const Type* param_type = infer(ctx, old_param->type, NULL);
            param_type = param_type ? param_type : inferred_arg_type.nodes[i];
            assert(is_subtype(param_type, inferred_arg_type.nodes[i]));
            nparams[i] = var(body_context.rewriter.dst_arena, param_type, old_param->name);
            register_processed(&body_context.rewriter, node->payload.anon_lam.params.nodes[i], nparams[i]);
        }
    }

    const Node* new_body = infer(&body_context, node->payload.anon_lam.body, NULL);
    return lambda(ctx->rewriter.dst_arena, nodes(arena, inferred_arg_type.count, nparams), new_body);
}

static const Node* _infer_basic_block(Context* ctx, const Node* node) {
    assert(is_basic_block(node));
    IrArena* arena = ctx->rewriter.dst_arena;

    Context body_context = *ctx;
    LARRAY(const Node*, nparams, node->payload.basic_block.params.count);
    for (size_t i = 0; i < node->payload.basic_block.params.count; i++) {
        const Variable* old_param = &node->payload.basic_block.params.nodes[i]->payload.var;
        // for the param type: use the inferred one if none is already provided
        // if one is provided, check the inferred argument type is a subtype of the param type
        const Type* param_type = infer(ctx, old_param->type, NULL);
        assert(param_type);
        nparams[i] = var(body_context.rewriter.dst_arena, param_type, old_param->name);
        register_processed(&body_context.rewriter, node->payload.basic_block.params.nodes[i], nparams[i]);
    }

    Node* fn = (Node*) infer(ctx, node->payload.basic_block.fn, NULL);
    Node* bb = basic_block(ctx->rewriter.dst_arena, fn, nodes(arena, node->payload.basic_block.params.count, nparams), node->payload.basic_block.name);
    assert(bb);
    register_processed(&ctx->rewriter, node, bb);

    bb->payload.basic_block.body = infer(&body_context, node->payload.basic_block.body, NULL);
    return bb;
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
        /*case neg_op:
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
            input_types = nodes(dst_arena, 2, (const Type*[]){ int32_type(dst_arena), int32_type(dst_arena) }); break;*/
        case push_stack_op:
        case push_stack_uniform_op: {
            assert(old_operands.count == 1);
            assert(type_args.count == 1);
            const Type* element_type = type_args.nodes[0];
            assert(is_data_type(element_type));
            new_inputs_scratch[0] = infer(ctx, old_operands.nodes[0], qualified_type_helper(element_type, false));
            goto skip_input_types;
        }
        case pop_stack_op:
        case pop_stack_uniform_op: {
            assert(old_operands.count == 0);
            assert(type_args.count == 1);
            const Type* element_type = type_args.nodes[0];
            assert(is_data_type(element_type));
            //new_inputs_scratch[0] = element_type;
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
            const Type* ptr_type = get_unqualified_type(new_inputs_scratch[0]->type);
            assert(ptr_type->tag == PtrType_TAG);
            new_inputs_scratch[1] = infer(ctx, old_operands.nodes[1], qualified_type_helper((&ptr_type->payload.ptr_type)->pointed_type, false));
            goto skip_input_types;
        }
        case alloca_op: {
            assert(type_args.count == 1);
            assert(old_operands.count == 0);
            const Type* element_type = type_args.nodes[0];
            assert(is_type(element_type));
            assert(is_data_type(element_type));
            goto skip_input_types;
        }
        case lea_op: {
            assert(old_operands.count >= 2);
            new_inputs_scratch[0] = infer(ctx, old_operands.nodes[0], NULL);
            new_inputs_scratch[1] = infer(ctx, old_operands.nodes[1], NULL);
            for (size_t i = 2; i < old_operands.count; i++) {
                new_inputs_scratch[i] = infer(ctx, old_operands.nodes[i], /*int32_type(dst_arena)*/ NULL);
            }

            const Type* base_datatype = remove_uniformity_qualifier(new_inputs_scratch[0]->type);
            assert(base_datatype->tag == PtrType_TAG);
            AddressSpace as = deconstruct_pointer_type(&base_datatype);
            const IntLiteral* lit = resolve_to_literal(new_inputs_scratch[1]);
            if ((!lit || lit->value.u64) != 0 && base_datatype->tag != ArrType_TAG) {
                warn_print("LEA used on a pointer to a non-array type!\n");
                BodyBuilder* bb = begin_body(dst_arena);
                const Node* cast_base = first(bind_instruction(bb, prim_op(dst_arena, (PrimOp) {
                    .op = reinterpret_op,
                    .type_arguments = singleton(ptr_type(dst_arena, (PtrType) {
                        .address_space = as,
                        .pointed_type = arr_type(dst_arena, (ArrType) {
                            .element_type = base_datatype,
                            .size = NULL
                        }),
                    })),
                    .operands = singleton(new_inputs_scratch[0]),
                })));
                Nodes final_lea_ops = mk_nodes(dst_arena, cast_base, new_inputs_scratch[1], int32_literal(dst_arena, 0));
                final_lea_ops = concat_nodes(dst_arena, final_lea_ops, nodes(dst_arena, old_operands.count - 2, new_inputs_scratch + 2));
                const Node* rslt = first(bind_instruction(bb, prim_op(dst_arena, (PrimOp) {
                        .op = lea_op,
                        .type_arguments = empty(dst_arena),
                        .operands = final_lea_ops
                })));
                return yield_values_and_wrap_in_block(bb, singleton(rslt));
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
            input_types = singleton(qualified_type_helper(bool_type(dst_arena), false));
            break;
        case mask_is_thread_active_op: {
            input_types = mk_nodes(dst_arena, qualified_type_helper(mask_type(dst_arena), false), qualified_type_helper(uint32_type(dst_arena), false));
            break;
        }
        default: {
            for (size_t i = 0; i < old_operands.count; i++) {
                new_inputs_scratch[i] = old_operands.nodes[i] ? infer(ctx, old_operands.nodes[i], NULL) : NULL;
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

static const Node* _infer_indirect_call(Context* ctx, const Node* node, const Type* expected_type) {
    assert(node->tag == Call_TAG);

    const Node* new_callee = infer(ctx, node->payload.call.callee, NULL);
    assert(is_value(new_callee));
    LARRAY(const Node*, new_args, node->payload.call.args.count);

    const Type* callee_type = get_unqualified_type(new_callee->type);
    if (callee_type->tag != PtrType_TAG)
        error("functions are called through function pointers");
    callee_type = callee_type->payload.ptr_type.pointed_type;

    if (callee_type->tag != FnType_TAG)
        error("Callees must have a function type");
    if (callee_type->payload.fn_type.param_types.count != node->payload.call.args.count)
        error("Mismatched argument counts");
    for (size_t i = 0; i < node->payload.call.args.count; i++) {
        const Node* arg = node->payload.call.args.nodes[i];
        assert(arg);
        new_args[i] = infer(ctx, node->payload.call.args.nodes[i], callee_type->payload.fn_type.param_types.nodes[i]);
        assert(new_args[i]->type);
    }

    return call(ctx->rewriter.dst_arena, (Call) {
        .callee = new_callee,
        .args = nodes(ctx->rewriter.dst_arena, node->payload.call.args.count, new_args)
    });
}

static const Node* _infer_if(Context* ctx, const Node* node, const Type* expected_type) {
    assert(node->tag == If_TAG);
    IrArena* arena = ctx->rewriter.dst_arena;
    const Node* condition = infer(ctx, node->payload.if_instr.condition, bool_type(ctx->rewriter.dst_arena));

    Nodes join_types = infer_nodes(ctx, node->payload.if_instr.yield_types);
    Context infer_if_body_ctx = *ctx;
    // When we infer the types of the arguments to a call to merge(), they are expected to be varying
    Nodes expected_join_types = annotate_all_types(ctx->rewriter.dst_arena, join_types, false);
    infer_if_body_ctx.merge_types = &expected_join_types;

    const Node* true_body = infer(&infer_if_body_ctx, node->payload.if_instr.if_true, wrap_multiple_yield_types(arena, nodes(ctx->rewriter.dst_arena, 0, NULL)));
    // don't allow seeing the variables made available in the true branch
    infer_if_body_ctx.rewriter = ctx->rewriter;
    const Node* false_body = node->payload.if_instr.if_false ? infer(&infer_if_body_ctx, node->payload.if_instr.if_false, wrap_multiple_yield_types(arena, nodes(ctx->rewriter.dst_arena, 0, NULL))) : NULL;

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

    Nodes old_params = get_abstraction_params(old_body);
    Nodes old_params_types = get_variables_types(arena, old_params);
    Nodes new_params_types = infer_nodes(ctx, old_params_types);

    Nodes old_initial_args = node->payload.loop_instr.initial_args;
    LARRAY(const Node*, new_initial_args, old_params.count);
    for (size_t i = 0; i < old_params.count; i++)
        new_initial_args[i] = infer(ctx, old_initial_args.nodes[i], new_params_types.nodes[i]);

    Nodes loop_yield_types = infer_nodes(ctx, node->payload.loop_instr.yield_types);

    loop_body_ctx.merge_types = NULL;
    loop_body_ctx.break_types = &loop_yield_types;
    loop_body_ctx.continue_types = &new_params_types;

    const Node* nbody = infer(&loop_body_ctx, old_body, wrap_multiple_yield_types(arena, new_params_types));
    // TODO check new body params match continue types

    return loop_instr(ctx->rewriter.dst_arena, (Loop) {
        .yield_types = loop_yield_types,
        .initial_args = nodes(ctx->rewriter.dst_arena, old_params.count, new_initial_args),
        .body = nbody,
    });
}

static const Node* _infer_control(Context* ctx, const Node* node, const Type* expected_type) {
    assert(node->tag == Control_TAG);
    IrArena* arena = ctx->rewriter.dst_arena;

    Nodes yield_types = infer_nodes(ctx, node->payload.control.yield_types);

    const Node* olam = node->payload.control.inside;
    const Node* ojp = first(get_abstraction_params(olam));

    Context joinable_ctx = *ctx;
    const Type* jpt = join_point_type(arena, (JoinPointType) {
            .yield_types = yield_types
    });
    jpt = qualified_type(arena, (QualifiedType) { .is_uniform = true, .type = jpt });
    const Node* jp = var(arena, jpt, ojp->payload.var.name);
    register_processed(&ctx->rewriter, ojp, jp);

    const Node* nlam = lambda(ctx->rewriter.dst_arena, singleton(jp), infer(&joinable_ctx, get_abstraction_body(olam), NULL));

    return control(ctx->rewriter.dst_arena, (Control) {
        .yield_types = yield_types,
        .inside = nlam
    });
}

static const Node* _infer_block(Context* ctx, const Node* node, const Type* expected_type) {
    assert(node->tag == Block_TAG);
    IrArena* arena = ctx->rewriter.dst_arena;

    const Node* olam = node->payload.block.inside;

    const Node* nlam = lambda(ctx->rewriter.dst_arena, empty(arena), infer(ctx, get_abstraction_body(olam), NULL));

    return control(ctx->rewriter.dst_arena, (Control) {
        .inside = nlam
    });
}

static const Node* _infer_instruction(Context* ctx, const Node* node, const Type* expected_type) {
    switch (is_instruction(node)) {
        case PrimOp_TAG:       return _infer_primop(ctx, node, expected_type);
        case Call_TAG:         return _infer_indirect_call(ctx, node, expected_type);
        case If_TAG:           return _infer_if    (ctx, node, expected_type);
        case Loop_TAG:         return _infer_loop  (ctx, node, expected_type);
        case Match_TAG:        error("TODO")
        case Control_TAG:      return _infer_control(ctx, node, expected_type);
        case Block_TAG:        return _infer_block  (ctx, node, expected_type);
        case Instruction_Comment_TAG: return recreate_node_identity(&ctx->rewriter, node);
        case NotAnInstruction: error("not an instruction");
    }
    SHADY_UNREACHABLE;
}

static const Node* _infer_terminator(Context* ctx, const Node* node) {
    IrArena* arena = ctx->rewriter.dst_arena;
    switch (is_terminator(node)) {
        case Terminator_LetMut_TAG:
        case NotATerminator: assert(false);
        case Let_TAG: {
            const Node* otail = node->payload.let.tail;
            Nodes annotated_types = get_variables_types(arena, otail->payload.anon_lam.params);
            const Node* inferred_instruction = infer(ctx, node->payload.let.instruction, wrap_multiple_yield_types(arena, annotated_types));
            Nodes inferred_yield_types = unwrap_multiple_yield_types(arena, inferred_instruction->type);
            for (size_t i = 0; i < inferred_yield_types.count; i++) {
                assert(is_value_type(inferred_yield_types.nodes[i]));
            }
            const Node* inferred_tail = infer(ctx, otail, wrap_multiple_yield_types(arena, inferred_yield_types));
            return let(arena, inferred_instruction, inferred_tail);
        }
        case Return_TAG: {
            const Node* imported_fn = infer(ctx, node->payload.fn_ret.fn, NULL);
            Nodes return_types = imported_fn->payload.fun.return_types;

            const Nodes* old_values = &node->payload.fn_ret.args;
            LARRAY(const Node*, nvalues, old_values->count);
            for (size_t i = 0; i < old_values->count; i++)
                nvalues[i] = infer(ctx, old_values->nodes[i], return_types.nodes[i]);
            return fn_ret(ctx->rewriter.dst_arena, (Return) {
                .args = nodes(ctx->rewriter.dst_arena, old_values->count, nvalues),
                .fn = NULL
            });
        }
        case Jump_TAG: {
            assert(is_basic_block(node->payload.jump.target));
            const Node* ntarget = infer(ctx, node->payload.jump.target, NULL);
            Nodes param_types = get_variables_types(arena, get_abstraction_params(ntarget));

            LARRAY(const Node*, tmp, node->payload.jump.args.count);
            for (size_t i = 0; i < node->payload.jump.args.count; i++)
                tmp[i] = infer(ctx, node->payload.jump.args.nodes[i], param_types.nodes[i]);

            Nodes new_args = nodes(ctx->rewriter.dst_arena, node->payload.jump.args.count, tmp);

            return jump(ctx->rewriter.dst_arena, (Jump) {
                .target = ntarget,
                .args = new_args
            });
        }
        case Branch_TAG: {
            const Node* ncond = infer(ctx, node->payload.branch.branch_condition, bool_type(ctx->rewriter.dst_arena));

            assert(is_basic_block(node->payload.branch.true_target));
            assert(is_basic_block(node->payload.branch.false_target));
            const Node* t_target = infer(ctx, node->payload.branch.true_target, NULL);
            const Node* f_target = infer(ctx, node->payload.branch.false_target, NULL);

            Nodes t_param_types = get_variables_types(arena, get_abstraction_params(t_target));
            Nodes f_param_types = get_variables_types(arena, get_abstraction_params(f_target));

            // TODO: unify the two target types

            LARRAY(const Node*, tmp, node->payload.branch.args.count);
            for (size_t i = 0; i < node->payload.branch.args.count; i++)
                tmp[i] = infer(ctx, node->payload.branch.args.nodes[i], t_param_types.nodes[i]);

            Nodes new_args = nodes(ctx->rewriter.dst_arena, node->payload.branch.args.count, tmp);

            return branch(ctx->rewriter.dst_arena, (Branch) {
                .branch_condition = ncond,
                .true_target = t_target,
                .false_target = f_target,
                .args = new_args
            });
        }
        case Terminator_Yield_TAG: {
            const Nodes* expected_types = ctx->merge_types;
            // TODO: block nodes should set merge types
            assert(expected_types && "Merge terminator found but we're not within a suitable if instruction !");
            const Nodes* old_args = &node->payload.yield.args;
            assert(expected_types->count == old_args->count);
            LARRAY(const Node*, new_args, old_args->count);
            for (size_t i = 0; i < old_args->count; i++)
                new_args[i] = infer(ctx, old_args->nodes[i], (*expected_types).nodes[i]);
            return yield(ctx->rewriter.dst_arena, (Yield) {
                .args = nodes(ctx->rewriter.dst_arena, old_args->count, new_args)
            });
        }
        case MergeContinue_TAG: {
            const Nodes* expected_types = ctx->continue_types;
            assert(expected_types && "Merge terminator found but we're not within a suitable loop instruction !");
            const Nodes* old_args = &node->payload.merge_continue.args;
            assert(expected_types->count == old_args->count);
            LARRAY(const Node*, new_args, old_args->count);
            for (size_t i = 0; i < old_args->count; i++)
                new_args[i] = infer(ctx, old_args->nodes[i], (*expected_types).nodes[i]);
            return merge_continue(ctx->rewriter.dst_arena, (MergeContinue) {
                .args = nodes(ctx->rewriter.dst_arena, old_args->count, new_args)
            });
        }
        case MergeBreak_TAG: {
            const Nodes* expected_types = ctx->break_types;
            assert(expected_types && "Merge terminator found but we're not within a suitable loop instruction !");
            const Nodes* old_args = &node->payload.merge_break.args;
            assert(expected_types->count == old_args->count);
            LARRAY(const Node*, new_args, old_args->count);
            for (size_t i = 0; i < old_args->count; i++)
                new_args[i] = infer(ctx, old_args->nodes[i], (*expected_types).nodes[i]);
            return merge_break(ctx->rewriter.dst_arena, (MergeBreak) {
                .args = nodes(ctx->rewriter.dst_arena, old_args->count, new_args)
            });
        }
        case Unreachable_TAG: return unreachable(ctx->rewriter.dst_arena);
        case Terminator_Join_TAG: return join(arena, (Join) {
            .join_point = infer(ctx, node->payload.join.join_point, NULL),
            .args = infer_nodes(ctx, node->payload.join.args),
        });
        case Terminator_TailCall_TAG:
        case Terminator_Switch_TAG: error("TODO")
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
        assert(is_value_type(value->type));
        return value;
    }else if (is_instruction(node))
        return _infer_instruction(&ctx, node, expect);
    else if (is_terminator(node)) {
        assert(expect == NULL);
        return _infer_terminator(&ctx, node);
    } else if (is_declaration(node)) {
        return _infer_decl(&ctx, node);
    } else if (is_annotation(node)) {
        assert(expect == NULL);
        return _infer_annotation(&ctx, node);
    } else if (is_anonymous_lambda(node)) {
        assert(expect != NULL);
        return _infer_anonymous_lambda(&ctx, node, expect);
    } else if (is_basic_block(node)) {
        return _infer_basic_block(&ctx, node);
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
