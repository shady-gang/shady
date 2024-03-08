#include "passes.h"

#include "log.h"
#include "portability.h"

#include "../type.h"
#include "../rewrite.h"
#include "../transform/ir_gen_helpers.h"

#include <assert.h>
#include <string.h>

#pragma GCC diagnostic error "-Wswitch"

static Nodes annotate_all_types(IrArena* a, Nodes types, bool uniform_by_default) {
    LARRAY(const Type*, ntypes, types.count);
    for (size_t i = 0; i < types.count; i++) {
        if (is_data_type(types.nodes[i]))
            ntypes[i] = qualified_type(a, (QualifiedType) {
                .type = types.nodes[i],
                .is_uniform = uniform_by_default,
            });
        else
            ntypes[i] = types.nodes[i];
    }
    return nodes(a, types.count, ntypes);
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
    IrArena* a = ctx->rewriter.dst_arena;
    assert(is_annotation(node));
    switch (node->tag) {
        case Annotation_TAG: return annotation(a, (Annotation) { .name = node->payload.annotation.name });
        case AnnotationValue_TAG: return annotation_value(a, (AnnotationValue) { .name = node->payload.annotation_value.name, .value = infer(ctx, node->payload.annotation_value.value, NULL) });
        case AnnotationValues_TAG: return annotation_values(a, (AnnotationValues) { .name = node->payload.annotation_values.name, .values = infer_nodes(ctx, node->payload.annotation_values.values) });
        case AnnotationCompound_TAG: return annotation_compound(a, (AnnotationCompound) { .name = node->payload.annotation_compound.name, .entries = infer_nodes(ctx, node->payload.annotation_compound.entries) });
        default: error("Not an annotation");
    }
}

static const Node* _infer_type(Context* ctx, const Type* type) {
    IrArena* a = ctx->rewriter.dst_arena;
    switch (type->tag) {
        case ArrType_TAG: {
            const Node* size = infer(ctx, type->payload.arr_type.size, NULL);
            return arr_type(a, (ArrType) {
                .size = size,
                .element_type = infer(ctx, type->payload.arr_type.element_type, NULL)
            });
        }
        case PtrType_TAG: {
            const Node* element_type = infer(ctx, type->payload.ptr_type.pointed_type, NULL);
            if (!element_type)
                element_type = unit_type(a);
            return ptr_type(a, (PtrType) { .pointed_type = element_type, .address_space = type->payload.ptr_type.address_space });
        }
        default: return recreate_node_identity(&ctx->rewriter, type);
    }
}

static const Node* _infer_decl(Context* ctx, const Node* node) {
    assert(is_declaration(node));
    const Node* already_done = search_processed(&ctx->rewriter, node);
    if (already_done)
        return already_done;

    if (lookup_annotation(node, "SkipOnInfer"))
        return NULL;

    IrArena* a = ctx->rewriter.dst_arena;
    switch (is_declaration(node)) {
        case Function_TAG: {
            Context body_context = *ctx;

            LARRAY(const Node*, nparams, node->payload.fun.params.count);
            for (size_t i = 0; i < node->payload.fun.params.count; i++) {
                const Variable* old_param = &node->payload.fun.params.nodes[i]->payload.var;
                const Type* imported_param_type = infer(ctx, old_param->type, NULL);
                nparams[i] = var(a, imported_param_type, old_param->name);
                register_processed(&body_context.rewriter, node->payload.fun.params.nodes[i], nparams[i]);
            }

            Nodes nret_types = annotate_all_types(a, infer_nodes(ctx, node->payload.fun.return_types), false);
            Node* fun = function(ctx->rewriter.dst_module, nodes(a, node->payload.fun.params.count, nparams), string(a, node->payload.fun.name), infer_nodes(ctx, node->payload.fun.annotations), nret_types);
            register_processed(&ctx->rewriter, node, fun);
            fun->payload.fun.body = infer(&body_context, node->payload.fun.body, NULL);
            return fun;
        }
        case Constant_TAG: {
            const Constant* oconstant = &node->payload.constant;
            const Type* imported_hint = infer(ctx, oconstant->type_hint, NULL);
            const Node* instruction;
            if (imported_hint) {
                assert(is_data_type(imported_hint));
                instruction = infer(ctx, oconstant->instruction, qualified_type_helper(imported_hint, true));
            } else {
                instruction = infer(ctx, oconstant->instruction, NULL);
            }
            imported_hint = get_unqualified_type(instruction->type);

            Node* nconstant = constant(ctx->rewriter.dst_module, infer_nodes(ctx, oconstant->annotations), imported_hint, oconstant->name);
            register_processed(&ctx->rewriter, node, nconstant);
            nconstant->payload.constant.instruction = instruction;

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
        case NominalType_TAG: {
            const NominalType* onom_type = &node->payload.nom_type;
            Node* nnominal_type = nominal_type(ctx->rewriter.dst_module, infer_nodes(ctx, onom_type->annotations), onom_type->name);
            register_processed(&ctx->rewriter, node, nnominal_type);
            nnominal_type->payload.nom_type.body = infer(ctx, onom_type->body, NULL);
            return nnominal_type;
        }
        case NotADeclaration: error("not a decl");
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

    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;
    switch (is_value(node)) {
        default: error("");
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
            return int_literal(a, (IntLiteral) {
                .width = node->payload.int_literal.width,
                .is_signed = node->payload.int_literal.is_signed,
                .value = node->payload.int_literal.value});
        }
        case UntypedNumber_TAG: {
            char* endptr;
            int64_t i = strtoll(node->payload.untyped_number.plaintext, &endptr, 10);
            if (!expected_type) {
                bool valid_int = *endptr == '\0';
                expected_type = valid_int ? int32_type(a) : fp32_type(a);
            }
            expected_type = remove_uniformity_qualifier(expected_type);
            if (expected_type->tag == Int_TAG) {
                // TODO chop off extra bits based on width ?
                return int_literal(a, (IntLiteral) {
                    .width = expected_type->payload.int_type.width,
                    .is_signed = expected_type->payload.int_literal.is_signed,
                    .value = i
                });
            } else if (expected_type->tag == Float_TAG) {
                uint64_t v;
                switch (expected_type->payload.float_type.width) {
                    case FloatTy16:
                        error("TODO: implement fp16 parsing");
                    case FloatTy32:
                        assert(sizeof(float) == sizeof(uint32_t));
                        float f = strtof(node->payload.untyped_number.plaintext, NULL);
                        memcpy(&v, &f, sizeof(uint32_t));
                        break;
                    case FloatTy64:
                        assert(sizeof(double) == sizeof(uint64_t));
                        double d = strtod(node->payload.untyped_number.plaintext, NULL);
                        memcpy(&v, &d, sizeof(uint64_t));
                        break;
                }
                return float_literal(a, (FloatLiteral) {.value = v, .width = expected_type->payload.float_type.width});
            }
        }
        case FloatLiteral_TAG: {
            if (expected_type) {
                expected_type = remove_uniformity_qualifier(expected_type);
                assert(expected_type->tag == Float_TAG);
                assert(expected_type->payload.float_type.width == node->payload.float_literal.width);
            }
            return float_literal(a, (FloatLiteral) { .width = node->payload.float_literal.width, .value = node->payload.float_literal.value });
        }
        case True_TAG: return true_lit(a);
        case False_TAG: return false_lit(a);
        case StringLiteral_TAG: return string_lit(a, (StringLiteral) { .string = string(a, node->payload.string_lit.string )});
        case RefDecl_TAG: {
            if (get_arena_config(ctx->rewriter.src_arena).untyped_ptrs) {
                const Node* ref_decl = recreate_node_identity(&ctx->rewriter, node);
                assert(ref_decl->tag == RefDecl_TAG);
                const Node* decl = ref_decl->payload.ref_decl.decl;
                if (decl->tag == GlobalVariable_TAG) {
                    AddressSpace as = decl->payload.global_variable.address_space;
                    if (is_physical_as(as)) {
                        const Node* untyped_ptr = ptr_type(a, (PtrType) {.address_space = as, .pointed_type = unit_type(a)});
                        Node* cast_constant = constant(ctx->rewriter.dst_module, empty(a), untyped_ptr, format_string_interned(a, "%s_cast", get_declaration_name(decl)));
                        cast_constant->payload.constant.instruction = prim_op_helper(a, reinterpret_op, singleton(untyped_ptr), singleton(ref_decl));
                        const Node* cast_ref_decl = ref_decl_helper(a, cast_constant);
                        register_processed(r, node, cast_ref_decl);
                        return cast_ref_decl;
                    }
                }
            }
            break;
        }
        case FnAddr_TAG: break;
        case Value_Undef_TAG: break;
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
                    inferred[i] = infer(ctx, omembers.nodes[i], qualified_type(a, (QualifiedType) { .is_uniform = uniform, .type = expected_members.nodes[i] }));
            } else {
                for (size_t i = 0; i < omembers.count; i++)
                    inferred[i] = infer(ctx, omembers.nodes[i], NULL);
            }
            Nodes nmembers = nodes(a, omembers.count, inferred);

            // Composites are tuples by default
            if (!elem_type)
                elem_type = record_type(a, (RecordType) { .members = strip_qualifiers(a, get_values_types(a, nmembers)) });

            return composite_helper(a, elem_type, nmembers);
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
            const Node* value = infer(ctx, node->payload.fill.value, qualified_type(a, (QualifiedType) { .is_uniform = uniform, .type = element_t }));
            return fill(a, (Fill) { .type = composite_t, .value = value });
        }
        case Value_NullPtr_TAG: break;
    }
    return recreate_node_identity(&ctx->rewriter, node);
}

static const Node* _infer_case(Context* ctx, const Node* node, const Node* expected) {
    IrArena* a = ctx->rewriter.dst_arena;
    assert(is_case(node));
    assert(expected);
    Nodes inferred_arg_type = unwrap_multiple_yield_types(a, expected);
    assert(inferred_arg_type.count == node->payload.case_.params.count || node->payload.case_.params.count == 0);

    Context body_context = *ctx;
    LARRAY(const Node*, nparams, inferred_arg_type.count);
    for (size_t i = 0; i < inferred_arg_type.count; i++) {
        if (node->payload.case_.params.count == 0) {
            // syntax sugar: make up a parameter if there was none
            nparams[i] = var(a, inferred_arg_type.nodes[i], unique_name(a, "_"));
        } else {
            const Variable* old_param = &node->payload.case_.params.nodes[i]->payload.var;
            // for the param type: use the inferred one if none is already provided
            // if one is provided, check the inferred argument type is a subtype of the param type
            const Type* param_type = infer(ctx, old_param->type, NULL);
            // and do not use the provided param type if it is an untyped ptr
            if (!param_type || param_type->tag != PtrType_TAG || param_type->payload.ptr_type.pointed_type)
                param_type = inferred_arg_type.nodes[i];
            assert(is_subtype(param_type, inferred_arg_type.nodes[i]));
            nparams[i] = var(a, param_type, old_param->name);
            register_processed(&body_context.rewriter, node->payload.case_.params.nodes[i], nparams[i]);
        }
    }

    const Node* new_body = infer(&body_context, node->payload.case_.body, NULL);
    return case_(a, nodes(a, inferred_arg_type.count, nparams), new_body);
}

static const Node* _infer_basic_block(Context* ctx, const Node* node) {
    assert(is_basic_block(node));
    IrArena* a = ctx->rewriter.dst_arena;

    Context body_context = *ctx;
    LARRAY(const Node*, nparams, node->payload.basic_block.params.count);
    for (size_t i = 0; i < node->payload.basic_block.params.count; i++) {
        const Variable* old_param = &node->payload.basic_block.params.nodes[i]->payload.var;
        // for the param type: use the inferred one if none is already provided
        // if one is provided, check the inferred argument type is a subtype of the param type
        const Type* param_type = infer(ctx, old_param->type, NULL);
        assert(param_type);
        nparams[i] = var(a, param_type, old_param->name);
        register_processed(&body_context.rewriter, node->payload.basic_block.params.nodes[i], nparams[i]);
    }

    Node* fn = (Node*) infer(ctx, node->payload.basic_block.fn, NULL);
    Node* bb = basic_block(a, fn, nodes(a, node->payload.basic_block.params.count, nparams), node->payload.basic_block.name);
    assert(bb);
    register_processed(&ctx->rewriter, node, bb);

    bb->payload.basic_block.body = infer(&body_context, node->payload.basic_block.body, NULL);
    return bb;
}

static const Type* type_untyped_ptr(const Type* untyped_ptr_t, const Type* element_type) {
    assert(element_type);
    IrArena* a = untyped_ptr_t->arena;
    assert(untyped_ptr_t->tag == PtrType_TAG);
    const Type* typed_ptr_t = ptr_type(a, (PtrType) { .pointed_type = element_type, .address_space = untyped_ptr_t->payload.ptr_type.address_space });
    return typed_ptr_t;
}

static const Node* reinterpret_cast_helper(BodyBuilder* bb, const Node* ptr, const Type* typed_ptr_t) {
    IrArena* a = ptr->arena;
    ptr = gen_reinterpret_cast(bb, typed_ptr_t, ptr);
    return ptr;
}

// Turns untyped pointers back into typed pointers
// For physical pointers, we can just reinterpret them
// For logical pointers, we need to do some stupid best-effort tricks and pray it works out
// the sort of casts we can allow are casting a pointer to a composite to a pointer to one of it's first elements
// we just attempt to do LEAs until we hit the right type, or some unrecoverable error
static void fix_source_pointer(BodyBuilder* bb, const Node** operand, const Type* element_type) {
    IrArena* a = element_type->arena;
    const Type* original_operand_t = get_unqualified_type((*operand)->type);
    assert(original_operand_t->tag == PtrType_TAG);
    if (is_physical_ptr_type(original_operand_t)) {
        // typed loads - normalise to typed ptrs instead by generating an extra cast!
        const Type *ptr_type = original_operand_t;
        ptr_type = type_untyped_ptr(ptr_type, element_type);
        *operand = reinterpret_cast_helper(bb, *operand, ptr_type);
    } else {
        // we can't insert a cast but maybe we can make this work
        do {
            const Node* ptr_t = get_unqualified_type((*operand)->type);
            const Type* pointee = get_pointer_type_element(ptr_t);
            if (pointee == element_type)
                return;
            pointee = get_maybe_nominal_type_body(pointee);
            if (pointee->tag == RecordType_TAG) {
                *operand = gen_lea(bb, *operand, int32_literal(a, 0), singleton(int32_literal(a, 0)));
                continue;
            }
            if (pointee->tag == ArrType_TAG) {
                *operand = gen_lea(bb, *operand, int32_literal(a, 0), singleton(int32_literal(a, 0)));
                continue;
            }
            // TODO: better diagnostics
            error_print("Fatal: Trying to type-pun a pointer in logical memory (%s)\n", get_address_space_name(get_pointer_type_address_space(ptr_t)));
            error_die();
        } while(true);
    }
}

static const Node* _infer_primop(Context* ctx, const Node* node, const Type* expected_type) {
    assert(node->tag == PrimOp_TAG);
    IrArena* a = ctx->rewriter.dst_arena;

    for (size_t i = 0; i < node->payload.prim_op.type_arguments.count; i++)
        assert(node->payload.prim_op.type_arguments.nodes[i] && is_type(node->payload.prim_op.type_arguments.nodes[i]));
    for (size_t i = 0; i < node->payload.prim_op.operands.count; i++)
        assert(node->payload.prim_op.operands.nodes[i] && is_value(node->payload.prim_op.operands.nodes[i]));

    Nodes old_type_args = node->payload.prim_op.type_arguments;
    Nodes type_args = infer_nodes(ctx, old_type_args);
    Nodes old_operands = node->payload.prim_op.operands;

    BodyBuilder* bb = begin_body(a);
    Op op = node->payload.prim_op.op;
    LARRAY(const Node*, new_operands, old_operands.count);
    Nodes input_types = empty(a);
    switch (node->payload.prim_op.op) {
        case push_stack_op: {
            assert(old_operands.count == 1);
            assert(type_args.count == 1);
            const Type* element_type = type_args.nodes[0];
            assert(is_data_type(element_type));
            new_operands[0] = infer(ctx, old_operands.nodes[0], qualified_type_helper(element_type, false));
            goto rebuild;
        }
        case pop_stack_op: {
            assert(old_operands.count == 0);
            assert(type_args.count == 1);
            const Type* element_type = type_args.nodes[0];
            assert(is_data_type(element_type));
            //new_inputs_scratch[0] = element_type;
            goto rebuild;
        }
        case load_op: {
            assert(old_operands.count == 1);
            assert(type_args.count <= 1);
            new_operands[0] = infer(ctx, old_operands.nodes[0], NULL);
            if (type_args.count == 1) {
                fix_source_pointer(bb, &new_operands[0], first(type_args));
                type_args = empty(a);
            }
            goto rebuild;
        }
        case store_op: {
            assert(old_operands.count == 2);
            assert(type_args.count <= 1);
            new_operands[0] = infer(ctx, old_operands.nodes[0], NULL);
            if (type_args.count == 1) {
                fix_source_pointer(bb, &new_operands[0], first(type_args));
                type_args = empty(a);
            }
            const Type* ptr_type = get_unqualified_type(new_operands[0]->type);
            assert(ptr_type->tag == PtrType_TAG);
            const Type* element_t = ptr_type->payload.ptr_type.pointed_type;
            assert(element_t);
            new_operands[1] = infer(ctx, old_operands.nodes[1], qualified_type_helper(element_t, false));
            goto rebuild;
        }
        case alloca_op: {
            assert(type_args.count == 1);
            assert(old_operands.count == 0);
            const Type* element_type = type_args.nodes[0];
            assert(is_type(element_type));
            assert(is_data_type(element_type));
            goto rebuild;
        }
        case reinterpret_op:
        case convert_op: {
            new_operands[0] = infer(ctx, old_operands.nodes[0], NULL);
            const Type* src_pointer_type = get_unqualified_type(new_operands[0]->type);
            const Type* old_dst_pointer_type = first(old_type_args);
            const Type* dst_pointer_type = first(type_args);

            if (is_generic_ptr_type(src_pointer_type) != is_generic_ptr_type(dst_pointer_type))
                op = convert_op;

            goto rebuild;
        }
        case lea_op: {
            assert(old_operands.count >= 2);
            assert(type_args.count <= 1);
            new_operands[0] = infer(ctx, old_operands.nodes[0], NULL);
            new_operands[1] = infer(ctx, old_operands.nodes[1], NULL);
            for (size_t i = 2; i < old_operands.count; i++) {
                new_operands[i] = infer(ctx, old_operands.nodes[i], NULL);
            }

            const Type* src_ptr = remove_uniformity_qualifier(new_operands[0]->type);
            const Type* base_datatype = src_ptr;
            assert(base_datatype->tag == PtrType_TAG);
            AddressSpace as = get_pointer_type_address_space(base_datatype);
            bool was_untyped = false;
            if (type_args.count == 1) {
                was_untyped = true;
                base_datatype = type_untyped_ptr(base_datatype, first(type_args));
                new_operands[0] = reinterpret_cast_helper(bb, new_operands[0], base_datatype);
                type_args = empty(a);
            }

            Nodes new_ops = nodes(a, old_operands.count, new_operands);

            const Node* offset = new_operands[1];
            const IntLiteral* offset_lit = resolve_to_int_literal(offset);
            if ((!offset_lit || offset_lit->value) != 0 && base_datatype->tag != ArrType_TAG) {
                warn_print("LEA used on a pointer to a non-array type!\n");
                const Type* arrayed_src_t = ptr_type(a, (PtrType) {
                        .address_space = as,
                        .pointed_type = arr_type(a, (ArrType) {
                            .element_type = get_pointer_type_element(base_datatype),
                            .size = NULL
                        }),
                });
                const Node* cast_base = gen_reinterpret_cast(bb, arrayed_src_t, first(new_ops));
                Nodes final_lea_ops = mk_nodes(a, cast_base, offset, int32_literal(a, 0));
                final_lea_ops = concat_nodes(a, final_lea_ops, nodes(a, old_operands.count - 2, new_operands + 2));
                new_ops = final_lea_ops;
            }

            const Node* result = first(bind_instruction(bb, prim_op(a, (PrimOp) {
                .op = lea_op,
                .type_arguments = empty(a),
                .operands = new_ops
            })));

            if (was_untyped && is_physical_as(get_pointer_type_address_space(src_ptr))) {
                const Type* result_t = type_untyped_ptr(base_datatype, unit_type(a));
                result = gen_reinterpret_cast(bb, result_t, result);
            }

            return yield_values_and_wrap_in_block(bb, singleton(result));
        }
        case empty_mask_op:
        case subgroup_active_mask_op:
        case subgroup_elect_first_op:
            input_types = nodes(a, 0, NULL);
            break;
        case subgroup_broadcast_first_op:
            new_operands[0] = infer(ctx, old_operands.nodes[0], NULL);
            goto rebuild;
        case subgroup_ballot_op:
            input_types = singleton(qualified_type_helper(bool_type(a), false));
            break;
        case mask_is_thread_active_op: {
            input_types = mk_nodes(a, qualified_type_helper(mask_type(a), false), qualified_type_helper(uint32_type(a), false));
            break;
        }
        case debug_printf_op: {
            String lit = get_string_literal(a, old_operands.nodes[0]);
            assert(lit && "debug_printf requires a string literal");
            new_operands[0] = string_lit_helper(a, lit);
            for (size_t i = 1; i < old_operands.count; i++)
                new_operands[i] = infer(ctx, old_operands.nodes[i], NULL);
            goto rebuild;
        }
        default: {
            for (size_t i = 0; i < old_operands.count; i++) {
                new_operands[i] = old_operands.nodes[i] ? infer(ctx, old_operands.nodes[i], NULL) : NULL;
            }
            goto rebuild;
        }
    }

    assert(input_types.count == old_operands.count);
    for (size_t i = 0; i < input_types.count; i++)
        new_operands[i] = infer(ctx, old_operands.nodes[i], input_types.nodes[i]);

    rebuild: {
        const Node* new_instruction = prim_op(a, (PrimOp) {
            .op = op,
            .type_arguments = type_args,
            .operands = nodes(a, old_operands.count, new_operands)
        });
        return bind_last_instruction_and_wrap_in_block(bb, new_instruction);
    }
}

static const Node* _infer_indirect_call(Context* ctx, const Node* node, const Type* expected_type) {
    assert(node->tag == Call_TAG);
    IrArena* a = ctx->rewriter.dst_arena;

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

    return call(a, (Call) {
        .callee = new_callee,
        .args = nodes(a, node->payload.call.args.count, new_args)
    });
}

static const Node* _infer_if(Context* ctx, const Node* node) {
    assert(node->tag == If_TAG);
    IrArena* a = ctx->rewriter.dst_arena;
    const Node* condition = infer(ctx, node->payload.structured_if.condition, bool_type(a));

    Nodes join_types = infer_nodes(ctx, node->payload.structured_if.yield_types);
    Context infer_if_body_ctx = *ctx;
    // When we infer the types of the arguments to a call to merge(), they are expected to be varying
    Nodes expected_join_types = annotate_all_types(a, join_types, false);
    infer_if_body_ctx.merge_types = &expected_join_types;

    const Node* true_body = infer(&infer_if_body_ctx, node->payload.structured_if.if_true, wrap_multiple_yield_types(a, nodes(a, 0, NULL)));
    // don't allow seeing the variables made available in the true branch
    infer_if_body_ctx.rewriter = ctx->rewriter;
    const Node* false_body = node->payload.structured_if.if_false ? infer(&infer_if_body_ctx, node->payload.structured_if.if_false, wrap_multiple_yield_types(a, nodes(a, 0, NULL))) : NULL;

    return structured_if(a, (If) {
        .yield_types = join_types,
        .condition = condition,
        .if_true = true_body,
        .if_false = false_body,
    });
}

static const Node* _infer_body(Context* ctx, const Node* node) {
    assert(node->tag == Body_TAG);
    IrArena* a = ctx->rewriter.dst_arena;

    Nodes oinstructions = node->payload.body.instructions;
    LARRAY(const Node*, instructions, oinstructions.count);
    for (size_t i = 0; i < oinstructions.count; i++)
        instructions[i] = infer(ctx, oinstructions.nodes[i], /*i + 1 == oinstructions.count ? expected_type : */NULL);
    return body(a, (Body) {
        .instructions = nodes(a, oinstructions.count, instructions),
        .terminator = infer(ctx, node->payload.body.terminator, NULL)
    });
}

static const Node* _infer_loop(Context* ctx, const Node* node) {
    assert(node->tag == Loop_TAG);
    IrArena* a = ctx->rewriter.dst_arena;
    Context loop_body_ctx = *ctx;
    const Node* old_body = node->payload.structured_loop.body;

    Nodes old_params = get_abstraction_params(old_body);
    Nodes old_params_types = get_variables_types(a, old_params);
    Nodes new_params_types = infer_nodes(ctx, old_params_types);

    Nodes old_initial_args = node->payload.structured_loop.initial_args;
    LARRAY(const Node*, new_initial_args, old_params.count);
    for (size_t i = 0; i < old_params.count; i++)
        new_initial_args[i] = infer(ctx, old_initial_args.nodes[i], new_params_types.nodes[i]);

    Nodes loop_yield_types = infer_nodes(ctx, node->payload.structured_loop.yield_types);

    loop_body_ctx.merge_types = NULL;
    loop_body_ctx.break_types = &loop_yield_types;
    loop_body_ctx.continue_types = &new_params_types;

    const Node* nbody = infer(&loop_body_ctx, old_body, wrap_multiple_yield_types(a, new_params_types));
    // TODO check new body params match continue types

    return structured_loop(a, (Loop) {
        .yield_types = loop_yield_types,
        .initial_args = nodes(a, old_params.count, new_initial_args),
        .body = nbody,
    });
}

static const Node* _infer_control(Context* ctx, const Node* node) {
    assert(node->tag == Control_TAG);
    IrArena* a = ctx->rewriter.dst_arena;

    Nodes yield_types = infer_nodes(ctx, node->payload.control.yield_types);

    const Node* olam = node->payload.control.inside;
    const Node* ojp = first(get_abstraction_params(olam));

    Context joinable_ctx = *ctx;
    const Type* jpt = join_point_type(a, (JoinPointType) {
            .yield_types = yield_types
    });
    jpt = qualified_type(a, (QualifiedType) { .is_uniform = true, .type = jpt });
    const Node* jp = var(a, jpt, ojp->payload.var.name);
    register_processed(&ctx->rewriter, ojp, jp);

    const Node* nlam = case_(a, singleton(jp), infer(&joinable_ctx, get_abstraction_body(olam), NULL));

    return control(a, (Control) {
        .yield_types = yield_types,
        .inside = nlam
    });
}

static const Node* _infer_instruction(Context* ctx, const Node* node, const Type* expected_type) {
    switch (is_instruction(node)) {
        case PrimOp_TAG:       return _infer_primop(ctx, node, expected_type);
        case Call_TAG:         return _infer_indirect_call(ctx, node, expected_type);
        case Comment_TAG: return recreate_node_identity(&ctx->rewriter, node);
        default:               error("TODO")
        case NotAnInstruction: error("not an instruction");
    }
    SHADY_UNREACHABLE;
}

static const Node* _infer_terminator(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    switch (is_terminator(node)) {
        case InsertHelperEnd_TAG: assert(false);
        case NotATerminator: assert(false);
        case Body_TAG:    return _infer_body  (ctx, node);
        case If_TAG:      return _infer_if    (ctx, node);
        case Loop_TAG:    return _infer_loop  (ctx, node);
        case Match_TAG:   error("TODO")
        case Control_TAG: return _infer_control(ctx, node);
        /*case Let_TAG: {
            const Node* otail = node->payload.let.tail;
            Nodes annotated_types = get_variables_types(a, otail->payload.case_.params);
            const Node* inferred_instruction = infer(ctx, node->payload.let.instruction, wrap_multiple_yield_types(a, annotated_types));
            Nodes inferred_yield_types = unwrap_multiple_yield_types(a, inferred_instruction->type);
            for (size_t i = 0; i < inferred_yield_types.count; i++) {
                assert(is_value_type(inferred_yield_types.nodes[i]));
            }
            const Node* inferred_tail = infer(ctx, otail, wrap_multiple_yield_types(a, inferred_yield_types));
            return let(a, inferred_instruction, inferred_tail);
        }*/
        case Return_TAG: {
            const Node* imported_fn = infer(ctx, node->payload.fn_ret.fn, NULL);
            Nodes return_types = imported_fn->payload.fun.return_types;

            const Nodes* old_values = &node->payload.fn_ret.args;
            LARRAY(const Node*, nvalues, old_values->count);
            for (size_t i = 0; i < old_values->count; i++)
                nvalues[i] = infer(ctx, old_values->nodes[i], return_types.nodes[i]);
            return fn_ret(a, (Return) {
                .args = nodes(a, old_values->count, nvalues),
                .fn = NULL
            });
        }
        case Jump_TAG: {
            assert(is_basic_block(node->payload.jump.target));
            const Node* ntarget = infer(ctx, node->payload.jump.target, NULL);
            Nodes param_types = get_variables_types(a, get_abstraction_params(ntarget));

            LARRAY(const Node*, tmp, node->payload.jump.args.count);
            for (size_t i = 0; i < node->payload.jump.args.count; i++)
                tmp[i] = infer(ctx, node->payload.jump.args.nodes[i], param_types.nodes[i]);

            Nodes new_args = nodes(a, node->payload.jump.args.count, tmp);

            return jump(a, (Jump) {
                .target = ntarget,
                .args = new_args
            });
        }
        case Branch_TAG:
        case Terminator_Switch_TAG: break;
        case Terminator_TailCall_TAG: break;
        case Terminator_Yield_TAG: {
            const Nodes* expected_types = ctx->merge_types;
            // TODO: block nodes should set merge types
            assert(expected_types && "Merge terminator found but we're not within a suitable if instruction !");
            const Nodes* old_args = &node->payload.yield.args;
            assert(expected_types->count == old_args->count);
            LARRAY(const Node*, new_args, old_args->count);
            for (size_t i = 0; i < old_args->count; i++)
                new_args[i] = infer(ctx, old_args->nodes[i], (*expected_types).nodes[i]);
            return yield(a, (Yield) {
                .args = nodes(a, old_args->count, new_args)
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
            return merge_continue(a, (MergeContinue) {
                .args = nodes(a, old_args->count, new_args)
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
            return merge_break(a, (MergeBreak) {
                .args = nodes(a, old_args->count, new_args)
            });
        }
        case Unreachable_TAG: return unreachable(a);
        case Terminator_Join_TAG: return join(a, (Join) {
            .join_point = infer(ctx, node->payload.join.join_point, NULL),
            .args = infer_nodes(ctx, node->payload.join.args),
        });
    }
    return recreate_node_identity(&ctx->rewriter, node);
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
    } else if (is_instruction(node)) {
        return _infer_instruction(&ctx, node, expect);
    } else if (is_value(node)) {
        const Node* value = _infer_value(&ctx, node, expect);
        assert(is_value_type(value->type));
        return value;
    } else if (is_terminator(node)) {
        assert(expect == NULL);
        return _infer_terminator(&ctx, node);
    } else if (is_declaration(node)) {
        return _infer_decl(&ctx, node);
    } else if (is_annotation(node)) {
        assert(expect == NULL);
        return _infer_annotation(&ctx, node);
    } else if (is_case(node)) {
        assert(expect != NULL);
        return _infer_case(&ctx, node, expect);
    } else if (is_basic_block(node)) {
        return _infer_basic_block(&ctx, node);
    }
    assert(false);
}

Module* infer_program(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    assert(!aconfig.check_types);
    aconfig.check_types = true;
    aconfig.untyped_ptrs = false;
    aconfig.allow_fold = true; // TODO was moved here because a refactor, does this cause issues ?
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));

    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process),
    };
    ctx.rewriter.config.search_map = false;
    ctx.rewriter.config.write_map = false;
    ctx.rewriter.config.rebind_let = true;
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
