#include "shady/pass.h"

#include "../shady/transform/ir_gen_helpers.h"
#include "../shady/check.h"

#include "log.h"
#include "portability.h"

#include <assert.h>
#include <string.h>

#pragma GCC diagnostic error "-Wswitch"

static Nodes annotate_all_types(IrArena* a, Nodes types, bool uniform_by_default) {
    LARRAY(const Type*, ntypes, types.count);
    for (size_t i = 0; i < types.count; i++) {
        if (shd_is_data_type(types.nodes[i]))
            ntypes[i] = qualified_type(a, (QualifiedType) {
                .type = types.nodes[i],
                .is_uniform = uniform_by_default,
            });
        else
            ntypes[i] = types.nodes[i];
    }
    return shd_nodes(a, types.count, ntypes);
}

typedef struct {
    Rewriter rewriter;

    const Node* current_fn;
    const Type* expected_type;
} Context;

static const Node* infer_value(Context* ctx, const Node* node, const Type* expected_type);
static const Node* infer_instruction(Context* ctx, const Node* node, const Node* expected_type);

static const Node* infer(Context* ctx, const Node* node, const Type* expect) {
    Context ctx2 = *ctx;
    ctx2.expected_type = expect;
    return shd_rewrite_node(&ctx2.rewriter, node);
}

static Nodes infer_nodes(Context* ctx, Nodes nodes) {
    Context ctx2 = *ctx;
    ctx2.expected_type = NULL;
    return shd_rewrite_nodes(&ctx->rewriter, nodes);
}

#define rewrite_node shd_error("don't use this directly, use the 'infer' and 'infer_node' helpers")
#define rewrite_nodes rewrite_node

static const Node* infer_annotation(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    assert(is_annotation(node));
    switch (node->tag) {
        case Annotation_TAG: return annotation(a, (Annotation) { .name = node->payload.annotation.name });
        case AnnotationValue_TAG: return annotation_value(a, (AnnotationValue) { .name = node->payload.annotation_value.name, .value = infer(ctx, node->payload.annotation_value.value, NULL) });
        case AnnotationValues_TAG: return annotation_values(a, (AnnotationValues) { .name = node->payload.annotation_values.name, .values = infer_nodes(ctx, node->payload.annotation_values.values) });
        case AnnotationCompound_TAG: return annotation_compound(a, (AnnotationCompound) { .name = node->payload.annotation_compound.name, .entries = infer_nodes(ctx, node->payload.annotation_compound.entries) });
        default: shd_error("Not an annotation");
    }
}

static const Node* infer_type(Context* ctx, const Type* type) {
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
            assert(element_type);
            //if (!element_type)
            //    element_type = unit_type(a);
            return ptr_type(a, (PtrType) { .pointed_type = element_type, .address_space = type->payload.ptr_type.address_space });
        }
        default: return shd_recreate_node(&ctx->rewriter, type);
    }
}

static const Node* infer_decl(Context* ctx, const Node* node) {
    assert(is_declaration(node));
    if (shd_lookup_annotation(node, "SkipOnInfer"))
        return NULL;

    IrArena* a = ctx->rewriter.dst_arena;
    switch (is_declaration(node)) {
        case Function_TAG: {
            Context body_context = *ctx;

            LARRAY(const Node*, nparams, node->payload.fun.params.count);
            for (size_t i = 0; i < node->payload.fun.params.count; i++) {
                const Param* old_param = &node->payload.fun.params.nodes[i]->payload.param;
                const Type* imported_param_type = infer(ctx, old_param->type, NULL);
                nparams[i] = param(a, imported_param_type, old_param->name);
                shd_register_processed(&body_context.rewriter, node->payload.fun.params.nodes[i], nparams[i]);
            }

            Nodes nret_types = annotate_all_types(a, infer_nodes(ctx, node->payload.fun.return_types), false);
            Node* fun = function(ctx->rewriter.dst_module, shd_nodes(a, node->payload.fun.params.count, nparams), string(a, node->payload.fun.name), infer_nodes(ctx, node->payload.fun.annotations), nret_types);
            shd_register_processed(&ctx->rewriter, node, fun);
            body_context.current_fn = fun;
            shd_set_abstraction_body(fun, infer(&body_context, node->payload.fun.body, NULL));
            return fun;
        }
        case Constant_TAG: {
            const Constant* oconstant = &node->payload.constant;
            const Type* imported_hint = infer(ctx, oconstant->type_hint, NULL);
            const Node* instruction = NULL;
            if (imported_hint) {
                assert(shd_is_data_type(imported_hint));
                const Node* s = shd_as_qualified_type(imported_hint, true);
                if (oconstant->value)
                    instruction = infer(ctx, oconstant->value, s);
            } else if (oconstant->value) {
                instruction = infer(ctx, oconstant->value, NULL);
            }
            if (instruction)
                imported_hint = shd_get_unqualified_type(instruction->type);
            assert(imported_hint);

            Node* nconstant = constant(ctx->rewriter.dst_module, infer_nodes(ctx, oconstant->annotations), imported_hint, oconstant->name);
            shd_register_processed(&ctx->rewriter, node, nconstant);
            nconstant->payload.constant.value = instruction;

            return nconstant;
        }
        case GlobalVariable_TAG: {
             const GlobalVariable* old_gvar = &node->payload.global_variable;
             const Type* imported_ty = infer(ctx, old_gvar->type, NULL);
             Node* ngvar = global_var(ctx->rewriter.dst_module, infer_nodes(ctx, old_gvar->annotations), imported_ty, old_gvar->name, old_gvar->address_space);
            shd_register_processed(&ctx->rewriter, node, ngvar);

             ngvar->payload.global_variable.init = infer(ctx, old_gvar->init, shd_as_qualified_type(imported_ty, true));
             return ngvar;
        }
        case NominalType_TAG: {
            const NominalType* onom_type = &node->payload.nom_type;
            Node* nnominal_type = nominal_type(ctx->rewriter.dst_module, infer_nodes(ctx, onom_type->annotations), onom_type->name);
            shd_register_processed(&ctx->rewriter, node, nnominal_type);
            nnominal_type->payload.nom_type.body = infer(ctx, onom_type->body, NULL);
            return nnominal_type;
        }
        case NotADeclaration: shd_error("not a decl");
    }
}

/// Like get_unqualified_type but won't error out if type wasn't qualified to begin with
static const Type* remove_uniformity_qualifier(const Node* type) {
    if (shd_is_value_type(type))
        return shd_get_unqualified_type(type);
    return type;
}

static const Node* infer_value(Context* ctx, const Node* node, const Type* expected_type) {
    if (!node) return NULL;

    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;
    switch (is_value(node)) {
        case NotAValue: shd_error("");
        case Param_TAG:
        case Value_ConstrainedValue_TAG: {
            const Type* type = infer(ctx, node->payload.constrained.type, NULL);
            bool expect_uniform = false;
            if (expected_type) {
                expect_uniform = shd_deconstruct_qualified_type(&expected_type);
                assert(shd_is_subtype(expected_type, type));
            }
            return infer(ctx, node->payload.constrained.value, shd_as_qualified_type(type, expect_uniform));
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
                expected_type = valid_int ? shd_int32_type(a) : shd_fp32_type(a);
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
                        shd_error("TODO: implement fp16 parsing");
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
        case RefDecl_TAG: break;
        case FnAddr_TAG: break;
        case Value_Undef_TAG: break;
        case Value_Composite_TAG: {
            const Node* elem_type = infer(ctx, node->payload.composite.type, NULL);
            bool uniform = false;
            if (elem_type && expected_type) {
                assert(shd_is_subtype(shd_get_unqualified_type(expected_type), elem_type));
            } else if (expected_type) {
                uniform = shd_deconstruct_qualified_type(&elem_type);
                elem_type = expected_type;
            }

            Nodes omembers = node->payload.composite.contents;
            LARRAY(const Node*, inferred, omembers.count);
            if (elem_type) {
                Nodes expected_members = shd_get_composite_type_element_types(elem_type);
                for (size_t i = 0; i < omembers.count; i++)
                    inferred[i] = infer(ctx, omembers.nodes[i], qualified_type(a, (QualifiedType) { .is_uniform = uniform, .type = expected_members.nodes[i] }));
            } else {
                for (size_t i = 0; i < omembers.count; i++)
                    inferred[i] = infer(ctx, omembers.nodes[i], NULL);
            }
            Nodes nmembers = shd_nodes(a, omembers.count, inferred);

            // Composites are tuples by default
            if (!elem_type)
                elem_type = record_type(a, (RecordType) { .members = shd_strip_qualifiers(a, shd_get_values_types(a, nmembers)) });

            return composite_helper(a, elem_type, nmembers);
        }
        case Value_Fill_TAG: {
            const Node* composite_t = infer(ctx, node->payload.fill.type, NULL);
            assert(composite_t);
            bool uniform = false;
            if (composite_t && expected_type) {
                assert(shd_is_subtype(shd_get_unqualified_type(expected_type), composite_t));
            } else if (expected_type) {
                uniform = shd_deconstruct_qualified_type(&composite_t);
                composite_t = expected_type;
            }
            assert(composite_t);
            const Node* element_t = shd_get_fill_type_element_type(composite_t);
            const Node* value = infer(ctx, node->payload.fill.value, qualified_type(a, (QualifiedType) { .is_uniform = uniform, .type = element_t }));
            return fill(a, (Fill) { .type = composite_t, .value = value });
        }
        default: break;
    }
    return shd_recreate_node(&ctx->rewriter, node);
}

static const Node* infer_case(Context* ctx, const Node* node, Nodes inferred_arg_type) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    assert(inferred_arg_type.count == node->payload.basic_block.params.count || node->payload.basic_block.params.count == 0);

    Context body_context = *ctx;
    LARRAY(const Node*, nparams, inferred_arg_type.count);
    for (size_t i = 0; i < inferred_arg_type.count; i++) {
        if (node->payload.basic_block.params.count == 0) {
            // syntax sugar: make up a parameter if there was none
            nparams[i] = param(a, inferred_arg_type.nodes[i], shd_make_unique_name(a, "_"));
        } else {
            const Param* old_param = &node->payload.basic_block.params.nodes[i]->payload.param;
            // for the param type: use the inferred one if none is already provided
            // if one is provided, check the inferred argument type is a subtype of the param type
            const Type* param_type = old_param->type ? infer_type(ctx, old_param->type) : NULL;
            // and do not use the provided param type if it is an untyped ptr
            if (!param_type || param_type->tag != PtrType_TAG || param_type->payload.ptr_type.pointed_type)
                param_type = inferred_arg_type.nodes[i];
            assert(shd_is_subtype(param_type, inferred_arg_type.nodes[i]));
            nparams[i] = param(a, param_type, old_param->name);
            shd_register_processed(&body_context.rewriter, node->payload.basic_block.params.nodes[i], nparams[i]);
        }
    }

    Node* new_case = basic_block(a, shd_nodes(a, inferred_arg_type.count, nparams), shd_get_abstraction_name_unsafe(node));
    shd_register_processed(r, node, new_case);
    shd_set_abstraction_body(new_case, infer(&body_context, node->payload.basic_block.body, NULL));
    return new_case;
}

static const Node* _infer_basic_block(Context* ctx, const Node* node) {
    assert(is_basic_block(node));
    IrArena* a = ctx->rewriter.dst_arena;

    Context body_context = *ctx;
    LARRAY(const Node*, nparams, node->payload.basic_block.params.count);
    for (size_t i = 0; i < node->payload.basic_block.params.count; i++) {
        const Param* old_param = &node->payload.basic_block.params.nodes[i]->payload.param;
        // for the param type: use the inferred one if none is already provided
        // if one is provided, check the inferred argument type is a subtype of the param type
        const Type* param_type = infer(ctx, old_param->type, NULL);
        assert(param_type);
        nparams[i] = param(a, param_type, old_param->name);
        shd_register_processed(&body_context.rewriter, node->payload.basic_block.params.nodes[i], nparams[i]);
    }

    Node* bb = basic_block(a, shd_nodes(a, node->payload.basic_block.params.count, nparams), node->payload.basic_block.name);
    assert(bb);
    shd_register_processed(&ctx->rewriter, node, bb);

    shd_set_abstraction_body(bb, infer(&body_context, node->payload.basic_block.body, NULL));
    return bb;
}

static const Node* infer_primop(Context* ctx, const Node* node, const Node* expected_type) {
    assert(node->tag == PrimOp_TAG);
    IrArena* a = ctx->rewriter.dst_arena;

    for (size_t i = 0; i < node->payload.prim_op.type_arguments.count; i++)
        assert(node->payload.prim_op.type_arguments.nodes[i] && is_type(node->payload.prim_op.type_arguments.nodes[i]));
    for (size_t i = 0; i < node->payload.prim_op.operands.count; i++)
        assert(node->payload.prim_op.operands.nodes[i] && is_value(node->payload.prim_op.operands.nodes[i]));

    Nodes old_type_args = node->payload.prim_op.type_arguments;
    Nodes type_args = infer_nodes(ctx, old_type_args);
    Nodes old_operands = node->payload.prim_op.operands;

    BodyBuilder* bb = shd_bld_begin_pure(a);
    Op op = node->payload.prim_op.op;
    LARRAY(const Node*, new_operands, old_operands.count);
    Nodes input_types = shd_empty(a);
    switch (node->payload.prim_op.op) {
        case reinterpret_op:
        case convert_op: {
            new_operands[0] = infer(ctx, old_operands.nodes[0], NULL);
            const Type* src_pointer_type = shd_get_unqualified_type(new_operands[0]->type);
            const Type* old_dst_pointer_type = shd_first(old_type_args);
            const Type* dst_pointer_type = shd_first(type_args);

            if (shd_is_generic_ptr_type(src_pointer_type) != shd_is_generic_ptr_type(dst_pointer_type))
                op = convert_op;

            goto rebuild;
        }
        case empty_mask_op:
        case mask_is_thread_active_op: {
            input_types = mk_nodes(a, shd_as_qualified_type(mask_type(a), false),
                                   shd_as_qualified_type(shd_uint32_type(a), false));
            break;
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
            .operands = shd_nodes(a, old_operands.count, new_operands)
        });
        return shd_bld_to_instr_with_last_instr(bb, new_instruction);
    }
}

static const Node* infer_indirect_call(Context* ctx, const Node* node, const Node* expected_type) {
    assert(node->tag == Call_TAG);
    IrArena* a = ctx->rewriter.dst_arena;

    const Node* new_callee = infer(ctx, node->payload.call.callee, NULL);
    assert(is_value(new_callee));
    LARRAY(const Node*, new_args, node->payload.call.args.count);

    const Type* callee_type = shd_get_unqualified_type(new_callee->type);
    if (callee_type->tag != PtrType_TAG)
        shd_error("functions are called through function pointers");
    callee_type = callee_type->payload.ptr_type.pointed_type;

    if (callee_type->tag != FnType_TAG)
        shd_error("Callees must have a function type");
    if (callee_type->payload.fn_type.param_types.count != node->payload.call.args.count)
        shd_error("Mismatched argument counts");
    for (size_t i = 0; i < node->payload.call.args.count; i++) {
        const Node* arg = node->payload.call.args.nodes[i];
        assert(arg);
        new_args[i] = infer(ctx, node->payload.call.args.nodes[i], callee_type->payload.fn_type.param_types.nodes[i]);
        assert(new_args[i]->type);
    }

    return call(a, (Call) {
        .callee = new_callee,
        .args = shd_nodes(a, node->payload.call.args.count, new_args),
        .mem = infer(ctx, node->payload.if_instr.mem, NULL),
    });
}

static const Node* infer_if(Context* ctx, const Node* node) {
    assert(node->tag == If_TAG);
    IrArena* a = ctx->rewriter.dst_arena;
    const Node* condition = infer(ctx, node->payload.if_instr.condition, shd_as_qualified_type(bool_type(a), false));

    Nodes join_types = infer_nodes(ctx, node->payload.if_instr.yield_types);
    Context infer_if_body_ctx = *ctx;
    // When we infer the types of the arguments to a call to merge(), they are expected to be varying
    Nodes expected_join_types = shd_add_qualifiers(a, join_types, false);

    const Node* true_body = infer_case(&infer_if_body_ctx, node->payload.if_instr.if_true, shd_nodes(a, 0, NULL));
    // don't allow seeing the variables made available in the true branch
    infer_if_body_ctx.rewriter = ctx->rewriter;
    const Node* false_body = node->payload.if_instr.if_false ? infer_case(&infer_if_body_ctx, node->payload.if_instr.if_false, shd_nodes(a, 0, NULL)) : NULL;

    return if_instr(a, (If) {
        .yield_types = join_types,
        .condition = condition,
        .if_true = true_body,
        .if_false = false_body,
        //.tail = infer_case(ctx, node->payload.if_instr.tail, expected_join_types)
        .tail = infer(ctx, node->payload.if_instr.tail, NULL),
        .mem = infer(ctx, node->payload.if_instr.mem, NULL),
    });
}

static const Node* infer_loop(Context* ctx, const Node* node) {
    assert(node->tag == Loop_TAG);
    IrArena* a = ctx->rewriter.dst_arena;
    Context loop_body_ctx = *ctx;
    const Node* old_body = node->payload.loop_instr.body;

    Nodes old_params = get_abstraction_params(old_body);
    Nodes old_params_types = shd_get_param_types(a, old_params);
    Nodes new_params_types = infer_nodes(ctx, old_params_types);
    new_params_types = annotate_all_types(a, new_params_types, false);

    Nodes old_initial_args = node->payload.loop_instr.initial_args;
    LARRAY(const Node*, new_initial_args, old_params.count);
    for (size_t i = 0; i < old_params.count; i++)
        new_initial_args[i] = infer(ctx, old_initial_args.nodes[i], new_params_types.nodes[i]);

    Nodes loop_yield_types = infer_nodes(ctx, node->payload.loop_instr.yield_types);
    Nodes qual_yield_types = shd_add_qualifiers(a, loop_yield_types, false);

    const Node* nbody = infer_case(&loop_body_ctx, old_body, new_params_types);
    // TODO check new body params match continue types

    return loop_instr(a, (Loop) {
        .yield_types = loop_yield_types,
        .initial_args = shd_nodes(a, old_params.count, new_initial_args),
        .body = nbody,
        //.tail = infer_case(ctx, node->payload.loop_instr.tail, qual_yield_types)
        .tail = infer(ctx, node->payload.loop_instr.tail, NULL),
        .mem = infer(ctx, node->payload.if_instr.mem, NULL),
    });
}

static const Node* infer_control(Context* ctx, const Node* node) {
    assert(node->tag == Control_TAG);
    IrArena* a = ctx->rewriter.dst_arena;

    Nodes yield_types = infer_nodes(ctx, node->payload.control.yield_types);

    const Node* olam = node->payload.control.inside;
    const Node* ojp = shd_first(get_abstraction_params(olam));

    Context joinable_ctx = *ctx;
    const Type* jpt = join_point_type(a, (JoinPointType) {
        .yield_types = yield_types
    });
    jpt = qualified_type(a, (QualifiedType) { .is_uniform = true, .type = jpt });
    const Node* jp = param(a, jpt, ojp->payload.param.name);
    shd_register_processed(&joinable_ctx.rewriter, ojp, jp);

    Node* new_case = basic_block(a, shd_singleton(jp), NULL);
    shd_register_processed(&joinable_ctx.rewriter, olam, new_case);
    shd_set_abstraction_body(new_case, infer(&joinable_ctx, get_abstraction_body(olam), NULL));

    return control(a, (Control) {
        .yield_types = yield_types,
        .inside = new_case,
        .tail = infer(ctx, get_structured_construct_tail(node), NULL /*add_qualifiers(a, yield_types, false)*/),
        .mem = infer(ctx, node->payload.if_instr.mem, NULL),
    });
}

static const Node* infer_instruction(Context* ctx, const Node* node, const Type* expected_type) {
    IrArena* a = ctx->rewriter.dst_arena;
    switch (is_instruction(node)) {
        case PrimOp_TAG:       return infer_primop(ctx, node, expected_type);
        case Call_TAG:         return infer_indirect_call(ctx, node, expected_type);
        case Instruction_Comment_TAG: return shd_recreate_node(&ctx->rewriter, node);
        case Instruction_Load_TAG: {
            return load(a, (Load) { .ptr = infer(ctx, node->payload.load.ptr, NULL), .mem = infer(ctx, node->payload.load.mem, NULL) });
        }
        case Instruction_Store_TAG: {
            Store payload = node->payload.store;
            const Node* ptr = infer(ctx, payload.ptr, NULL);
            const Type* ptr_type = shd_get_unqualified_type(ptr->type);
            assert(ptr_type->tag == PtrType_TAG);
            const Type* element_t = ptr_type->payload.ptr_type.pointed_type;
            assert(element_t);
            const Node* value = infer(ctx, payload.value, shd_as_qualified_type(element_t, false));
            return store(a, (Store) { .ptr = ptr, .value = value, .mem = infer(ctx, node->payload.store.mem, NULL) });
        }
        case Instruction_StackAlloc_TAG: {
            const Type* element_type = node->payload.stack_alloc.type;
            assert(is_type(element_type));
            assert(shd_is_data_type(element_type));
            return stack_alloc(a, (StackAlloc) { .type = infer_type(ctx, element_type), .mem = infer(ctx, node->payload.stack_alloc.mem, NULL) });
        }
        default: break;
        case NotAnInstruction: shd_error("not an instruction");
    }
    return shd_recreate_node(&ctx->rewriter, node);
}

static const Node* infer_terminator(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    switch (is_terminator(node)) {
        case NotATerminator: assert(false);
        case If_TAG:           return infer_if    (ctx, node);
        case Match_TAG:        shd_error("TODO")
        case Loop_TAG:         return infer_loop  (ctx, node);
        case Control_TAG:      return infer_control(ctx, node);
        case Return_TAG: {
            const Node* imported_fn = ctx->current_fn;
            Nodes return_types = imported_fn->payload.fun.return_types;

            Return payload = node->payload.fn_ret;
            LARRAY(const Node*, nvalues, payload.args.count);
            for (size_t i = 0; i < payload.args.count; i++)
                nvalues[i] = infer(ctx, payload.args.nodes[i], return_types.nodes[i]);
            return fn_ret(a, (Return) {
                .args = shd_nodes(a, payload.args.count, nvalues),
                .mem = infer(ctx, payload.mem, NULL),
            });
        }
        default: break;
    }
    return shd_recreate_node(&ctx->rewriter, node);
}

static const Node* process(Context* src_ctx, const Node* node) {
    const Node* expected_type = src_ctx->expected_type;
    Context ctx = *src_ctx;
    ctx.expected_type = NULL;

    IrArena* a = ctx.rewriter.dst_arena;

    if (is_type(node)) {
        assert(expected_type == NULL);
        return infer_type(&ctx, node);
    } else if (is_instruction(node)) {
        if (expected_type) {
            return infer_instruction(&ctx, node, expected_type);
        }
        return infer_instruction(&ctx, node, NULL);
    } else if (is_value(node)) {
        const Node* value = infer_value(&ctx, node, expected_type);
        assert(shd_is_value_type(value->type));
        return value;
    } else if (is_terminator(node)) {
        assert(expected_type == NULL);
        return infer_terminator(&ctx, node);
    } else if (is_declaration(node)) {
        return infer_decl(&ctx, node);
    } else if (is_annotation(node)) {
        assert(expected_type == NULL);
        return infer_annotation(&ctx, node);
    } else if (is_basic_block(node)) {
        return _infer_basic_block(&ctx, node);
    }else if (is_mem(node)) {
        return shd_recreate_node(&ctx.rewriter, node);
    }
    assert(false);
}

Module* slim_pass_infer(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    assert(!aconfig.check_types);
    aconfig.check_types = true;
    aconfig.allow_fold = true; // TODO was moved here because a refactor, does this cause issues ?
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
    };
    //ctx.rewriter.config.search_map = false;
    //ctx.rewriter.config.write_map = false;
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
