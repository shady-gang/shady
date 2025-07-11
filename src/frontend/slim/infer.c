#include "shady/pass.h"

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
                .scope = uniform_by_default ? ShdScopeSubgroup : ShdScopeInvocation,
            });
        else
            ntypes[i] = types.nodes[i];
    }
    return shd_nodes(a, types.count, ntypes);
}

typedef struct {
    Rewriter rewriter;
    const TargetConfig* target;

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

static Nodes infer_params(Context* ctx, Nodes params, bool entry_pt) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    LARRAY(const Node*, nparams, params.count);
    for (size_t i = 0; i < params.count; i++) {
        const Param* old_param = &params.nodes[i]->payload.param;
        const Type* imported_param_type = infer(ctx, old_param->type, NULL);
        if (imported_param_type->tag != QualifiedType_TAG) {
            if (entry_pt)
                imported_param_type = qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.constants, imported_param_type);
            else
                imported_param_type = qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.bottom, imported_param_type);
        }
        nparams[i] = param_helper(a, imported_param_type);
        shd_register_processed(r, params.nodes[i], nparams[i]);
        shd_rewrite_annotations(r, params.nodes[i], nparams[i]);
    }
    return shd_nodes(a, params.count, nparams);
}

static const Node* infer_decl(Context* ctx, const Node* node) {
    if (shd_lookup_annotation(node, "SkipOnInfer"))
        return NULL;

    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    switch (node->tag) {
        case Function_TAG: {
            Context body_context = *ctx;

            Nodes nparams = infer_params(&body_context, get_abstraction_params(node), shd_lookup_annotation(node, "EntryPoint"));

            Nodes nret_types = annotate_all_types(a, infer_nodes(ctx, node->payload.fun.return_types), false);
            Node* fun = function_helper(ctx->rewriter.dst_module, nparams, nret_types);
            shd_register_processed(r, node, fun);
            body_context.current_fn = fun;
            shd_set_abstraction_body(fun, infer(&body_context, node->payload.fun.body, NULL));
            shd_rewrite_annotations(r, node, fun);
            return fun;
        }
        case Constant_TAG: {
            const Constant* oconstant = &node->payload.constant;
            const Type* imported_hint = infer(ctx, oconstant->type_hint, NULL);
            const Node* instruction = NULL;
            if (imported_hint) {
                assert(shd_is_data_type(imported_hint));
                const Node* s = qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.constants, imported_hint);
                if (oconstant->value)
                    instruction = infer(ctx, oconstant->value, s);
            } else if (oconstant->value) {
                instruction = infer(ctx, oconstant->value, NULL);
            }
            if (instruction)
                imported_hint = shd_get_unqualified_type(instruction->type);
            assert(imported_hint);

            Node* nconstant = constant_helper(ctx->rewriter.dst_module, imported_hint);
            shd_register_processed(r, node, nconstant);
            nconstant->payload.constant.value = instruction;
            shd_rewrite_annotations(r, node, nconstant);

            return nconstant;
        }
        case GlobalVariable_TAG: {
             GlobalVariable old_payload = node->payload.global_variable;
             Node* ngvar = shd_recreate_node_head(r, node);
             shd_register_processed(r, node, ngvar);

             ngvar->payload.global_variable.init = infer(ctx, old_payload.init, qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.constants, ngvar->payload.global_variable.type));
             return ngvar;
        }
        default: shd_error("not a decl");
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
            }
            if (expected_type->tag == Float_TAG) {
                uint64_t v;
                switch (expected_type->payload.float_type.width) {
                    case ShdFloatFormat16:
                        shd_error("TODO: implement fp16 parsing");
                    case ShdFloatFormat32:
                        assert(sizeof(float) == sizeof(uint32_t));
                        float f = strtof(node->payload.untyped_number.plaintext, NULL);
                        memcpy(&v, &f, sizeof(uint32_t));
                        break;
                    case ShdFloatFormat64:
                        assert(sizeof(double) == sizeof(uint64_t));
                        double d = strtod(node->payload.untyped_number.plaintext, NULL);
                        memcpy(&v, &d, sizeof(uint64_t));
                        break;
                }
                return float_literal(a, (FloatLiteral) {.value = v, .width = expected_type->payload.float_type.width});
            }
            shd_error("Expected type of untyped number is not integer of float");
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
        case StringLiteral_TAG: return string_lit(a, (StringLiteral) { .string = shd_string(a, node->payload.string_lit.string )});
        case FnAddr_TAG: break;
        case Value_Undef_TAG: break;
        case Value_Composite_TAG: {
            const Node* elem_type = infer(ctx, node->payload.composite.type, NULL);
            ShdScope scope = ctx->target->scopes.constants;
            if (elem_type && expected_type) {
                assert(shd_is_subtype(shd_get_unqualified_type(expected_type), elem_type));
            } else if (expected_type) {
                scope = shd_combine_scopes(scope, shd_deconstruct_qualified_type(&elem_type));
                elem_type = expected_type;
            }

            Nodes omembers = node->payload.composite.contents;
            LARRAY(const Node*, inferred, omembers.count);
            if (elem_type) {
                Nodes expected_members = shd_get_composite_type_element_types(elem_type);
                for (size_t i = 0; i < omembers.count; i++)
                    inferred[i] = infer(ctx, omembers.nodes[i], qualified_type(a, (QualifiedType) { .scope = scope, .type = expected_members.nodes[i] }));
            } else {
                for (size_t i = 0; i < omembers.count; i++)
                    inferred[i] = infer(ctx, omembers.nodes[i], NULL);
            }
            Nodes nmembers = shd_nodes(a, omembers.count, inferred);

            // Composites are tuples by default
            if (!elem_type)
                elem_type = tuple_type(a, (TupleType) { .members = shd_strip_qualifiers(a, shd_get_values_types(a, nmembers)) });

            return composite_helper(a, elem_type, nmembers);
        }
        case Value_Fill_TAG: {
            const Node* composite_t = infer(ctx, node->payload.fill.type, NULL);
            assert(composite_t);
            ShdScope scope = ctx->target->scopes.constants;
            if (composite_t && expected_type) {
                assert(shd_is_subtype(shd_get_unqualified_type(expected_type), composite_t));
            } else if (expected_type) {
                scope = shd_combine_scopes(scope, shd_deconstruct_qualified_type(&composite_t));
                composite_t = expected_type;
            }
            assert(composite_t);
            const Node* element_t = shd_get_fill_type_element_type(composite_t);
            const Node* value = infer(ctx, node->payload.fill.value, qualified_type(a, (QualifiedType) { .scope = scope, .type = element_t }));
            return fill(a, (Fill) { .type = composite_t, .value = value });
        }
        case Value_BitCast_TAG: {
            BitCast payload = node->payload.bit_cast;
            const Type* dst_t = infer(ctx, payload.type, NULL);
            const Type* src = infer(ctx, payload.src, dst_t);
            return bit_cast_helper(a, dst_t, src);
        }
        /*case Value_Conversion_TAG: {
            Conversion payload = node->payload.conversion;
            const Node* src = infer(ctx, payload.src, NULL);
            const Type* src_pointer_type = shd_get_unqualified_type(new_operands[0]->type);
            const Type* old_dst_pointer_type = shd_first(old_type_args);
            const Type* dst_pointer_type = shd_first(type_args);

            // TODO: this is vestigial (reinterpret_op), was it still needed ?
            if (shd_is_generic_ptr_type(src_pointer_type) != shd_is_generic_ptr_type(dst_pointer_type))
                op = convert_op;

            goto rebuild;
        }*/
        default: break;
    }
    return shd_recreate_node(&ctx->rewriter, node);
}

static const Node* infer_basic_block(Context* ctx, const Node* node) {
    assert(is_basic_block(node));
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    Context body_context = *ctx;
    Nodes nparams = infer_params(ctx, get_abstraction_params(node), false);

    Node* bb = basic_block_helper(a, nparams);
    assert(bb);
    shd_register_processed(&ctx->rewriter, node, bb);
    shd_rewrite_annotations(r, node, bb);

    shd_set_abstraction_body(bb, infer(&body_context, node->payload.basic_block.body, NULL));
    return bb;
}

static const Node* infer_primop(Context* ctx, const Node* node, const Node* expected_type) {
    assert(node->tag == PrimOp_TAG);
    IrArena* a = ctx->rewriter.dst_arena;

    for (size_t i = 0; i < node->payload.prim_op.operands.count; i++)
        assert(node->payload.prim_op.operands.nodes[i] && is_value(node->payload.prim_op.operands.nodes[i]));

    Nodes old_operands = node->payload.prim_op.operands;

    BodyBuilder* bb = shd_bld_begin_pure(a);
    Op op = node->payload.prim_op.op;
    LARRAY(const Node*, new_operands, old_operands.count);
    Nodes input_types = shd_empty(a);
    switch (node->payload.prim_op.op) {
        case empty_mask_op:
        case mask_is_thread_active_op: {
            input_types = mk_nodes(a, qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.bottom, shd_get_exec_mask_type(a)), qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.bottom, shd_uint32_type(a)));
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
            .operands = shd_nodes(a, old_operands.count, new_operands)
        });
        return shd_bld_to_instr_with_last_instr(bb, new_instruction);
    }
}

static const Node* infer_indirect_call(Context* ctx, const Node* node, const Node* expected_type) {
    assert(node->tag == IndirectCall_TAG);
    IrArena* a = ctx->rewriter.dst_arena;

    const Node* new_callee = infer(ctx, node->payload.indirect_call.callee, NULL);
    assert(is_value(new_callee));
    LARRAY(const Node*, new_args, node->payload.indirect_call.args.count);

    const Type* callee_type = shd_get_unqualified_type(new_callee->type);
    if (callee_type->tag != PtrType_TAG)
        shd_error("functions are called through function pointers");
    callee_type = callee_type->payload.ptr_type.pointed_type;

    if (callee_type->tag != FnType_TAG)
        shd_error("Callees must have a function type");
    if (callee_type->payload.fn_type.param_types.count != node->payload.indirect_call.args.count)
        shd_error("Mismatched argument counts");
    for (size_t i = 0; i < node->payload.indirect_call.args.count; i++) {
        const Node* arg = node->payload.indirect_call.args.nodes[i];
        assert(arg);
        new_args[i] = infer(ctx, node->payload.indirect_call.args.nodes[i], callee_type->payload.fn_type.param_types.nodes[i]);
        assert(new_args[i]->type);
    }

    return indirect_call(a, (IndirectCall) {
        .callee = new_callee,
        .args = shd_nodes(a, node->payload.indirect_call.args.count, new_args),
        .mem = infer(ctx, node->payload.if_instr.mem, NULL),
    });
}

static const Node* infer_if(Context* ctx, const Node* node) {
    assert(node->tag == If_TAG);
    IrArena* a = ctx->rewriter.dst_arena;
    const Node* condition = infer(ctx, node->payload.if_instr.condition, qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.bottom, bool_type(a)));

    Nodes join_types = infer_nodes(ctx, node->payload.if_instr.yield_types);
    Context infer_if_body_ctx = *ctx;
    // When we infer the types of the arguments to a call to merge(), they are expected to be varying
    Nodes expected_join_types = shd_add_qualifiers(a, join_types, shd_get_arena_config(a)->target.scopes.bottom);

    const Node* true_body = infer_basic_block(&infer_if_body_ctx, node->payload.if_instr.if_true);
    // don't allow seeing the variables made available in the true branch
    infer_if_body_ctx.rewriter = ctx->rewriter;
    const Node* false_body = node->payload.if_instr.if_false ? infer_basic_block(&infer_if_body_ctx, node->payload.if_instr.if_false) : NULL;

    return if_instr(a, (If) {
        .yield_types = join_types,
        .condition = condition,
        .if_true = true_body,
        .if_false = false_body,
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

    const Node* nbody = infer_basic_block(&loop_body_ctx, old_body);
    // TODO check new body params match continue types

    return loop_instr(a, (Loop) {
        .yield_types = loop_yield_types,
        .initial_args = shd_nodes(a, old_params.count, new_initial_args),
        .body = nbody,
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
    jpt = qualified_type(a, (QualifiedType) { .scope = ctx->target->scopes.gang, .type = jpt });
    const Node* jp = param_helper(a, jpt);
    shd_register_processed(&joinable_ctx.rewriter, ojp, jp);

    Node* new_case = basic_block_helper(a, shd_singleton(jp));
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
        case PrimOp_TAG: return infer_primop(ctx, node, expected_type);
        case Instruction_IndirectCall_TAG: return infer_indirect_call(ctx, node, expected_type);
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
            const Node* value = infer(ctx, payload.value, qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.bottom, element_t));
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

    if (is_declaration(node)) {
        return infer_decl(&ctx, node);
    } else if (is_type(node)) {
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
    } else if (is_annotation(node)) {
        assert(expected_type == NULL);
        return infer_annotation(&ctx, node);
    } else if (is_basic_block(node)) {
        return infer_basic_block(&ctx, node);
    } else if (is_mem(node)) {
        return shd_recreate_node(&ctx.rewriter, node);
    }
    assert(false);
}

Module* slim_pass_infer(SHADY_UNUSED const CompilerConfig* config, SHADY_UNUSED const void* unused, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    assert(!aconfig.check_types);
    aconfig.check_types = true;
    aconfig.allow_fold = true; // TODO was moved here because a refactor, does this cause issues ?
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .target = &aconfig.target,
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
