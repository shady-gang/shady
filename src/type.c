#include "type.h"

#include "implem.h"

#include "dict.h"
#include "murmur3.h"

#include <string.h>
#include <assert.h>

bool is_type(const Node* node) {
    switch (node->tag) {
#define NODEDEF(_, _2, _3, name, _4) case name##_TAG:
TYPE_NODES()
#undef NODEDEF
                 return true;
        default: return false;
    }
}

bool is_subtype(const Type* supertype, const Type* type) {
    if (supertype->tag != type->tag)
        return false;
    switch (supertype->tag) {
        case QualifiedType_TAG: {
            // uniform T <: varying T
            if (supertype->payload.qualified_type.is_uniform && !type->payload.qualified_type.is_uniform)
                return false;
            return is_subtype(supertype->payload.qualified_type.type, type->payload.qualified_type.type);
        }
        case RecordType_TAG: {
            const Nodes* supermembers = &supertype->payload.record_type.members;
            const Nodes* members = &type->payload.record_type.members;
            for (size_t i = 0; i < members->count; i++) {
                if (!is_subtype(supermembers->nodes[i], members->nodes[i]))
                    return false;
            }
            goto post_switch;
        }
        case FnType_TAG:
            if (supertype->payload.fn.return_types.count != type->payload.fn.return_types.count)
                return false;
            for (size_t i = 0; i < type->payload.fn.return_types.count; i++)
                if (!is_subtype(supertype->payload.fn.return_types.nodes[i], type->payload.fn.return_types.nodes[i]))
                    return false;

            const Nodes* superparams = &supertype->payload.fn_type.param_types;
            const Nodes* params = &type->payload.fn_type.param_types;
            goto check_params;
        case ContType_TAG:
            superparams = &supertype->payload.fn_type.param_types;
            params = &type->payload.fn_type.param_types;
            goto check_params;
        case PtrType_TAG: SHADY_NOT_IMPLEM;
        default: goto post_switch;
        check_params:
            if (params->count != superparams->count)
                return false;

            for (size_t i = 0; i < params->count; i++) {
                if (!is_subtype(params->nodes[i], superparams->nodes[i]))
                    return false;
            }
        return false;
    }
    post_switch:
    return true;
}

void check_subtype(const Type* supertype, const Type* type) {
    if (!is_subtype(supertype, type))
        error("is not a subtype")
}

const Type* strip_qualifier(const Type* type, DivergenceQualifier* qual_out) {
    if (type->tag == QualifiedType_TAG) {
        *qual_out = type->payload.qualified_type.is_uniform ? Uniform : Varying;
        return type->payload.qualified_type.type;
    } else {
        *qual_out = Unknown;
        return type;
    }
}

DivergenceQualifier get_qualifier(const Type* type) {
    DivergenceQualifier result;
    strip_qualifier(type, &result);
    return result;
}

Nodes extract_variable_types(IrArena* arena, const Nodes* variables) {
    LARRAY(const Type*, arr, variables->count);
    for (size_t i = 0; i < variables->count; i++)
        arr[i] = variables->nodes[i]->payload.var.type;
    return nodes(arena, variables->count, arr);
}

const Type* derive_fn_type(IrArena* arena, const Function* fn) {
    return fn_type(arena, (FnType) { .param_types = extract_variable_types(arena, &fn->params), .return_types = fn->return_types });
}

#define empty() nodes(arena, 0, NULL)
#define singleton(t) singleton_impl(arena, t)
Nodes singleton_impl(IrArena* arena, const Type* type) {
    const Type* arr[] = { type };
    return nodes(arena, 1, arr);
}

const Type* ensure_value_t(Nodes yields) {
    assert(yields.count == 1);
    return yields.nodes[0];
}

Nodes check_type_call(IrArena* arena, Call call) {
    const Type* callee_type = ensure_value_t(call.callee->yields);
    if (callee_type->tag != FnType_TAG)
        error("Callees must have a function type");
    if (callee_type->payload.fn_type.param_types.count != call.args.count)
        error("Mismatched argument counts");
    for (size_t i = 0; i < call.args.count; i++) {
        // TODO
    }
    return callee_type->payload.fn.return_types;
}

Nodes check_type_fn(IrArena* arena, Function fn) {
    return singleton(qualified_type(arena, (QualifiedType) {
        .is_uniform = true,
        .type = derive_fn_type(arena, &fn)
    }));
}

Nodes check_type_var_decl(IrArena* arena, VariableDecl decl) {
    SHADY_NOT_IMPLEM
    //return ptr_type(arena, (PtrType) { .address_space = decl.address_space, .pointed_type = decl.variable->type });
}

Nodes check_type_expr_eval(IrArena* arena, ExpressionEval expr) {
    SHADY_NOT_IMPLEM;
}

Nodes check_type_var(IrArena* arena, Variable variable) {
    assert(get_qualifier(variable.type) != Unknown);
    return singleton(variable.type);
}

Nodes check_type_untyped_number(IrArena* arena, UntypedNumber untyped) {
    error("should never happen");
}

Nodes check_type_int_literal(IrArena* arena, IntLiteral lit) {
    return singleton(qualified_type(arena, (QualifiedType) {
        .is_uniform = true,
        .type = int_type(arena)
    }));
}

Nodes check_type_true_lit(IrArena* arena) { return singleton(bool_type(arena)); }
Nodes check_type_false_lit(IrArena* arena) { return singleton(bool_type(arena)); }

Nodes check_type_let(IrArena* arena, Let let) {
    Nodes var_tys = extract_variable_types(arena, &let.variables);
    Nodes yields = let.target->yields;
    if (yields.count != var_tys.count)
        error("let variables count != yield count from operation")
    for (size_t i = 0; i < var_tys.count; i++)
        check_subtype(var_tys.nodes[i], yields.nodes[i]);
    return var_tys;
}

Nodes check_type_fn_ret(IrArena* arena, Return fn_ret) {
    return empty();
}

Nodes op_params(IrArena* arena, Op op) {
    switch (op) {
        case add_op:
        case sub_op: return nodes(arena, 2, (const Type*[]){ int_type(arena), int_type(arena) });
            default: error("unhandled op params");
    }
}

Nodes op_yields(IrArena* arena, Op op) {
    switch (op) {
        case add_op:
        case sub_op: return nodes(arena, 1, (const Type*[]) { int_type(arena) });
        default: error("unhandled op yield");
    }
}

Nodes check_type_primop(IrArena* arena, PrimOp primop) {
    // TODO check params
    switch (primop.op) {
        case sub_op:
        case add_op: {
            bool is_result_uniform = true;
            for (size_t i = 0; i < primop.args.count; i++) {
                const Node* arg = primop.args.nodes[i];
                DivergenceQualifier op_div = get_qualifier(ensure_value_t(arg->yields));
                assert(op_div != Unknown); // we expect all operands to be clearly known !
                is_result_uniform ^= op_div == Uniform;
            }

            return singleton(qualified_type(arena, (QualifiedType) { .is_uniform = is_result_uniform, .type = int_type(arena) }));
        }
        default: SHADY_NOT_IMPLEM;
    }
}

Nodes check_type_root(IrArena* arena, Root program) {
    return empty();
}
