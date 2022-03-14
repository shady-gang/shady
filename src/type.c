#include "type.h"

#include "implem.h"

#include "dict.h"
#include "murmur3.h"

#include <string.h>
#include <assert.h>

bool is_type(const Node* node) {
    switch (node->tag) {
#define NODEDEF(_, _2, name, _3) case name##_TAG:
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

DivergenceQualifier resolve_divergence_impl(const Type* type, bool allow_qualifier_types) {
    switch (type->tag) {
        case QualifiedType_TAG: {
            if (!allow_qualifier_types)
                error("Uniformity qualifier information found in inappropriate context...")
            return resolve_divergence_impl(type->payload.qualified_type.type, false);
        }
        case NoRet_TAG:
        case Int_TAG:
        case Float_TAG:
            return Unknown;

        default: SHADY_NOT_IMPLEM;
    }
}

DivergenceQualifier resolve_divergence(const Type* type) {
    return resolve_divergence_impl(type, true);
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

const Type* check_type_call(IrArena* arena, Call call) {
    const Type* callee_type = call.callee->type;
    if (callee_type->tag != FnType_TAG)
        error("Callees must have a function type");
    if (callee_type->payload.fn_type.param_types.count != call.args.count)
        error("Mismatched argument counts");
    for (size_t i = 0; i < call.args.count; i++) {
        // TODO
    }
    //return callee_type->payload.fn.return_type;
    return NULL;
}

// This is a pretty good helper fn
const Type* derive_fn_type(IrArena* arena, const Function* fn) {
    const Type* ptypes[fn->params.count];
    for (size_t i = 0; i < fn->params.count; i++)
        ptypes[i] = fn->params.nodes[i]->type;
    return fn_type(arena, (FnType) { .param_types = nodes(arena, fn->params.count, ptypes), .return_types = fn->return_types });
}

const Type* check_type_fn(IrArena* arena, Function fn) {
    return derive_fn_type(arena, &fn);
}

const Type* check_type_var_decl(IrArena* arena, VariableDecl decl) {
    return ptr_type(arena, (PtrType) { .address_space = decl.address_space, .pointed_type = decl.variable->type });
}

const Type* check_type_expr_eval(IrArena* arena, ExpressionEval expr) {
    SHADY_NOT_IMPLEM;
}

const Type* check_type_var(IrArena* arena, Variable variable) {
    return variable.type;
}

const Type* check_type_untyped_number(IrArena* arena, UntypedNumber untyped) {
    error("should never happen");
}

const Type* check_type_int_literal(IrArena* arena, IntLiteral lit) {
    return int_type(arena);
}

const Type* check_type_let(IrArena* arena, Let let) {
    return let.target->type;
}

const Type* check_type_fn_ret(IrArena* arena, Return fn_ret) {
    return noret_type(arena);
}

const Type* check_type_primop(IrArena* arena, PrimOp primop) {
    switch (primop.op) {
        case sub_op:
        case add_op: {
            bool is_result_uniform = true;
            for (size_t i = 0; i < primop.args.count; i++) {
                DivergenceQualifier op_div = resolve_divergence(primop.args.nodes[i]->type);
                assert(op_div != Unknown); // we expect all operands to be clearly known !
                is_result_uniform ^= op_div == Uniform;
            }

            return qualified_type(arena, (QualifiedType) { .is_uniform = is_result_uniform, .type = int_type(arena) });
        }
        default: SHADY_NOT_IMPLEM;
    }
}

const Type* check_type_root(IrArena* arena, Root program) {
    return NULL;
}
