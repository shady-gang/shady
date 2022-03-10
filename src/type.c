#include "type.h"

#include "implem.h"

#include "dict.h"
#include "murmur3.h"

#include <string.h>
#include <assert.h>

bool is_subtype(const struct Type* supertype, const struct Type* type) {
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
            const struct Types* supermembers = &supertype->payload.record_type.members;
            const struct Types* members = &type->payload.record_type.members;
            for (size_t i = 0; i < members->count; i++) {
                if (!is_subtype(supermembers->types[i], members->types[i]))
                    return false;
            }
            goto post_switch;
        }
        case FnType_TAG:
            if (!is_subtype(supertype->payload.fn.return_type, type->payload.fn.return_type))
                return false;

            const struct Types* superparams = &supertype->payload.fn_type.param_types;
            const struct Types* params = &type->payload.fn_type.param_types;
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
                if (!is_subtype(params->types[i], superparams->types[i]))
                    return false;
            }
        return false;
    }
    post_switch:
    return true;
}

void check_subtype(const struct Type* supertype, const struct Type* type) {
    if (!is_subtype(supertype, type))
        error("is not a subtype")
}

enum DivergenceQualifier resolve_divergence_impl(const struct Type* type, bool allow_qualifier_types) {
    switch (type->tag) {
        case QualifiedType_TAG: {
            if (!allow_qualifier_types)
                error("Uniformity qualifier information found in inappropriate context...")
            return resolve_divergence_impl(type->payload.qualified_type.type, false);
        }
        case Void_TAG:
            return Uniform;
        case NoRet_TAG:
        case Int_TAG:
        case Float_TAG:
            return Unknown;

        default: SHADY_NOT_IMPLEM;
    }
}

enum DivergenceQualifier resolve_divergence(const struct Type* type) {
    return resolve_divergence_impl(type, true);
}

const struct Type* strip_qualifier(const struct Type* type, enum DivergenceQualifier* qual_out) {
    if (type->tag == QualifiedType_TAG) {
        *qual_out = type->payload.qualified_type.is_uniform ? Uniform : Varying;
        return type->payload.qualified_type.type;
    } else {
        *qual_out = Unknown;
        return type;
    }
}

const struct Type* check_type_call(struct IrArena* arena, struct Call call) {
    const struct Type* callee_type = call.callee->type;
    if (callee_type->tag != FnType_TAG)
        error("Callees must have a function type");
    if (callee_type->payload.fn_type.param_types.count != call.args.count)
        error("Mismatched argument counts");
    for (size_t i = 0; i < call.args.count; i++) {
        // TODO
    }
    return callee_type->payload.fn.return_type;
}

// This is a pretty good helper fn
const struct Type* derive_fn_type(struct IrArena* arena, const struct Function* fn) {
    const struct Type* ptypes[fn->params.count];
    for (size_t i = 0; i < fn->params.count; i++)
        ptypes[i] = fn->params.nodes[i]->type;
    return fn_type(arena, (struct FnType) { .param_types = types(arena, fn->params.count, ptypes), .return_type = fn->return_type });
}

const struct Type* check_type_fn(struct IrArena* arena, struct Function fn) {
    return derive_fn_type(arena, &fn);
}

const struct Type* check_type_var_decl(struct IrArena* arena, struct VariableDecl decl) {
    return ptr_type(arena, (struct PtrType) { .address_space = decl.address_space, .pointed_type = decl.variable->type });
}

const struct Type* check_type_expr_eval(struct IrArena* arena, struct ExpressionEval expr) {
    SHADY_NOT_IMPLEM;
}

const struct Type* check_type_var(struct IrArena* arena, struct Variable variable) {
    return variable.type;
}

const struct Type* check_type_untyped_number(struct IrArena* arena, struct UntypedNumber untyped) {
    error("should never happen");
}

const struct Type* check_type_let(struct IrArena* arena, struct Let let) {
    return let.target->type;
}

const struct Type* check_type_fn_ret(struct IrArena* arena, struct Return fn_ret) {
    return noret_type(arena);
}

const struct Type* check_type_primop(struct IrArena* arena, struct PrimOp primop) {
    switch (primop.op) {
        case sub_op:
        case add_op: {
            bool is_result_uniform = true;
            for (size_t i = 0; i < primop.args.count; i++) {
                enum DivergenceQualifier op_div = resolve_divergence(primop.args.nodes[i]->type);
                assert(op_div != Unknown); // we expect all operands to be clearly known !
                is_result_uniform ^= op_div == Uniform;
            }

            return qualified_type(arena, (struct QualifiedType) { .is_uniform = is_result_uniform, .type = int_type(arena) });
        }
        default: SHADY_NOT_IMPLEM;
    }
}

const struct Type* check_type_root(struct IrArena* arena, struct Root program) {
    return NULL;
}
