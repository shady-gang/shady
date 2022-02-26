#include "implem.h"

bool is_subtype(struct Type* supertype, struct Type* type) {
    // uniform T <: varying T
    if (supertype->uniform && !type->uniform)
        return false;
    if (supertype->tag != type->tag)
        return false;
    switch (supertype->tag) {
        case RecordType: {
            struct Types* supermembers = &supertype->payload.record.members;
            struct Types* members = &type->payload.record.members;
            for (int i = 0; i < members->ntypes; i++) {
                if (!is_subtype(&supermembers->types[i], &members->types[i]))
                    return false;
            }
        }
        case FnType:
            if (!is_subtype(supertype->payload.fn.return_type, type->payload.fn.return_type))
                return false;

            struct Types* superparams = &supertype->payload.fn.param_types;
            struct Types* params = &type->payload.fn.param_types;
            goto check_params;
        case ContType:
            superparams = &supertype->payload.fn.param_types;
            params = &type->payload.fn.param_types;
            goto check_params;
        default: goto post_switch;
        check_params:
            if (params->ntypes != superparams->ntypes)
                return false;

            for (int i = 0; i < params->ntypes; i++) {
                if (!is_subtype(&params->types[i], &superparams->types[i]))
                    return false;
            }
        return false;
    }
    post_switch:
    return true;
}

void check_subtype(struct Type* supertype, struct Type* type) {
    if (!is_subtype(supertype, type))
        error("is not a subtype")
}

struct Type* infer_call(struct IrArena* arena, struct Call call) {
    struct Type* callee_type = call.callee->type;
    if (callee_type->tag != FnType)
        error("Callees must have a function type");
    if (callee_type->payload.fn.param_types.ntypes != call.args.count)
        error("Mismatched argument counts");
    for (int i = 0; i < call.args.count; i++) {
        // TODO
    }
    return callee_type->payload.fn.return_type;
}

struct Type* infer_var_decl(struct IrArena* arena, struct VariableDecl call) {

}

struct Type* infer_expr_eval(struct IrArena* arena, struct ExpressionEval call) {

}

struct Type* infer_var(struct IrArena* arena, struct Variable variable) {
    return variable.type;
}

struct Type* void_type(struct IrArena* arena) {
    struct Type* type = (struct Type*) arena_alloc(arena, sizeof(struct Type));
    type->tag = Void;
    type->uniform = true;
    return type;
}

struct Type* noret_type(struct IrArena* arena) {
    struct Type* type = (struct Type*) arena_alloc(arena, sizeof(struct Type));
    type->tag = NoRet;
    type->uniform = true;
    return type;
}

struct Type* int_type(struct IrArena* arena, bool uniform) {
    struct Type* type = (struct Type*) arena_alloc(arena, sizeof(struct Type));
    type->tag = Int;
    type->uniform = uniform;
    return type;
}

struct Type* float_type(struct IrArena* arena, bool uniform) {
    struct Type* type = (struct Type*) arena_alloc(arena, sizeof(struct Type));
    type->tag = Float;
    type->uniform = uniform;
    return type;
}

struct Type* record_type(struct IrArena* arena, char* name, struct Types members) {
    struct Type* type = (struct Type*) arena_alloc(arena, sizeof(struct Type));
    type->tag = RecordType;
    bool uniform = true;
    for (int i = 0; i < members.ntypes; i++) {
        uniform &= members.types[i].uniform;
    }
    type->uniform = uniform;
    type->payload.record.name = name;
    type->payload.record.members = members;
    return type;
}

struct Type* cont_type(struct IrArena* arena, bool uniform, struct Types params) {
    struct Type* type = (struct Type*) arena_alloc(arena, sizeof(struct Type));
    type->tag = ContType;
    type->uniform = uniform;
    type->payload.cont.param_types = params;
    return type;
}

struct Type* fn_type(struct IrArena* arena, bool uniform, struct Types params, struct Type* return_type) {
    struct Type* type = (struct Type*) arena_alloc(arena, sizeof(struct Type));
    type->tag = FnType;
    type->uniform = uniform;
    type->payload.fn.param_types = params;
    type->payload.fn.return_type = return_type;
    return type;
}
