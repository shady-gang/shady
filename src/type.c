#include "implem.h"

bool is_subtype(struct Type* supertype, struct Type* type) {
    return true; // TODO
}

void check_subtype(struct Type* supertype, struct Type* type) {
    if (!is_subtype(supertype, type))
        error("is not a subtype")
}

struct Type* infer_call(struct IrArena* arena, struct Call call) {
    struct Type* callee_type = &call.callee->type;
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