#include "type.h"

#include "implem.h"

#include "dict.h"
#include "murmur3.h"

#include <string.h>
#include <assert.h>

KeyHash hash_type(struct Type** type) {
    uint32_t out[4];
    MurmurHash3_x64_128(*type, sizeof(struct Type), 0x1234567, &out);
    uint32_t final = 0;
    final ^= out[0];
    final ^= out[1];
    final ^= out[2];
    final ^= out[3];
    //printf("hash of :");
    //print_type(*type);
    //printf(" = [%u] %u\n", final, final % 32);
    return final;
}

bool compare_type(struct Type** a, struct Type** b) {
    return memcmp(*a, *b, sizeof(struct Type)) == 0;
}

struct TypeTable {
    struct Dict* set;
};

struct TypeTable* new_type_table() {
    struct TypeTable* table = (struct TypeTable*) malloc(sizeof (struct TypeTable));
    *table = (struct TypeTable) {
        .set = new_set(struct Type*, (HashFn) hash_type, (CmpFn) compare_type)
    };
    return table;
}
void destroy_type_table(struct TypeTable* table) {
    destroy_dict(table->set);
    free(table);
}

bool is_subtype(const struct Type* supertype, const struct Type* type) {
    if (supertype->tag != type->tag)
        return false;
    switch (supertype->tag) {
        case QualType: {
            // uniform T <: varying T
            if (supertype->payload.qualified.is_uniform && !type->payload.qualified.is_uniform)
                return false;
            return is_subtype(supertype->payload.qualified.type, type->payload.qualified.type);
        }
        case RecordType: {
            const struct Types* supermembers = &supertype->payload.record.members;
            const struct Types* members = &type->payload.record.members;
            for (size_t i = 0; i < members->count; i++) {
                if (!is_subtype(supermembers->types[i], members->types[i]))
                    return false;
            }
            goto post_switch;
        }
        case FnType:
            if (!is_subtype(supertype->payload.fn.return_type, type->payload.fn.return_type))
                return false;

            const struct Types* superparams = &supertype->payload.fn.param_types;
            const struct Types* params = &type->payload.fn.param_types;
            goto check_params;
        case ContType:
            superparams = &supertype->payload.fn.param_types;
            params = &type->payload.fn.param_types;
            goto check_params;
        case PtrType: SHADY_NOT_IMPLEM;
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
        case QualType: {
            if (!allow_qualifier_types)
                error("Uniformity qualifier information found in inappropriate context...")
            return resolve_divergence_impl(type->payload.qualified.type, false);
        }
        case Void:
            return Uniform;
        case NoRet:
        case Int:
        case Float:
            return Unknown;

        default: SHADY_NOT_IMPLEM;
    }
}

enum DivergenceQualifier resolve_divergence(const struct Type* type) {
    return resolve_divergence_impl(type, true);
}

const struct Type* strip_qualifier(const struct Type* type, enum DivergenceQualifier* qual_out) {
    if (type->tag == QualType) {
        *qual_out = type->payload.qualified.is_uniform ? Uniform : Varying;
        return type->payload.qualified.type;
    } else {
        *qual_out = Unknown;
        return type;
    }
}

const struct Type* infer_call(struct IrArena* arena, struct Call call) {
    const struct Type* callee_type = call.callee->type;
    if (callee_type->tag != FnType)
        error("Callees must have a function type");
    if (callee_type->payload.fn.param_types.count != call.args.count)
        error("Mismatched argument counts");
    for (size_t i = 0; i < call.args.count; i++) {
        // TODO
    }
    return callee_type->payload.fn.return_type;
}

// This is a pretty good helper fn
const struct Type* derive_fn_type(struct IrArena* arena, const struct Function* fn) {
    struct Types types = reserve_types(arena, fn->params.count);
    for (size_t i = 0; i < types.count; i++)
        types.types[i] = fn->params.nodes[i]->type;
    return fn_type(arena, types, fn->return_type);
}

const struct Type* infer_fn(struct IrArena* arena, struct Function fn) {
    return derive_fn_type(arena, &fn);
}

const struct Type* infer_var_decl(struct IrArena* arena, struct VariableDecl decl) {
    return ptr_type(arena, decl.variable->type, decl.address_space);
}

const struct Type* infer_expr_eval(struct IrArena* arena, struct ExpressionEval expr) {
    SHADY_NOT_IMPLEM;
}

const struct Type* infer_var(struct IrArena* arena, struct Variable variable) {
    return variable.type;
}

const struct Type* infer_untyped_number(struct IrArena* arena, struct UntypedNumber untyped) {
    error("should never happen");
}

const struct Type* infer_let(struct IrArena* arena, struct Let let) {
    return let.target->type;
}

const struct Type* infer_fn_ret(struct IrArena* arena, struct Return fn_ret) {
    return noret_type(arena);
}

const struct Type* infer_primop(struct IrArena* arena, struct PrimOp primop) {
    switch (primop.op) {
        case sub_op:
        case add_op: {
            bool is_result_uniform = true;
            for (size_t i = 0; i < primop.args.count; i++) {
                enum DivergenceQualifier op_div = resolve_divergence(primop.args.nodes[i]->type);
                assert(op_div != Unknown); // we expect all operands to be clearly known !
                is_result_uniform ^= op_div == Uniform;
            }

            return qualified_type(arena, is_result_uniform, int_type(arena));
        }
        default: SHADY_NOT_IMPLEM;
    }
}

const struct Type* infer_root(struct IrArena* arena, struct Root program) {
    return NULL;
}

#define type_ctor_prelude struct Type type; \
memset((void*)&type, 0, sizeof(struct Type));

#define type_ctor_epilogue struct Type* localptr = &type;                            \
struct Type** found = find_key_dict(struct Type*, arena->type_table->set, localptr); \
if (found) return *found;                                                            \
struct Type* globalptr = arena_alloc(arena, sizeof(struct Type));                    \
*globalptr = type;                                                                   \
bool result = insert_set_get_result(struct Type*, arena->type_table->set, globalptr);    \
assert(result);                                                                      \
return globalptr;                                                                    \

const struct Type* void_type(struct IrArena* arena) {
    type_ctor_prelude

    type.tag = Void;

    type_ctor_epilogue
}

const struct Type* noret_type(struct IrArena* arena) {
    type_ctor_prelude

    type.tag = NoRet;

    type_ctor_epilogue
}

const struct Type* int_type(struct IrArena* arena) {
    type_ctor_prelude

    type.tag = Int;

    type_ctor_epilogue
}

const struct Type* float_type(struct IrArena* arena) {
    type_ctor_prelude

    type.tag = Float;

    type_ctor_epilogue
}

const struct Type* record_type(struct IrArena* arena, const char* name, struct Types members) {
    type_ctor_prelude

    type.tag = RecordType;
    type.payload.record.name = name;
    type.payload.record.members = members;

    type_ctor_epilogue
}

const struct Type* cont_type(struct IrArena* arena, struct Types params) {
    type_ctor_prelude

    type.tag = ContType;
    type.payload.cont.param_types = params;

    type_ctor_epilogue
}

const struct Type* fn_type(struct IrArena* arena, struct Types params, const struct Type* return_type) {
    type_ctor_prelude

    type.tag = FnType;
    type.payload.fn.param_types = params;
    type.payload.fn.return_type = return_type;

    type_ctor_epilogue
}

const struct Type* ptr_type(struct IrArena* arena, const struct Type* pointed_type, enum AddressSpace address_space) {
    type_ctor_prelude

    type.tag = PtrType;
    type.payload.ptr.pointed_type = pointed_type;
    type.payload.ptr.address_space = address_space;

    type_ctor_epilogue
}

const struct Type* qualified_type(struct IrArena* arena, bool is_uniform, const struct Type* unqualified) {
    type_ctor_prelude

    // TODO check unqualified is truly unqualified

    type.tag = QualType;
    type.payload.qualified.is_uniform = is_uniform;
    type.payload.qualified.type = unqualified;

    type_ctor_epilogue
}
