#include "implem.h"

#include "dict.h"
#include "murmur3.h"

#include <string.h>
#include <assert.h>

KeyHash hash_type_ptr(struct Type** type) {
    uint32_t out[4];
    MurmurHash3_x64_128(*type, sizeof(struct Type), 0x1234567, &out);
    uint32_t final = 0;
    final ^= out[0];
    final ^= out[1];
    final ^= out[2];
    final ^= out[3];
    printf("hash of :");
    print_type(*type);
    printf(" = [%u] %u\n", final, final % 32);
    return final;
}

bool compare_type_ptr(struct Type** a, struct Type** b) {
    return memcmp(*a, *b, sizeof(struct Type)) == 0;
}

struct TypeTable {
    struct Dict* set;
};

struct TypeTable* new_type_table() {
    struct TypeTable* table = (struct TypeTable*) malloc(sizeof (struct TypeTable));
    *table = (struct TypeTable) {
        .set = new_set(struct Type*, hash_type_ptr, compare_type_ptr)
    };
    return table;
}
void destroy_type_table(struct TypeTable* table) {
    destroy_dict(table->set);
    free(table);
}

bool is_subtype(const struct Type* supertype, const struct Type* type) {
    // uniform T <: varying T
    if (supertype->uniform && !type->uniform)
        return false;
    if (supertype->tag != type->tag)
        return false;
    switch (supertype->tag) {
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

const struct Type* infer_fn(struct IrArena* arena, struct Function fn) {
    // TODO check function
    struct Types types = reserve_types(arena, fn.params.count);
    for (size_t i = 0; i < types.count; i++)
        types.types[i] = fn.params.variables[i]->type;
    return fn_type(arena, true, types, fn.return_type);
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

#define type_ctor_prelude struct Type type; \
memset((void*)&type, 0, sizeof(struct Type));

#define type_ctor_epilogue struct Type* localptr = &type;                            \
struct Type** found = find_key_dict(struct Type*, arena->type_table->set, localptr); \
if (found) return *found;                                                            \
struct Type* globalptr = arena_alloc(arena, sizeof(struct Type));                    \
*globalptr = type;                                                                   \
bool result = insert_or_get_set(struct Type*, arena->type_table->set, globalptr);    \
assert(result);                                                                      \
return globalptr;                                                                    \

const struct Type* void_type(struct IrArena* arena) {
    type_ctor_prelude

    type.tag = Void;
    type.uniform = true;

    type_ctor_epilogue
}

const struct Type* noret_type(struct IrArena* arena) {
    type_ctor_prelude

    type.tag = NoRet;
    type.uniform = true;

    type_ctor_epilogue
}

const struct Type* int_type(struct IrArena* arena, bool uniform) {
    type_ctor_prelude

    type.tag = Int;
    type.uniform = uniform;

    type_ctor_epilogue
}

const struct Type* float_type(struct IrArena* arena, bool uniform) {
    type_ctor_prelude

    type.tag = Float;
    type.uniform = uniform;

    type_ctor_epilogue
}

const struct Type* record_type(struct IrArena* arena, const char* name, struct Types members) {
    type_ctor_prelude

    type.tag = RecordType;
    bool uniform = true;
    for (size_t i = 0; i < members.count; i++) {
        uniform &= members.types[i]->uniform;
    }
    type.uniform = uniform;
    type.payload.record.name = name;
    type.payload.record.members = members;

    type_ctor_epilogue
}

const struct Type* cont_type(struct IrArena* arena, bool uniform, struct Types params) {
    type_ctor_prelude

    type.tag = ContType;
    type.uniform = uniform;
    type.payload.cont.param_types = params;

    type_ctor_epilogue
}

const struct Type* fn_type(struct IrArena* arena, bool uniform, struct Types params, const struct Type* return_type) {
    type_ctor_prelude

    type.tag = FnType;
    type.uniform = uniform;
    type.payload.fn.param_types = params;
    type.payload.fn.return_type = return_type;

    type_ctor_epilogue
}

const struct Type* ptr_type(struct IrArena* arena, const struct Type* pointed_type, enum AddressSpace address_space) {
    type_ctor_prelude

    type.tag = PtrType;
    type.uniform = pointed_type;
    type.payload.ptr.pointed_type = pointed_type;
    type.payload.ptr.address_space = address_space;

    type_ctor_epilogue
}
