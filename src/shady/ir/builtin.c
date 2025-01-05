#include "shady/ir/builtin.h"
#include "shady/ir/annotation.h"
#include "shady/ir/module.h"
#include "shady/ir/mem.h"
#include "shady/ir/arena.h"
#include "shady/ir/decl.h"
#include "shady/ir/int.h"
#include "shady/ir/float.h"

#include "log.h"
#include "portability.h"

#include <spirv/unified1/spirv.h>

#include <string.h>

static AddressSpace builtin_as[] = {
#define BUILTIN(_, as, _2, _3) as,
SHADY_BUILTINS()
#undef BUILTIN
};

AddressSpace shd_get_builtin_address_space(Builtin builtin) {
    if (builtin >= BuiltinsCount)
        return AsGeneric;
    return builtin_as[builtin];
}

static ShdScope builtin_scope[] = {
#define BUILTIN(_, _2, scope, _3) ShdScope##scope,
SHADY_BUILTINS()
#undef BUILTIN
};

ShdScope shd_get_builtin_scope(Builtin builtin) {
    if (builtin >= BuiltinsCount)
        return ShdScopeBottom;
    return builtin_scope[builtin];
}

static String builtin_names[] = {
#define BUILTIN(name, _, _2, _3) #name,
SHADY_BUILTINS()
#undef BUILTIN
};

String shd_get_builtin_name(Builtin builtin) {
    if (builtin >= BuiltinsCount)
        return "";
    return builtin_names[builtin];
}

const Type* shd_get_builtin_type(IrArena* arena, Builtin builtin) {
    switch (builtin) {
#define BUILTIN(name, _, _2, datatype) case Builtin##name: return datatype;
SHADY_BUILTINS()
#undef BUILTIN
        default: shd_error("Unhandled builtin")
    }
}

// What's the decoration for the builtin
static SpvBuiltIn spv_builtins[] = {
#define BUILTIN(name, _, _2, _3) SpvBuiltIn##name,
SHADY_BUILTINS()
#undef BUILTIN
};

Builtin shd_get_builtin_by_name(String s) {
    for (size_t i = 0; i < BuiltinsCount; i++) {
        if (strcmp(s, builtin_names[i]) == 0) {
            return i;
        }
    }
    return BuiltinsCount;
}

Builtin shd_get_builtin_by_spv_id(SpvBuiltIn id) {
    Builtin b = BuiltinsCount;
    for (size_t i = 0; i < BuiltinsCount; i++) {
        if (id == spv_builtins[i]) {
            b = i;
            break;
        }
    }
    return b;
}

Builtin shd_get_decl_builtin(const Node* decl) {
    const Node* a = shd_lookup_annotation(decl, "Builtin");
    if (!a)
        return BuiltinsCount;
    String payload = shd_get_annotation_string_payload(a);
    return shd_get_builtin_by_name(payload);
}


bool shd_is_decl_builtin(const Node* decl) {
    return shd_get_decl_builtin(decl) != BuiltinsCount;
}

int32_t shd_get_builtin_spv_id(Builtin builtin) {
    if (builtin >= BuiltinsCount)
        return 0;
    return spv_builtins[builtin];
}

bool shd_is_builtin_load_op(const Node* n, Builtin* out) {
    assert(is_instruction(n));
    if (n->tag == Load_TAG) {
        const Node* src = n->payload.load.ptr;
        if (src->tag == BuiltinRef_TAG) {
            BuiltinRef payload = src->payload.builtin_ref;
            *out = payload.builtin;
            return true;
        }
    }
    return false;
}

const Node* shd_get_or_create_builtin(Module* m, Builtin b, String n) {
    return builtin_ref_helper(shd_module_get_arena(m), b);
}

const Node* shd_bld_builtin_load(Module* m, BodyBuilder* bb, Builtin b) {
    return shd_bld_load(bb, shd_get_or_create_builtin(m, b, NULL));
}
