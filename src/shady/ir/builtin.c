#include "shady/ir/builtin.h"
#include "shady/ir/annotation.h"

#include "log.h"
#include "portability.h"

#include <spirv/unified1/spirv.h>

#include <string.h>

static AddressSpace builtin_as[] = {
#define BUILTIN(_, as, _2) as,
SHADY_BUILTINS()
#undef BUILTIN
};

AddressSpace shd_get_builtin_address_space(Builtin builtin) {
    if (builtin >= BuiltinsCount)
        return AsGeneric;
    return builtin_as[builtin];
}

static String builtin_names[] = {
#define BUILTIN(name, _, _2) #name,
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
#define BUILTIN(name, _, datatype) case Builtin##name: return datatype;
SHADY_BUILTINS()
#undef BUILTIN
        default: shd_error("Unhandled builtin")
    }
}

// What's the decoration for the builtin
static SpvBuiltIn spv_builtins[] = {
#define BUILTIN(name, _, _2) SpvBuiltIn##name,
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
        if (src->tag == RefDecl_TAG)
            src = src->payload.ref_decl.decl;
        if (src->tag == GlobalVariable_TAG) {
            const Node* a = shd_lookup_annotation(src, "Builtin");
            if (a) {
                String bn = shd_get_annotation_string_payload(a);
                assert(bn);
                Builtin b = shd_get_builtin_by_name(bn);
                if (b != BuiltinsCount) {
                    *out = b;
                    return true;
                }
            }
        }
    }
    return false;
}
