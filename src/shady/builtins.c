#include "shady/builtins.h"
#include "spirv/unified1/spirv.h"

#include "log.h"
#include "portability.h"
#include <string.h>

AddressSpace builtin_as[] = {
#define BUILTIN(_, as, _2) as,
SHADY_BUILTINS()
#undef BUILTIN
};

String builtin_names[] = {
#define BUILTIN(name, _, _2) #name,
SHADY_BUILTINS()
#undef BUILTIN
};

const Type* get_builtin_type(IrArena* arena, Builtin builtin) {
    switch (builtin) {
#define BUILTIN(name, _, datatype) case Builtin##name: return datatype;
SHADY_BUILTINS()
#undef BUILTIN
        default: error("Unhandled builtin")
    }
}

// What's the decoration for the builtin
SpvBuiltIn spv_builtins[] = {
#define BUILTIN(name, _, _2) SpvBuiltIn##name,
SHADY_BUILTINS()
#undef BUILTIN
};

Builtin get_builtin_by_name(String s) {
    for (size_t i = 0; i < BuiltinsCount; i++) {
        if (strcmp(s, builtin_names[i]) == 0) {
            return i;
        }
    }
    return BuiltinsCount;
}

Builtin get_builtin_by_spv_id(SpvBuiltIn id) {
    Builtin b = BuiltinsCount;
    for (size_t i = 0; i < BuiltinsCount; i++) {
        if (id == spv_builtins[i]) {
            b = i;
            break;
        }
    }
    return b;
}
