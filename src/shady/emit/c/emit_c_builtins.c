#include "emit_c.h"

#include "log.h"

#pragma GCC diagnostic error "-Wswitch"

static String ispc_builtins[BuiltinsCount] = {
    [BuiltinSubgroupLocalInvocationId] = "programIndex",
};

static String glsl_builtins[BuiltinsCount] = {
    [BuiltinSubgroupLocalInvocationId] = "gl_SubgroupInvocationID",
    [BuiltinLocalInvocationId] = "gl_LocalInvocationID",
    [BuiltinWorkgroupId] = "gl_WorkGroupID",
    [BuiltinNumWorkgroups] = "gl_NumWorkGroups",
    [BuiltinWorkgroupSize] = "gl_WorkGroupSize",
    [BuiltinGlobalInvocationId] = "gl_GlobalInvocationID",
};

CTerm c_emit_builtin(Emitter* emitter, Builtin b) {
    String name = NULL;
    switch(emitter->config.dialect) {
        case ISPC: name = ispc_builtins[b]; break;
        case GLSL: name = glsl_builtins[b]; break;
        default: break;
    }
    if (name)
        return term_from_cvar(name);
    return term_from_cvar(get_builtin_name(b));
}
