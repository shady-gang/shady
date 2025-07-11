#include "emit_c.h"

#include "log.h"

#pragma GCC diagnostic error "-Wswitch"

static String ispc_builtins[ShdBuiltinsCount] = {
    [ShdBuiltinSubgroupLocalInvocationId] = "programIndex",
};

static String glsl_builtins[ShdBuiltinsCount] = {
    [ShdBuiltinSubgroupLocalInvocationId] = "gl_SubgroupInvocationID",
    [ShdBuiltinLocalInvocationId] = "gl_LocalInvocationID",
    [ShdBuiltinWorkgroupId] = "gl_WorkGroupID",
    [ShdBuiltinNumWorkgroups] = "gl_NumWorkGroups",
    [ShdBuiltinWorkgroupSize] = "gl_WorkGroupSize",
    [ShdBuiltinGlobalInvocationId] = "gl_GlobalInvocationID",
    [ShdBuiltinPosition] = "gl_Position",
};

CTerm shd_c_emit_builtin(Emitter* emitter, ShdBuiltin b) {
    String name = NULL;
    switch(emitter->backend_config.dialect) {
        case CDialect_ISPC: name = ispc_builtins[b]; break;
        case CDialect_GLSL: name = glsl_builtins[b]; break;
        default: break;
    }
    if (name)
        return term_from_cvar(name);
    return term_from_cvar(shd_get_builtin_name(b));
}
