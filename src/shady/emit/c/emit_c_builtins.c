#include "emit_c.h"

#include "log.h"

#pragma GCC diagnostic error "-Wswitch"

CTerm emit_c_builtin(Emitter* emitter, Builtin b) {
    switch (b) {
        case BuiltinBaseInstance:
        case BuiltinBaseVertex:
        case BuiltinDeviceIndex:
        case BuiltinDrawIndex:
        case BuiltinFragCoord:
        case BuiltinFragDepth:
        case BuiltinInstanceId:
        case BuiltinInvocationId:
        case BuiltinInstanceIndex:
        case BuiltinLocalInvocationId:
        case BuiltinLocalInvocationIndex:
        case BuiltinGlobalInvocationId:
        case BuiltinWorkgroupId:
        case BuiltinWorkgroupSize:
        case BuiltinNumSubgroups:
        case BuiltinNumWorkgroups:
        case BuiltinPosition:
        case BuiltinPrimitiveId:
        case BuiltinSubgroupLocalInvocationId:
        case BuiltinSubgroupId:
        case BuiltinVertexIndex:
        case BuiltinSubgroupSize: {
            return term_from_cvar(get_builtin_name(b));
        }
        case BuiltinsCount: error("")
    }
}

/*case subgroup_local_id_op: {
    switch (emitter->config.dialect) {
        case ISPC: final_expression = "programIndex"; break;
        case C: error("TODO");
        case GLSL: final_expression = "gl_SubgroupInvocationID"; break;
    }
    break;
}
case subgroup_id_op:        final_expression = "gl_SubgroupID";         break;
case workgroup_id_op:       final_expression = "gl_WorkGroupID";        break;
case workgroup_local_id_op: final_expression = "gl_LocalInvocationID";  break;
case workgroup_num_op:      final_expression = "gl_NumWorkGroups";      break;
case workgroup_size_op:     final_expression = "gl_WorkGroupSize";      break;
case global_id_op:          final_expression = "gl_GlobalInvocationID"; break;*/