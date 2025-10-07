#include "shady/pass.h"
#include "shady/ir/type.h"

#include "log.h"
#include "portability.h"

#include "spirv/unified1/spirv.h"

#include <string.h>

typedef struct {
    Rewriter rewriter;
} Context;

typedef struct {
    SpvOp spv_op;
    Op scalar;
    const Node* (*I)(IrArena*, const Type* t);
} GroupOp;

static GroupOp group_operations[] = {
    { .spv_op = SpvOpGroupIAdd, add_op },
    { .spv_op = SpvOpGroupFAdd, add_op },
    { .spv_op = SpvOpGroupFMin, min_op },
    { .spv_op = SpvOpGroupUMin, min_op },
    { .spv_op = SpvOpGroupSMin, min_op },
    { .spv_op = SpvOpGroupFMax, max_op, },
    { .spv_op = SpvOpGroupUMax, max_op },
    { .spv_op = SpvOpGroupSMax, max_op },
    { .spv_op = SpvOpGroupNonUniformBallotBitCount, /* todo */ },
    { .spv_op = SpvOpGroupNonUniformIAdd, add_op },
    { .spv_op = SpvOpGroupNonUniformFAdd, add_op },
    { .spv_op = SpvOpGroupNonUniformIMul, mul_op },
    { .spv_op = SpvOpGroupNonUniformFMul, mul_op },
    { .spv_op = SpvOpGroupNonUniformSMin, min_op },
    { .spv_op = SpvOpGroupNonUniformUMin, min_op },
    { .spv_op = SpvOpGroupNonUniformFMin, min_op },
    { .spv_op = SpvOpGroupNonUniformSMax, max_op },
    { .spv_op = SpvOpGroupNonUniformUMax, max_op },
    { .spv_op = SpvOpGroupNonUniformFMax, max_op },
    { .spv_op = SpvOpGroupNonUniformBitwiseAnd, and_op },
    { .spv_op = SpvOpGroupNonUniformBitwiseOr, or_op },
    { .spv_op = SpvOpGroupNonUniformBitwiseXor, xor_op },
    { .spv_op = SpvOpGroupNonUniformLogicalAnd, and_op },
    { .spv_op = SpvOpGroupNonUniformLogicalOr, or_op },
    { .spv_op = SpvOpGroupNonUniformLogicalXor, xor_op },
    { .spv_op = SpvOpGroupIAddNonUniformAMD, /* todo: map to std */ },
    { .spv_op = SpvOpGroupFAddNonUniformAMD, /* todo: map to std */ },
    { .spv_op = SpvOpGroupFMinNonUniformAMD, /* todo: map to std */ },
    { .spv_op = SpvOpGroupUMinNonUniformAMD, /* todo: map to std */ },
    { .spv_op = SpvOpGroupSMinNonUniformAMD, /* todo: map to std */ },
    { .spv_op = SpvOpGroupFMaxNonUniformAMD, /* todo: map to std */ },
    { .spv_op = SpvOpGroupUMaxNonUniformAMD, /* todo: map to std */ },
    { .spv_op = SpvOpGroupSMaxNonUniformAMD, /* todo: map to std */ },
    { .spv_op = SpvOpGroupIMulKHR, /* todo: map to std */ },
    { .spv_op = SpvOpGroupFMulKHR, /* todo: map to std */ },
    { .spv_op = SpvOpGroupBitwiseAndKHR, /* todo: map to std */ },
    { .spv_op = SpvOpGroupBitwiseOrKHR, /* todo: map to std */ },
    { .spv_op = SpvOpGroupBitwiseXorKHR, /* todo: map to std */ },
    { .spv_op = SpvOpGroupLogicalAndKHR, /* todo: map to std */ },
    { .spv_op = SpvOpGroupLogicalOrKHR, /* todo: map to std */ },
    { .spv_op = SpvOpGroupLogicalXorKHR, /* todo: map to std */ },
};

enum {
    NumGroupOps = sizeof(group_operations) / sizeof(group_operations[0])
};

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    switch (node->tag) {
        case ExtInstr_TAG: {
            ExtInstr payload = node->payload.ext_instr;
            ExtSpvOp op = payload.op->payload.ext_spv_op;
            if (strcmp(op.set, "spirv.core") == 0) {
                for (size_t i = 0; i < NumGroupOps; i++) {
                    if (op.opcode == group_operations[i].spv_op) {
                        if (shd_get_int_value(payload.arguments.nodes[1], false) == SpvGroupOperationInclusiveScan) {
                            //assert(group_operations[i].I);
                            IrArena* oa = node->arena;
                            payload.arguments = shd_change_node_at_index(oa, payload.arguments, 1, shd_uint32_literal(a, SpvGroupOperationExclusiveScan));
                            const Node* new = shd_recreate_node(r, ext_instr(oa, payload));
                            // new = prim_op_helper(a, group_operations[i].scalar, shd_empty(a), mk_nodes(a, new, group_operations[i].I(a, new->type) ));
                            new = prim_op_helper(a, group_operations[i].scalar, mk_nodes(a, new, shd_recreate_node(r, payload.arguments.nodes[2]) ));
                            new = mem_and_value_helper(a, shd_rewrite_node(r, payload.mem), new);
                            return new;
                        }
                    }
                }
            }
        }
        default: break;
    }

    return shd_recreate_node(r, node);
}

/// Transforms
/// SpvOpGroupXXX(Scope, 'GroupOperationInclusiveScan', v)
/// into
/// SpvOpGroupXXX(Scope, 'GroupOperationExclusiveScan', v) op v
Module* shd_pass_lower_inclusive_scan(SHADY_UNUSED const CompilerConfig* config, SHADY_UNUSED const void* unused, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
