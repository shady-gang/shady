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

static const Node* zero(IrArena* a, const Type* t) {
    t = shd_get_unqualified_type(t);
    assert(t->tag == Int_TAG);
    Int t_payload = t->payload.int_type;
    IntLiteral lit = {
        .width = t_payload.width,
        .is_signed = t_payload.is_signed,
        .value = 0
    };
    return int_literal(a, lit);
}

static const Node* one(IrArena* a, const Type* t) {
    t = shd_get_unqualified_type(t);
    assert(t->tag == Int_TAG);
    Int t_payload = t->payload.int_type;
    IntLiteral lit = {
        .width = t_payload.width,
        .is_signed = t_payload.is_signed,
        .value = 1
    };
    return int_literal(a, lit);
}

static GroupOp group_operations[] = {
    { SpvOpGroupIAdd, add_op },
    { SpvOpGroupFAdd, add_op },
    { SpvOpGroupFMin, min_op },
    { SpvOpGroupUMin, min_op },
    { SpvOpGroupSMin, min_op },
    { SpvOpGroupFMax, max_op, },
    { SpvOpGroupUMax, max_op },
    { SpvOpGroupSMax, max_op },
    { SpvOpGroupNonUniformBallotBitCount, /* todo */ },
    { SpvOpGroupNonUniformIAdd, add_op },
    { SpvOpGroupNonUniformFAdd, add_op },
    { SpvOpGroupNonUniformIMul, mul_op },
    { SpvOpGroupNonUniformFMul, mul_op },
    { SpvOpGroupNonUniformSMin, min_op },
    { SpvOpGroupNonUniformUMin, min_op },
    { SpvOpGroupNonUniformFMin, min_op },
    { SpvOpGroupNonUniformSMax, max_op },
    { SpvOpGroupNonUniformUMax, max_op },
    { SpvOpGroupNonUniformFMax, max_op },
    { SpvOpGroupNonUniformBitwiseAnd, and_op },
    { SpvOpGroupNonUniformBitwiseOr, or_op },
    { SpvOpGroupNonUniformBitwiseXor, xor_op },
    { SpvOpGroupNonUniformLogicalAnd, and_op },
    { SpvOpGroupNonUniformLogicalOr, or_op },
    { SpvOpGroupNonUniformLogicalXor, xor_op },
    { SpvOpGroupIAddNonUniformAMD, /* todo: map to std */ },
    { SpvOpGroupFAddNonUniformAMD, /* todo: map to std */ },
    { SpvOpGroupFMinNonUniformAMD, /* todo: map to std */ },
    { SpvOpGroupUMinNonUniformAMD, /* todo: map to std */ },
    { SpvOpGroupSMinNonUniformAMD, /* todo: map to std */ },
    { SpvOpGroupFMaxNonUniformAMD, /* todo: map to std */ },
    { SpvOpGroupUMaxNonUniformAMD, /* todo: map to std */ },
    { SpvOpGroupSMaxNonUniformAMD, /* todo: map to std */ },
    { SpvOpGroupIMulKHR, /* todo: map to std */ },
    { SpvOpGroupFMulKHR, /* todo: map to std */ },
    { SpvOpGroupBitwiseAndKHR, /* todo: map to std */ },
    { SpvOpGroupBitwiseOrKHR, /* todo: map to std */ },
    { SpvOpGroupBitwiseXorKHR, /* todo: map to std */ },
    { SpvOpGroupLogicalAndKHR, /* todo: map to std */ },
    { SpvOpGroupLogicalOrKHR, /* todo: map to std */ },
    { SpvOpGroupLogicalXorKHR, /* todo: map to std */ },
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
            if (strcmp(payload.set, "spirv.core") == 0) {
                for (size_t i = 0; i < NumGroupOps; i++) {
                    if (payload.opcode == group_operations[i].spv_op) {
                        if (shd_get_int_value(payload.operands.nodes[1], false) == SpvGroupOperationInclusiveScan) {
                            //assert(group_operations[i].I);
                            IrArena* oa = node->arena;
                            payload.operands = shd_change_node_at_index(oa, payload.operands, 1, shd_uint32_literal(a, SpvGroupOperationExclusiveScan));
                            const Node* new = shd_recreate_node(r, ext_instr(oa, payload));
                            // new = prim_op_helper(a, group_operations[i].scalar, shd_empty(a), mk_nodes(a, new, group_operations[i].I(a, new->type) ));
                            new = prim_op_helper(a, group_operations[i].scalar, mk_nodes(a, new, shd_recreate_node(r, payload.operands.nodes[2]) ));
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
Module* shd_pass_lower_inclusive_scan(SHADY_UNUSED const CompilerConfig* config, Module* src) {
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
