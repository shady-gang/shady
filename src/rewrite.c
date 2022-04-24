#include "log.h"
#include "local_array.h"
#include "type.h"

#include "list.h"

Nodes rewrite_nodes(Rewriter* rewriter, Nodes old_nodes) {
    size_t count = old_nodes.count;
    LARRAY(const Node*, arr, count);
    for (size_t i = 0; i < count; i++)
        arr[i] = rewriter->rewrite_fn(rewriter, old_nodes.nodes[i]);
    return nodes(rewriter->dst_arena, count, arr);
}

const Node* rewrite_node(Rewriter* rewriter, const Node* node) { return rewriter->rewrite_fn(rewriter, node); }

const Node* recreate_node_identity(Rewriter* rewriter, const Node* node) {
    if (node == NULL)
        return NULL;
    switch (node->tag) {
        case Root_TAG:          return root(rewriter->dst_arena, (Root) {
            .declarations = rewrite_nodes(rewriter, node->payload.root.declarations)
        });
        case Block_TAG:         return block(rewriter->dst_arena, (Block) {
            .instructions = rewrite_nodes(rewriter, node->payload.block.instructions),
            .terminator = rewriter->rewrite_fn(rewriter, node->payload.block.terminator)
        });
        case Constant_TAG:
        case Function_TAG: error("nominal nodes need custom rewrite logic")
        case UntypedNumber_TAG: return untyped_number(rewriter->dst_arena, (UntypedNumber) {
            .plaintext = string(rewriter->dst_arena, node->payload.untyped_number.plaintext)
        });
        case IntLiteral_TAG:    return int_literal(rewriter->dst_arena, node->payload.int_literal);
        case True_TAG:          return true_lit(rewriter->dst_arena);
        case False_TAG:         return false_lit(rewriter->dst_arena);
        case Variable_TAG:      return var_with_id(rewriter->dst_arena, rewriter->rewrite_fn(rewriter, node->payload.var.type), string(rewriter->dst_arena, node->payload.var.name), node->payload.var.id);
        case Let_TAG:           return let(rewriter->dst_arena, (Let) {
            .variables = rewrite_nodes(rewriter, node->payload.let.variables),
            .instruction = rewriter->rewrite_fn(rewriter, node->payload.let.instruction)
        });
        case PrimOp_TAG:        return prim_op(rewriter->dst_arena, (PrimOp) {
            .op = node->payload.prim_op.op,
            .operands = rewrite_nodes(rewriter, node->payload.prim_op.operands)
        });
        case Call_TAG:          return call_instr(rewriter->dst_arena, (Call) {
            .callee = rewriter->rewrite_fn(rewriter, node->payload.call_instr.callee),
            .args = rewrite_nodes(rewriter, node->payload.call_instr.args)
        });
        case If_TAG:            return if_instr(rewriter->dst_arena, (If) {
            .yield_types = rewrite_nodes(rewriter, node->payload.if_instr.yield_types),
            .condition = rewriter->rewrite_fn(rewriter, node->payload.if_instr.condition),
            .if_true = rewriter->rewrite_fn(rewriter, node->payload.if_instr.if_true),
            .if_false = rewriter->rewrite_fn(rewriter, node->payload.if_instr.if_false),
        });
        case Jump_TAG:          return jump(rewriter->dst_arena, (Jump) {
            .target = rewriter->rewrite_fn(rewriter, node->payload.jump.target),
            .args = rewrite_nodes(rewriter, node->payload.jump.args)
        });
        case Return_TAG:        return fn_ret(rewriter->dst_arena, (Return) {
            .fn = NULL,
            .values = rewrite_nodes(rewriter, node->payload.fn_ret.values)
        });
        case Unreachable_TAG:   return unreachable(rewriter->dst_arena);
        case Join_TAG:          return join(rewriter->dst_arena, (Join) {
            .args = rewrite_nodes(rewriter, node->payload.join.args)
        });
        case NoRet_TAG:         return noret_type(rewriter->dst_arena);
        case Int_TAG:           return int_type(rewriter->dst_arena);
        case Bool_TAG:          return bool_type(rewriter->dst_arena);
        case Float_TAG:         return float_type(rewriter->dst_arena);
        case RecordType_TAG:    return record_type(rewriter->dst_arena, (RecordType) {
                                    .members = rewrite_nodes(rewriter, node->payload.record_type.members)});
        case FnType_TAG:        return fn_type(rewriter->dst_arena, (FnType) {
                                    .is_continuation = node->payload.fn_type.is_continuation,
                                    .param_types = rewrite_nodes(rewriter, node->payload.fn_type.param_types),
                                    .return_types = rewrite_nodes(rewriter, node->payload.fn_type.return_types)});
        case PtrType_TAG:       return ptr_type(rewriter->dst_arena, (PtrType) {
                                    .address_space = node->payload.ptr_type.address_space,
                                    .pointed_type = rewriter->rewrite_fn(rewriter, node->payload.ptr_type.pointed_type)});
        case QualifiedType_TAG: return qualified_type(rewriter->dst_arena, (QualifiedType) {
                                    .is_uniform = node->payload.qualified_type.is_uniform,
                                    .type = rewriter->rewrite_fn(rewriter, node->payload.qualified_type.type)});
        default: error("unhandled node for rewrite %s", node_tags[node->tag]);
    }
}