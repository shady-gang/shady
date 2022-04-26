#include "shady/ir.h"

#include "analysis/scope.h"

#include <assert.h>

static void visit_nodes(Visitor* visitor, Nodes nodes) {
    for (size_t i = 0; i < nodes.count; i++) {
        visitor->visit_fn(visitor, nodes.nodes[i]);
    }
}

#define visit(t) if (t) visitor->visit_fn(visitor, t);

void visit_fn_blocks_except_head(Visitor* visitor, const Node* function) {
    assert(function->tag == Function_TAG);
    assert(!function->payload.fn.atttributes.is_continuation);
    Scope scope = build_scope(function);
    assert(scope.rpo[0]->node == function);
    for (size_t i = 1; i < scope.size; i++) {
        visit(scope.rpo[i]->node);
    }
    dispose_scope(&scope);
}

void visit_children(Visitor* visitor, const Node* node) {
    switch(node->tag) {
        case Constant_TAG: {
            visit(node->payload.constant.value);
            visit(node->payload.constant.type_hint);
            break;
        }
        case Function_TAG: {
            visit_nodes(visitor, node->payload.fn.params);
            visit_nodes(visitor, node->payload.fn.return_types);
            visit(node->payload.fn.block);

            if (visitor->visit_fn_scope_rpo && !node->payload.fn.atttributes.is_continuation)
                visit_fn_blocks_except_head(visitor, node);

            break;
        }
        case Block_TAG: {
            visit_nodes(visitor, node->payload.block.instructions);
            visit(node->payload.block.terminator);
            break;
        }
        case ParsedBlock_TAG: {
            visit_nodes(visitor, node->payload.parsed_block.instructions);
            visit(node->payload.parsed_block.terminator);
            visit_nodes(visitor, node->payload.parsed_block.continuations_vars);
            visit_nodes(visitor, node->payload.parsed_block.continuations);
            break;
        }
        case Root_TAG: {
            visit_nodes(visitor, node->payload.root.declarations);
            break;
        }
        case Let_TAG: {
            visit_nodes(visitor, node->payload.let.variables);
            visit(node->payload.let.instruction);
            break;
        }
        case PrimOp_TAG: {
            visit_nodes(visitor, node->payload.prim_op.operands);
            break;
        }
        case Call_TAG: {
            visit(node->payload.call_instr.callee);
            visit_nodes(visitor, node->payload.call_instr.args);
            break;
        }
        case If_TAG: {
            visit_nodes(visitor, node->payload.if_instr.yield_types);
            visit(node->payload.if_instr.condition);
            visit(node->payload.if_instr.if_true);
            visit(node->payload.if_instr.if_false);
            break;
        }
        case Match_TAG: {
            visit_nodes(visitor, node->payload.match_instr.yield_types);
            visit(node->payload.match_instr.inspect);
            visit_nodes(visitor, node->payload.match_instr.cases);
            visit(node->payload.match_instr.default_case);
            break;
        }
        case Loop_TAG: {
            visit_nodes(visitor, node->payload.loop_instr.yield_types);
            visit_nodes(visitor, node->payload.loop_instr.params);
            visit(node->payload.loop_instr.body);
            visit_nodes(visitor, node->payload.loop_instr.initial_args);
            break;
        }
        case Return_TAG: {
            if (visitor->visit_return_fn_annotation)
                visit(node->payload.fn_ret.fn);
            visit_nodes(visitor, node->payload.fn_ret.values);
            break;
        }
        case Jump_TAG: {
            if (visitor->visit_cf_targets)
                visit(node->payload.jump.target);
            visit_nodes(visitor, node->payload.jump.args);
            break;
        }
        case Branch_TAG: {
            visit(node->payload.branch.condition);
            if (visitor->visit_cf_targets) {
                visit(node->payload.branch.true_target);
                visit(node->payload.branch.false_target);
            }
            visit_nodes(visitor, node->payload.branch.args);
            break;
        }
        case Join_TAG: {
            visit_nodes(visitor, node->payload.join.args);
            break;
        }
        case Callf_TAG: {
            if (visitor->visit_callf_return_fn_annotation)
                visit(node->payload.callf.ret_fn);
            visit(node->payload.callf.callee);
            visit_nodes(visitor, node->payload.callf.args);
            break;
        }
        case Callc_TAG: {
            if (visitor->visit_cf_targets)
                visit(node->payload.callc.ret_cont);
            visit(node->payload.callc.callee);
            visit_nodes(visitor, node->payload.callc.args);
            break;
        }
        default: break;
    }
}
