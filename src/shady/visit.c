#include "shady/ir.h"
#include "log.h"
#include "visit.h"

#include "analysis/scope.h"

#include <assert.h>

#define visit(t) if (t && visitor->visit_fn) visitor->visit_fn(visitor, t);

static void visit_nodes(Visitor* visitor, Nodes nodes) {
    for (size_t i = 0; i < nodes.count; i++) {
         visit(nodes.nodes[i]);
    }
}

void visit_fn_blocks_except_head(Visitor* visitor, const Node* function) {
    assert(function->tag == Function_TAG);
    assert(!function->payload.fn.is_basic_block);
    Scope scope = build_scope(function);
    assert(scope.rpo[0]->node == function);
    for (size_t i = 1; i < scope.size; i++) {
        visit(scope.rpo[i]->node);
    }
    dispose_scope(&scope);
}

#pragma GCC diagnostic error "-Wswitch"

void visit_children(Visitor* visitor, const Node* node) {
    if (!node_type_has_payload[node->tag])
        return;

    switch(node->tag) {
        // Types
        case MaskType_TAG:
        case NoRet_TAG:
        case Unit_TAG:
        case Int_TAG:
        case Float_TAG:
        case Bool_TAG: break;
        case RecordType_TAG: {
            visit_nodes(visitor, node->payload.record_type.members);
            break;
        }
        case FnType_TAG: {
            visit_nodes(visitor, node->payload.fn_type.param_types);
            visit_nodes(visitor, node->payload.fn_type.return_types);
            break;
        }
        case PtrType_TAG: {
            visit(node->payload.ptr_type.pointed_type);
            break;
        }
        case QualifiedType_TAG: {
            visit(node->payload.qualified_type.type);
            break;
        }
        case ArrType_TAG: {
            visit(node->payload.arr_type.element_type);
            visit(node->payload.arr_type.size);
            break;
        }
        case PackType_TAG: {
            visit(node->payload.pack_type.element_type);
            break;
        }
        // Values
        case Variable_TAG: {
            visit(node->payload.var.type);
            break;
        }
        case Unbound_TAG:
        case IntLiteral_TAG:
        case UntypedNumber_TAG:
        case True_TAG:
        case False_TAG:
        case StringLiteral_TAG: break;
        case Tuple_TAG: {
            visit_nodes(visitor, node->payload.tuple.contents);
            break;
        }
        case ArrayLiteral_TAG: {
            visit(node->payload.arr_lit.element_type);
            visit_nodes(visitor, node->payload.arr_lit.contents);
            break;
        }
        case RefDecl_TAG: {
            if (visitor->visit_referenced_decls)
                visit(node->payload.ref_decl.decl);
        }
        case FnAddr_TAG: {
            visit(node->payload.fn_addr.fn);
            break;
        }
        // Instructions
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
            // visit(node->payload.call_instr.callee);
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
        // Terminators
        case Branch_TAG: {
            switch (node->payload.branch.branch_mode) {
                case BrTailcall:
                case BrJump: visit(node->payload.branch.target); break;
                case BrIfElse: {
                    visit(node->payload.branch.branch_condition);
                    if (visitor->visit_cf_targets) {
                        visit(node->payload.branch.true_target);
                        visit(node->payload.branch.false_target);
                    }
                    break;
                }
                case BrSwitch: error("TODO");
            }

            visit_nodes(visitor, node->payload.branch.args);
            break;
        }
        case Join_TAG: {
            if (visitor->visit_cf_targets)
                visit(node->payload.join.join_at);
            visit_nodes(visitor, node->payload.join.args);
            break;
        }
        case Callc_TAG: {
            if (visitor->visit_cf_targets) {
                visit(node->payload.callc.ret_cont);
                //visit(node->payload.callc.callee);
            }
            visit_nodes(visitor, node->payload.callc.args);
            break;
        }
        case Return_TAG: {
            if (visitor->visit_return_fn_annotation)
                visit(node->payload.fn_ret.fn);
            visit_nodes(visitor, node->payload.fn_ret.values);
            break;
        }
        case MergeConstruct_TAG: {
            visit_nodes(visitor, node->payload.merge_construct.args);
            break;
        }
        case Unreachable_TAG: break;

        // Decls
        case Constant_TAG: {
            visit_nodes(visitor, node->payload.constant.annotations);
            visit(node->payload.constant.value);
            visit(node->payload.constant.type_hint);
            break;
        }
        case Function_TAG: {
            visit_nodes(visitor, node->payload.fn.annotations);
            visit_nodes(visitor, node->payload.fn.params);
            visit_nodes(visitor, node->payload.fn.return_types);
            visit(node->payload.fn.block);

            if (visitor->visit_fn_scope_rpo && !node->payload.fn.is_basic_block)
                visit_fn_blocks_except_head(visitor, node);

            break;
        }
        case GlobalVariable_TAG: {
            visit_nodes(visitor, node->payload.global_variable.annotations);
            visit(node->payload.global_variable.type);
            visit(node->payload.global_variable.init);
            break;
        }
        // Misc.
        case Annotation_TAG: {
            switch (node->payload.annotation.payload_type) {
                case AnPayloadNone: break;
                case AnPayloadValue: visit(node->payload.annotation.value); break;
                case AnPayloadValues:
                case AnPayloadMap: visit_nodes(visitor, node->payload.annotation.values); break;
                default: error("TODO");
            }
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
    }
}
