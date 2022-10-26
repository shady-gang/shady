#include "shady/ir.h"
#include "log.h"
#include "visit.h"

#include "analysis/scope.h"

#include <assert.h>

#define visit(t) if (t && visitor->visit_fn) visitor->visit_fn(visitor, t);

void visit_nodes(Visitor* visitor, Nodes nodes) {
    for (size_t i = 0; i < nodes.count; i++) {
         visit(nodes.nodes[i]);
    }
}

void visit_fn_blocks_except_head(Visitor* visitor, const Node* function) {
    assert(function->tag == Lambda_TAG);
    assert(function->payload.lam.tier == FnTier_Function);
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
        case InvalidNode_TAG:
        case MaskType_TAG:
        case NoRet_TAG:
        case Unit_TAG:
        case Int_TAG:
        case Float_TAG:
        case Bool_TAG: break;
        case JoinPointType_TAG: {
            visit_nodes(visitor, node->payload.join_point_type.yield_types);
            break;
        }
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
        case NominalType_TAG: {
            visit(node->payload.nom_type.body);
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
            if (visitor->visit_fn_addr)
                visit(node->payload.fn_addr.fn);
            break;
        }
        // Instructions
        case Let_TAG: {
            visit(node->payload.let.instruction);
            visit(node->payload.let.tail);
            break;
        }
        case PrimOp_TAG: {
            visit_nodes(visitor, node->payload.prim_op.operands);
            break;
        }
        case Call_TAG: {
            if (visitor->visit_continuations || node->payload.call_instr.is_indirect)
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
            visit_nodes(visitor, node->payload.loop_instr.initial_args);
            visit(node->payload.loop_instr.body);
            break;
        }
        case Control_TAG: {
            visit_nodes(visitor, node->payload.control.yield_types);
            if (visitor->visit_continuations && node->payload.control.inside->payload.lam.tier == FnTier_BasicBlock) {
                visit(node->payload.control.inside);
            }
            break;
        }
        // Terminators
        case TailCall_TAG: {
            visit(node->payload.tail_call.target);
            break;
        }
        case Branch_TAG: {
            switch (node->payload.branch.branch_mode) {
                case BrJump: {
                    if (visitor->visit_continuations)
                        visit(node->payload.branch.target);
                    break;
                } case BrIfElse: {
                    visit(node->payload.branch.branch_condition);
                    if (visitor->visit_continuations) {
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
            visit(node->payload.join.join_point);
            visit_nodes(visitor, node->payload.join.args);
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
        case Lambda_TAG: {
            visit_nodes(visitor, node->payload.lam.annotations);
            visit_nodes(visitor, node->payload.lam.params);
            visit_nodes(visitor, node->payload.lam.return_types);
            visit(node->payload.lam.body);

            if (visitor->visit_fn_scope_rpo && node->payload.lam.tier == FnTier_Function)
                visit_fn_blocks_except_head(visitor, node);

            // TODO flag for visiting children_conts ?

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
    }
}

void visit_module(Visitor* visitor, Module* mod) {
    Nodes decls = get_module_declarations(mod);
    visit_nodes(visitor, decls);
}
