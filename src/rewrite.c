#include "rewrite.h"

#include "log.h"
#include "portability.h"
#include "type.h"

#include "dict.h"

#include <assert.h>

const Node* rewrite_node(Rewriter* rewriter, const Node* node) {
    if (node)
        return rewriter->rewrite_fn(rewriter, node);
    else
        return NULL;
}

Nodes rewrite_nodes(Rewriter* rewriter, Nodes old_nodes) {
    size_t count = old_nodes.count;
    LARRAY(const Node*, arr, count);
    for (size_t i = 0; i < count; i++)
        arr[i] = rewrite_node(rewriter, old_nodes.nodes[i]);
    return nodes(rewriter->dst_arena, count, arr);
}

const Node* search_processed(const Rewriter* ctx, const Node* old) {
    assert(ctx->processed && "this rewriter has no processed cache");
    const Node** found = find_value_dict(const Node*, const Node*, ctx->processed, old);
    return found ? *found : NULL;
}

const Node* find_processed(const Rewriter* ctx, const Node* old) {
    const Node* found = search_processed(ctx, old);
    assert(found && "this node was supposed to have been processed before");
    return found;
}

void register_processed(Rewriter* ctx, const Node* old, const Node* new) {
    assert(ctx->processed && "this rewriter has no processed cache");
    bool r = insert_dict_and_get_result(const Node*, const Node*, ctx->processed, old, new);
    assert(r && "registered the same node as processed twice");
}

const Node* recreate_variable(Rewriter* rewriter, const Node* old) {
    assert(old->tag == Variable_TAG);
    return var(rewriter->dst_arena, rewrite_node(rewriter, old->payload.var.type), old->payload.var.name);
}

Nodes recreate_variables(Rewriter* rewriter, Nodes old) {
    LARRAY(const Node*, nvars, old.count);
    for (size_t i = 0; i < old.count; i++)
        nvars[i] = recreate_variable(rewriter, old.nodes[i]);
    return nodes(rewriter->dst_arena, old.count, nvars);
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

Node* recreate_decl_header_identity(Rewriter* rewriter, const Node* old) {
    Node* new = NULL;
    switch (old->tag) {\
        case GlobalVariable_TAG: new = global_var(rewriter->dst_arena, rewrite_node(rewriter, old->payload.global_variable.type), old->payload.global_variable.name, old->payload.global_variable.address_space); break;
        case Constant_TAG: new = constant(rewriter->dst_arena, old->payload.constant.name); break;
        case Function_TAG: new = fn(rewriter->dst_arena, old->payload.fn.atttributes, old->payload.fn.name, recreate_variables(rewriter, old->payload.fn.params), rewrite_nodes(rewriter, old->payload.fn.return_types)); break;
        default: error("not a decl");
    }
    assert(new);
    register_processed(rewriter, old, new);
    return new;
}

void recreate_decl_body_identity(Rewriter* rewriter, const Node* old, Node* new) {
    switch (old->tag) {
        case Constant_TAG: {
            new->payload.constant.type_hint = rewrite_node(rewriter, old->payload.constant.type_hint);
            new->payload.constant.value     = rewrite_node(rewriter, old->payload.constant.value);
            new->type = new->payload.constant.value->type;
            break;
        }
        case Function_TAG: {
            //struct Dict* old_processed = rewriter->processed;
            //rewriter->processed = clone_dict(rewriter->processed);
            for (size_t i = 0; i < new->payload.fn.params.count; i++)
                register_processed(rewriter, old->payload.fn.params.nodes[i], new->payload.fn.params.nodes[i]);
            new->payload.fn.block = rewrite_node(rewriter, old->payload.fn.block);
            //destroy_dict(rewriter->processed);
            //rewriter->processed = old_processed;
            break;
        }
        case GlobalVariable_TAG: {
            new->payload.global_variable.init = rewrite_node(rewriter, old->payload.global_variable.init);
            break;
        }
        default: error("not a decl");
    }
}

const Node* recreate_node_identity(Rewriter* rewriter, const Node* node) {
    if (node == NULL)
        return NULL;

    const Node* already_done_before = rewriter->processed ? search_processed(rewriter, node) : NULL;
    if (already_done_before)
        return already_done_before;

    switch (node->tag) {
        case Root_TAG: {
            Nodes decls = rewrite_nodes(rewriter, node->payload.root.declarations);

            if (rewriter->rewrite_decl_body) {
                for (size_t i = 0; i < decls.count; i++)
                    rewriter->rewrite_decl_body(rewriter, node->payload.root.declarations.nodes[i], (Node*) decls.nodes[i]);
            }

            return root(rewriter->dst_arena, (Root) {
                .declarations = decls,
            });
        }
        case Block_TAG:         return block(rewriter->dst_arena, (Block) {
            .instructions = rewrite_nodes(rewriter, node->payload.block.instructions),
            .terminator = rewrite_node(rewriter, node->payload.block.terminator)
        });
        case GlobalVariable_TAG:
        case Constant_TAG:
        case Function_TAG:      error("Declarations are not handled");
        case UntypedNumber_TAG: return untyped_number(rewriter->dst_arena, (UntypedNumber) {
            .plaintext = string(rewriter->dst_arena, node->payload.untyped_number.plaintext)
        });
        case IntLiteral_TAG:    return int_literal(rewriter->dst_arena, node->payload.int_literal);
        case True_TAG:          return true_lit(rewriter->dst_arena);
        case False_TAG:         return false_lit(rewriter->dst_arena);
        case Variable_TAG:      error("We expect variables to be available for us in the `processed` set");
        case Let_TAG:           {
            const Node* ninstruction = rewrite_node(rewriter, node->payload.let.instruction);
            const Nodes output_types = typecheck_instruction(rewriter->dst_arena, ninstruction);
            Nodes oldvars = node->payload.let.variables;
            assert(output_types.count == oldvars.count);

            // TODO: pull into a helper fn
            LARRAY(const char*, old_names, oldvars.count);
            for (size_t i = 0; i < oldvars.count; i++) {
                assert(oldvars.nodes[i]->tag == Variable_TAG);
                old_names[i] = oldvars.nodes[i]->payload.var.name;
            }

            const Node* rewritten = let(rewriter->dst_arena, ninstruction, output_types.count, old_names);
            for (size_t i = 0; i < oldvars.count; i++)
                register_processed(rewriter, oldvars.nodes[i], rewritten->payload.let.variables.nodes[i]);

            return rewritten;
        }
        case PrimOp_TAG:        return prim_op(rewriter->dst_arena, (PrimOp) {
            .op = node->payload.prim_op.op,
            .operands = rewrite_nodes(rewriter, node->payload.prim_op.operands)
        });
        case Call_TAG:          return call_instr(rewriter->dst_arena, (Call) {
            .callee = rewrite_node(rewriter, node->payload.call_instr.callee),
            .args = rewrite_nodes(rewriter, node->payload.call_instr.args)
        });
        case If_TAG:            return if_instr(rewriter->dst_arena, (If) {
            .yield_types = rewrite_nodes(rewriter, node->payload.if_instr.yield_types),
            .condition = rewrite_node(rewriter, node->payload.if_instr.condition),
            .if_true = rewrite_node(rewriter, node->payload.if_instr.if_true),
            .if_false = rewrite_node(rewriter, node->payload.if_instr.if_false),
        });
        case Loop_TAG: {
            Nodes oparams = node->payload.loop_instr.params;
            Nodes nparams = rewrite_nodes(rewriter, oparams);

            //struct Dict* old_processed = rewriter->processed;
            //rewriter->processed = clone_dict(rewriter->processed);
            for (size_t i = 0; i < oparams.count; i++)
                register_processed(rewriter, oparams.nodes[i], nparams.nodes[i]);
            const Node* nbody = rewrite_node(rewriter, node->payload.loop_instr.body);
            //destroy_dict(rewriter->processed);
            //rewriter->processed = old_processed;

            return loop_instr(rewriter->dst_arena, (Loop) {
                .yield_types = rewrite_nodes(rewriter, node->payload.loop_instr.yield_types),
                .params = nparams,
                .initial_args = rewrite_nodes(rewriter, node->payload.loop_instr.initial_args),
                .body = nbody,
            });
        }
        case Match_TAG:         return match_instr(rewriter->dst_arena, (Match) {
            .yield_types = rewrite_nodes(rewriter, node->payload.match_instr.yield_types),
            .inspect = rewrite_node(rewriter, node->payload.match_instr.inspect),
            .literals = rewrite_nodes(rewriter, node->payload.match_instr.literals),
            .cases = rewrite_nodes(rewriter, node->payload.match_instr.cases),
            .default_case = rewrite_node(rewriter, node->payload.match_instr.default_case),
        });
        case Branch_TAG: switch (node->payload.branch.branch_mode) {
            case BrTailcall:
            case BrJump: return branch(rewriter->dst_arena, (Branch) {
                .branch_mode = node->payload.branch.branch_mode,
                .yield = node->payload.branch.yield,

                .target = rewrite_node(rewriter, node->payload.branch.target),
                .args = rewrite_nodes(rewriter, node->payload.branch.args)
            });
            case BrIfElse: return branch(rewriter->dst_arena, (Branch) {
                .branch_mode = node->payload.branch.branch_mode,
                .yield = node->payload.branch.yield,

                .branch_condition = rewrite_node(rewriter, node->payload.branch.branch_condition),
                .true_target = rewrite_node(rewriter, node->payload.branch.true_target),
                .false_target = rewrite_node(rewriter, node->payload.branch.false_target),
                .args = rewrite_nodes(rewriter, node->payload.branch.args)
            });
            case BrSwitch: return branch(rewriter->dst_arena, (Branch) {
                .branch_mode = node->payload.branch.branch_mode,
                .yield = node->payload.branch.yield,

                .switch_value = rewrite_node(rewriter, node->payload.branch.switch_value),
                .default_target = rewrite_node(rewriter, node->payload.branch.default_target),
                .case_values = rewrite_nodes(rewriter, node->payload.branch.case_values),
                .case_targets = rewrite_nodes(rewriter, node->payload.branch.case_targets)
            });
        }
        case Join_TAG:        return join(rewriter->dst_arena, (Join) {
            .is_indirect = node->payload.join.is_indirect,
            .join_at = rewrite_node(rewriter, node->payload.join.join_at),
            .desired_mask = rewrite_node(rewriter, node->payload.join.desired_mask),
            .args = rewrite_nodes(rewriter, node->payload.branch.args)
        });
        case Return_TAG:        return fn_ret(rewriter->dst_arena, (Return) {
            .fn = NULL,
            .values = rewrite_nodes(rewriter, node->payload.fn_ret.values)
        });
        case Unreachable_TAG:   return unreachable(rewriter->dst_arena);
        case MergeConstruct_TAG: return merge_construct(rewriter->dst_arena, (MergeConstruct) {
            .construct = node->payload.merge_construct.construct,
            .args = rewrite_nodes(rewriter, node->payload.merge_construct.args)
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
                                    .pointed_type = rewrite_node(rewriter, node->payload.ptr_type.pointed_type)});
        case QualifiedType_TAG: return qualified_type(rewriter->dst_arena, (QualifiedType) {
                                    .is_uniform = node->payload.qualified_type.is_uniform,
                                    .type = rewrite_node(rewriter, node->payload.qualified_type.type)});
        case ArrType_TAG:       return arr_type(rewriter->dst_arena, (ArrType) {
                                    .element_type = rewrite_node(rewriter, node->payload.arr_type.element_type),
                                    .size = rewrite_node(rewriter, node->payload.arr_type.size),
        });
        default: error("unhandled node for rewrite %s", node_tags[node->tag]);
    }
}
