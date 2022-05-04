#include "rewrite.h"

#include "log.h"
#include "local_array.h"
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
    bool r = insert_dict(const Node*, const Node*, ctx->processed, old, new);
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
            break;
        }
        case Function_TAG: {
            struct Dict* old_processed = rewriter->processed;
            rewriter->processed = clone_dict(rewriter->processed);
            for (size_t i = 0; i < new->payload.fn.params.count; i++)
                register_processed(rewriter, old->payload.fn.params.nodes[i], new->payload.fn.params.nodes[i]);
            new->payload.fn.block = rewrite_node(rewriter, old->payload.fn.block);
            destroy_dict(rewriter->processed);
            rewriter->processed = old_processed;
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
            Nodes oldvars = node->payload.let.variables;
            Nodes nvars = recreate_variables(rewriter, oldvars);
            const Node* nlet = let(rewriter->dst_arena, (Let) {
                .variables = nvars,
                .instruction = rewrite_node(rewriter, node->payload.let.instruction)
            });
            for (size_t i = 0; i < oldvars.count; i++)
                register_processed(rewriter, oldvars.nodes[i], nvars.nodes[i]);
            return nlet;
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
        case Jump_TAG:          return jump(rewriter->dst_arena, (Jump) {
            .target = rewrite_node(rewriter, node->payload.jump.target),
            .args = rewrite_nodes(rewriter, node->payload.jump.args)
        });
        case Branch_TAG:        return branch(rewriter->dst_arena, (Branch) {
            .condition = rewrite_node(rewriter, node->payload.branch.condition),
            .true_target = rewrite_node(rewriter, node->payload.branch.true_target),
            .false_target = rewrite_node(rewriter, node->payload.branch.false_target),
            .args = rewrite_nodes(rewriter, node->payload.branch.args)
        });
        case Return_TAG:        return fn_ret(rewriter->dst_arena, (Return) {
            .fn = NULL,
            .values = rewrite_nodes(rewriter, node->payload.fn_ret.values)
        });
        case Unreachable_TAG:   return unreachable(rewriter->dst_arena);
        case Merge_TAG:         return merge(rewriter->dst_arena, (Merge) {
            .what = node->payload.merge.what,
            .args = rewrite_nodes(rewriter, node->payload.merge.args)
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
        default: error("unhandled node for rewrite %s", node_tags[node->tag]);
    }
}
