#include "rewrite.h"

#include "log.h"
#include "ir_private.h"
#include "portability.h"
#include "type.h"

#include "dict.h"

#include <assert.h>

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

Rewriter create_rewriter(IrArena* src, IrArena* dst, RewriteFn fn) {
    return (Rewriter) {
        .src_arena = src,
        .dst_arena = dst,
        .rewrite_fn = fn,
        .processed = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node)
    };
}

Rewriter create_importer(IrArena* src, IrArena* dst) {
    return create_rewriter(src, dst, recreate_node_identity);
}

static const Node* recreate_node_substitutions_only(Rewriter* rewriter, const Node* node) {
    assert(rewriter->dst_arena == rewriter->src_arena);
    const Node* found = rewriter->processed ? search_processed(rewriter, node) : NULL;
    if (found)
        return found;

    if (is_declaration(node))
        return node;
    if (node->tag == Variable_TAG)
        return node;
    return recreate_node_identity(rewriter, node);
}

Rewriter create_substituter(IrArena* arena) {
    return create_rewriter(arena, arena, recreate_node_substitutions_only);
}

void destroy_rewriter(Rewriter* r) {
    assert(r->processed);
    destroy_dict(r->processed);
}

const Node* rewrite_node(Rewriter* rewriter, const Node* node) {
    assert(rewriter->rewrite_fn);
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

Strings import_strings(IrArena* dst_arena, Strings old_strings) {
    size_t count = old_strings.count;
    LARRAY(String, arr, count);
    for (size_t i = 0; i < count; i++)
        arr[i] = string(dst_arena, old_strings.strings[i]);
    return strings(dst_arena, count, arr);
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
    assert(old->arena == ctx->src_arena);
    assert(new->arena == ctx->dst_arena);
#ifndef NDEBUG
    const Node* found = search_processed(ctx, old);
    if (found) {
        error_print("Trying to replace ");
        error_node(old);
        error_print(" with ");
        error_node(new);
        error_print(" but there was already ");
        error_node(found);
        error_print("\n");
        error("The same node got processed twice !");
    }
#endif
    assert(ctx->processed && "this rewriter has no processed cache");
    bool r = insert_dict_and_get_result(const Node*, const Node*, ctx->processed, old, new);
    assert(r);
}

void register_processed_list(Rewriter* rewriter, Nodes old, Nodes new) {
    assert(old.count == new.count);
    for (size_t i = 0; i < old.count; i++)
        register_processed(rewriter, old.nodes[i], new.nodes[i]);
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
    switch (old->tag) {
        case GlobalVariable_TAG: new = global_var(rewriter->dst_arena, rewrite_nodes(rewriter, old->payload.global_variable.annotations), rewrite_node(rewriter, old->payload.global_variable.type), old->payload.global_variable.name, old->payload.global_variable.address_space); break;
        case Constant_TAG: new = constant(rewriter->dst_arena, rewrite_nodes(rewriter, old->payload.constant.annotations), old->payload.constant.name); break;
        case Lambda_TAG: {
            Nodes new_params = recreate_variables(rewriter, old->payload.lam.params);
            switch (old->payload.lam.tier) {
                case FnTier_Lambda:
                    new = lambda(rewriter->dst_arena, new_params);
                    break;
                case FnTier_BasicBlock:
                    new = basic_block(rewriter->dst_arena, new_params, old->payload.lam.name);
                    break;
                case FnTier_Function:
                    new = function(rewriter->dst_arena, new_params, old->payload.lam.name, rewrite_nodes(rewriter, old->payload.lam.annotations), rewrite_nodes(rewriter, old->payload.lam.return_types));
                    break;
            }
            assert(new && new->tag == Lambda_TAG);
            register_processed_list(rewriter, old->payload.lam.params, new->payload.lam.params);
            break;
        }
        default: error("not a decl");
    }
    assert(new);
    register_processed(rewriter, old, new);
    return new;
}

void recreate_decl_body_identity(Rewriter* rewriter, const Node* old, Node* new) {
    // assert(is_declaration(new) && is_declaration(old));
    switch (old->tag) {
        case GlobalVariable_TAG: {
            new->payload.global_variable.init = rewrite_node(rewriter, old->payload.global_variable.init);
            break;
        }
        case Constant_TAG: {
            new->payload.constant.type_hint = rewrite_node(rewriter, old->payload.constant.type_hint);
            new->payload.constant.value     = rewrite_node(rewriter, old->payload.constant.value);
            new->type                       = rewrite_node(rewriter, new->payload.constant.value->type);
            break;
        }
        case Lambda_TAG: {
            assert(new->payload.lam.body == NULL);
            new->payload.lam.body = rewrite_node(rewriter, old->payload.lam.body);
            break;
        }
        default: error("not a decl");
    }
}

#pragma GCC diagnostic error "-Wswitch"

const Node* recreate_node_identity(Rewriter* rewriter, const Node* node) {
    if (node == NULL)
        return NULL;

    const Node* already_done_before = rewriter->processed ? search_processed(rewriter, node) : NULL;
    if (already_done_before)
        return already_done_before;

    switch (node->tag) {
        case InvalidNode_TAG:   assert(false);
        case NoRet_TAG:         return noret_type(rewriter->dst_arena);
        case Int_TAG:           return int_type(rewriter->dst_arena, node->payload.int_type);
        case Bool_TAG:          return bool_type(rewriter->dst_arena);
        case Float_TAG:         return float_type(rewriter->dst_arena);
        case Unit_TAG:          return unit_type(rewriter->dst_arena);
        case MaskType_TAG:      return mask_type(rewriter->dst_arena);
        case JoinPointType_TAG: return join_point_type(rewriter->dst_arena, (JoinPointType) { .yield_types = rewrite_nodes(rewriter, node->payload.join_point_type.yield_types )});
        case RecordType_TAG:    return record_type(rewriter->dst_arena, (RecordType) {
                                    .members = rewrite_nodes(rewriter, node->payload.record_type.members),
                                    .names = import_strings(rewriter->dst_arena, node->payload.record_type.names),
                                    .special = node->payload.record_type.special});
        case FnType_TAG:        return fn_type(rewriter->dst_arena, (FnType) {
                                    .tier = node->payload.fn_type.tier,
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
        case PackType_TAG:      return pack_type(rewriter->dst_arena, (PackType) {
                                    .element_type = rewrite_node(rewriter, node->payload.pack_type.element_type),
                                    .width = node->payload.pack_type.width
        });
        case NominalType_TAG: {
            Type* new = nominal_type(rewriter->dst_arena, node->payload.nom_type.name);
                register_processed(rewriter, node, new);
            new->payload.nom_type.body = rewrite_node(rewriter, node->payload.nom_type.body);
            return new;
        }

        case Variable_TAG:      error("We expect variables to be available for us in the `processed` set");
        case Unbound_TAG:       return unbound(rewriter->dst_arena, (Unbound) { .name = string(rewriter->dst_arena, node->payload.unbound.name) });
        case UntypedNumber_TAG: return untyped_number(rewriter->dst_arena, (UntypedNumber) { .plaintext = string(rewriter->dst_arena, node->payload.untyped_number.plaintext) });
        case IntLiteral_TAG:    return int_literal(rewriter->dst_arena, node->payload.int_literal);
        case True_TAG:          return true_lit(rewriter->dst_arena);
        case False_TAG:         return false_lit(rewriter->dst_arena);
        case StringLiteral_TAG: return string_lit(rewriter->dst_arena, (StringLiteral) { .string = string(rewriter->dst_arena, node->payload.string_lit.string )});
        case ArrayLiteral_TAG:  return arr_lit(rewriter->dst_arena, (ArrayLiteral) {
            .element_type = rewrite_node(rewriter, node->payload.arr_lit.element_type),
            .contents = rewrite_nodes(rewriter, node->payload.arr_lit.contents)
        });
        case Tuple_TAG:         return tuple(rewriter->dst_arena, rewrite_nodes(rewriter, node->payload.tuple.contents));
        case FnAddr_TAG:        return fn_addr(rewriter->dst_arena, (FnAddr) { .fn = rewrite_node(rewriter, node->payload.fn_addr.fn) });
        case RefDecl_TAG:       return ref_decl(rewriter->dst_arena, (RefDecl) { .decl = rewrite_node(rewriter, node->payload.ref_decl.decl) });

        case Let_TAG: {
            assert(!node->payload.let.is_mutable);
            const Node* ninstruction = rewrite_node(rewriter, node->payload.let.instruction);
            const Node* tail = rewrite_node(rewriter, node->payload.let.tail);
            return let(rewriter->dst_arena, false, ninstruction, tail);
        }
        case PrimOp_TAG:        return prim_op(rewriter->dst_arena, (PrimOp) {
            .op = node->payload.prim_op.op,
            .operands = rewrite_nodes(rewriter, node->payload.prim_op.operands)
        });
        case Call_TAG:          return call_instr(rewriter->dst_arena, (Call) {
            .is_indirect = node->payload.call_instr.is_indirect,
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
            const Node* nbody = rewrite_node(rewriter, node->payload.loop_instr.body);

            return loop_instr(rewriter->dst_arena, (Loop) {
                .yield_types = rewrite_nodes(rewriter, node->payload.loop_instr.yield_types),
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
        case TailCall_TAG: return tail_call(rewriter->dst_arena, (TailCall) {
            .target = rewrite_node(rewriter, node->payload.tail_call.target),
            .args = rewrite_nodes(rewriter, node->payload.tail_call.args),
        });
        case Branch_TAG: switch (node->payload.branch.branch_mode) {
            case BrJump: return branch(rewriter->dst_arena, (Branch) {
                .branch_mode = node->payload.branch.branch_mode,

                .target = rewrite_node(rewriter, node->payload.branch.target),
                .args = rewrite_nodes(rewriter, node->payload.branch.args)
            });
            case BrIfElse: return branch(rewriter->dst_arena, (Branch) {
                .branch_mode = node->payload.branch.branch_mode,

                .branch_condition = rewrite_node(rewriter, node->payload.branch.branch_condition),
                .true_target = rewrite_node(rewriter, node->payload.branch.true_target),
                .false_target = rewrite_node(rewriter, node->payload.branch.false_target),
                .args = rewrite_nodes(rewriter, node->payload.branch.args)
            });
            case BrSwitch: return branch(rewriter->dst_arena, (Branch) {
                .branch_mode = node->payload.branch.branch_mode,

                .switch_value = rewrite_node(rewriter, node->payload.branch.switch_value),
                .default_target = rewrite_node(rewriter, node->payload.branch.default_target),
                .case_values = rewrite_nodes(rewriter, node->payload.branch.case_values),
                .case_targets = rewrite_nodes(rewriter, node->payload.branch.case_targets)
            });
            default: SHADY_UNREACHABLE;
        }
        case Control_TAG:     return control(rewriter->dst_arena, (Control) {
            .yield_types = rewrite_nodes(rewriter, node->payload.control.yield_types),
            .inside = rewrite_node(rewriter, node->payload.control.inside),
        });
        case Join_TAG:        return join(rewriter->dst_arena, (Join) {
            .join_point = rewrite_node(rewriter, node->payload.join.join_point),
            .args = rewrite_nodes(rewriter, node->payload.join.args)
        });
        case Return_TAG:        return fn_ret(rewriter->dst_arena, (Return) {
            .fn = rewrite_node(rewriter, node->payload.fn_ret.fn),
            .values = rewrite_nodes(rewriter, node->payload.fn_ret.values)
        });
        case Unreachable_TAG:   return unreachable(rewriter->dst_arena);
        case MergeConstruct_TAG: return merge_construct(rewriter->dst_arena, (MergeConstruct) {
            .construct = node->payload.merge_construct.construct,
            .args = rewrite_nodes(rewriter, node->payload.merge_construct.args)
        });

        case GlobalVariable_TAG:
        case Constant_TAG:
        case Lambda_TAG: {
            Node* new = recreate_decl_header_identity(rewriter, node);
            recreate_decl_body_identity(rewriter, node, new);
            return new;
        }

        case Root_TAG: {
            Nodes decls = rewrite_nodes(rewriter, node->payload.root.declarations);
            return root(rewriter->dst_arena, (Root) {
                .declarations = decls,
            });
        }
        case Annotation_TAG: switch (node->payload.annotation.payload_type) {
            case AnPayloadNone: return annotation(rewriter->dst_arena, (Annotation) {
                                    .payload_type = node->payload.annotation.payload_type,
                                    .name = string(rewriter->dst_arena, node->payload.annotation.name),
                                });
            case AnPayloadValue: return annotation(rewriter->dst_arena, (Annotation) {
                                    .payload_type = node->payload.annotation.payload_type,
                                    .name = string(rewriter->dst_arena, node->payload.annotation.name),
                                    .value = rewrite_node(rewriter, node->payload.annotation.value)
                                });
            case AnPayloadValues: return annotation(rewriter->dst_arena, (Annotation) {
                                    .payload_type = node->payload.annotation.payload_type,
                                    .name = string(rewriter->dst_arena, node->payload.annotation.name),
                                    .values = rewrite_nodes(rewriter, node->payload.annotation.values)
                                });
            case AnPayloadMap: return annotation(rewriter->dst_arena, (Annotation) {
                                    .payload_type = node->payload.annotation.payload_type,
                                    .name = string(rewriter->dst_arena, node->payload.annotation.name),
                                    .labels = import_strings(rewriter->dst_arena, node->payload.annotation.labels),
                                    .values = rewrite_nodes(rewriter, node->payload.annotation.values)
                                });
            default: error("Unknown annotation payload type");
        }
    }
    SHADY_UNREACHABLE;
}
