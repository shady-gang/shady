#include "implem.h"
#include "type.h"

#include "list.h"

Nodes rewrite_nodes(Rewriter* rewriter, Nodes old_nodes) {
    size_t count = old_nodes.count;
    const Node* arr[count];
    for (size_t i = 0; i < count; i++)
        arr[i] = rewriter->rewrite_fn(rewriter, old_nodes.nodes[i]);
    return nodes(rewriter->dst_arena, count, arr);
}

Strings import_strings(Rewriter* rewriter, Strings old_strings) {
    size_t count = old_strings.count;
    String arr[count];
    for (size_t i = 0; i < count; i++)
        arr[i] = string(rewriter->dst_arena, old_strings.strings[i]);
    return strings(rewriter->dst_arena, count, arr);
}

const Node* rewrite_node(Rewriter* rewriter, const Node* node) { return rewriter->rewrite_fn(rewriter, node); }

const Node* recreate_node_identity(Rewriter* rewriter, const Node* node) {
    if (node == NULL)
        return NULL;
    switch (node->tag) {
        case Root_TAG: {
            size_t count = node->payload.root.variables.count;
            const Node* new_variables[count];
            const Node* new_definitions[count];

            for (size_t i = 0; i < count; i++) {
                new_variables[i] = rewriter->rewrite_fn(rewriter, node->payload.root.variables.nodes[i]);
                new_definitions[i] = rewriter->rewrite_fn(rewriter, node->payload.root.definitions.nodes[i]);
            }

            return root(rewriter->dst_arena, (Root) {
                .variables = nodes(rewriter->dst_arena, count, new_variables),
                .definitions = nodes(rewriter->dst_arena, count, new_definitions)
            });
        }
        case Function_TAG:      return fn(rewriter->dst_arena, (Function) {
           .return_type = rewriter->rewrite_fn(rewriter, node->payload.fn.return_type),
           .instructions = rewrite_nodes(rewriter, node->payload.fn.instructions),
           .params = rewrite_nodes(rewriter, node->payload.fn.params),
        });
        case UntypedNumber_TAG: return untyped_number(rewriter->dst_arena, (UntypedNumber) {
            .plaintext = string(rewriter->dst_arena, node->payload.untyped_number.plaintext)
        });
        case Variable_TAG:      return var(rewriter->dst_arena, (Variable) {
            .name = string(rewriter->dst_arena, node->payload.var.name),
            .type = rewriter->rewrite_fn(rewriter, node->payload.var.type)
        });
        case VariableDecl_TAG:  return var_decl(rewriter->dst_arena, (VariableDecl) {
            .address_space = node->payload.var_decl.address_space,
            .variable = rewriter->rewrite_fn(rewriter, node->payload.var_decl.variable),
            .init = rewriter->rewrite_fn(rewriter, node->payload.var_decl.init),
        });
        case Call_TAG:          return call(rewriter->dst_arena, (Call) {
            .callee = rewriter->rewrite_fn(rewriter, node->payload.call.callee),
            .args = rewrite_nodes(rewriter, node->payload.call.args)
        });
        case Let_TAG:           return let(rewriter->dst_arena, (Let) {
            .variables = rewrite_nodes(rewriter, node->payload.let.variables),
            .target = rewriter->rewrite_fn(rewriter, node->payload.let.target)
        });
        case Return_TAG:        return fn_ret(rewriter->dst_arena, (Return) {
            .values = rewrite_nodes(rewriter, node->payload.fn_ret.values)
        });
        case PrimOp_TAG:        return primop(rewriter->dst_arena, (PrimOp) {
            .op = node->payload.primop.op,
            .args = rewrite_nodes(rewriter, node->payload.primop.args)
        });
        case NoRet_TAG:         return noret_type(rewriter->dst_arena);
        case Void_TAG:          return void_type(rewriter->dst_arena);
        case Int_TAG:           return int_type(rewriter->dst_arena);
        case Float_TAG:         return float_type(rewriter->dst_arena);
        case RecordType_TAG:    return record_type(rewriter->dst_arena, (RecordType) {
                                    .name = string(rewriter->dst_arena, node->payload.record_type.name),
                                    .members = rewrite_nodes(rewriter, node->payload.record_type.members)});
        case ContType_TAG:      return cont_type(rewriter->dst_arena, (ContType) { .param_types = rewrite_nodes(rewriter, node->payload.cont_type.param_types) });
        case FnType_TAG:        return fn_type(rewriter->dst_arena, (FnType) {
                                    .param_types = rewrite_nodes(rewriter, node->payload.fn_type.param_types),
                                    .return_type = rewriter->rewrite_fn(rewriter, node->payload.fn_type.return_type)});
        case PtrType_TAG:       return ptr_type(rewriter->dst_arena, (PtrType) {
                                    .address_space = node->payload.ptr_type.address_space,
                                    .pointed_type = rewriter->rewrite_fn(rewriter, node->payload.ptr_type.pointed_type)});
        case QualifiedType_TAG: return qualified_type(rewriter->dst_arena, (QualifiedType) {
                                    .is_uniform = node->payload.qualified_type.is_uniform,
                                    .type = rewriter->rewrite_fn(rewriter, node->payload.qualified_type.type)});
        default: error("unhandled node for rewrite");
    }
}