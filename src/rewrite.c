#include "implem.h"
#include "type.h"

#include "list.h"

struct Nodes rewrite_nodes(struct Rewriter* rewriter, struct Nodes old_nodes) {
    size_t count = old_nodes.count;
    const struct Node* arr[count];
    for (size_t i = 0; i < count; i++)
        arr[i] = rewriter->rewrite_node_fn(rewriter, old_nodes.nodes[i]);
    return nodes(rewriter->dst_arena, count, arr);
}

struct Strings import_strings(struct Rewriter* rewriter, struct Strings old_strings) {
    size_t count = old_strings.count;
    String arr[count];
    for (size_t i = 0; i < count; i++)
        arr[i] = string(rewriter->dst_arena, old_strings.strings[i]);
    return strings(rewriter->dst_arena, count, arr);
}

const struct Node* rewrite_node(struct Rewriter* rewriter, const struct Node* node) { return rewriter->rewrite_node_fn(rewriter, node); }
const struct Type* rewrite_type(struct Rewriter* rewriter, const struct Type* type) { return rewriter->rewrite_type_fn(rewriter, type); }

const struct Node* recreate_node_identity(struct Rewriter* rewriter, const struct Node* node) {
    if (node == NULL)
        return NULL;
    switch (node->tag) {
        case Root_TAG: {
            size_t count = node->payload.root.variables.count;
            const struct Node* new_variables[count];
            const struct Node* new_definitions[count];

            for (size_t i = 0; i < count; i++) {
                new_variables[i] = rewriter->rewrite_node_fn(rewriter, node->payload.root.variables.nodes[i]);
                new_definitions[i] = rewriter->rewrite_node_fn(rewriter, node->payload.root.definitions.nodes[i]);
            }

            return root(rewriter->dst_arena, (struct Root) {
                .variables = nodes(rewriter->dst_arena, count, new_variables),
                .definitions = nodes(rewriter->dst_arena, count, new_definitions)
            });
        }
        case Function_TAG:      return fn(rewriter->dst_arena, (struct Function) {
           .return_type = rewriter->rewrite_type_fn(rewriter, node->payload.fn.return_type),
           .instructions = rewrite_nodes(rewriter, node->payload.fn.instructions),
           .params = rewrite_nodes(rewriter, node->payload.fn.params),
        });
        case UntypedNumber_TAG: return untyped_number(rewriter->dst_arena, (struct UntypedNumber) {
            .plaintext = string(rewriter->dst_arena, node->payload.untyped_number.plaintext)
        });
        case Variable_TAG:      return var(rewriter->dst_arena, (struct Variable) {
            .name = string(rewriter->dst_arena, node->payload.var.name),
            .type = rewriter->rewrite_type_fn(rewriter, node->payload.var.type)
        });
        case VariableDecl_TAG:  return var_decl(rewriter->dst_arena, (struct VariableDecl) {
            .address_space = node->payload.var_decl.address_space,
            .variable = rewriter->rewrite_node_fn(rewriter, node->payload.var_decl.variable),
            .init = rewriter->rewrite_node_fn(rewriter, node->payload.var_decl.init),
        });
        case Call_TAG:          return call(rewriter->dst_arena, (struct Call) {
            .callee = rewriter->rewrite_node_fn(rewriter, node->payload.call.callee),
            .args = rewrite_nodes(rewriter, node->payload.call.args)
        });
        case Let_TAG:           return let(rewriter->dst_arena, (struct Let) {
            .variables = rewrite_nodes(rewriter, node->payload.let.variables),
            .target = rewriter->rewrite_node_fn(rewriter, node->payload.let.target)
        });
        case Return_TAG:        return fn_ret(rewriter->dst_arena, (struct Return) {
            .values = rewrite_nodes(rewriter, node->payload.fn_ret.values)
        });
        case PrimOp_TAG:        return primop(rewriter->dst_arena, (struct PrimOp) {
            .op = node->payload.primop.op,
            .args = rewrite_nodes(rewriter, node->payload.primop.args)
        });
        default: error("unhandled node for rewrite");
    }
}

const struct Type* recreate_type_identity(struct Rewriter* rewriter, const struct Type* type) {
    if (type == NULL)
        return NULL;
    switch (type->tag) {
        case NoRet_TAG:         return noret_type(rewriter->dst_arena);
        case Void_TAG:          return void_type(rewriter->dst_arena);
        case Int_TAG:           return int_type(rewriter->dst_arena);
        case Float_TAG:         return float_type(rewriter->dst_arena);
        case RecordType_TAG:    return record_type(rewriter->dst_arena, (struct RecordType) {
                                    .name = string(rewriter->dst_arena, type->payload.record_type.name),
                                    .members = rewrite_nodes(rewriter, type->payload.record_type.members)});
        case ContType_TAG:      return cont_type(rewriter->dst_arena, (struct ContType) { .param_types = rewrite_nodes(rewriter, type->payload.cont_type.param_types) });
        case FnType_TAG:        return fn_type(rewriter->dst_arena, (struct FnType) {
                                    .param_types = rewrite_nodes(rewriter, type->payload.fn_type.param_types),
                                    .return_type = rewriter->rewrite_type_fn(rewriter, type->payload.fn_type.return_type)});
        case PtrType_TAG:       return ptr_type(rewriter->dst_arena, (struct PtrType) {
                                    .address_space = type->payload.ptr_type.address_space,
                                    .pointed_type = rewriter->rewrite_type_fn(rewriter, type->payload.ptr_type.pointed_type)});
        case QualifiedType_TAG: return qualified_type(rewriter->dst_arena, (struct QualifiedType) {
                                    .is_uniform = type->payload.qualified_type.is_uniform,
                                    .type = rewriter->rewrite_type_fn(rewriter, type->payload.qualified_type.type)});
        default: error("unhandled type");
    }
}
