#include "implem.h"
#include "type.h"

#include "list.h"

struct Nodes rewrite_nodes(struct Rewriter* rewriter, struct Nodes old_nodes) {
    size_t count = old_nodes.count;
    const struct Node* arr[count];
    for (size_t i = 0; i < count; i++)
        arr[i] = rewriter->rewrite_node(rewriter, old_nodes.nodes[i]);
    return nodes(rewriter->dst_arena, count, arr);
}

struct Types rewrite_types(struct Rewriter* rewriter, struct Types old_types) {
    size_t count = old_types.count;
    const struct Type* arr[count];
    for (size_t i = 0; i < count; i++)
        arr[i] = rewriter->rewrite_type(rewriter, old_types.types[i]);
    return types(rewriter->dst_arena, count, arr);
}

struct Strings import_strings(struct Rewriter* rewriter, struct Strings old_strings) {
    size_t count = old_strings.count;
    String arr[count];
    for (size_t i = 0; i < count; i++)
        arr[i] = string(rewriter->dst_arena, old_strings.strings[i]);
    return strings(rewriter->dst_arena, count, arr);
}

const struct Node* recreate_node_identity(struct Rewriter* rewriter, const struct Node* node) {
    if (node == NULL)
        return NULL;
    switch (node->tag) {
        case Root_TAG: {
            size_t count = node->payload.root.variables.count;
            const struct Node* new_variables[count];
            const struct Node* new_definitions[count];

            for (size_t i = 0; i < count; i++) {
                new_variables[i] = rewriter->rewrite_node(rewriter, node->payload.root.variables.nodes[i]);
                new_definitions[i] = rewriter->rewrite_node(rewriter, node->payload.root.definitions.nodes[i]);
            }

            return root(rewriter->dst_arena, (struct Root) {
                .variables = nodes(rewriter->dst_arena, count, new_variables),
                .definitions = nodes(rewriter->dst_arena, count, new_definitions)
            });
        }
        case Function_TAG:      return fn(rewriter->dst_arena, (struct Function) {
           .return_type = rewriter->rewrite_type(rewriter, node->payload.fn.return_type),
           .instructions = rewrite_nodes(rewriter, node->payload.fn.instructions),
           .params = rewrite_nodes(rewriter, node->payload.fn.params),
        });
        case UntypedNumber_TAG: return untyped_number(rewriter->dst_arena, (struct UntypedNumber) {
            .plaintext = string(rewriter->dst_arena, node->payload.untyped_number.plaintext)
        });
        case Variable_TAG:      return var(rewriter->dst_arena, (struct Variable) {
            .name = string(rewriter->dst_arena, node->payload.var.name),
            .type = rewriter->rewrite_type(rewriter, node->payload.var.type)
        });
        case VariableDecl_TAG:  return var_decl(rewriter->dst_arena, (struct VariableDecl) {
            .address_space = node->payload.var_decl.address_space,
            .variable = rewriter->rewrite_node(rewriter, node->payload.var_decl.variable),
            .init = rewriter->rewrite_node(rewriter, node->payload.var_decl.init),
        });
        case Call_TAG:          return call(rewriter->dst_arena, (struct Call) {
            .callee = rewriter->rewrite_node(rewriter, node->payload.call.callee),
            .args = rewrite_nodes(rewriter, node->payload.call.args)
        });
        case Let_TAG:           return let(rewriter->dst_arena, (struct Let) {
            .variables = rewrite_nodes(rewriter, node->payload.let.variables),
            .target = rewriter->rewrite_node(rewriter, node->payload.let.target)
        });
        case Return_TAG:        return fn_ret(rewriter->dst_arena, (struct Return) {
            .values = rewrite_nodes(rewriter, node->payload.fn_ret.values)
        });
        case PrimOp_TAG:        return primop(rewriter->dst_arena, (struct PrimOp) {
            .op = node->payload.primop.op,
            .args = rewrite_nodes(rewriter, node->payload.primop.args)
        });
        default: error("unhandled node");
    }
}

const struct Type* recreate_type_identity(struct Rewriter* rewriter, const struct Type* type) {
    if (type == NULL)
        return NULL;
    switch (type->tag) {
        case NoRet:      return noret_type(rewriter->dst_arena);
        case Void:       return void_type(rewriter->dst_arena);
        case Int:        return int_type(rewriter->dst_arena);
        case Float:      return float_type(rewriter->dst_arena);
        case RecordType: return record_type(rewriter->dst_arena,
                                            string(rewriter->dst_arena, type->payload.record.name),
                                            rewrite_types(rewriter, type->payload.record.members));
        case ContType:   return cont_type(rewriter->dst_arena, rewrite_types(rewriter, type->payload.cont.param_types));
        case FnType:     return fn_type(rewriter->dst_arena, rewrite_types(rewriter, type->payload.fn.param_types), rewriter->rewrite_type(rewriter, type->payload.fn.return_type));
        case PtrType:    return ptr_type(rewriter->dst_arena,
                                               rewriter->rewrite_type(rewriter, type->payload.ptr.pointed_type),
                                               type->payload.ptr.address_space);
        case QualType:   return qualified_type(rewriter->dst_arena,
                                               type->payload.qualified.is_uniform,
                                               rewriter->rewrite_type(rewriter, type->payload.qualified.type));
        default: error("unhandled type");
    }
}
