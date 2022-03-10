#include "implem.h"

void print_param_list(const struct Nodes vars, bool use_names) {
    printf("(");
    for (size_t i = 0; i < vars.count; i++) {
        print_type(vars.nodes[i]->payload.var.type);
        if (use_names)
            printf(" %s", vars.nodes[i]->payload.var.name);
        if (i < vars.count - 1)
            printf(", ");
    }
    printf(")");
}

static int indent = 0;

void print_instructions(const struct Nodes instructions) {

}

void print_node_impl(const struct Node* node, const char* def_name) {
    switch (node->tag) {
        case Root_TAG: {
            const struct Root* top_level = &node->payload.root;
            for (size_t i = 0; i < top_level->variables.count; i++) {
                // Some top-level variables do not have definitions !
                if (top_level->definitions.nodes[i])
                    print_node_impl(top_level->definitions.nodes[i], top_level->variables.nodes[i]->payload.var.name);
                else print_node_impl(top_level->variables.nodes[i], top_level->variables.nodes[i]->payload.var.name);

                if (i < top_level->variables.count - 1)
                    printf("\n");
            }
            break;
        }
        case VariableDecl_TAG:
            print_type(node->payload.var_decl.variable->payload.var.type);
            printf(" %s", node->payload.var_decl.variable->payload.var.name);
            if (node->payload.var_decl.init) {
                printf(" = ");
                print_node(node->payload.var_decl.init);
            }
            printf(";\n");
            break;
        case Variable_TAG:
            if (def_name) {
                printf("var ");
                print_type(node->payload.var.type);
                printf(" %s;\n", node->payload.var.name);
            } else
                printf(" %s", node->payload.var.name);
            break;
        case Function_TAG:
            print_type(node->payload.fn.return_type);
            if (def_name)
                printf(" %s", def_name);
            print_param_list(node->payload.fn.params, true);
            printf(" {\n");
            indent++;
            print_instructions(node->payload.fn.instructions);
            indent--;
            printf("}\n");
            break;
        default: error("dunno how to print this");
    }
}

void print_node(const struct Node* node) {
    return print_node_impl(node, NULL);
}

void print_type(const struct Type* type) {
    if (type == NULL) {
        printf("?");
        return;
    }
    switch (type->tag) {
        case QualifiedType_TAG:
            if (type->payload.qualified_type.is_uniform)
                printf("uniform ");
            else
                printf("varying ");
            print_type(type->payload.qualified_type.type);
            break;
        case NoRet_TAG:
            printf("!");
            break;
        case Void_TAG:
            printf("void");
            break;
        case Int_TAG:
            printf("int");
            break;
        case Float_TAG:
            printf("float");
            break;
        case RecordType_TAG:
            printf("struct %s", type->payload.record_type.name);
            break;
        case ContType_TAG: {
            printf("cont (");
            const struct Types *params = &type->payload.cont_type.param_types;
            for (size_t i = 0; i < params->count; i++) {
                print_type(params->types[i]);
                if (i < params->count - 1)
                    printf(", ");
            }
            printf(")");
            break;
        } case FnType_TAG: {
            printf("fn (");
            const struct Types *params = &type->payload.fn_type.param_types;
            for (size_t i = 0; i < params->count; i++) {
                print_type(params->types[i]);
                if (i < params->count - 1)
                    printf(", ");
            }
            printf(") ");
            print_type(type->payload.fn_type.return_type);
            break;
        }
        case PtrType_TAG: {
            printf("ptr[");
            print_type(type->payload.ptr_type.pointed_type);
            printf("]");
            break;
        }
    }
}
