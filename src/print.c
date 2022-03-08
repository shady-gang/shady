#include "implem.h"

void print_program(const struct Program* program) {
    for (size_t i = 0; i < program->variables.count; i++) {
        // Some top-level variables do not have definitions !
        if (program->definitions.nodes[i])
            print_node(program->definitions.nodes[i], program->variables.nodes[i]->payload.var.name);
        else print_node(program->variables.nodes[i], program->variables.nodes[i]->payload.var.name);

        if (i < program->variables.count - 1)
            printf("\n");
    }
}

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

void print_node(const struct Node* node, const char* def_name) {
    switch (node->tag) {
        case VariableDecl_TAG:
            print_type(node->payload.var_decl.variable->payload.var.type);
            printf(" %s", node->payload.var_decl.variable->payload.var.name);
            if (node->payload.var_decl.init) {
                printf(" = ");
                print_node(node->payload.var_decl.init, NULL);
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

void print_type(const struct Type* type) {
    if (type == NULL) {
        printf("?");
        return;
    }
    switch (type->tag) {
        case QualType:
            if (type->payload.qualified.is_uniform)
                printf("uniform ");
            else
                printf("varying ");
            print_type(type->payload.qualified.type);
            break;
        case NoRet:
            printf("!");
            break;
        case Void:
            printf("void");
            break;
        case Int:
            printf("int");
            break;
        case Float:
            printf("float");
            break;
        case RecordType:
            printf("struct %s", type->payload.record.name);
            break;
        case ContType: {
            printf("cont (");
            const struct Types *params = &type->payload.cont.param_types;
            for (size_t i = 0; i < params->count; i++) {
                print_type(params->types[i]);
                if (i < params->count - 1)
                    printf(", ");
            }
            printf(")");
            break;
        } case FnType: {
            printf("fn (");
            const struct Types *params = &type->payload.fn.param_types;
            for (size_t i = 0; i < params->count; i++) {
                print_type(params->types[i]);
                if (i < params->count - 1)
                    printf(", ");
            }
            printf(") ");
            print_type(type->payload.fn.return_type);
            break;
        }
        case PtrType: {
            printf("ptr[");
            print_type(type->payload.ptr.pointed_type);
            printf("]");
            break;
        }
    }
}
