#include "implem.h"

void print_program(const struct Program* program) {
    for (size_t i = 0; i < program->declarations_and_definitions.count; i++) {
        print_node(program->declarations_and_definitions.nodes[i], true);

        if (i < program->declarations_and_definitions.count - 1)
            printf("\n");
    }
}

void print_param_list(const struct Variables vars, bool use_names) {
    printf("(");
    for (size_t i = 0; i < vars.count; i++) {
        print_type(vars.variables[i]->type);
        if (use_names)
            printf(" %s", vars.variables[i]->name);
        if (i < vars.count - 1)
            printf(", ");
    }
    printf(")");
}

static int indent = 0;

void print_instructions(const struct Nodes instructions) {

}

void print_node(const struct Node* node, bool is_decl) {
    switch (node->tag) {
        case VariableDecl_TAG:
            print_type(node->payload.var_decl.variable->type);
            printf(" %s", node->payload.var_decl.variable->payload.var.name);
            if (node->payload.var_decl.init) {
                printf(" = ");
                print_node(node->payload.var_decl.init, false);
            }
            printf(";\n");
            break;
        case Function_TAG:
            print_type(node->payload.fn.return_type);
            printf(" %s", node->payload.fn.name);
            print_param_list(node->payload.fn.params, true);
            printf(" {\n");
            indent++;
            print_instructions(node->payload.fn.instructions);
            indent--;
            printf("}\n");
            break;
        default: error("unhandled node");
    }
}

void print_type(const struct Type* type) {
    if (type->uniform)
        printf("uniform ");
    else
        printf("varying ");

    switch (type->tag) {
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
    }
}
