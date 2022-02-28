#include "implem.h"

void print_node(const struct Node* node) {
    switch (node->tag) {
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
            for (int i = 0; i < params->count; i++) {
                print_type(params->types[i]);
                if (i < params->count - 1)
                    printf(", ");
            }
            printf(")");
            break;
        } case FnType: {
            printf("fn (");
            const struct Types *params = &type->payload.fn.param_types;
            for (int i = 0; i < params->count; i++) {
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
