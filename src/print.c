#include "implem.h"

extern const char* node_tags[];

void print_param_list(const Nodes vars, bool use_names) {
    printf("(");
    for (size_t i = 0; i < vars.count; i++) {
        print_node(vars.nodes[i]->payload.var.type);
        if (use_names)
            printf(" %s", vars.nodes[i]->payload.var.name);
        if (i < vars.count - 1)
            printf(", ");
    }
    printf(")");
}

static int indent = 0;

void print_instructions(const Nodes instructions) {
    for(size_t i = 0; i < instructions.count; i++) {
        printf("   ");
        print_node(instructions.nodes[i]);
        printf(";\n");
    }
}

bool pretty_print = false;

void print_node_impl(const Node* node, const char* def_name) {
    if (node == NULL) {
        printf("?");
        return;
    }
    switch (node->tag) {
        case Root_TAG: {
            const Root* top_level = &node->payload.root;
            for (size_t i = 0; i < top_level->variables.count; i++) {
                const Variable* var = &top_level->variables.nodes[i]->payload.var;
                // Some top-level variables do not have definitions !
                if (pretty_print) {
                    if (top_level->definitions.nodes[i])
                        print_node_impl(top_level->definitions.nodes[i], var->name);
                    else print_node_impl(top_level->variables.nodes[i], var->name);
                } else {
                    printf("let ");
                    print_node_impl(var->type, NULL);
                    printf(" %s", var->name);
                    if (top_level->definitions.nodes[i]) {
                        printf(" = ");
                        print_node_impl(top_level->definitions.nodes[i], NULL);
                    }
                    printf(";");
                }

                if (i < top_level->variables.count - 1)
                    printf("\n");
            }
            break;
        }
        case VariableDecl_TAG:
            print_node(node->payload.var_decl.variable->payload.var.type);
            printf(" %s", node->payload.var_decl.variable->payload.var.name);
            if (node->payload.var_decl.init) {
                printf(" = ");
                print_node(node->payload.var_decl.init);
            }
            printf(";\n");
            break;
        case Variable_TAG:
            printf("%s_%d", node->payload.var.name, node->payload.var.id);
            break;
        case Unbound_TAG:
            printf("`%s`", node->payload.unbound.name);
            break;
        case Function_TAG:
            printf("fn ");
            const Nodes* returns = &node->payload.fn.return_types;
            for (size_t i = 0; i < returns->count; i++) {
                print_node(returns->nodes[i]);
                if (i < returns->count - 1)
                    printf(", ");
            }
            if (def_name)
                printf(" %s", def_name);
            print_param_list(node->payload.fn.params, true);
            printf(" {\n");
            indent++;
            print_instructions(node->payload.fn.instructions);
            indent--;
            printf("}\n");
            break;
        case UntypedNumber_TAG:
            printf("%s", node->payload.untyped_number.plaintext);
            break;
        case IntLiteral_TAG:
            printf("%d", node->payload.int_literal.value);
            break;
        case True_TAG:
            printf("true");
            break;
        case False_TAG:
            printf("false");
            break;
        case PrimOp_TAG:
            printf("%s", primop_names[node->payload.primop.op]);
            for (size_t i = 0; i < node->payload.primop.args.count; i++) {
                printf(" ");
                print_node(node->payload.primop.args.nodes[i]);
            }
            break;
        case Let_TAG:
            printf("let");
            for (size_t i = 0; i < node->payload.let.variables.count; i++) {
                printf(" ");
                print_node(node->payload.let.variables.nodes[i]->payload.var.type);
                printf(" %s", node->payload.let.variables.nodes[i]->payload.var.name);
            }
            printf(" = ");
            print_node(node->payload.let.target);
            break;
        case Return_TAG:
            printf("return");
            for (size_t i = 0; i < node->payload.fn_ret.values.count; i++) {
                printf(" ");
                print_node(node->payload.fn_ret.values.nodes[i]);
            }
            break;
        case QualifiedType_TAG:
            if (node->payload.qualified_type.is_uniform)
                printf("uniform ");
            else
                printf("varying ");
            print_node(node->payload.qualified_type.type);
            break;
        case NoRet_TAG:
            printf("!");
            break;
        case Int_TAG:
            printf("int");
            break;
        case Bool_TAG:
            printf("bool");
            break;
        case Float_TAG:
            printf("float");
            break;
        case RecordType_TAG:
            printf("struct {");
            const Nodes* members = &node->payload.record_type.members;
            for (size_t i = 0; i < members->count; i++) {
                print_node(members->nodes[i]);
                if (i < members->count - 1)
                    printf(", ");
            }
            printf("}");
            break;
        case ContType_TAG: {
            printf("cont (");
            const Nodes* params = &node->payload.cont_type.param_types;
            for (size_t i = 0; i < params->count; i++) {
                print_node(params->nodes[i]);
                if (i < params->count - 1)
                    printf(", ");
            }
            printf(")");
            break;
        } case FnType_TAG: {
            printf("fn ");
            const Nodes* returns = &node->payload.fn_type.return_types;
            for (size_t i = 0; i < returns->count; i++) {
                print_node(returns->nodes[i]);
                if (i < returns->count - 1)
                    printf(", ");
            }
            printf("(");
            const Nodes* params = &node->payload.fn_type.param_types;
            for (size_t i = 0; i < params->count; i++) {
                print_node(params->nodes[i]);
                if (i < params->count - 1)
                    printf(", ");
            }
            printf(") ");
            break;
        }
        case PtrType_TAG: {
            printf("ptr[");
            print_node(node->payload.ptr_type.pointed_type);
            printf("]");
            break;
        }
        default: error("dunno how to print %s\n", node_tags[node->tag]);
    }
}

void print_node(const Node* node) {
    print_node_impl(node, NULL);
}
