#include "implem.h"
#include "dict.h"

extern const char* node_tags[];

struct PrinterCtx {
    unsigned int indent;
    bool pretty_print;
    struct Dict* emitted_fns;
};

void print_node_impl(struct PrinterCtx* ctx, const Node* node, const char* def_name);

void print_param_list(const Nodes vars) {
    printf("(");
    for (size_t i = 0; i < vars.count; i++) {
        const Variable* var = &vars.nodes[i]->payload.var;
        print_node(var->type);
        printf(" %s_%d", var->name, var->id);
        if (i < vars.count - 1)
            printf(", ");
    }
    printf(")");
}

//static int indent = 0;
#define INDENT for (int j = 0; j < ctx->indent; j++) \
    printf("   ");

#define print_node(n) print_node_impl(ctx, n, NULL)

void print_node_impl(struct PrinterCtx* ctx, const Node* node, const char* def_name) {
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
                if (ctx->pretty_print) {
                    if (top_level->definitions.nodes[i])
                        print_node_impl(ctx, top_level->definitions.nodes[i], var->name);
                    else print_node_impl(ctx, top_level->variables.nodes[i], var->name);
                } else {
                    printf("let ");
                    print_node(var->type);
                    printf(" %s", var->name);
                    if (top_level->definitions.nodes[i]) {
                        printf(" = ");
                        print_node(top_level->definitions.nodes[i]);
                    }
                    printf(";");
                }

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
            if (find_key_dict(const Node*, ctx->emitted_fns, node) != NULL)
                printf("%s", node->payload.fn.name);
            else {
                insert_set_get_result(const Node*, ctx->emitted_fns, node);
                if (node->payload.fn.is_continuation)
                    printf("cont ");
                else {
                    printf("fn ");
                    const Nodes* returns = &node->payload.fn.return_types;
                    for (size_t i = 0; i < returns->count; i++) {
                        print_node(returns->nodes[i]);
                        if (i < returns->count - 1)
                            printf(", ");
                        else
                            printf(" ");
                    }
                }
                printf("%s ", node->payload.fn.name);
                print_param_list(node->payload.fn.params);
                printf(" {\n");
                ctx->indent++;
                print_node(node->payload.fn.block);
                ctx->indent--;
                INDENT printf("}\n");
            }
            break;
        case Block_TAG: {
            const Block* block = &node->payload.block;
            for(size_t i = 0; i < block->instructions.count; i++) {
                INDENT
                print_node(block->instructions.nodes[i]);
                printf(";\n");
            }
            break;
        }
        case ParsedBlock_TAG: {
            const ParsedBlock* pblock = &node->payload.parsed_block;
            for(size_t i = 0; i < pblock->instructions.count; i++) {
                INDENT
                print_node(pblock->instructions.nodes[i]);
                printf(";\n");
            }

            if (pblock->continuations.count > 0) {
                printf("\n");
            }
            for(size_t i = 0; i < pblock->continuations.count; i++) {
                INDENT
                print_node_impl(ctx, pblock->continuations.nodes[i], pblock->continuations_vars.nodes[i]->payload.var.name);
            }
            break;
        }
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
        case Let_TAG:
            printf("let");
            for (size_t i = 0; i < node->payload.let.variables.count; i++) {
                printf(" ");
                print_node(node->payload.let.variables.nodes[i]->payload.var.type);
                printf(" %s", node->payload.let.variables.nodes[i]->payload.var.name);
            }
            printf(" = ");

            printf("%s", primop_names[node->payload.let.op]);
            for (size_t i = 0; i < node->payload.let.args.count; i++) {
                printf(" ");
                print_node(node->payload.let.args.nodes[i]);
            }

            break;
        case StructuredSelection_TAG:
            printf("if ");
            print_node(node->payload.selection.condition);
            printf(" {\n");
            ctx->indent++;
            print_node(node->payload.selection.ifTrue);
            ctx->indent--;
            INDENT printf("} else {\n");
            ctx->indent++;
            print_node(node->payload.selection.ifFalse);
            ctx->indent--;
            INDENT printf("}");
            break;
        case Return_TAG:
            printf("return");
            for (size_t i = 0; i < node->payload.fn_ret.values.count; i++) {
                printf(" ");
                print_node(node->payload.fn_ret.values.nodes[i]);
            }
            break;
        case Jump_TAG:
            printf("jump ");
            print_node(node->payload.jump.target);
            for (size_t i = 0; i < node->payload.jump.args.count; i++) {
                printf(" ");
                print_node(node->payload.jump.args.nodes[i]);
            }
            break;
        case Branch_TAG:
            printf("branch ");
            print_node(node->payload.branch.condition);
            printf(" ");
            print_node(node->payload.branch.trueTarget);
            printf(" ");
            print_node(node->payload.branch.falseTarget);
            printf(" ");
            for (size_t i = 0; i < node->payload.branch.args.count; i++) {
                printf(" ");
                print_node(node->payload.branch.args.nodes[i]);
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
        case FnType_TAG: {
            if (node->payload.fn_type.is_continuation)
                printf("cont");
            else {
                printf("fn ");
                const Nodes* returns = &node->payload.fn_type.return_types;
                for (size_t i = 0; i < returns->count; i++) {
                    print_node(returns->nodes[i]);
                    if (i < returns->count - 1)
                        printf(", ");
                }
            }
            printf("(");
            const Nodes* params = &node->payload.fn_type.param_types;
            for (size_t i = 0; i < params->count; i++) {
                print_node(params->nodes[i]);
                if (i < params->count - 1)
                    printf(", ");
            }
            printf(")");
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

#undef print_node

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

void print_node(const Node* node) {
    struct PrinterCtx ctx = {
        .indent = 0,
        .emitted_fns = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node)
    };
    print_node_impl(&ctx, node, NULL);
    destroy_dict(ctx.emitted_fns);
}
