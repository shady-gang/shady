#include "shady/ir.h"
#include "analysis/scope.h"

#include "log.h"
#include "dict.h"

struct PrinterCtx {
    FILE* output;
    unsigned int indent;
    bool pretty_print;
    struct Dict* emitted_fns;
};

#define printf(...) fprintf(ctx->output, __VA_ARGS__)
#define print_node(n) print_node_impl(ctx, n, NULL)

void print_node_impl(struct PrinterCtx* ctx, const Node* node, const char* def_name);

void print_param_list(struct PrinterCtx* ctx, const Nodes vars) {
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
#define INDENT for (unsigned int j = 0; j < ctx->indent; j++) \
    printf("   ");

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
                const Node* def = top_level->definitions.nodes[i];
                if (!def) {
                    printf("var ");
                    print_node(var->type);
                } else if (def->tag == Function_TAG) {
                    printf("fn");
                } else {
                    printf("const ");
                    print_node(var->type);
                }
                printf(" %s", var->name);
                if (top_level->definitions.nodes[i]) {
                    printf(" = ");
                    print_node_impl(ctx, def, var->name);
                }
                printf(";\n");

                if (def && def->tag == Function_TAG)
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
                if (node->payload.fn.is_continuation) {
                    printf("%s", node->payload.fn.name);
                    break;
                }

                const Nodes* returns = &node->payload.fn.return_types;
                for (size_t i = 0; i < returns->count; i++) {
                    print_node(returns->nodes[i]);
                    if (i < returns->count - 1)
                        printf(", ");
                    else
                        printf(" ");
                }
                print_param_list(ctx, node->payload.fn.params);
                printf(" {\n");
                ctx->indent++;
                print_node(node->payload.fn.block);

                if (node->type != NULL) {
                    bool section_space = false;
                    Scope scope = build_scope(node);
                    for (size_t i = 1; i < scope.size; i++) {
                        if (!section_space) {
                            printf("\n");
                            section_space = true;
                        }

                        const CFNode* cfnode = read_list(CFNode*, scope.contents)[i];
                        INDENT
                        printf("cont %s = ", cfnode->node->payload.fn.name);
                        print_param_list(ctx, cfnode->node->payload.fn.params);
                        printf(" {\n");
                        ctx->indent++;
                        print_node(cfnode->node->payload.fn.block);
                        ctx->indent--;
                        INDENT
                        printf("} \n");
                    }
                    dispose_scope(&scope);
                }

                ctx->indent--;
                INDENT printf("}");
            }
            break;
        case Block_TAG: {
            const Block* block = &node->payload.block;
            for(size_t i = 0; i < block->instructions.count; i++) {
                INDENT
                print_node(block->instructions.nodes[i]);
                printf(";\n");
            }
            INDENT
            print_node(block->terminator);
            printf("\n");
            break;
        }
        case ParsedBlock_TAG: {
            const ParsedBlock* pblock = &node->payload.parsed_block;
            for(size_t i = 0; i < pblock->instructions.count; i++) {
                INDENT
                print_node(pblock->instructions.nodes[i]);
                printf(";\n");
            }
            INDENT
            print_node(pblock->terminator);
            printf("\n");

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
        // ----------------- INSTRUCTIONS
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
        case IfInstr_TAG:
            printf("if ");
            print_node(node->payload.if_instr.condition);
            printf(" {\n");
            ctx->indent++;
            print_node(node->payload.if_instr.if_true);
            ctx->indent--;
            INDENT printf("} else {\n");
            ctx->indent++;
            print_node(node->payload.if_instr.if_false);
            ctx->indent--;
            INDENT printf("}");
            break;
        // --------------------- TERMINATORS
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
            print_node(node->payload.branch.true_target);
            printf(" ");
            print_node(node->payload.branch.false_target);
            printf(" ");
            for (size_t i = 0; i < node->payload.branch.args.count; i++) {
                printf(" ");
                print_node(node->payload.branch.args.nodes[i]);
            }
            break;
        case Callf_TAG:
            printf("callf ");
            print_node(node->payload.callf.ret_cont);
            printf(" ");
            print_node(node->payload.callf.target);
            for (size_t i = 0; i < node->payload.callf.args.count; i++) {
                printf(" ");
                print_node(node->payload.callf.args.nodes[i]);
            }
            break;
        case Unreachable_TAG:
            printf("unreachable ");
            break;
        case Join_TAG:
            printf("join ");
            break;
        // --------------------------- TYPES
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
        default: error("dunno how to print %s", node_tags[node->tag]);
    }
}

#undef print_node
#undef printf

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

static void print_node_in_output(FILE* output, const Node* node) {
    struct PrinterCtx ctx = {
        .output = output,
        .indent = 0,
        .emitted_fns = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node)
    };
    print_node_impl(&ctx, node, NULL);
    destroy_dict(ctx.emitted_fns);
}

void print_node(const Node* node) {
    print_node_in_output(stdout, node);
}

void log_node(LogLevel level, const Node* node) {
    if (level >= log_level)
        print_node_in_output(stderr, node);
}

void dump_node(const Node* node) {
    print_node(node);
    printf("\n");
}
