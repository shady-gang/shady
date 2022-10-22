#include "shady/ir.h"
#include "analysis/scope.h"

#include "log.h"
#include "list.h"
#include "growy.h"
#include "printer.h"

#include <assert.h>
#include <inttypes.h>

typedef struct PrinterCtx_ PrinterCtx;
typedef void (*PrintFn)(PrinterCtx* ctx, char* format, ...);

struct PrinterCtx_ {
    Printer* printer;
    bool print_ptrs;
    bool color;
};

#define COLOR(x) (ctx->color ? (x) : "")

#define RESET    COLOR("\033[0m")
#define RED      COLOR("\033[0;31m")
#define GREEN    COLOR("\033[0;32m")
#define YELLOW   COLOR("\033[0;33m")
#define BLUE     COLOR("\033[0;34m")
#define MANGENTA COLOR("\033[0;35m")
#define CYAN     COLOR("\033[0;36m")
#define WHITE    COLOR("\033[0;37m")

#define GREY     COLOR("\033[0;90m")
#define BRED     COLOR("\033[0;91m")
#define BGREEN   COLOR("\033[0;92m")
#define BYELLOW  COLOR("\033[0;93m")
#define BBLUE    COLOR("\033[0;94m")
#define BMAGENTA COLOR("\033[0;95m")
#define BCYAN    COLOR("\033[0;96m")
#define BWHITE   COLOR("\033[0;97m")

#define printf(...) print(ctx->printer, __VA_ARGS__)
#define print_node(n) print_node_impl(ctx, n)

static void print_node_impl(PrinterCtx* ctx, const Node* node);

#pragma GCC diagnostic error "-Wswitch"

static void print_storage_qualifier_for_global(PrinterCtx* ctx, AddressSpace as) {
    printf(BLUE);
    switch (as) {
        case AsGeneric:             printf("generic"); break;

        case AsFunctionLogical:  printf("l_function"); break;
        case AsPrivateLogical:      printf("private"); break;
        case AsSharedLogical:        printf("shared"); break;
        case AsGlobalLogical:        printf("global"); break;

        case AsPrivatePhysical:   printf("p_private"); break;
        case AsSubgroupPhysical: printf("p_subgroup"); break;
        case AsSharedPhysical:     printf("p_shared"); break;
        case AsGlobalPhysical:     printf("p_global"); break;

        case AsInput:                 printf("input"); break;
        case AsOutput:               printf("output"); break;
        case AsExternal:           printf("external"); break;
        case AsProgramCode:    printf("program_code"); break;
    }
    printf(RESET);
}

static void print_ptr_addr_space(PrinterCtx* ctx, AddressSpace as) {
    printf(GREY);
    switch (as) {
        case AsGeneric:             printf("generic"); break;

        case AsFunctionLogical:  printf("l_function"); break;
        case AsPrivateLogical:    printf("l_private"); break;
        case AsSharedLogical:      printf("l_shared"); break;
        case AsGlobalLogical:      printf("l_global"); break;

        case AsPrivatePhysical:     printf("private"); break;
        case AsSubgroupPhysical:   printf("subgroup"); break;
        case AsSharedPhysical:       printf("shared"); break;
        case AsGlobalPhysical:       printf("global"); break;

        case AsInput:                 printf("input"); break;
        case AsOutput:               printf("output"); break;
        case AsExternal:           printf("external"); break;
        case AsProgramCode:    printf("program_code"); break;
    }
    printf(RESET);
}

static void print_param_list(PrinterCtx* ctx, Nodes vars, const Nodes* defaults) {
    if (defaults != NULL)
        assert(defaults->count == vars.count);
    printf("(");
    for (size_t i = 0; i < vars.count; i++) {
        if (ctx->print_ptrs) printf("%zu::", (size_t)(void*)vars.nodes[i]);
        const Variable* var = &vars.nodes[i]->payload.var;
        print_node(var->type);
        printf(YELLOW);
        printf(" %s~%d", var->name, var->id);
        printf(RESET);
        if (defaults) {
            printf(" = ");
            print_node(defaults->nodes[i]);
        }
        if (i < vars.count - 1)
            printf(", ");
    }
    printf(")");
}

static void print_param_types(PrinterCtx* ctx, Nodes param_types) {
    printf("(");
    for (size_t i = 0; i < param_types.count; i++) {
        print_node(param_types.nodes[i]);
        if (i < param_types.count - 1)
            printf(", ");
    }
    printf(")");
}

static void print_args_list(PrinterCtx* ctx, Nodes args) {
    printf("(");
    for (size_t i = 0; i < args.count; i++) {
        if (ctx->print_ptrs) printf("%zu::", (size_t)(void*)args.nodes[i]);
        print_node(args.nodes[i]);
        if (i < args.count - 1)
            printf(", ");
    }
    printf(")");
}

static void print_yield_types(PrinterCtx* ctx, Nodes types) {
    bool initial_space = false;
    for (size_t i = 0; i < types.count; i++) {
        if (!initial_space) {
            printf(" ");
            initial_space = true;
        }
        print_node(types.nodes[i]);
        if (i < types.count - 1)
            printf(" ");
    }
}

static void print_function(PrinterCtx* ctx, const Node* node) {
    assert(node->tag == Lambda_TAG && node->payload.lam.tier == FnTier_Function);
    print_yield_types(ctx, node->payload.lam.return_types);
    print_param_list(ctx, node->payload.lam.params, NULL);
    if (!node->payload.lam.body)
        return;

    printf(" {");
    indent(ctx->printer);

    printf("\n");
    print_node(node->payload.lam.body);
    printf(";");

    if (node->type != NULL && node->payload.lam.body) {
        bool section_space = false;
        Scope scope = build_scope_from_basic_block(node);
        for (size_t i = 1; i < scope.size; i++) {
            if (!section_space) {
                printf("\n");
                section_space = true;
            }

            const CFNode* cfnode = read_list(CFNode*, scope.contents)[i];
            printf("\ncont %s = ", cfnode->location.head->payload.lam.name);
            print_param_list(ctx, cfnode->location.head->payload.lam.params, NULL);
            print_node(cfnode->location.head->payload.lam.body);
        }
        dispose_scope(&scope);
    }

    if (node->payload.lam.children_continuations.count > 0)
        printf("\n");
    for(size_t i = 0; i < node->payload.lam.children_continuations.count; i++) {
        printf("\n");
        print_node_impl(ctx, node->payload.lam.children_continuations.nodes[i]);
    }

    deindent(ctx->printer);
    printf("\n}");
}

static void print_lambda_body(PrinterCtx* ctx, const Node* body) {
    assert(is_terminator(body));
    printf("{");
    indent(ctx->printer);
    printf("\n");
    print_node(body);
    printf(";");
    deindent(ctx->printer);
    printf("\n}");
}

static void print_annotations(PrinterCtx* ctx, Nodes annotations) {
    for (size_t i = 0; i < annotations.count; i++) {
        print_node(annotations.nodes[i]);
        printf(" ");
    }
}

static void print_type(PrinterCtx* ctx, const Node* node) {
    printf(BCYAN);
    switch (is_type(node)) {
        case NotAType: assert(false); break;
        case Unit_TAG: printf("()"); break;
        case NoRet_TAG: printf("!"); break;
        case Bool_TAG: printf("bool"); break;
        case Float_TAG: printf("float"); break;
        case MaskType_TAG: printf("mask"); break;
        case QualifiedType_TAG:
            printf(node->payload.qualified_type.is_uniform ? "uniform" : "varying");
            printf(" ");
            printf(RESET);
            print_node(node->payload.qualified_type.type);
            break;
        case Int_TAG:
            switch (node->payload.int_literal.width) {
                case IntTy8:  printf("i8");  break;
                case IntTy16: printf("i16"); break;
                case IntTy32: printf("i32"); break;
                case IntTy64: printf("i64"); break;
                default: error("Not a known valid int width")
            }
            break;
        case RecordType_TAG:
            printf("struct");
            printf(RESET);
            printf(" {");
            const Nodes* members = &node->payload.record_type.members;
            for (size_t i = 0; i < members->count; i++) {
                print_node(members->nodes[i]);
                printf(RESET);
                if (i < members->count - 1)
                    printf(", ");
            }
            printf("}");
            break;
        case JoinPointType_TAG:
            printf("join_token");
            printf(RESET);
            print_param_types(ctx, node->payload.join_point_type.yield_types);
            break;
        case FnType_TAG: {
            if (node->payload.fn_type.tier != FnTier_Function) {
                printf(node->payload.fn_type.tier == FnTier_Lambda ? "lambda" : "cont");
                printf(RESET);
            } else {
                printf("fn");
                printf(RESET);
                print_yield_types(ctx, node->payload.fn_type.return_types);
            }
            print_param_types(ctx, node->payload.fn_type.param_types);
            break;
        }
        case PtrType_TAG: {
            printf("ptr");
            printf(RESET);
            printf("(");
            print_ptr_addr_space(ctx, node->payload.ptr_type.address_space);
            printf(", ");
            print_node(node->payload.ptr_type.pointed_type);
            printf(")");
            break;
        }
        case ArrType_TAG: {
            printf("[");
            print_node(node->payload.arr_type.element_type);
            if (node->payload.arr_type.size) {
                printf("; ");
                print_node(node->payload.arr_type.size);
            }
            printf("]");
            break;
        }
        case PackType_TAG: {
            printf("pack");
            printf(RESET);
            printf("(%d", node->payload.pack_type.width);
            printf(", ");
            print_node(node->payload.pack_type.element_type);
            printf(")");
            break;
        }
        case NominalType_TAG: {
            printf("%s", node->payload.nom_type.name);
        }
    }
    printf(RESET);
}

static void print_value(PrinterCtx* ctx, const Node* node) {
    switch (is_value(node)) {
        case NotAValue: assert(false); break;
        case Variable_TAG:
            printf(YELLOW);
            printf("%s~%d", node->payload.var.name, node->payload.var.id);
            printf(RESET);
            break;
        case Unbound_TAG:
            printf(YELLOW);
            printf("`%s`", node->payload.unbound.name);
            printf(RESET);
            break;
        case UntypedNumber_TAG:
            printf(BBLUE);
            printf("%s", node->payload.untyped_number.plaintext);
            printf(RESET);
            break;
        case IntLiteral_TAG:
            printf(BBLUE);
            switch (node->payload.int_literal.width) {
                case IntTy8:  printf("%" PRIu8 ,  node->payload.int_literal.value_i8);  break;
                case IntTy16: printf("%" PRIu16, node->payload.int_literal.value_i16); break;
                case IntTy32: printf("%" PRIu32, node->payload.int_literal.value_i32); break;
                case IntTy64: printf("%" PRIu64, node->payload.int_literal.value_i64); break;
                default: error("Not a known valid int width")
            }
            printf(RESET);
            break;
        case True_TAG:
            printf(BBLUE);
            printf("true");
            printf(RESET);
            break;
        case False_TAG:
            printf(BBLUE);
            printf("false");
            printf(RESET);
            break;
        case StringLiteral_TAG:
            printf(BBLUE);
            printf("\"%s\"", node->payload.string_lit.string);
            printf(RESET);
            break;
        case ArrayLiteral_TAG:
            printf(BBLUE);
            printf("array ");
            printf(RESET);
            printf("(");
            print_node(node->payload.arr_lit.element_type);
            printf(")");
            printf(" {");
            Nodes nodes = node->payload.arr_lit.contents;
            for (size_t i = 0; i < nodes.count; i++) {
                print_node(nodes.nodes[i]);
                if (i + 1 < nodes.count)
                    printf(", ");
            }
            printf("}");
            printf(RESET);
            break;
        case Value_Tuple_TAG: error("TODO")
        case Value_RefDecl_TAG: {
            printf(BYELLOW);
            printf((char*) get_decl_name(node->payload.ref_decl.decl));
            printf(RESET);
            break;
        }
        case FnAddr_TAG:
            printf(BYELLOW);
            printf("&");
            printf((char*) get_decl_name(node->payload.fn_addr.fn));
            printf(RESET);
            break;
    }
}

static void print_instruction(PrinterCtx* ctx, const Node* node) {
    switch (is_instruction(node)) {
        case NotAnInstruction: assert(false); break;
        case PrimOp_TAG:
            printf(GREEN);
            printf("%s", primop_names[node->payload.prim_op.op]);
            printf(RESET);
            printf("(");
            for (size_t i = 0; i < node->payload.prim_op.operands.count; i++) {
                print_node(node->payload.prim_op.operands.nodes[i]);
                if (i + 1 < node->payload.prim_op.operands.count)
                    printf(", ");
            }
            printf(")");
            break;
        case Call_TAG:
            printf(GREEN);
            printf("call ");
            printf(RESET);
            const Node* callee = node->payload.call_instr.callee;
            print_node(callee);
            printf("(");
            for (size_t i = 0; i < node->payload.call_instr.args.count; i++) {
                print_node(node->payload.call_instr.args.nodes[i]);
                if (i + 1 < node->payload.call_instr.args.count)
                    printf(", ");
            }
            printf(")");
            break;
        case If_TAG:
            printf(GREEN);
            printf("if");
            printf(RESET);
            print_yield_types(ctx, node->payload.if_instr.yield_types);
            printf("(");
            print_node(node->payload.if_instr.condition);
            printf(") ");
            print_lambda_body(ctx, node->payload.if_instr.if_true);
            if (node->payload.if_instr.if_false) {
                printf(GREEN);
                printf(" else ");
                printf(RESET);
                print_lambda_body(ctx, node->payload.if_instr.if_false);
            }
            break;
        case Loop_TAG:
            printf(GREEN);
            printf("loop");
            printf(RESET);
            print_yield_types(ctx, node->payload.loop_instr.yield_types);
            const Node* body = node->payload.loop_instr.body;
            assert(body->tag == Lambda_TAG && body->payload.lam.tier == FnTier_Lambda);
            print_param_list(ctx, body->payload.lam.params, &node->payload.loop_instr.initial_args);
            print_lambda_body(ctx, body->payload.lam.body);
            break;
        case Match_TAG:
            printf(GREEN);
            printf("match");
            printf(RESET);
            print_yield_types(ctx, node->payload.match_instr.yield_types);
            printf("(");
            print_node(node->payload.match_instr.inspect);
            printf(")");
            printf(" {");
            indent(ctx->printer);
            for (size_t i = 0; i < node->payload.match_instr.literals.count; i++) {
                printf("\n");
                printf(GREEN);
                printf("case");
                printf(RESET);
                printf(" ");
                print_node(node->payload.match_instr.literals.nodes[i]);
                printf(": ");
                print_node(node->payload.match_instr.cases.nodes[i]);
            }

            printf("\n");
            printf(GREEN);
            printf("default");
            printf(RESET);
            printf(": ");
            print_node(node->payload.match_instr.default_case);

            deindent(ctx->printer);
            printf("\n}");
            break;
        case Control_TAG:
            printf(BGREEN);
            printf("control");
            printf(RESET);
            print_yield_types(ctx, node->payload.control.yield_types);
            printf("(");
            printf(")");
            break;
    }
}

static void print_terminator(PrinterCtx* ctx, const Node* node) {
    switch (is_terminator(node)) {
        case NotATerminator: assert(false);
        case Let_TAG: {
            const Node* tail = node->payload.let.tail;
            if (is_anonymous_lambda(tail)) {
                // if the let tail is a lambda, we apply some syntactic sugar
                if (tail->payload.lam.params.count > 0) {
                    printf(GREEN);
                    if (node->payload.let.is_mutable)
                        printf("var");
                    else
                        printf("val");
                    printf(RESET);
                    Nodes params = tail->payload.lam.params;
                    for (size_t i = 0; i < params.count; i++) {
                        printf(" ");
                        print_node(params.nodes[i]->payload.var.type);
                        printf(YELLOW);
                        printf(" %s", params.nodes[i]->payload.var.name);
                        printf("~%d", params.nodes[i]->payload.var.id);
                        printf(RESET);
                    }
                    printf(" = ");
                }
                print_node(node->payload.let.instruction);
                printf(";\n");
                print_node(node->payload.let.tail->payload.lam.body);
            } else {
                printf(GREEN);
                if (node->payload.let.is_mutable)
                    printf("let mut");
                else
                    printf("let");
                printf(RESET);
                print_node(node->payload.let.instruction);
                printf(GREEN);
                printf(" in ");
                printf(RESET);
                print_node(node->payload.let.tail);
            }
            break;
        } case Return_TAG:
            printf(BGREEN);
            printf("return");
            printf(RESET);
            for (size_t i = 0; i < node->payload.fn_ret.values.count; i++) {
                printf(" ");
                print_node(node->payload.fn_ret.values.nodes[i]);
            }
            break;
        case Terminator_TailCall_TAG:
            printf(BGREEN);
            printf("tail_call ");
            printf(RESET);
            print_node(node->payload.tail_call.target);
            print_args_list(ctx, node->payload.tail_call.args);
            break;
        case Branch_TAG:
            printf(BGREEN);
            switch (node->payload.branch.branch_mode) {
                case BrJump:     printf("jump "     ); break;
                case BrIfElse:   printf("br_ifelse "); break;
                case BrSwitch:   printf("br_switch "); break;
                default: error("unknown branch mode");
            }
            printf(RESET);
            switch (node->payload.branch.branch_mode) {
                case BrJump: {
                    print_node(node->payload.branch.target);
                    break;
                }
                case BrIfElse: {
                    printf("(");
                    print_node(node->payload.branch.branch_condition);
                    printf(", ");
                    print_node(node->payload.branch.true_target);
                    printf(", ");
                    print_node(node->payload.branch.false_target);
                    printf(")");
                    break;
                }
                case BrSwitch: {
                    printf("(");
                    print_node(node->payload.branch.switch_value);
                    printf(", ");
                    for (size_t i = 0; i < node->payload.branch.case_values.count; i++) {
                        print_node(node->payload.branch.case_values.nodes[i]);
                        printf(", ");
                        print_node(node->payload.branch.case_targets.nodes[i]);
                        if (i + 1 < node->payload.branch.case_values.count)
                            printf(", ");
                    }
                    printf(", ");
                    print_node(node->payload.branch.default_target);
                    printf(") ");
                }
            }
            print_args_list(ctx, node->payload.branch.args);
            break;
        case Join_TAG:
            printf(BGREEN);
            printf("join");
            printf(RESET);
            printf("(");
            print_node(node->payload.join.join_point);
            printf(")");
            print_args_list(ctx, node->payload.join.args);
            break;
        case Unreachable_TAG:
            printf(BGREEN);
            printf("unreachable ");
            printf(RESET);
            break;
        case MergeConstruct_TAG:
            printf(BGREEN);
            printf("%s", merge_what_string[node->payload.merge_construct.construct]);
            printf(RESET);
            print_args_list(ctx, node->payload.merge_construct.args);
            break;
    }
}

static void print_decl(PrinterCtx* ctx, const Node* node) {
    switch (node->tag) {
        case GlobalVariable_TAG: {
            const GlobalVariable* gvar = &node->payload.global_variable;
            print_annotations(ctx, gvar->annotations);
            print_storage_qualifier_for_global(ctx, gvar->address_space);
            printf(" ");
            print_node(gvar->type);
            printf(BYELLOW);
            printf(" %s", gvar->name);
            printf(RESET);
            if (gvar->init) {
                printf(" = ");
                print_node(gvar->init);
            }
            printf(";\n");
            break;
        }
        case Constant_TAG: {
            const Constant* cnst = &node->payload.constant;
            print_annotations(ctx, cnst->annotations);
            printf(BLUE);
            printf("const ");
            printf(RESET);
            print_node(node->type);
            printf(BYELLOW);
            printf(" %s", cnst->name);
            printf(RESET);
            printf(" = ");
            print_node(cnst->value);
            printf(";\n");
            break;
        }
        case Lambda_TAG: {
            const Lambda* fun = &node->payload.lam;
            assert(fun->tier == FnTier_Function && "basic blocks aren't supposed to be found at the top level");
            print_annotations(ctx, fun->annotations);
            printf(BLUE);
            printf("fn");
            printf(RESET);
            printf(BYELLOW);
            printf(" %s", fun->name);
            printf(RESET);
            print_function(ctx, node);
            printf(";\n\n");
            break;
        }
        default: error("Not a decl");
    }
}

static void print_node_impl(PrinterCtx* ctx, const Node* node) {
    if (node == NULL) {
        printf("?");
        return;
    }

    if (ctx->print_ptrs) printf("%zu::", (size_t)(void*)node);

    if (is_type(node))
        print_type(ctx, node);
    else if (is_value(node))
        print_value(ctx, node);
    else if (is_instruction(node))
        print_instruction(ctx, node);
    else if (is_terminator(node))
        print_terminator(ctx, node);
    else if (node->tag == Lambda_TAG && node->payload.lam.tier) {
        printf(BYELLOW);
        printf("lambda ");
        printf(RESET);
        print_param_list(ctx, node->payload.lam.params, NULL);
        indent(ctx->printer);
        printf(" {\n");
        print_node(node->payload.lam.body);
        printf(";");
        deindent(ctx->printer);
        printf("\n}");
    } else if (is_declaration(node)) {
        printf(BYELLOW);
        printf("%s", get_decl_name(node));
        printf(RESET);
    } else switch (node->tag) {
        case Root_TAG: {
            const Root* top_level = &node->payload.root;
            for (size_t i = 0; i < top_level->declarations.count; i++) {
                const Node* decl = top_level->declarations.nodes[i];
                print_decl(ctx, decl);
            }
            break;
        }
        case Annotation_TAG: {
            const Annotation* annotation = &node->payload.annotation;
            printf(RED);
            printf("@%s", annotation->name);
            printf(RESET);
            switch (annotation->payload_type) {
                case AnPayloadValue:
                    printf("(");
                    print_node(annotation->value);
                    printf(")");
                    break;
                default: break;
            }
            break;
        }
        default: error("dunno how to print %s", node_tags[node->tag]);
    }
}

#undef print_node
#undef printf

void print_node_into_str(const Node* node, char** str_ptr, size_t* size) {
    Growy* g = new_growy();
    PrinterCtx ctx = {
        .printer = open_growy_as_printer(g),
        .print_ptrs = false,
        .color = false,
    };
    print_node_impl(&ctx, node);

    *size = growy_size(g);
    *str_ptr = growy_deconstruct(g);
    destroy_printer(ctx.printer);
}

static void print_node_in_output(FILE* output, const Node* node, bool dump_ptrs) {
    PrinterCtx ctx = {
        .printer = open_file_as_printer(output),
        .print_ptrs = dump_ptrs,
        .color = true,
    };
    print_node_impl(&ctx, node);
    flush(ctx.printer);
    destroy_printer(ctx.printer);
}

void print_node(const Node* node) {
    print_node_in_output(stdout, node, false);
}

void log_node(LogLevel level, const Node* node) {
    if (level >= get_log_level())
        print_node_in_output(stderr, node, false);
}

void dump_node(const Node* node) {
    print_node(node);
    printf("\n");
}
