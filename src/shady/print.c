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
    const Node* fn;
    Scope* scope;
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

static void print_ty_args_list(PrinterCtx* ctx, Nodes args) {
    printf("[");
    for (size_t i = 0; i < args.count; i++) {
        if (ctx->print_ptrs) printf("%zu::", (size_t)(void*)args.nodes[i]);
        print_node(args.nodes[i]);
        if (i < args.count - 1)
            printf(", ");
    }
    printf("]");
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

static void print_abs_body(PrinterCtx* ctx, const Node* block) {
    assert(!ctx->fn || is_function(ctx->fn));
    assert(is_abstraction(block));

    print_node(get_abstraction_body(block));
    printf(";");

    if (ctx->scope != NULL) {
        const CFNode* dominator = scope_lookup(ctx->scope, block);
        assert(dominator);
        for (size_t i = 0; i < dominator->dominates->elements_count; i++) {
            const CFNode* cfnode = read_list(const CFNode*, dominator->dominates)[i];

            if (is_basic_block(cfnode->node)) {
                printf(GREEN);
                printf("\n\ncont");
                printf(BYELLOW);
                printf(" %s", cfnode->node->payload.basic_block.name);
                printf(RESET);
                print_param_list(ctx, cfnode->node->payload.basic_block.params, NULL);

                printf(" {");
                indent(ctx->printer);
                printf("\n");
                print_abs_body(ctx, cfnode->node);
                deindent(ctx->printer);
                printf("\n}");
            }
        }
    }
}

static void print_lambda_body(PrinterCtx* ctx, const Node* lam) {
    assert(is_anonymous_lambda(lam));
    printf(" {");
    indent(ctx->printer);
    printf("\n");
    print_abs_body(ctx, lam);
    deindent(ctx->printer);
    printf("\n}");
}

static void print_function(PrinterCtx* ctx, const Node* node) {
    assert(is_function(node));
    print_yield_types(ctx, node->payload.fun.return_types);
    print_param_list(ctx, node->payload.fun.params, NULL);
    if (!node->payload.fun.body)
        return;

    printf(" {");
    indent(ctx->printer);
    printf("\n");

    Scope scope = build_scope(node);
    ctx->scope = &scope;
    ctx->fn = node;
    print_abs_body(ctx, node);
    dispose_scope(&scope);

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
            printf("fn");
            printf(RESET);
            print_yield_types(ctx, node->payload.fn_type.return_types);
            print_param_types(ctx, node->payload.fn_type.param_types);
            break;
        }
        case BBType_TAG: {
            printf("cont");
            printf(RESET);
            print_param_types(ctx, node->payload.bb_type.param_types);
            break;
        }
        case LamType_TAG: {
            printf("lambda");
            print_param_types(ctx, node->payload.lam_type.param_types);
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
                case IntTy8:  printf("%" PRIu8 ,  node->payload.int_literal.value.i8);  break;
                case IntTy16: printf("%" PRIu16, node->payload.int_literal.value.i16); break;
                case IntTy32: printf("%" PRIu32, node->payload.int_literal.value.i32); break;
                case IntTy64: printf("%" PRIu64, node->payload.int_literal.value.i64); break;
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
            Nodes ty_args = node->payload.prim_op.type_arguments;
            if (ty_args.count > 0)
                print_ty_args_list(ctx, node->payload.prim_op.type_arguments);
            print_args_list(ctx, node->payload.prim_op.operands);
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
            assert(is_anonymous_lambda(body));
            print_param_list(ctx, body->payload.anon_lam.params, &node->payload.loop_instr.initial_args);
            print_lambda_body(ctx, body);
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
                print_lambda_body(ctx, node->payload.match_instr.cases.nodes[i]);
            }

            printf("\n");
            printf(GREEN);
            printf("default");
            printf(RESET);
            printf(": ");
            print_lambda_body(ctx, node->payload.match_instr.default_case);

            deindent(ctx->printer);
            printf("\n}");
            break;
        case Control_TAG:
            printf(BGREEN);
            printf("control");
            printf(RESET);
            print_yield_types(ctx, node->payload.control.yield_types);
            print_param_list(ctx, node->payload.control.inside->payload.anon_lam.params, NULL);
            print_lambda_body(ctx, node->payload.control.inside);
            break;
    }
}

static void print_terminator(PrinterCtx* ctx, const Node* node) {
    TerminatorTag tag = is_terminator(node);
    switch (tag) {
        case NotATerminator: assert(false);
        case LetMut_TAG:
        case LetIndirect_TAG:
        case Let_TAG: {
            const Node* tail = node->payload.let.tail;
            if (is_anonymous_lambda(tail)) {
                assert(tag != LetIndirect_TAG);
                // if the let tail is a lambda, we apply some syntactic sugar
                if (tail->payload.anon_lam.params.count > 0) {
                    printf(GREEN);
                    if (tag == LetMut_TAG)
                        printf("var");
                    else
                        printf("val");
                    printf(RESET);
                    Nodes params = tail->payload.anon_lam.params;
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
                print_node(node->payload.let.tail->payload.anon_lam.body);
            } else {
                assert(tag == LetIndirect_TAG);
                printf(GREEN);
                printf("let ");
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
            for (size_t i = 0; i < node->payload.fn_ret.args.count; i++) {
                printf(" ");
                print_node(node->payload.fn_ret.args.nodes[i]);
            }
            break;
        case Terminator_TailCall_TAG:
            printf(BGREEN);
            printf("tail_call ");
            printf(RESET);
            print_node(node->payload.tail_call.target);
            print_args_list(ctx, node->payload.tail_call.args);
            break;
        case Jump_TAG:
            printf(BGREEN);
            printf("jump ");
            printf(RESET);
            print_node(node->payload.jump.target);
            print_args_list(ctx, node->payload.jump.args);
            break;
        case Branch_TAG:
            printf(BGREEN);
            printf("branch ");
            printf(RESET);
            printf("(");
            print_node(node->payload.branch.branch_condition);
            printf(", ");
            print_node(node->payload.branch.true_target);
            printf(", ");
            print_node(node->payload.branch.false_target);
            printf(")");
            print_args_list(ctx, node->payload.branch.args);
            break;
        case Switch_TAG:
            printf(BGREEN);
            printf("br_switch ");
            printf(RESET);
            printf("(");
            print_node(node->payload.br_switch.switch_value);
            printf(", ");
            for (size_t i = 0; i < node->payload.br_switch.case_values.count; i++) {
                print_node(node->payload.br_switch.case_values.nodes[i]);
                printf(", ");
                print_node(node->payload.br_switch.case_targets.nodes[i]);
                if (i + 1 < node->payload.br_switch.case_values.count)
                    printf(", ");
            }
            printf(", ");
            print_node(node->payload.br_switch.default_target);
            printf(") ");
            print_args_list(ctx, node->payload.br_switch.args);
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
            printf("unreachable");
            printf(RESET);
            break;
        case MergeSelection_TAG:
        case MergeContinue_TAG:
        case MergeBreak_TAG:
            printf(BGREEN);
            printf("%s", node_tags[node->tag]);
            printf(RESET);
            print_args_list(ctx, node->payload.merge_selection.args);
            break;
    }
}

static void print_decl(PrinterCtx* ctx, const Node* node) {
    assert(is_declaration(node));
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
        case Function_TAG: {
            const Function* fun = &node->payload.fun;
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
    else if (node->tag == AnonLambda_TAG) {
        printf(BYELLOW);
        printf("lambda ");
        printf(RESET);
        print_param_list(ctx, node->payload.anon_lam.params, NULL);
        indent(ctx->printer);
        printf(" {\n");
        print_node(node->payload.anon_lam.body);
        printf(";");
        deindent(ctx->printer);
        printf("\n}");
    } else if (is_declaration(node)) {
        printf(BYELLOW);
        printf("%s", get_decl_name(node));
        printf(RESET);
    } else switch (node->tag) {
        case Annotation_TAG: {
            const Annotation* annotation = &node->payload.annotation;
            printf(RED);
            printf("@%s", annotation->name);
            printf(RESET);
            break;
        }
        case AnnotationValue_TAG: {
            const AnnotationValue* annotation = &node->payload.annotation_value;
            printf(RED);
            printf("@%s", annotation->name);
            printf(RESET);
            printf("(");
            print_node(annotation->value);
            printf(")");
            break;
        }
        default: error("dunno how to print %s", node_tags[node->tag]);
    }
}

static void print_mod_impl(PrinterCtx* ctx, Module* mod) {
    Nodes decls = get_module_declarations(mod);
    for (size_t i = 0; i < decls.count; i++) {
        const Node* decl = decls.nodes[i];
        print_decl(ctx, decl);
    }
}

#undef print_node
#undef printf

static void print_helper(Printer* printer, const Node* node, Module* mod, bool dump_ptrs) {
    PrinterCtx ctx = {
        .printer = printer,
        .print_ptrs = dump_ptrs,
        .color = true,
    };
    if (node)
        print_node_impl(&ctx, node);
    if (mod)
        print_mod_impl(&ctx, mod);
    flush(ctx.printer);
    destroy_printer(ctx.printer);
}

void print_node_into_str(const Node* node, char** str_ptr, size_t* size) {
    Growy* g = new_growy();
    print_helper(open_growy_as_printer(g), node, NULL, false);
    *size = growy_size(g);
    *str_ptr = growy_deconstruct(g);
}

void print_module_into_str(Module* mod, char** str_ptr, size_t* size) {
    Growy* g = new_growy();
    print_helper(open_growy_as_printer(g), NULL, mod, false);
    *size = growy_size(g);
    *str_ptr = growy_deconstruct(g);
}

void dump_node(const Node* node) {
    print_helper(open_file_as_printer(stdout), node, NULL, false);
    printf("\n");
}

void dump_module(Module* mod) {
    print_helper(open_file_as_printer(stdout), NULL, mod, false);
    printf("\n");
}

void log_node(LogLevel level, const Node* node) {
    if (level >= get_log_level())
        print_helper(open_file_as_printer(stderr), node, NULL, false);
}

void log_module(LogLevel level, Module* mod) {
    if (level >= get_log_level())
        print_helper(open_file_as_printer(stderr), NULL, mod, false);
}
