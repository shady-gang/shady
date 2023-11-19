#include "ir_private.h"
#include "analysis/scope.h"
#include "analysis/uses.h"
#include "analysis/leak.h"

#include "log.h"
#include "list.h"
#include "dict.h"
#include "growy.h"
#include "printer.h"

#include "type.h"

#include <assert.h>
#include <inttypes.h>
#include <string.h>

typedef struct PrinterCtx_ PrinterCtx;
typedef void (*PrintFn)(PrinterCtx* ctx, char* format, ...);

typedef struct {
    bool skip_builtin;
    bool skip_internal;
    bool skip_generated;
    bool print_ptrs;
    bool color;
    bool reparseable;
} PrintConfig;

struct PrinterCtx_ {
    Printer* printer;
    const Node* fn;
    Scope* scope;
    const UsesMap* scope_uses;
    long int min_rpo;
    PrintConfig config;
};

#define COLOR(x) (ctx->config.color ? (x) : "")

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

static void print_param_list(PrinterCtx* ctx, Nodes params, const Nodes* defaults) {
    if (defaults != NULL)
        assert(defaults->count == params.count);
    printf("(");
    for (size_t i = 0; i < params.count; i++) {
        const Node* param = params.nodes[i];
        if (ctx->config.print_ptrs) printf("%zu::", (size_t)(void*) param);
        print_node(param->payload.var.type);
        printf(" ");
        print_node(param);
        printf(RESET);
        if (defaults) {
            printf(" = ");
            print_node(defaults->nodes[i]);
        }
        if (i < params.count - 1)
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
        if (ctx->config.print_ptrs) printf("%zu::", (size_t)(void*)args.nodes[i]);
        print_node(args.nodes[i]);
        if (i < args.count - 1)
            printf(", ");
    }
    printf(")");
}

static void print_ty_args_list(PrinterCtx* ctx, Nodes args) {
    printf("[");
    for (size_t i = 0; i < args.count; i++) {
        if (ctx->config.print_ptrs) printf("%zu::", (size_t)(void*)args.nodes[i]);
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

static void print_abs_body(PrinterCtx* ctx, const Node* block);

static void print_basic_block(PrinterCtx* ctx, const Node* bb) {
    printf(GREEN);
    printf("\n\ncont");
    printf(BYELLOW);
    printf(" %s", bb->payload.basic_block.name);
    printf(RESET);
    if (ctx->config.print_ptrs) {
        printf(" %zu:: ", (size_t)(void*)bb);
        printf(" fn=%zu:: ", (size_t)(void*)bb->payload.basic_block.fn);
    }
    print_param_list(ctx, bb->payload.basic_block.params, NULL);

    printf(" {");
    indent(ctx->printer);
    printf("\n");
    print_abs_body(ctx, bb);
    deindent(ctx->printer);
    printf("\n}");
}

static void print_dominated_bbs(PrinterCtx* ctx, const CFNode* dominator) {
    assert(dominator);
    for (size_t i = 0; i < dominator->dominates->elements_count; i++) {
        const CFNode* cfnode = read_list(const CFNode*, dominator->dominates)[i];
        // ignore cases that make up basic structural dominance
        if (find_key_dict(const Node*, dominator->structurally_dominates, cfnode->node))
            continue;
        assert(is_basic_block(cfnode->node));
        print_basic_block(ctx, cfnode->node);
    }
}

static void print_abs_body(PrinterCtx* ctx, const Node* block) {
    assert(!ctx->fn || is_function(ctx->fn));
    assert(is_abstraction(block));

    print_node(get_abstraction_body(block));

    // TODO: it's likely cleaner to instead print things according to the dominator tree in the first place.
    if (ctx->scope != NULL) {
        const CFNode* dominator = scope_lookup(ctx->scope, block);
        if (ctx->min_rpo < ((long int) dominator->rpo_index)) {
            size_t save_rpo = ctx->min_rpo;
            ctx->min_rpo = dominator->rpo_index;
            print_dominated_bbs(ctx, dominator);
            ctx->min_rpo = save_rpo;
        }
    }
}

static void print_case_body(PrinterCtx* ctx, const Node* case_) {
    assert(is_case(case_));
    printf(" {");
    indent(ctx->printer);
    printf("\n");
    print_abs_body(ctx, case_);
    deindent(ctx->printer);
    printf("\n}");
}

static void print_function(PrinterCtx* ctx, const Node* node) {
    assert(is_function(node));

    PrinterCtx sub_ctx = *ctx;
    if (node->arena->config.check_op_classes) {
        Scope* scope = new_scope(node);
        sub_ctx.scope = scope;
        sub_ctx.fn = node;
        if (node->arena->config.check_types && node->arena->config.allow_fold) {
            sub_ctx.scope_uses = create_uses_map(node, (NcDeclaration | NcType));
        }
    }
    ctx = &sub_ctx;
    ctx->min_rpo = -1;

    print_yield_types(ctx, node->payload.fun.return_types);
    print_param_list(ctx, node->payload.fun.params, NULL);
    if (!node->payload.fun.body) {
        printf(";");
        return;
    }

    printf(" {");
    indent(ctx->printer);
    printf("\n");

    print_abs_body(ctx, node);

    deindent(ctx->printer);
    printf("\n}");

    if (node->arena->config.check_op_classes) {
        if (sub_ctx.scope_uses)
            destroy_uses_map(sub_ctx.scope_uses);
        destroy_scope(sub_ctx.scope);
    }
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
        case NoRet_TAG: printf("!"); break;
        case Bool_TAG: printf("bool"); break;
        case Float_TAG:
            printf("f");
            switch (node->payload.float_type.width) {
                // case FloatTy8:  printf("8");  break;
                case FloatTy16: printf("16"); break;
                case FloatTy32: printf("32"); break;
                case FloatTy64: printf("64"); break;
                default: error("Not a known valid float width")
            }
            break;
        case MaskType_TAG: printf("mask"); break;
        case QualifiedType_TAG:
            printf(node->payload.qualified_type.is_uniform ? "uniform" : "varying");
            printf(" ");
            printf(RESET);
            print_node(node->payload.qualified_type.type);
            break;
        case Int_TAG:
            printf(node->payload.int_type.is_signed ? "i" : "u");
            switch (node->payload.int_type.width) {
                case IntTy8:  printf("8");  break;
                case IntTy16: printf("16"); break;
                case IntTy32: printf("32"); break;
                case IntTy64: printf("64"); break;
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
            printf("case_");
            print_param_types(ctx, node->payload.lam_type.param_types);
            break;
        }
        case PtrType_TAG: {
            printf("ptr");
            printf(RESET);
            printf("(");
            printf(BLUE);
            printf(get_address_space_name(node->payload.ptr_type.address_space));
            printf(RESET);
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
        case Type_ImageType_TAG: {
            switch (node->payload.image_type.sampled) {
                case 0: printf("texture_or_image_type");
                case 1: printf("texture_type");
                case 2: printf("image_type");
            }
            printf(RESET);
            printf("[");
            print_node(node->payload.image_type.sampled_type);
            printf(RESET);
            printf(", %d, %d, %d, %d]", node->payload.image_type.dim, node->payload.image_type.depth, node->payload.image_type.onion, node->payload.image_type.multisample);
            break;
        }
        case Type_SamplerType_TAG: {
            printf("sampler_type");
            break;
        }
        case Type_CombinedImageSamplerType_TAG: {
            printf("sampled");
            printf(RESET);
            printf("[");
            print_node(node->payload.combined_image_sampler_type.image_type);
            printf(RESET);
            printf("]");
            break;
        }
        case TypeDeclRef_TAG: {
            printf("%s", get_decl_name(node->payload.type_decl_ref.decl));
            break;
        }
    }
    printf(RESET);
}

static void print_string_lit(PrinterCtx* ctx, const char* string) {
    printf("\"");
    while (*string) {
        switch (*string) {
            case '\a': printf("\\a");         break;
            case '\b': printf("\\b");         break;
            case '\f': printf("\\f");         break;
            case '\r': printf("\\r");         break;
            case '\n': printf("\\n");         break;
            case '\t': printf("\\t");         break;
            case '\v': printf("\\v");         break;
            case '\\': printf("\\\\");        break;
            case '\'': printf("\\\'");        break;
            case '\"': printf("\\\"");        break;
            default:   printf("%c", *string); break;
        }
        ++string;
    }
    printf("\"");
}

static void print_value(PrinterCtx* ctx, const Node* node) {
    switch (is_value(node)) {
        case NotAValue: assert(false); break;
        case ConstrainedValue_TAG: {
            print_node(node->payload.constrained.type);
            printf(" ");
            print_node(node->payload.constrained.value);
            break;
        }
        case Variable_TAG:
            if (ctx->scope_uses) {
                // if ((*find_value_dict(const Node*, Uses*, ctx->uses->map, node))->escapes_defining_block)
                //     printf(MANGENTA);
                // else
                    printf(YELLOW);
            } else
                printf(YELLOW);
            if (ctx->config.reparseable)
                printf("%s_%d", node->payload.var.name, node->payload.var.id);
            else
                printf("%s~%d", node->payload.var.name, node->payload.var.id);
            printf(RESET);
            break;
        case UntypedNumber_TAG:
            printf(BBLUE);
            printf("%s", node->payload.untyped_number.plaintext);
            printf(RESET);
            break;
        case IntLiteral_TAG:
            printf(BBLUE);
            uint64_t v = get_int_literal_value(node, false);
            switch (node->payload.int_literal.width) {
                case IntTy8:  printf("%" PRIu8,  (uint8_t)  v);  break;
                case IntTy16: printf("%" PRIu16, (uint16_t) v); break;
                case IntTy32: printf("%" PRIu32, (uint32_t) v); break;
                case IntTy64: printf("%" PRIu64, v); break;
                default: error("Not a known valid int width")
            }
            printf(RESET);
            break;
        case FloatLiteral_TAG:
            printf(BBLUE);
            switch (node->payload.float_literal.width) {
                case FloatTy16: printf("%" PRIu16, (uint16_t) node->payload.float_literal.value); break;
                case FloatTy32: {
                    float f;
                    memcpy(&f, &node->payload.float_literal.value, sizeof(uint32_t));
                    double d = (double) f;
                    printf("%.9g", d); break;
                }
                case FloatTy64: {
                    double d;
                    memcpy(&d, &node->payload.float_literal.value, sizeof(uint64_t));
                    printf("%.17g", d); break;
                }
                default: error("Not a known valid float width")
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
            print_string_lit(ctx, node->payload.string_lit.string);
            printf(RESET);
            break;
        case Value_Undef_TAG: {
            const Type* type = node->payload.undef.type;
            printf(BBLUE);
            printf("undef");
            printf(RESET);
            printf("[");
            print_node(type);
            printf(RESET);
            printf("]");
            break;
        }
        case Value_NullPtr_TAG: {
            const Type* type = node->payload.undef.type;
            printf(BBLUE);
            printf("null");
            printf(RESET);
            printf("[");
            print_node(type);
            printf(RESET);
            printf("]");
            break;
        }
        case Value_Composite_TAG: {
            const Type* type = node->payload.composite.type;
            printf(BBLUE);
            printf("composite");
            printf(RESET);
            printf("[");
            print_node(type);
            printf("]");
            print_args_list(ctx, node->payload.composite.contents);
            break;
        }
        case Value_Fill_TAG: {
            const Type* type = node->payload.fill.type;
            printf(BBLUE);
            printf("fill");
            printf(RESET);
            printf("[");
            print_node(type);
            printf(RESET);
            printf("]");
            printf("(");
            print_node(node->payload.fill.value);
            printf(")");
            break;
        }
        case Value_RefDecl_TAG: {
            printf(BYELLOW);
            printf((char*) get_decl_name(node->payload.ref_decl.decl));
            printf(RESET);
            break;
        }
        case FnAddr_TAG:
            printf(BYELLOW);
            printf((char*) get_decl_name(node->payload.fn_addr.fn));
            printf(RESET);
            break;
    }
}

static void print_instruction(PrinterCtx* ctx, const Node* node) {
    switch (is_instruction(node)) {
        case NotAnInstruction: assert(false); break;
        case Instruction_Comment_TAG: {
            printf(GREY);
            printf("/* %s */", node->payload.comment.string);
            printf(RESET);
            break;
        } case PrimOp_TAG: {
            printf(GREEN);
            printf("%s", primop_names[node->payload.prim_op.op]);
            printf(RESET);
            Nodes ty_args = node->payload.prim_op.type_arguments;
            if (ty_args.count > 0)
                print_ty_args_list(ctx, node->payload.prim_op.type_arguments);
            print_args_list(ctx, node->payload.prim_op.operands);
            break;
        } case Call_TAG: {
            printf(GREEN);
            printf("call");
            printf(RESET);
            printf(" (");
            print_node(node->payload.call.callee);
            printf(")");
            print_args_list(ctx, node->payload.call.args);
            break;
        } case If_TAG: {
            printf(GREEN);
            printf("if");
            printf(RESET);
            print_yield_types(ctx, node->payload.if_instr.yield_types);
            printf("(");
            print_node(node->payload.if_instr.condition);
            printf(") ");
            print_case_body(ctx, node->payload.if_instr.if_true);
            if (node->payload.if_instr.if_false) {
                printf(GREEN);
                printf(" else ");
                printf(RESET);
                print_case_body(ctx, node->payload.if_instr.if_false);
            }
            break;
        } case Loop_TAG: {
            printf(GREEN);
            printf("loop");
            printf(RESET);
            print_yield_types(ctx, node->payload.loop_instr.yield_types);
            const Node* body = node->payload.loop_instr.body;
            assert(is_case(body));
            print_param_list(ctx, body->payload.case_.params, &node->payload.loop_instr.initial_args);
            print_case_body(ctx, body);
            break;
        } case Match_TAG: {
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
                print_case_body(ctx, node->payload.match_instr.cases.nodes[i]);
            }

            printf("\n");
            printf(GREEN);
            printf("default");
            printf(RESET);
            printf(": ");
            print_case_body(ctx, node->payload.match_instr.default_case);

            deindent(ctx->printer);
            printf("\n}");
            break;
        } case Control_TAG: {
            printf(BGREEN);
            if (ctx->scope_uses) {
                if (is_control_static(ctx->scope_uses, node))
                    printf("static ");
            }
            printf("control");
            printf(RESET);
            print_yield_types(ctx, node->payload.control.yield_types);
            print_param_list(ctx, node->payload.control.inside->payload.case_.params, NULL);
            print_case_body(ctx, node->payload.control.inside);
            break;
        } case Block_TAG: {
            printf(BGREEN);
            printf("block");
            printf(RESET);
            print_case_body(ctx, node->payload.block.inside);
            break;
        }
    }
}

static void print_jump(PrinterCtx* ctx, const Node* node) {
    assert(node->tag == Jump_TAG);
    print_node(node->payload.jump.target);
    print_args_list(ctx, node->payload.jump.args);
}

static void print_terminator(PrinterCtx* ctx, const Node* node) {
    TerminatorTag tag = is_terminator(node);
    switch (tag) {
        case NotATerminator: assert(false);
        case Let_TAG:
        case LetMut_TAG: {
            const Node* instruction = get_let_instruction(node);
            const Node* tail = get_let_tail(node);
            if (!ctx->config.reparseable) {
                // if the let tail is a case, we apply some syntactic sugar
                if (tail->payload.case_.params.count > 0) {
                    printf(GREEN);
                    if (tag == LetMut_TAG)
                        printf("var");
                    else
                        printf("val");
                    printf(RESET);
                    Nodes params = tail->payload.case_.params;
                    for (size_t i = 0; i < params.count; i++) {
                        if (tag == LetMut_TAG || !ctx->config.reparseable) {
                            printf(" ");
                            print_node(params.nodes[i]->payload.var.type);
                        }
                        printf(" ");
                        print_node(params.nodes[i]);
                        printf(RESET);
                    }
                    printf(" = ");
                }
                print_node(instruction);
                printf(";\n");
                print_abs_body(ctx, tail);
            } else {
                printf(GREEN);
                printf("let");
                printf(RESET);
                printf(" ");
                print_node(instruction);
                printf(GREEN);
                printf(" in ");
                printf(RESET);
                print_node(tail);
                printf(";");
            }
            break;
        } case Return_TAG:
            printf(BGREEN);
            printf("return");
            printf(RESET);
            print_args_list(ctx, node->payload.fn_ret.args);
            printf(";");
            break;
        case Terminator_TailCall_TAG:
            printf(BGREEN);
            printf("tail_call ");
            printf(RESET);
            print_node(node->payload.tail_call.target);
            print_args_list(ctx, node->payload.tail_call.args);
            printf(";");
            break;
        case Jump_TAG:
            printf(BGREEN);
            printf("jump");
            printf(RESET);
            printf(" ");
            print_jump(ctx, node);
            printf(";");
            break;
        case Branch_TAG:
            printf(BGREEN);
            printf("branch ");
            printf(RESET);
            printf("(");
            print_node(node->payload.branch.branch_condition);
            printf(", ");
            print_jump(ctx, node->payload.branch.true_jump);
            printf(", ");
            print_jump(ctx, node->payload.branch.false_jump);
            printf(")");
            printf(";");
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
                print_jump(ctx, node->payload.br_switch.case_jumps.nodes[i]);
                if (i + 1 < node->payload.br_switch.case_values.count)
                    printf(", ");
            }
            printf(", ");
            print_jump(ctx, node->payload.br_switch.default_jump);
            printf(") ");
            printf(";");
            break;
        case Join_TAG:
            printf(BGREEN);
            printf("join");
            printf(RESET);
            printf("(");
            print_node(node->payload.join.join_point);
            printf(")");
            print_args_list(ctx, node->payload.join.args);
            printf(";");
            break;
        case Unreachable_TAG:
            printf(BGREEN);
            printf("unreachable");
            printf(RESET);
            printf(";");
            break;
        case MergeContinue_TAG:
        case MergeBreak_TAG:
        case Terminator_Yield_TAG:
            printf(BGREEN);
            printf("%s", node_tags[node->tag]);
            printf(RESET);
            print_args_list(ctx, node->payload.yield.args);
            printf(";");
            break;
    }
}

static void print_decl(PrinterCtx* ctx, const Node* node) {
    assert(is_declaration(node));
    if (ctx->config.skip_generated && lookup_annotation(node, "Generated"))
        return;
    if (ctx->config.skip_internal && lookup_annotation(node, "Internal"))
        return;
    if (ctx->config.skip_builtin && lookup_annotation(node, "Builtin"))
        return;

    PrinterCtx sub_ctx = *ctx;
    sub_ctx.scope = NULL;
    ctx = &sub_ctx;

    switch (node->tag) {
        case GlobalVariable_TAG: {
            const GlobalVariable* gvar = &node->payload.global_variable;
            print_annotations(ctx, gvar->annotations);
            printf(BLUE);
            printf(get_address_space_name(gvar->address_space));
            printf(RESET);
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
            if (get_quoted_value(cnst->instruction))
                print_node(get_quoted_value(cnst->instruction));
            else
                print_node(cnst->instruction);
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
            printf("\n\n");
            break;
        }
        case NominalType_TAG: {
            const NominalType* nom = &node->payload.nom_type;
            print_annotations(ctx, nom->annotations);
            printf(BLUE);
            printf("type");
            printf(RESET);
            printf(BYELLOW);
            printf(" %s", nom->name);
            printf(RESET);
            printf(" = ");
            print_node(nom->body);
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

    if (ctx->config.print_ptrs) printf("%zu::", (size_t)(void*)node);

    if (is_type(node))
        print_type(ctx, node);
    else if (is_value(node))
        print_value(ctx, node);
    else if (is_instruction(node))
        print_instruction(ctx, node);
    else if (is_terminator(node))
        print_terminator(ctx, node);
    else if (node->tag == Case_TAG) {
        printf(BYELLOW);
        printf("case_ ");
        printf(RESET);
        print_param_list(ctx, node->payload.case_.params, NULL);
        indent(ctx->printer);
        printf(" {\n");
        print_abs_body(ctx, node);
        // printf(";");
        deindent(ctx->printer);
        printf("\n}");
    } else if (is_declaration(node)) {
        printf(BYELLOW);
        printf("%s", get_decl_name(node));
        printf(RESET);
    } else if (node->tag == Unbound_TAG) {
        printf(YELLOW);
        printf("`%s`", node->payload.unbound.name);
        printf(RESET);
    } else if (node->tag == UnboundBBs_TAG) {
        print_node(node->payload.unbound_bbs.body);
        for (size_t i = 0; i < node->payload.unbound_bbs.children_blocks.count; i++)
            print_basic_block(ctx, node->payload.unbound_bbs.children_blocks.nodes[i]);
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
        case AnnotationValues_TAG: {
            const AnnotationValues* annotation = &node->payload.annotation_values;
            printf(RED);
            printf("@%s", annotation->name);
            printf(RESET);
            print_args_list(ctx, annotation->values);
            break;
        }
        case AnnotationCompound_TAG: {
            const AnnotationValues* annotation = &node->payload.annotation_values;
            printf(RED);
            printf("@%s", annotation->name);
            printf(RESET);
            print_args_list(ctx, annotation->values);
            break;
        }
        case BasicBlock_TAG: {
            printf(BYELLOW);
            printf("%s", node->payload.basic_block.name);
            printf(RESET);
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

static void print_helper(Printer* printer, const Node* node, Module* mod, PrintConfig config) {
    PrinterCtx ctx = {
        .printer = printer,
        .config = config,
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
    print_helper(open_growy_as_printer(g), node, NULL, (PrintConfig) { .reparseable = true });
    *size = growy_size(g);
    *str_ptr = growy_deconstruct(g);
}

void print_module_into_str(Module* mod, char** str_ptr, size_t* size) {
    Growy* g = new_growy();
    print_helper(open_growy_as_printer(g), NULL, mod, (PrintConfig) { .reparseable = true, });
    *size = growy_size(g);
    *str_ptr = growy_deconstruct(g);
}

void dump_node(const Node* node) {
    print_helper(open_file_as_printer(stdout), node, NULL, (PrintConfig) { .color = true });
    printf("\n");
}

void dump_module(Module* mod) {
    print_helper(open_file_as_printer(stdout), NULL, mod, (PrintConfig) { .color = true });
    printf("\n");
}

void log_node(LogLevel level, const Node* node) {
    if (level >= get_log_level())
        print_helper(open_file_as_printer(stderr), node, NULL, (PrintConfig) { .color = true });
}

void log_module(LogLevel level, CompilerConfig* compiler_cfg, Module* mod) {
    PrintConfig config = { .color = true };
    if (compiler_cfg) {
        config.skip_generated = compiler_cfg->logging.skip_generated;
        config.skip_builtin = compiler_cfg->logging.skip_builtin;
        config.skip_internal = compiler_cfg->logging.skip_internal;
    }
    if (level >= get_log_level())
        print_helper(open_file_as_printer(stderr), NULL, mod, config);
}
