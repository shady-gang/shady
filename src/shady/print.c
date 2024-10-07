#include "shady/print.h"

#include "ir_private.h"
#include "analysis/cfg.h"
#include "analysis/scheduler.h"
#include "analysis/uses.h"
#include "analysis/leak.h"

#include "log.h"
#include "list.h"
#include "dict.h"
#include "growy.h"
#include "printer.h"

#include "type.h"
#include "visit.h"

#include <assert.h>
#include <inttypes.h>
#include <string.h>

typedef struct PrinterCtx_ PrinterCtx;

struct PrinterCtx_ {
    Printer* printer;
    NodePrintConfig config;
    const Node* fn;
    CFG* cfg;
    Scheduler* scheduler;
    const UsesMap* uses;

    Growy* root_growy;
    Printer* root_printer;

    Growy** bb_growies;
    Printer** bb_printers;
    struct Dict* emitted;
};

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

static bool print_node_impl(PrinterCtx* ctx, const Node* node);
static void print_terminator(PrinterCtx* ctx, const Node* node);
static void print_mod_impl(PrinterCtx* ctx, Module* mod);

static String emit_node(PrinterCtx* ctx, const Node* node);
static void print_mem(PrinterCtx* ctx, const Node* node);

static PrinterCtx make_printer_ctx(Printer* printer, NodePrintConfig config) {
    PrinterCtx ctx = {
        .printer = printer,
        .config = config,
        .emitted = shd_new_dict(const Node*, String, (HashFn) hash_node, (CmpFn) compare_node),
        .root_growy = shd_new_growy(),
    };
    ctx.root_printer = shd_new_printer_from_growy(ctx.root_growy);
    return ctx;
}

static void destroy_printer_ctx(PrinterCtx ctx) {
    shd_destroy_dict(ctx.emitted);
}

void shd_print_module(Printer* printer, NodePrintConfig config, Module* mod) {
    PrinterCtx ctx = make_printer_ctx(printer, config);
    print_mod_impl(&ctx, mod);
    String s = shd_printer_growy_unwrap(ctx.root_printer);
    shd_print(ctx.printer, "%s", s);
    free((void*)s);
    shd_printer_flush(ctx.printer);
    destroy_printer_ctx(ctx);
}

void shd_print_node(Printer* printer, NodePrintConfig config, const Node* node) {
    PrinterCtx ctx = make_printer_ctx(printer, config);
    String emitted = emit_node(&ctx, node);
    String s = shd_printer_growy_unwrap(ctx.root_printer);
    shd_print(ctx.printer, "%s%s", s, emitted);
    free((void*)s);
    shd_printer_flush(ctx.printer);
    destroy_printer_ctx(ctx);
}

void shd_print_node_into_str(const Node* node, char** str_ptr, size_t* size) {
    Growy* g = shd_new_growy();
    Printer* p = shd_new_printer_from_growy(g);
    if (node)
        shd_print(p, "%%%d ", node->id);
    shd_print_node(p, (NodePrintConfig) {.reparseable = true}, node);
    shd_destroy_printer(p);
    *size = shd_growy_size(g);
    *str_ptr = shd_growy_deconstruct(g);
}

void shd_print_module_into_str(Module* mod, char** str_ptr, size_t* size) {
    Growy* g = shd_new_growy();
    Printer* p = shd_new_printer_from_growy(g);
    shd_print_module(p, (NodePrintConfig) {.reparseable = true,}, mod);
    shd_destroy_printer(p);
    *size = shd_growy_size(g);
    *str_ptr = shd_growy_deconstruct(g);
}

void dump_node(const Node* node) {
    Printer* p = shd_new_printer_from_file(stdout);
    if (node)
        shd_print(p, "%%%d ", node->id);
    shd_print_node(p, (NodePrintConfig) {.color = true}, node);
    printf("\n");
}

void dump_module(Module* mod) {
    Printer* p = shd_new_printer_from_file(stdout);
    shd_print_module(p, (NodePrintConfig) {.color = true}, mod);
    shd_destroy_printer(p);
    printf("\n");
}

void shd_log_node(LogLevel level, const Node* node) {
    if (level <= shd_log_get_level()) {
        Printer* p = shd_new_printer_from_file(stderr);
        shd_print_node(p, (NodePrintConfig) {.color = true}, node);
        shd_destroy_printer(p);
    }
}

void shd_log_module(LogLevel level, const CompilerConfig* compiler_cfg, Module* mod) {
    NodePrintConfig config = { .color = true };
    if (compiler_cfg) {
        config.print_generated = compiler_cfg->logging.print_generated;
        config.print_builtin = compiler_cfg->logging.print_builtin;
        config.print_internal = compiler_cfg->logging.print_internal;
    }
    if (level <= shd_log_get_level()) {
        Printer* p = shd_new_printer_from_file(stderr);
        shd_print_module(p, config, mod);
        shd_destroy_printer(p);
    }
}

#define COLOR(x) (ctx->config.color ? (x) : "")

#define RESET    COLOR("\033[0m")
#define RED      COLOR("\033[0;31m")
#define GREEN    COLOR("\033[0;32m")
#define YELLOW   COLOR("\033[0;33m")
#define BLUE     COLOR("\033[0;34m")
#define MAGENTA  COLOR("\033[0;35m")
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

#define printf(...) shd_print(ctx->printer, __VA_ARGS__)
#define print_node(n) print_operand_helper(ctx, 0, n)
#define print_operand(nc, n) print_operand_helper(ctx, nc, n)

static void print_operand_helper(PrinterCtx* ctx, NodeClass nc, const Node* op);

void print_node_operand(PrinterCtx* ctx, const Node* node, String op_name, NodeClass op_class, const Node* op);
void print_node_operand_list(PrinterCtx* ctx, const Node* node, String op_name, NodeClass op_class, Nodes ops);

void print_node_generated(PrinterCtx* ctx, const Node* node);

#pragma GCC diagnostic error "-Wswitch"

static void print_param_list(PrinterCtx* ctx, Nodes params, const Nodes* defaults) {
    if (defaults != NULL)
        assert(defaults->count == params.count);
    printf("(");
    for (size_t i = 0; i < params.count; i++) {
        const Node* param = params.nodes[i];
        if (ctx->config.print_ptrs) printf("%zu::", (size_t)(void*) param);
        print_node(param->payload.param.type);
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

static String emit_abs_body(PrinterCtx* ctx, const Node* abs);

static void print_basic_block(PrinterCtx* ctx, const CFNode* node) {
    const Node* bb = node->node;
    printf(GREEN);
    printf("\ncont");
    printf(BYELLOW);
    if (bb->payload.basic_block.name && strlen(bb->payload.basic_block.name) > 0)
        printf(" %s", bb->payload.basic_block.name);
    else
        printf(" %%%d", bb->id);
    printf(RESET);
    if (ctx->config.print_ptrs) {
        printf(" %zu:: ", (size_t)(void*)bb);
    }
    print_param_list(ctx, bb->payload.basic_block.params, NULL);

    printf(" {");
    shd_printer_indent(ctx->printer);
    printf("\n");
    printf("%s", emit_abs_body(ctx, bb));
    shd_printer_deindent(ctx->printer);
    printf("\n}");
}

static String emit_abs_body(PrinterCtx* ctx, const Node* abs) {
    Growy* g = shd_new_growy();
    Printer* p = shd_new_printer_from_growy(g);
    CFNode* cfnode = ctx->cfg ? cfg_lookup(ctx->cfg, abs) : NULL;
    if (cfnode)
        ctx->bb_printers[cfnode->rpo_index] = p;

    emit_node(ctx, get_abstraction_body(abs));

    if (cfnode) {
        Growy* g2 = shd_new_growy();
        Printer* p2 = shd_new_printer_from_growy(g2);
        size_t count = cfnode->dominates->elements_count;
        for (size_t i = 0; i < count; i++) {
            const CFNode* dominated = shd_read_list(const CFNode*, cfnode->dominates)[i];
            assert(is_basic_block(dominated->node));
            PrinterCtx bb_ctx = *ctx;
            bb_ctx.printer = p2;
            print_basic_block(&bb_ctx, dominated);
            if (i + 1 < count)
                shd_newline(bb_ctx.printer);
        }

        String bbs = shd_printer_growy_unwrap(p2);
        shd_print(p, "%s", bbs);
        free((void*) bbs);
    }

    String s = shd_printer_growy_unwrap(p);
    String s2 = string(ctx->fn->arena, s);
    if (cfnode)
        ctx->bb_printers[cfnode->rpo_index] = NULL;
    free((void*) s);
    return s2;
}

static void print_function(PrinterCtx* ctx, const Node* node) {
    assert(is_function(node));

    PrinterCtx sub_ctx = *ctx;
    sub_ctx.fn = node;
    if (node->arena->config.name_bound) {
        CFGBuildConfig cfg_config = structured_scope_cfg_build();
        CFG* cfg = build_cfg(node, node, cfg_config);
        sub_ctx.cfg = cfg;
        sub_ctx.scheduler = new_scheduler(cfg);
        sub_ctx.bb_growies = calloc(sizeof(size_t), cfg->size);
        sub_ctx.bb_printers = calloc(sizeof(size_t), cfg->size);
        if (node->arena->config.check_types && node->arena->config.allow_fold) {
            sub_ctx.uses = create_fn_uses_map(node, (NcDeclaration | NcType));
        }
    }
    ctx = &sub_ctx;

    print_yield_types(ctx, node->payload.fun.return_types);
    print_param_list(ctx, node->payload.fun.params, NULL);
    if (!get_abstraction_body(node)) {
        printf(";");
    } else {
        printf(" {");
        shd_printer_indent(ctx->printer);
        printf("\n");

        printf("%s", emit_abs_body(ctx, node));

        shd_printer_deindent(ctx->printer);
        printf("\n}");
    }

    if (sub_ctx.cfg) {
        if (sub_ctx.uses)
            destroy_uses_map(sub_ctx.uses);
        free(sub_ctx.bb_printers);
        free(sub_ctx.bb_growies);
        destroy_cfg(sub_ctx.cfg);
        destroy_scheduler(sub_ctx.scheduler);
    }
}

static void print_nodes(PrinterCtx* ctx, Nodes nodes) {
    for (size_t i = 0; i < nodes.count; i++) {
        print_node(nodes.nodes[i]);
        if (i + 1 < nodes.count)
            printf(" ");
    }
}

static bool print_type(PrinterCtx* ctx, const Node* node) {
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
                default: shd_error("Not a known valid float width")
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
                default: shd_error("Not a known valid int width")
            }
            break;
        case RecordType_TAG:
            if (node->payload.record_type.members.count == 0) {
                printf("unit_t");
                break;
            } else if (node->payload.record_type.special & DecorateBlock) {
                printf("block");
            } else {
                printf("struct");
            }
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
            printf(node->payload.ptr_type.is_reference ? "ref" : "ptr");
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
            printf(", %d, %d, %d, %d]", node->payload.image_type.dim, node->payload.image_type.depth, node->payload.image_type.arrayed, node->payload.image_type.ms);
            break;
        }
        case Type_SamplerType_TAG: {
            printf("sampler_type");
            break;
        }
        case Type_SampledImageType_TAG: {
            printf("sampled");
            printf(RESET);
            printf("[");
            print_node(node->payload.sampled_image_type.image_type);
            printf(RESET);
            printf("]");
            break;
        }
        case TypeDeclRef_TAG: {
            printf("%s", get_declaration_name(node->payload.type_decl_ref.decl));
            break;
        }
        default:
            print_node_generated(ctx, node);
            break;
    }
    printf(RESET);
    return true;
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

static bool print_value(PrinterCtx* ctx, const Node* node) {
    switch (is_value(node)) {
        case NotAValue: assert(false); break;
        case ConstrainedValue_TAG: {
            print_node(node->payload.constrained.type);
            printf(" ");
            print_node(node->payload.constrained.value);
            break;
        }
        case Value_Param_TAG:
            printf(YELLOW);
            String name = get_value_name_unsafe(node);
            if (name && strlen(name) > 0)
                printf("%s_", name);
            printf("%%%d", node->id);
            printf(RESET);
            return true;
        case UntypedNumber_TAG:
            printf(BBLUE);
            printf("%s", node->payload.untyped_number.plaintext);
            printf(RESET);
            break;
        case IntLiteral_TAG:
            printf(BBLUE);
            uint64_t v = get_int_literal_value(node->payload.int_literal, false);
            switch (node->payload.int_literal.width) {
                case IntTy8:  printf("%" PRIu8,  (uint8_t)  v);  break;
                case IntTy16: printf("%" PRIu16, (uint16_t) v); break;
                case IntTy32: printf("%" PRIu32, (uint32_t) v); break;
                case IntTy64: printf("%" PRIu64, v); break;
                default: shd_error("Not a known valid int width")
            }
            printf(RESET);
            return true;
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
                default: shd_error("Not a known valid float width")
            }
            printf(RESET);
            return true;
        case True_TAG:
            printf(BBLUE);
            printf("true");
            printf(RESET);
            return true;
        case False_TAG:
            printf(BBLUE);
            printf("false");
            printf(RESET);
            return true;
        case StringLiteral_TAG:
            printf(BBLUE);
            print_string_lit(ctx, node->payload.string_lit.string);
            printf(RESET);
            return true;
        case Value_Undef_TAG: {
            const Type* type = node->payload.undef.type;
            printf(BBLUE);
            printf("undef");
            printf(RESET);
            printf("[");
            print_node(type);
            printf(RESET);
            printf("]");
            return true;
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
            return true;
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
            return true;
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
            return true;
        }
        case Value_RefDecl_TAG: {
            printf(BYELLOW);
            printf("%s", (char*) get_declaration_name(node->payload.ref_decl.decl));
            printf(RESET);
            return true;
        }
        case FnAddr_TAG:
            printf(BYELLOW);
            printf("%s", (char*) get_declaration_name(node->payload.fn_addr.fn));
            printf(RESET);
            return true;
        default:
            print_node_generated(ctx, node);
            break;
    }
    return false;
}

static void print_instruction(PrinterCtx* ctx, const Node* node) {
    //printf("%%%d = ", node->id);
    switch (is_instruction(node)) {
        case NotAnInstruction: assert(false); break;
        // case Instruction_Comment_TAG: {
        //     printf(MAGENTA);
        //     printf("/* %s */", node->payload.comment.string);
        //     printf(RESET);
        //     break;
        // } case PrimOp_TAG: {
        //     printf(GREEN);
        //     printf("%s", get_primop_name(node->payload.prim_op.op));
        //     printf(RESET);
        //     Nodes ty_args = node->payload.prim_op.type_arguments;
        //     if (ty_args.count > 0)
        //         print_ty_args_list(ctx, node->payload.prim_op.type_arguments);
        //     print_args_list(ctx, node->payload.prim_op.operands);
        //     break;
        // } case Call_TAG: {
        //     printf(GREEN);
        //     printf("call");
        //     printf(RESET);
        //     printf(" (");
        //     shd_print_node(node->payload.call.callee);
        //     printf(")");
        //     print_args_list(ctx, node->payload.call.args);
        //     break;
        // }
        default: print_node_generated(ctx, node);
    }
    //printf("\n");
}

static void print_jump(PrinterCtx* ctx, const Node* node) {
    assert(node->tag == Jump_TAG);
    print_node(node->payload.jump.target);
    print_args_list(ctx, node->payload.jump.args);
}

static void print_structured_construct_results(PrinterCtx* ctx, const Node* tail_case) {
    Nodes params = get_abstraction_params(tail_case);
    if (params.count > 0) {
        printf(GREEN);
        printf("val");
        printf(RESET);
        for (size_t i = 0; i < params.count; i++) {
            // TODO: fix let mut
            if (tail_case->arena->config.check_types) {
                printf(" ");
                print_node(params.nodes[i]->type);
            }
            printf(" ");
            print_node(params.nodes[i]);
            printf(RESET);
        }
        printf(" = ");
    }
}

static void print_terminator(PrinterCtx* ctx, const Node* node) {
    TerminatorTag tag = is_terminator(node);
    switch (tag) {
        case NotATerminator: assert(false);
        /*
        case If_TAG: {
            print_structured_construct_results(ctx, get_structured_construct_tail(node));
            printf(GREEN);
            printf("if");
            printf(RESET);
            print_yield_types(ctx, node->payload.if_instr.yield_types);
            printf("(");
            print_node(node->payload.if_instr.condition);
            printf(") ");
            if (ctx->config.in_cfg)
                break;
            print_case_body(ctx, node->payload.if_instr.if_true);
            if (node->payload.if_instr.if_false) {
                printf(GREEN);
                printf(" else ");
                printf(RESET);
                print_case_body(ctx, node->payload.if_instr.if_false);
            }
            printf("\n");
            print_abs_body(ctx, get_structured_construct_tail(node));
            break;
        } case Match_TAG: {
            print_structured_construct_results(ctx, get_structured_construct_tail(node));
            printf(GREEN);
            printf("match");
            printf(RESET);
            print_yield_types(ctx, node->payload.match_instr.yield_types);
            printf("(");
            print_node(node->payload.match_instr.inspect);
            printf(")");
            if (ctx->config.in_cfg)
                break;
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
            printf("\n");
            print_abs_body(ctx, get_structured_construct_tail(node));
            break;
        } case Loop_TAG: {
            print_structured_construct_results(ctx, get_structured_construct_tail(node));
            printf(GREEN);
            printf("loop");
            printf(RESET);
            print_yield_types(ctx, node->payload.loop_instr.yield_types);
            if (ctx->config.in_cfg)
                break;
            const Node* body = node->payload.loop_instr.body;
            print_param_list(ctx, get_abstraction_params(body), &node->payload.loop_instr.initial_args);
            print_case_body(ctx, body);
            printf("\n");
            print_abs_body(ctx, get_structured_construct_tail(node));
            break;
        } case Control_TAG: {
            print_structured_construct_results(ctx, get_structured_construct_tail(node));
            printf(BGREEN);
            if (ctx->uses) {
                if (is_control_static(ctx->uses, node))
                    printf("static ");
            }
            printf("control");
            printf(RESET);
            print_yield_types(ctx, node->payload.control.yield_types);
            if (ctx->config.in_cfg)
                break;
            print_param_list(ctx, get_abstraction_params(node->payload.control.inside), NULL);
            print_case_body(ctx, node->payload.control.inside);
            printf("\n");
            print_abs_body(ctx, get_structured_construct_tail(node));
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
            print_node(node->payload.branch.condition);
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
            shd_print_node(node->payload.join.join_point);
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
        case Terminator_MergeSelection_TAG:
            printf(BGREEN);
            printf("%s", node_tags[node->tag]);
            printf(RESET);
            print_args_list(ctx, node->payload.merge_selection.args);
            printf(";");
            break;*/
        default:
            print_node_generated(ctx, node);
            return;
    }
    emit_node(ctx, get_terminator_mem(node));
}

static void print_decl(PrinterCtx* ctx, const Node* node) {
    assert(is_declaration(node));
    if (!ctx->config.print_generated && lookup_annotation(node, "Generated"))
        return;
    if (!ctx->config.print_internal && lookup_annotation(node, "Internal"))
        return;
    if (!ctx->config.print_builtin && lookup_annotation(node, "Builtin"))
        return;

    PrinterCtx sub_ctx = *ctx;
    sub_ctx.cfg = NULL;
    sub_ctx.scheduler = NULL;
    ctx = &sub_ctx;

    switch (node->tag) {
        case GlobalVariable_TAG: {
            const GlobalVariable* gvar = &node->payload.global_variable;
            print_nodes(ctx, gvar->annotations);
            printf("\n");
            printf(BLUE);
            printf("var ");
            printf(BLUE);
            printf(get_address_space_name(gvar->address_space));
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
            print_nodes(ctx, cnst->annotations);
            printf("\n");
            printf(BLUE);
            printf("const ");
            printf(RESET);
            print_node(node->type);
            printf(BYELLOW);
            printf(" %s", cnst->name);
            printf(RESET);
            if (cnst->value) {
                printf(" = %s", emit_node(ctx, cnst->value));
            }
            printf(";\n");
            break;
        }
        case Function_TAG: {
            const Function* fun = &node->payload.fun;
            print_nodes(ctx, fun->annotations);
            printf("\n");
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
            print_nodes(ctx, nom->annotations);
            printf("\n");
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
        default: shd_error("Not a decl");
    }
}

static void print_annotation(PrinterCtx* ctx, const Node* node) {
    switch (is_annotation(node)) {
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
        case NotAnAnnotation: shd_error("");
    }
}

static String emit_node(PrinterCtx* ctx, const Node* node) {
    if (node == NULL) {
        return "?";
    }

    String* found = shd_dict_find_value(const Node*, String, ctx->emitted, node);
    if (found)
        return *found;

    bool print_def = true;
    String r;
    if (is_declaration(node)) {
        r = get_declaration_name(node);
        print_def = false;
    } else if (is_param(node) || is_basic_block(node) || node->tag == RefDecl_TAG || node->tag == FnAddr_TAG) {
        print_def = false;
        r = shd_fmt_string_irarena(node->arena, "%%%d", node->id);
    } else {
        r = shd_fmt_string_irarena(node->arena, "%%%d", node->id);
    }
    shd_dict_insert(const Node*, String, ctx->emitted, node, r);

    Growy* g = shd_new_growy();
    PrinterCtx ctx2 = *ctx;
    ctx2.printer = shd_new_printer_from_growy(g);
    bool print_inline = print_node_impl(&ctx2, node);

    String s = shd_printer_growy_unwrap(ctx2.printer);
    Printer* p = ctx->root_printer;
    if (ctx->scheduler) {
        CFNode* dst = schedule_instruction(ctx->scheduler, node);
        if (dst)
            p = ctx2.bb_printers[dst->rpo_index];
    }

    if (print_def)
        shd_print(p, "%%%d = %s\n", node->id, s);

    if (print_inline) {
        String is = string(node->arena, s);
        shd_dict_insert(const Node*, String, ctx->emitted, node, is);
        free((void*) s);
        return is;
    } else {
        free((void*) s);
        return r;
    }
}

static bool print_node_impl(PrinterCtx* ctx, const Node* node) {
    assert(node);

    if (ctx->config.print_ptrs) printf("%zu::", (size_t)(void*)node);

    if (is_type(node)) {
        return print_type(ctx, node);
    } else if (is_instruction(node))
        print_instruction(ctx, node);
    else if (is_value(node))
        return print_value(ctx, node);
    else if (is_terminator(node))
        print_terminator(ctx, node);
    else if (is_declaration(node)) {
        printf(BYELLOW);
        printf("%s", get_declaration_name(node));
        printf(RESET);
    } else if (is_annotation(node)) {
        print_annotation(ctx, node);
        return true;
    } else if (is_basic_block(node)) {
        printf(BYELLOW);
        if (node->payload.basic_block.name && strlen(node->payload.basic_block.name) > 0)
            printf("%s", node->payload.basic_block.name);
        else
            printf("%%%d", node->id);
        printf(RESET);
        return true;
    } else {
        print_node_generated(ctx, node);
    }
    return false;
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

typedef struct {
    Visitor v;
    PrinterCtx* ctx;
} PrinterVisitor;

static void print_mem_visitor(PrinterVisitor* ctx, NodeClass nc, String opname, const Node* op, size_t i) {
    if (nc == NcMem)
        print_mem(ctx->ctx, op);
}

static void print_mem(PrinterCtx* ctx, const Node* mem) {
    PrinterVisitor pv = {
        .v = {
            .visit_op_fn = (VisitOpFn) print_mem_visitor,
        },
        .ctx = ctx,
    };
    switch (is_mem(mem)) {
        case Mem_AbsMem_TAG: return;
        case Mem_MemAndValue_TAG: return print_mem(ctx, mem->payload.mem_and_value.mem);
        default: {
            assert(is_instruction(mem));
            visit_node_operands((Visitor*) &pv, 0, mem);
            return;
        }
    }
}

static void print_operand_name_helper(PrinterCtx* ctx, String name) {
    shd_print(ctx->printer, GREY);
    shd_print(ctx->printer, "%s", name);
    shd_print(ctx->printer, RESET);
    shd_print(ctx->printer, ": ", name);
}

static void print_operand_helper(PrinterCtx* ctx, NodeClass nc, const Node* op) {
    if (getenv("SHADY_SUPER_VERBOSE_NODE_DEBUG")) {
        if (op && (is_value(op) || is_instruction(op)))
            shd_print(ctx->printer, "%%%d ", op->id);
        shd_print(ctx->printer, "%s", emit_node(ctx, op));
    } else {
        shd_print(ctx->printer, "%s", emit_node(ctx, op));
    }
}

void print_node_operand(PrinterCtx* ctx, const Node* n, String name, NodeClass op_class, const Node* op) {
    print_operand_name_helper(ctx, name);
    if (op_class == NcBasic_block)
        shd_print(ctx->printer, BYELLOW);
    print_operand_helper(ctx, op_class, op);
    shd_print(ctx->printer, RESET);
}

void print_node_operand_list(PrinterCtx* ctx, const Node* n, String name, NodeClass op_class, Nodes ops) {
    print_operand_name_helper(ctx, name);
    shd_print(ctx->printer, "[");
    for (size_t i = 0; i < ops.count; i++) {
        print_operand_helper(ctx, op_class, ops.nodes[i]);
        if (i + 1 < ops.count)
            shd_print(ctx->printer, ", ");
    }
    shd_print(ctx->printer, "]");
}

void print_node_operand_AddressSpace(PrinterCtx* ctx, const Node* n, String name, AddressSpace as) {
    print_operand_name_helper(ctx, name);
    shd_print(ctx->printer, "%s", get_address_space_name(as));
}

void print_node_operand_Op(PrinterCtx* ctx, const Node* n, String name, Op op) {
    print_operand_name_helper(ctx, name);
    shd_print(ctx->printer, "%s", get_primop_name(op));
}

void print_node_operand_RecordSpecialFlag(PrinterCtx* ctx, const Node* n, String name, RecordSpecialFlag flags) {
    print_operand_name_helper(ctx, name);
    if (flags & DecorateBlock)
        shd_print(ctx->printer, "DecorateBlock");
}

void print_node_operand_uint32_t(PrinterCtx* ctx, const Node* n, String name, uint32_t i) {
    print_operand_name_helper(ctx, name);
    shd_print(ctx->printer, "%u", i);
}

void print_node_operand_uint64_t(PrinterCtx* ctx, const Node* n, String name, uint64_t i) {
    print_operand_name_helper(ctx, name);
    shd_print(ctx->printer, "%zu", i);
}

void print_node_operand_IntSizes(PrinterCtx* ctx, const Node* n, String name, IntSizes s) {
    print_operand_name_helper(ctx, name);
    switch (s) {
        case IntTy8: shd_print(ctx->printer, "8");  break;
        case IntTy16: shd_print(ctx->printer, "16"); break;
        case IntTy32: shd_print(ctx->printer, "32"); break;
        case IntTy64: shd_print(ctx->printer, "64"); break;
    }
}

void print_node_operand_FloatSizes(PrinterCtx* ctx, const Node* n, String name, FloatSizes s) {
    print_operand_name_helper(ctx, name);
    switch (s) {
        case FloatTy16: shd_print(ctx->printer, "16"); break;
        case FloatTy32: shd_print(ctx->printer, "32"); break;
        case FloatTy64: shd_print(ctx->printer, "64"); break;
    }
}

void print_node_operand_String(PrinterCtx* ctx, const Node* n, String name, String s ){
    print_operand_name_helper(ctx, name);
    shd_print(ctx->printer, "\"%s\"", s);
}

void print_node_operand_Strings(PrinterCtx* ctx, const Node* n, String name, Strings ops) {
    print_operand_name_helper(ctx, name);
    shd_print(ctx->printer, "[");
    for (size_t i = 0; i < ops.count; i++) {
        shd_print(ctx->printer, "\"%s\"", (size_t) ops.strings[i]);
        if (i + 1 < ops.count)
            shd_print(ctx->printer, ", ");
    }
    shd_print(ctx->printer, "]");
}

void print_node_operand_bool(PrinterCtx* ctx, const Node* n, String name, bool b) {
    print_operand_name_helper(ctx, name);
    if (b)
        shd_print(ctx->printer, "true");
    else
        shd_print(ctx->printer, "false");
}

#include "print_generated.c"
