#include "ir_private.h"

#include "shady/print.h"
#include "shady/analysis/uses.h"
#include "shady/visit.h"

#include "analysis/cfg.h"
#include "analysis/scheduler.h"
#include "analysis/leak.h"

#include "log.h"
#include "list.h"
#include "dict.h"
#include "growy.h"
#include "printer.h"

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

    int depth;

    Growy* root_growy;
    Printer* root_printer;

    Growy** bb_growies;
    Printer** bb_printers;
    struct Dict* emitted;
};

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

static bool print_node_impl(PrinterCtx* ctx, const Node* node);
static void print_terminator(PrinterCtx* ctx, const Node* node);
static void print_mod_impl(PrinterCtx* ctx, Module* mod);

static String emit_node(PrinterCtx* ctx, const Node* node);
static void print_mem(PrinterCtx* ctx, const Node* node);

static PrinterCtx make_printer_ctx(Printer* printer, NodePrintConfig config) {
    PrinterCtx ctx = {
        .printer = printer,
        .config = config,
        .emitted = shd_new_dict(const Node*, String, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
        .root_growy = shd_new_growy(),
        .depth = 0
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
    shd_print(ctx.printer, "%s\n", s);
    free((void*)s);
    shd_printer_flush(ctx.printer);
    destroy_printer_ctx(ctx);
}

void shd_print_node(Printer* printer, NodePrintConfig config, const Node* node) {
    PrinterCtx ctx = make_printer_ctx(printer, config);
    String emitted = emit_node(&ctx, node);
    String s = shd_printer_growy_unwrap(ctx.root_printer);
    if (strlen(s) > 0 && !config.only_immediate)
        shd_print(ctx.printer, "%s\n", s);
    shd_print(ctx.printer, "%s", emitted);
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
    shd_print_module(p, (NodePrintConfig) { .reparseable = true }, mod);
    shd_destroy_printer(p);
    *size = shd_growy_size(g);
    *str_ptr = shd_growy_deconstruct(g);
}

void shd_dump_module(Module* mod) {
    Printer* p = shd_new_printer_from_file(stdout);
    shd_print_module(p, (NodePrintConfig) { .color = true, .print_internal = true }, mod);
    shd_destroy_printer(p);
    printf("\n");
}

void shd_dump_module_unscheduled(Module* mod) {
    Printer* p = shd_new_printer_from_file(stdout);
    shd_print_module(p, (NodePrintConfig) { .color = true, .print_internal = true, .no_scheduling = true }, mod);
    shd_destroy_printer(p);
    printf("\n");
}

void shd_dump(const Node* node) {
    Printer* p = shd_new_printer_from_file(stdout);
    if (node)
        shd_print(p, "%%%d ", node->id);
    shd_print_node(p, (NodePrintConfig) { .color = true }, node);
    printf("\n");
}

void shd_dump_unscheduled(const Node* node) {
    Printer* p = shd_new_printer_from_file(stdout);
    if (node)
        shd_print(p, "%%%d ", node->id);
    shd_print_node(p, (NodePrintConfig) { .color = true, .no_scheduling = true }, node);
    printf("\n");
}

void shd_log_node(LogLevel level, const Node* node) {
    if (level <= shd_log_get_level()) {
        Printer* p = shd_new_printer_from_file(stderr);
        shd_print_node(p, (NodePrintConfig) { .color = true, .max_depth = 1, .only_immediate = true }, node);
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

#define ANNOTATION_COLOR RED
#define LITERAL_COLOR BBLUE
#define OPERAND_NAME_COLOR GREY

#define FUNCTION_COLOR BMAGENTA
#define BASIC_BLOCK_COLOR MAGENTA

#define TYPE_COLOR BCYAN
#define SCOPE_COLOR CYAN

#define MEM_COLOR YELLOW
#define VALUE_COLOR BYELLOW

#define printf(...) shd_print(ctx->printer, __VA_ARGS__)
#define print_node(n) print_operand_helper(ctx, 0, n)
#define print_operand(nc, n) print_operand_helper(ctx, nc, n)

static void print_operand_helper(PrinterCtx* ctx, NodeClass oc, const Node* op);

void _shd_print_node_operand(PrinterCtx* ctx, const Node* node, String op_name, NodeClass op_class, const Node* op);
void _shd_print_node_operand_list(PrinterCtx* ctx, const Node* node, String op_name, NodeClass op_class, Nodes ops);

void _shd_print_node_generated(PrinterCtx* ctx, const Node* node);

#pragma GCC diagnostic error "-Wswitch"

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

static void print_terminator_op(PrinterCtx* ctx, const Node* term) {
    if (!term) {
        printf("null");
    } else {
        printf("{");
        shd_printer_indent(ctx->printer);
        printf("\n");

        Growy* g = shd_new_growy();
        Printer* p = shd_new_printer_from_growy(g);
        CFNode* cfnode = ctx->scheduler ? shd_schedule_instruction(ctx->scheduler, term) : NULL;
        if (cfnode) {
            ctx->bb_printers[cfnode->rpo_index] = p;
            ctx->bb_growies[cfnode->rpo_index] = g;
        }

        String t = emit_node(ctx, term);

        if (cfnode) {
            Growy* g2 = shd_new_growy();
            Printer* p2 = shd_new_printer_from_growy(g2);
            size_t count = cfnode->dominates->elements_count;
            for (size_t i = 0; i < count; i++) {
                const CFNode* dominated = shd_read_list(const CFNode*, cfnode->dominates)[i];
                assert(is_basic_block(dominated->node));
                PrinterCtx bbCtx = *ctx;
                bbCtx.printer = p2;
                emit_node(&bbCtx, dominated->node);
            }

            String bbs = shd_printer_growy_unwrap(p2);
            shd_print(p, "%s", bbs);
            free((void*) bbs);
        }

        String s = shd_printer_growy_unwrap(p);
        if (cfnode)
            ctx->bb_printers[cfnode->rpo_index] = NULL;
        printf("%s", s);
        free((void*) s);
        printf("\n%s", t);

        shd_printer_deindent(ctx->printer);
        printf("\n}");
    }
}

static void print_function_body(PrinterCtx* ctx, const Node* node) {
    PrinterCtx sub_ctx = *ctx;
    sub_ctx.fn = node;
    if (node->arena->config.name_bound && !ctx->config.no_scheduling) {
        CFGBuildConfig cfg_config = structured_scope_cfg_build();
        CFG* cfg = shd_new_cfg(node, node, cfg_config);
        sub_ctx.cfg = cfg;
        sub_ctx.scheduler = shd_new_scheduler(cfg);
        sub_ctx.bb_growies = calloc(sizeof(size_t), cfg->size);
        sub_ctx.bb_printers = calloc(sizeof(size_t), cfg->size);
        if (node->arena->config.check_types && node->arena->config.allow_fold) {
            sub_ctx.uses = shd_new_uses_map_fn(node, (NcFunction | NcType));
        }
    }
    ctx = &sub_ctx;

    _shd_print_node_generated(ctx, node);

    if (sub_ctx.cfg) {
        if (sub_ctx.uses)
            shd_destroy_uses_map(sub_ctx.uses);
        free(sub_ctx.bb_printers);
        free(sub_ctx.bb_growies);
        shd_destroy_cfg(sub_ctx.cfg);
        shd_destroy_scheduler(sub_ctx.scheduler);
    }
}

String shd_get_scope_name(ShdScope scope) {
    switch (scope) {
        case ShdScopeTop:        return "CrossDevice";
        case ShdScopeDevice:     return "Device";
        case ShdScopeWorkgroup:  return "Workgroup";
        case ShdScopeSubgroup:   return "Subgroup";
        case ShdScopeInvocation: return "Invocation";
    }
}

static void print_scope(PrinterCtx* ctx, ShdScope scope) {
    printf("%s", shd_get_scope_name(scope));
}

static bool print_type(PrinterCtx* ctx, const Node* node) {
    printf(TYPE_COLOR);
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
            printf(SCOPE_COLOR);
            print_scope(ctx, node->payload.qualified_type.scope);
            printf(RESET);
            printf(" ");
            print_operand_helper(ctx, NcType, node->payload.qualified_type.type);
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
            printf(shd_get_address_space_name(node->payload.ptr_type.address_space));
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
        default:_shd_print_node_generated(ctx, node);
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
        case Value_Param_TAG:
            printf(VALUE_COLOR);
            String name = shd_get_node_name_unsafe(node);
            if (name && strlen(name) > 0)
                printf("%s_", name);
            printf("%%%d", node->id);
            printf(RESET);
            return true;
        case UntypedNumber_TAG:
            printf(LITERAL_COLOR);
            printf("%s", node->payload.untyped_number.plaintext);
            printf(RESET);
            break;
        case IntLiteral_TAG:
            printf(LITERAL_COLOR);
            uint64_t v = shd_get_int_literal_value(node->payload.int_literal, false);
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
            printf(LITERAL_COLOR);
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
            printf(LITERAL_COLOR);
            printf("true");
            printf(RESET);
            return true;
        case False_TAG:
            printf(LITERAL_COLOR);
            printf("false");
            printf(RESET);
            return true;
        case StringLiteral_TAG:
            printf(LITERAL_COLOR);
            print_string_lit(ctx, node->payload.string_lit.string);
            printf(RESET);
            return true;
        case Value_Undef_TAG: {
            const Type* type = node->payload.undef.type;
            printf(LITERAL_COLOR);
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
            printf(LITERAL_COLOR);
            printf("null");
            printf(RESET);
            printf("[");
            print_node(type);
            printf(RESET);
            printf("]");
            return true;
        }
        /*case Value_Composite_TAG: {
            const Type* type = node->payload.composite.type;
            printf(LITERAL_COLOR);
            printf("composite");
            printf(RESET);
            printf("[");
            print_node(type);
            printf("]");
            print_args_list(ctx, node->payload.composite.contents);
            return false;
        }
        case Value_Fill_TAG: {
            const Type* type = node->payload.fill.type;
            printf(LITERAL_COLOR);
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
        }*/
        case FnAddr_TAG:
            printf(GREEN);
            printf("FnAddr");
            printf(RESET);
            printf("(");
            printf(FUNCTION_COLOR);
            printf("%s", (char*) emit_node(ctx, node->payload.fn_addr.fn));
            printf(RESET);
            printf(")");
            return true;
        default:
            _shd_print_node_generated(ctx, node);
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
        default: _shd_print_node_generated(ctx, node);
    }
    //printf("\n");
}

static void print_terminator(PrinterCtx* ctx, const Node* node) {
    emit_node(ctx, get_terminator_mem(node));
    TerminatorTag tag = is_terminator(node);
    switch (tag) {
        case NotATerminator: assert(false);
        default:
            _shd_print_node_generated(ctx, node);
            return;
    }
}

static void print_decl(PrinterCtx* ctx, const Node* node) {
    assert(is_declaration(node));

    PrinterCtx sub_ctx = *ctx;
    sub_ctx.cfg = NULL;
    sub_ctx.scheduler = NULL;
    ctx = &sub_ctx;

    switch (node->tag) {
        case Function_TAG: {
            print_function_body(ctx, node);
            break;
        }
        default:
            _shd_print_node_generated(ctx, node);
            break;
    }
}

static void print_annotation(PrinterCtx* ctx, const Node* node) {
    switch (is_annotation(node)) {
        case Annotation_TAG: {
            const Annotation* annotation = &node->payload.annotation;
            printf(ANNOTATION_COLOR);
            printf("@%s", annotation->name);
            printf(RESET);
            break;
        }
        case Annotation_AnnotationId_TAG: {
            const AnnotationId* annotation = &node->payload.annotation_id;
            printf(ANNOTATION_COLOR);
            printf("@%s", annotation->name);
            printf(RESET);
            printf("(");
            print_node(annotation->id);
            printf(")");
            break;
        }
        case AnnotationValue_TAG: {
            const AnnotationValue* annotation = &node->payload.annotation_value;
            printf(ANNOTATION_COLOR);
            printf("@%s", annotation->name);
            printf(RESET);
            printf("(");
            print_node(annotation->value);
            printf(")");
            break;
        }
        case AnnotationValues_TAG: {
            const AnnotationValues* annotation = &node->payload.annotation_values;
            printf(ANNOTATION_COLOR);
            printf("@%s", annotation->name);
            printf(RESET);
            print_args_list(ctx, annotation->values);
            break;
        }
        case NotAnAnnotation: shd_error("");
    }
}

static void print_node_name(PrinterCtx* ctx, const Node* node) {
    if (!node) {
        printf("null");
        return;
    }
    // avoid the safe version overhead
    String name = shd_get_node_name_unsafe(node);
    if (name && strlen(name) > 0)
        printf("%s", name);
    else
        printf("%%%d", node->id);
}

static bool print_node_impl(PrinterCtx* ctx, const Node* node) {
    assert(node);

    if (ctx->config.print_ptrs) printf("%zu::", (size_t) (void*) node);

    if (is_declaration(node)) {
        print_decl(ctx, node);
    } else if (is_type(node))
        return print_type(ctx, node);
    else if (is_instruction(node))
        print_instruction(ctx, node);
    else if (is_value(node))
        return print_value(ctx, node);
    else if (is_terminator(node)) {
        print_terminator(ctx, node);
        return true;
    } else if (is_annotation(node)) {
        print_annotation(ctx, node);
        return true;
    } else {
        _shd_print_node_generated(ctx, node);
    }
    return false;
}

static void print_mod_impl(PrinterCtx* ctx, Module* mod) {
    Nodes decls = shd_module_get_all_exported(mod);
    for (size_t i = 0; i < decls.count; i++) {
        const Node* decl = decls.nodes[i];
        emit_node(ctx, decl);
    }
}

static String emit_node(PrinterCtx* ctx, const Node* node) {
    if (node == NULL) {
        return "null";
    }

    String* found = shd_dict_find_value(const Node*, String, ctx->emitted, node);
    if (found)
        return *found;

    String printed_node_name = NULL;
    if (shd_is_node_tag_recursive(node->tag)) {
        Growy* g2 = shd_new_growy();
        PrinterCtx ctx2 = *ctx;
        ctx2.printer = shd_new_printer_from_growy(g2);

        print_node_name(&ctx2, node);
        String s = shd_printer_growy_unwrap(ctx2.printer);
        printed_node_name = shd_string(node->arena, s);
        shd_dict_insert(const Node*, String, ctx->emitted, node, printed_node_name);
        free((void*) s);
    }

    bool skip = false;
    if (!ctx->config.print_generated && shd_lookup_annotation(node, "Generated"))
        skip = true;
    if (!ctx->config.print_internal && shd_lookup_annotation(node, "Internal"))
        skip = true;
    if (!ctx->config.print_builtin && node->tag == BuiltinRef_TAG)
        skip = true;

    if (!printed_node_name) {
        printed_node_name = shd_get_node_name_safe(node);
    }

    if (skip) {
        shd_dict_insert(const Node*, String, ctx->emitted, node, printed_node_name);
        return printed_node_name;
    }

    Growy* g3 = shd_new_growy();
    PrinterCtx ctx3 = *ctx;
    ctx3.depth++;
    ctx3.printer = shd_new_printer_from_growy(g3);
    bool print_inline = print_node_impl(&ctx3, node);
    String s = shd_printer_growy_unwrap(ctx3.printer);

    // if (node->tag != Param_TAG)
    //     print_inline = false;

    if (print_inline) {
        String printed_node = shd_string(node->arena, s);
        shd_dict_insert(const Node*, String, ctx->emitted, node, printed_node);
        free((void*) s);
        return printed_node;
    }

    Printer* p = ctx->root_printer;
    Growy* g = ctx->root_growy;
    if (ctx->scheduler) {
        CFNode* dst = node->tag == BasicBlock_TAG ? shd_cfg_lookup(ctx->cfg, node)->idom : shd_schedule_instruction(ctx->scheduler, node);
        if (dst) {
            p = ctx3.bb_printers[dst->rpo_index];
            g = ctx3.bb_growies[dst->rpo_index];
            assert(p);
        }
    }

    if (shd_growy_size(g) > 0)
        shd_print(p, "\n");

    bool first = true;
    for (size_t i = 0; i < node->annotations.count; i++) {
        if (first) first = false;
        else shd_print(p, " ");
        shd_print(p, "%s", emit_node(ctx, node->annotations.nodes[i]));
    }

    if (!first)
        shd_print(p, " ");

    if (is_value(node))
        shd_print(p, VALUE_COLOR);
    else if (is_mem(node))
        shd_print(p, MEM_COLOR);
    else if (is_basic_block(node))
        shd_print(p, BASIC_BLOCK_COLOR);
    else if (is_function(node))
        shd_print(p, FUNCTION_COLOR);
    else if (is_type(node))
        shd_print(p, TYPE_COLOR);

    if (node->type && is_value(node)) {
        shd_print(p, "%s", printed_node_name);
        shd_print(p, RESET);
        String t = emit_node(ctx, node->type);
        shd_print(p, ": %s = %s", t, s);
    } else {
        shd_print(p, "%s", printed_node_name);
        shd_print(p, RESET);
        shd_print(p, " = ");
        shd_print(p, "%s", s);
    }
    free((void*) s);
    shd_dict_insert(const Node*, String, ctx->emitted, node, printed_node_name);
    return printed_node_name;
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
            shd_visit_node_operands((Visitor*) &pv, 0, mem);
            return;
        }
    }
}

static void print_operand_name_helper(PrinterCtx* ctx, String name) {
    shd_print(ctx->printer, OPERAND_NAME_COLOR);
    shd_print(ctx->printer, "%s", name);
    shd_print(ctx->printer, RESET);
    shd_print(ctx->printer, ": ", name);
}

static void print_operand_helper(PrinterCtx* ctx, NodeClass oc, const Node* op) {
    if (getenv("SHADY_SUPER_VERBOSE_NODE_DEBUG")) {
        if (op && (is_value(op) || is_instruction(op)))
            shd_print(ctx->printer, "%%%d ", op->id);
    }

    if (oc == NcBasic_block)
        shd_print(ctx->printer, BASIC_BLOCK_COLOR);
    else if (oc == NcMem)
        shd_print(ctx->printer, MEM_COLOR);
    else if (oc == NcValue)
        shd_print(ctx->printer, VALUE_COLOR);
    else if (oc == NcType)
        shd_print(ctx->printer, TYPE_COLOR);

    if (!ctx->config.max_depth || ctx->depth < ctx->config.max_depth) {
        if (oc == NcTerminator) {
            print_terminator_op(ctx, op);
        } else {
            shd_print(ctx->printer, "%s", emit_node(ctx, op));
        }
    } else {
        print_node_name(ctx, op);
    }
    shd_print(ctx->printer, RESET);

    if (oc == NcParam) {
        shd_print(ctx->printer, ": %s", emit_node(ctx, op->payload.param.type));
    }
}

void _shd_print_node_operand(PrinterCtx* ctx, const Node* n, String name, NodeClass op_class, const Node* op) {
    print_operand_name_helper(ctx, name);
    print_operand_helper(ctx, op_class, op);
    shd_print(ctx->printer, RESET);
}

void _shd_print_node_operand_list(PrinterCtx* ctx, const Node* n, String name, NodeClass op_class, Nodes ops) {
    print_operand_name_helper(ctx, name);
    shd_print(ctx->printer, "[");
    for (size_t i = 0; i < ops.count; i++) {
        print_operand_helper(ctx, op_class, ops.nodes[i]);
        if (i + 1 < ops.count)
            shd_print(ctx->printer, ", ");
    }
    shd_print(ctx->printer, "]");
}

void _shd_print_node_operand_AddressSpace(PrinterCtx* ctx, const Node* n, String name, AddressSpace as) {
    print_operand_name_helper(ctx, name);
    shd_print(ctx->printer, BLUE);
    shd_print(ctx->printer, "%s", shd_get_address_space_name(as));
    shd_print(ctx->printer, RESET);
}

void _shd_print_node_operand_Builtin(PrinterCtx* ctx, const Node* n, String name, Builtin b) {
    print_operand_name_helper(ctx, name);
    shd_print(ctx->printer, BLUE);
    shd_print(ctx->printer, "%s", shd_get_builtin_name(b));
    shd_print(ctx->printer, RESET);
}

void _shd_print_node_operand_Op(PrinterCtx* ctx, const Node* n, String name, Op op) {
    print_operand_name_helper(ctx, name);
    shd_print(ctx->printer, BLUE);
    shd_print(ctx->printer, "%s", shd_get_primop_name(op));
    shd_print(ctx->printer, RESET);
}

void _shd_print_node_operand_RecordSpecialFlag(PrinterCtx* ctx, const Node* n, String name, RecordSpecialFlag flags) {
    print_operand_name_helper(ctx, name);
    if (flags & DecorateBlock)
        shd_print(ctx->printer, "DecorateBlock");
}

void _shd_print_node_operand_uint32_t(PrinterCtx* ctx, const Node* n, String name, uint32_t i) {
    print_operand_name_helper(ctx, name);
    shd_print(ctx->printer, "%u", i);
}

void _shd_print_node_operand_uint64_t(PrinterCtx* ctx, const Node* n, String name, uint64_t i) {
    print_operand_name_helper(ctx, name);
    shd_print(ctx->printer, "%zu", i);
}

void _shd_print_node_operand_IntSizes(PrinterCtx* ctx, const Node* n, String name, IntSizes s) {
    print_operand_name_helper(ctx, name);
    switch (s) {
        case IntTy8: shd_print(ctx->printer, "8");  break;
        case IntTy16: shd_print(ctx->printer, "16"); break;
        case IntTy32: shd_print(ctx->printer, "32"); break;
        case IntTy64: shd_print(ctx->printer, "64"); break;
    }
}

void _shd_print_node_operand_FloatSizes(PrinterCtx* ctx, const Node* n, String name, FloatSizes s) {
    print_operand_name_helper(ctx, name);
    switch (s) {
        case FloatTy16: shd_print(ctx->printer, "16"); break;
        case FloatTy32: shd_print(ctx->printer, "32"); break;
        case FloatTy64: shd_print(ctx->printer, "64"); break;
    }
}

void _shd_print_node_operand_String(PrinterCtx* ctx, const Node* n, String name, String s ){
    print_operand_name_helper(ctx, name);
    shd_print(ctx->printer, LITERAL_COLOR);
    if (s)
        shd_print(ctx->printer, "\"%s\"", s);
    else
        shd_print(ctx->printer, "null");
    shd_print(ctx->printer, RESET);
}

void _shd_print_node_operand_Strings(PrinterCtx* ctx, const Node* n, String name, Strings ops) {
    print_operand_name_helper(ctx, name);
    shd_print(ctx->printer, "[");
    for (size_t i = 0; i < ops.count; i++) {
        shd_print(ctx->printer, LITERAL_COLOR);
        if (ops.strings[i])
            shd_print(ctx->printer, "\"%s\"", ops.strings[i]);
        else
            shd_print(ctx->printer, "null");
        shd_print(ctx->printer, RESET);
        if (i + 1 < ops.count)
            shd_print(ctx->printer, ", ");
    }
    shd_print(ctx->printer, "]");
}

void _shd_print_node_operand_bool(PrinterCtx* ctx, const Node* n, String name, bool b) {
    print_operand_name_helper(ctx, name);
    if (b)
        shd_print(ctx->printer, "true");
    else
        shd_print(ctx->printer, "false");
}

static void _shd_print_node_operand_ShdScope(PrinterCtx* ctx, const Node* n, String name, ShdScope scope) {
    print_operand_name_helper(ctx, name);
    shd_print(ctx->printer, SCOPE_COLOR);
    print_scope(ctx, scope);
    shd_print(ctx->printer, RESET);
}


#include "print_generated.c"
