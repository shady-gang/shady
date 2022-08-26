#include "shady/ir.h"
#include "analysis/scope.h"

#include "log.h"
#include "list.h"

#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

typedef struct PrinterCtx {
    void (*print_fn)(struct PrinterCtx*, char* format, ...);
    unsigned int indent;
    bool print_ptrs;
    bool color;
} PrinterCtx;

#define COLOR(x) (ctx->color ? x : "")

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

//#define printf(...) fprintf(ctx->output, __VA_ARGS__)
#define printf(...) ctx->print_fn(ctx, __VA_ARGS__)
#define print_node(n) print_node_impl(ctx, n)

static void print_node_impl(struct PrinterCtx* ctx, const Node* node);

#define INDENT for (unsigned int j = 0; j < ctx->indent; j++) \
    printf("   ");

static void print_storage_qualifier_for_global(struct PrinterCtx* ctx, AddressSpace as) {
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
        default: error("Unknown address space: %d", (int) as);
    }
    printf(RESET);
}

static void print_ptr_addr_space(struct PrinterCtx* ctx, AddressSpace as) {
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
        default: error("Unknown address space: %d", (int) as);
    }
    printf(RESET);
}

static void print_param_list(struct PrinterCtx* ctx, Nodes vars, const Nodes* defaults) {
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

static void print_yield_types(struct PrinterCtx* ctx, Nodes types) {
    bool space = false;
    for (size_t i = 0; i < types.count; i++) {
        if (!space) {
            printf(" ");
            space = true;
        }
        print_node(types.nodes[i]);
        if (i < types.count - 1)
            printf(" ");
        //else
        //    printf("");
    }
}

static void print_function(struct PrinterCtx* ctx, const Node* node) {
    print_yield_types(ctx, node->payload.fn.return_types);
    print_param_list(ctx, node->payload.fn.params, NULL);
    printf(" {\n");
    ctx->indent++;
    print_node(node->payload.fn.block);

    if (node->type != NULL && node->payload.fn.block) {
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
            print_param_list(ctx, cfnode->node->payload.fn.params, NULL);
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

static void print_annotations(struct PrinterCtx* ctx, Nodes annotations) {
    for (size_t i = 0; i < annotations.count; i++) {
        print_node(annotations.nodes[i]);
        printf(" ");
    }
}

static void print_node_impl(struct PrinterCtx* ctx, const Node* node) {

    if (node == NULL) {
        printf("?");
        return;
    }

    if (ctx->print_ptrs) printf("%zu::", (size_t)(void*)node);

    switch (node->tag) {
        // --------------------------- TYPES
        case QualifiedType_TAG:
            printf(CYAN);
            if (node->payload.qualified_type.is_uniform)
                printf("uniform ");
            else
                printf("varying ");
            print_node(node->payload.qualified_type.type);
            printf(RESET);
            break;
        case NoRet_TAG:
            printf(BCYAN);
            printf("!");
            printf(RESET);
            break;
        case Int_TAG:
            printf(BCYAN);
            switch (node->payload.int_literal.width) {
                case IntTy8:  printf("i8");  break;
                case IntTy16: printf("i16"); break;
                case IntTy32: printf("i32"); break;
                case IntTy64: printf("i64"); break;
                default: error("Not a known valid int width")
            }
            printf(RESET);
            break;
        case Bool_TAG:
            printf(BCYAN);
            printf("bool");
            printf(RESET);
            break;
        case Float_TAG:
            printf(BCYAN);
            printf("float");
            printf(RESET);
            break;
        case MaskType_TAG:
            printf(BCYAN);
            printf("mask");
            printf(RESET);
            break;
        case RecordType_TAG:
            printf(BCYAN);
            printf("struct");
            printf(RESET);
            printf(" {");
            const Nodes* members = &node->payload.record_type.members;
            for (size_t i = 0; i < members->count; i++) {
                print_node(members->nodes[i]);
                if (i < members->count - 1)
                    printf(", ");
            }
            printf(RESET);
            printf("}");
            break;
        case FnType_TAG: {
            printf(BCYAN);
            if (node->payload.fn_type.is_basic_block) {
                printf("cont");
                printf(RESET);
            } else {
                printf("fn ");
                printf(RESET);
                const Nodes* returns = &node->payload.fn_type.return_types;
                for (size_t i = 0; i < returns->count; i++) {
                    print_node(returns->nodes[i]);
                    //if (i < returns->count - 1)
                    //    printf(", ");
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
            printf(BCYAN);
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
            printf(RESET);
            break;
        }
        case Unit_TAG: {
            printf(BCYAN);
            printf("()");
            printf(RESET);
            break;
        }
        case PackType_TAG: {
            printf(BCYAN);
            printf("pack");
            printf(RESET);
            printf("(%d", node->payload.pack_type.width);
            printf(", ");
            print_node(node->payload.pack_type.element_type);
            printf(")");
            break;
        }

        case Root_TAG: {
            const Root* top_level = &node->payload.root;
            for (size_t i = 0; i < top_level->declarations.count; i++) {
                const Node* decl = top_level->declarations.nodes[i];
                if (ctx->print_ptrs) printf("%zu::", (size_t)(void*)decl);
                if (decl->tag == GlobalVariable_TAG) {
                    const GlobalVariable* gvar = &decl->payload.global_variable;
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
                } else if (decl->tag == Function_TAG) {
                    const Function* fun = &decl->payload.fn;
                    assert(!fun->is_basic_block && "basic blocks aren't supposed to be found at the top level");
                    print_annotations(ctx, fun->annotations);
                    printf(BLUE);
                    printf("fn");
                    printf(RESET);
                    printf(BYELLOW);
                    printf(" %s", fun->name);
                    printf(RESET);
                    print_function(ctx, decl);
                    printf(";\n\n");
                } else if (decl->tag == Constant_TAG) {
                    const Constant* cnst = &decl->payload.constant;
                    print_annotations(ctx, cnst->annotations);
                    printf(BLUE);
                    printf("const ");
                    printf(RESET);
                    print_node(decl->type);
                    printf(BYELLOW);
                    printf(" %s", cnst->name);
                    printf(RESET);
                    printf(" = ");
                    print_node(cnst->value);
                    printf(";\n");
                } else error("Unammed node at the top level")
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

        case Constant_TAG:
            printf(BYELLOW);
            printf("%s", node->payload.constant.name);
            printf(RESET);
            break;
        case GlobalVariable_TAG:
            printf(BYELLOW);
            printf("%s", node->payload.global_variable.name);
            printf(RESET);
            break;
        case Function_TAG:
            printf(BYELLOW);
            printf("%s", node->payload.fn.name);
            printf(RESET);
            break;

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
        case FnAddr_TAG:
            printf("&");
            print_node(node->payload.fn_addr.fn);
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
                print_node_impl(ctx, pblock->continuations.nodes[i]);
            }
            break;
        }
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
        // ----------------- INSTRUCTIONS
        case Let_TAG:
            if (node->payload.let.variables.count > 0) {
                printf(GREEN);
                if (node->payload.let.is_mutable)
                    printf("var");
                else
                    printf("let");
                printf(RESET);
                for (size_t i = 0; i < node->payload.let.variables.count; i++) {
                    printf(" ");
                    print_node(node->payload.let.variables.nodes[i]->payload.var.type);
                    printf(YELLOW);
                    printf(" %s", node->payload.let.variables.nodes[i]->payload.var.name);
                    printf("~%d", node->payload.let.variables.nodes[i]->payload.var.id);
                    printf(RESET);
                }
                printf(" = ");
            }
            print_node(node->payload.let.instruction);
            break;
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
            print_node(node->payload.call_instr.callee);
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
            printf(")");
            printf(" {\n");
            ctx->indent++;
            print_node(node->payload.if_instr.if_true);
            ctx->indent--;
            if (node->payload.if_instr.if_false) {
                INDENT printf("} ");
                printf(GREEN);
                printf("else");
                printf(RESET);
                printf(" {\n");
                ctx->indent++;
                print_node(node->payload.if_instr.if_false);
                ctx->indent--;
            } // else if (node->payload.if_instr.)
            INDENT printf("}");
            break;
        case Loop_TAG:
            printf(GREEN);
            printf("loop");
            printf(RESET);
            print_yield_types(ctx, node->payload.loop_instr.yield_types);
            print_param_list(ctx, node->payload.loop_instr.params, &node->payload.loop_instr.initial_args);
            printf(" {\n");
            ctx->indent++;
            print_node(node->payload.loop_instr.body);
            ctx->indent--;
            INDENT printf("}");
            break;
        case Match_TAG:
            printf(GREEN);
            printf("match");
            printf(RESET);
            print_yield_types(ctx, node->payload.match_instr.yield_types);
            printf("(");
            print_node(node->payload.match_instr.inspect);
            printf(")");
            printf(" {\n");
            ctx->indent++;
            for (size_t i = 0; i < node->payload.match_instr.literals.count; i++) {
                INDENT
                printf("case ");
                print_node(node->payload.match_instr.literals.nodes[i]);
                printf(": {\n");
                ctx->indent++;
                print_node(node->payload.match_instr.cases.nodes[i]);
                ctx->indent--;
                INDENT printf("}\n");
            }

            INDENT
            printf("default");
            printf(": {\n");
            ctx->indent++;
            print_node(node->payload.match_instr.default_case);
            ctx->indent--;
            INDENT printf("}\n");

            ctx->indent--;
            INDENT printf("}");
            break;
        // --------------------- TERMINATORS
        case Return_TAG:
            printf(BGREEN);
            printf("return");
            printf(RESET);
            for (size_t i = 0; i < node->payload.fn_ret.values.count; i++) {
                printf(" ");
                print_node(node->payload.fn_ret.values.nodes[i]);
            }
            break;
        case Branch_TAG:
            printf(BGREEN);
            switch (node->payload.branch.branch_mode) {
                case BrTailcall: printf("tail_call ");   break;
                case BrJump:     printf("jump "     );        break;
                case BrIfElse:   printf("br_ifelse ");   break;
                case BrSwitch:   printf("br_switch ");   break;
                default: error("unknown branch mode");
            }
            if (node->payload.branch.yield)
                printf("yield ");
            printf(RESET);
            switch (node->payload.branch.branch_mode) {
                case BrTailcall:
                case BrJump: {
                    print_node(node->payload.branch.target);
                    break;
                }
                case BrIfElse: {
                    printf("(");
                    print_node(node->payload.branch.branch_condition);
                    printf(" ? ");
                    print_node(node->payload.branch.true_target);
                    printf(" : ");
                    print_node(node->payload.branch.false_target);
                    printf(")");
                    break;
                }
                case BrSwitch: {
                    print_node(node->payload.branch.switch_value);
                    printf(" ? (");
                    for (size_t i = 0; i < node->payload.branch.case_values.count; i++) {
                        print_node(node->payload.branch.case_values.nodes[i]);
                        printf(" ");
                        print_node(node->payload.branch.case_targets.nodes[i]);
                        if (i + 1 < node->payload.branch.case_values.count)
                            printf(", ");
                    }
                    printf(" : ");
                    print_node(node->payload.branch.default_target);
                    printf(") ");
                }
            }
            for (size_t i = 0; i < node->payload.branch.args.count; i++) {
                printf(" ");
                print_node(node->payload.branch.args.nodes[i]);
            }
            break;
        case Join_TAG:
            printf(BGREEN);
            if (node->payload.join.is_indirect)
                printf("joinf ");
            else
                printf("joinc ");
            printf(RESET);
            print_node(node->payload.join.join_at);
            printf(" ");
            print_node(node->payload.join.desired_mask);
            for (size_t i = 0; i < node->payload.join.args.count; i++) {
                printf(" ");
                print_node(node->payload.join.args.nodes[i]);
            }
            break;
        case Callc_TAG:
            printf(BGREEN);
            if (node->payload.callc.is_return_indirect)
                printf("callf ");
            else
                printf("callc ");
            printf(RESET);
            print_node(node->payload.callc.ret_cont);
            printf(" ");
            print_node(node->payload.callc.callee);
            for (size_t i = 0; i < node->payload.callc.args.count; i++) {
                printf(" ");
                print_node(node->payload.callc.args.nodes[i]);
            }
            break;
        case Unreachable_TAG:
            printf(BGREEN);
            printf("unreachable ");
            printf(RESET);
            break;
        case MergeConstruct_TAG:
            printf(BGREEN);
            printf("%s ", merge_what_string[node->payload.merge_construct.construct]);
            printf(RESET);
            for (size_t i = 0; i < node->payload.merge_construct.args.count; i++) {
                print_node(node->payload.merge_construct.args.nodes[i]);
                printf(" ");
            }
            break;
        default: error("dunno how to print %s", node_tags[node->tag]);
    }
}

#undef print_node
#undef printf

typedef struct {
    char* alloc;
    size_t size;
    size_t used;
} GrowyBuffer;

typedef struct {
    PrinterCtx base;
    GrowyBuffer buffer;
} StringPrinterCtx;

static GrowyBuffer new_growy_buffer(size_t base_size) {
    GrowyBuffer buf = {
        .used = 0,
        .size = base_size,
        .alloc = malloc(base_size)
    };
    return buf;
}

static void grow_growy_buffer(GrowyBuffer* buf, const void* src, size_t size) {
    if (buf->used + size >= buf->size) {
        buf->size *= 2;
        buf->alloc = realloc(buf->alloc, buf->size);
    }

    memcpy(buf->alloc + buf->used, src, size);
    buf->used += size;
}

static void print_into_string(StringPrinterCtx* ctx, const char* format, ...) {
    va_list args;
    size_t buf_len = 256;
    char* buf = malloc(buf_len);
    size_t real_len;
    while (true) {
        va_start(args, format);
        int written = vsnprintf(buf, buf_len, format, args);
        va_end(args);
        if (written < buf_len) {
            real_len = written;
            break;
        } else {
            buf_len *= 2;
            buf = realloc(buf, buf_len);
        }
    }

    grow_growy_buffer(&ctx->buffer, buf, real_len);

    free(buf);
}

void print_node_into_str(const Node* node, char** str_ptr, size_t* size) {
    StringPrinterCtx ctx = {
        .base = {
            .print_fn = print_into_string,
            .indent = 0,
            .print_ptrs = false,
            .color = false,
        },
        .buffer = new_growy_buffer(1024)
    };
    print_node_impl(&ctx.base, node);

    *str_ptr = ctx.buffer.alloc;
    *size = ctx.buffer.used;
}

typedef struct {
    PrinterCtx base;
    FILE* file;
} FilePrinterCtx;

static void print_into_file(FilePrinterCtx* ctx, const char* format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(ctx->file, format, args);
    va_end(args);
}

static void print_node_in_output(FILE* output, const Node* node, bool dump_ptrs) {
    FilePrinterCtx ctx = {
        .base = {
            .print_fn = print_into_file,
            .indent = 0,
            .print_ptrs = dump_ptrs,
            .color = true,
        },
        .file = output
    };
    print_node_impl(&ctx.base, node);
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
