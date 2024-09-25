#include "l2s_private.h"

#include "ir_private.h"
#include "type.h"
#include "analysis/verify.h"

#include "log.h"
#include "dict.h"
#include "list.h"
#include "util.h"
#include "portability.h"

#include "llvm-c/IRReader.h"

#include <assert.h>
#include <string.h>
#include <stdlib.h>

typedef struct OpaqueRef* OpaqueRef;

static KeyHash hash_opaque_ptr(OpaqueRef* pvalue) {
    if (!pvalue)
        return 0;
    size_t ptr = *(size_t*) pvalue;
    return hash_murmur(&ptr, sizeof(size_t));
}

static bool cmp_opaque_ptr(OpaqueRef* a, OpaqueRef* b) {
    if (a == b)
        return true;
    if (!a ^ !b)
        return false;
    return *a == *b;
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

#ifdef LLVM_VERSION_MAJOR
int vcc_get_linked_major_llvm_version() {
    return LLVM_VERSION_MAJOR;
}
#else
#error "wat"
#endif

static void write_bb_body(Parser* p, FnParseCtx* fn_ctx, BBParseCtx* bb_ctx) {
    bb_ctx->builder = begin_body_with_mem(bb_ctx->nbb->arena, get_abstraction_mem(bb_ctx->nbb));
    LLVMValueRef instr;
    LLVMBasicBlockRef bb = bb_ctx->bb;
    for (instr = bb_ctx->instr; instr; instr = LLVMGetNextInstruction(instr)) {
        bool last = instr == LLVMGetLastInstruction(bb);
        if (last)
            assert(LLVMGetBasicBlockTerminator(bb) == instr);
        // LLVMDumpValue(instr);
        // printf("\n");
        if (LLVMIsATerminatorInst(instr))
            return;
        const Node* emitted = convert_instruction(p, fn_ctx, bb_ctx->nbb, bb_ctx->builder, instr);
        if (!emitted)
            continue;
        insert_dict(LLVMValueRef, const Node*, p->map, instr, emitted);
    }
    log_string(ERROR, "Reached end of LLVM basic block without encountering a terminator!");
    SHADY_UNREACHABLE;
}

static void write_bb_tail(Parser* p, FnParseCtx* fn_ctx, BBParseCtx* bb_ctx) {
    LLVMBasicBlockRef bb = bb_ctx->bb;
    LLVMValueRef instr = LLVMGetLastInstruction(bb);
    set_abstraction_body(bb_ctx->nbb, finish_body(bb_ctx->builder, convert_instruction(p, fn_ctx, bb_ctx->nbb, bb_ctx->builder, instr)));
}

static void prepare_bb(Parser* p, FnParseCtx* fn_ctx, BBParseCtx* ctx, LLVMBasicBlockRef bb) {
    IrArena* a = get_module_arena(p->dst);
    debug_print("l2s: preparing BB %s %d\n", LLVMGetBasicBlockName(bb), bb);
    if (get_log_level() >= DEBUG)
        LLVMDumpValue((LLVMValueRef)bb);

    struct List* phis = shd_new_list(LLVMValueRef);
    Nodes params = empty(a);
    LLVMValueRef instr = LLVMGetFirstInstruction(bb);
    while (instr) {
        switch (LLVMGetInstructionOpcode(instr)) {
            case LLVMPHI: {
                const Node* nparam = param(a, qualified_type_helper(convert_type(p, LLVMTypeOf(instr)), false), "phi");
                insert_dict(LLVMValueRef, const Node*, p->map, instr, nparam);
                shd_list_append(LLVMValueRef, phis, instr);
                params = append_nodes(a, params, nparam);
                break;
            }
            default: goto after_phis;
        }
        instr = LLVMGetNextInstruction(instr);
    }
    after_phis:
    {
        String name = LLVMGetBasicBlockName(bb);
        if (strlen(name) == 0)
            name = NULL;
        Node* nbb = basic_block(a, params, name);
        insert_dict(LLVMValueRef, const Node*, p->map, bb, nbb);
        insert_dict(const Node*, struct List*, fn_ctx->phis, nbb, phis);
        *ctx = (BBParseCtx) {
            .bb = bb,
            .instr = instr,
            .nbb = nbb,
        };
    }
}

static BBParseCtx* get_bb_ctx(Parser* p, FnParseCtx* fn_ctx, LLVMBasicBlockRef bb) {
    BBParseCtx** found = find_value_dict(LLVMValueRef, BBParseCtx*, fn_ctx->bbs, bb);
    if (found) return *found;

    BBParseCtx* ctx = shd_arena_alloc(p->annotations_arena, sizeof(BBParseCtx));
    prepare_bb(p, fn_ctx, ctx, bb);
    insert_dict(LLVMBasicBlockRef, BBParseCtx*, fn_ctx->bbs, bb, ctx);

    return ctx;
}

const Node* convert_basic_block_header(Parser* p, FnParseCtx* fn_ctx, LLVMBasicBlockRef bb) {
    const Node** found = find_value_dict(LLVMValueRef, const Node*, p->map, bb);
    if (found) return *found;

    BBParseCtx* ctx = get_bb_ctx(p, fn_ctx, bb);
    return ctx->nbb;
}

const Node* convert_basic_block_body(Parser* p, FnParseCtx* fn_ctx, LLVMBasicBlockRef bb) {
    BBParseCtx* ctx = get_bb_ctx(p, fn_ctx, bb);
    if (ctx->translated)
        return ctx->nbb;

    ctx->translated = true;
    write_bb_body(p, fn_ctx, ctx);
    write_bb_tail(p, fn_ctx, ctx);
    return ctx->nbb;
}

const Node* convert_function(Parser* p, LLVMValueRef fn) {
    if (is_llvm_intrinsic(fn)) {
        warn_print("Skipping unknown LLVM intrinsic function: %s\n", LLVMGetValueName(fn));
        return NULL;
    }
    if (is_shady_intrinsic(fn)) {
        warn_print("Skipping shady intrinsic function: %s\n", LLVMGetValueName(fn));
        return NULL;
    }

    const Node** found = find_value_dict(LLVMValueRef, const Node*, p->map, fn);
    if (found) return *found;
    IrArena* a = get_module_arena(p->dst);
    debug_print("Converting function: %s\n", LLVMGetValueName(fn));

    Nodes params = empty(a);
    for (LLVMValueRef oparam = LLVMGetFirstParam(fn); oparam; oparam = LLVMGetNextParam(oparam)) {
        LLVMTypeRef ot = LLVMTypeOf(oparam);
        const Type* t = convert_type(p, ot);
        const Node* nparam = param(a, qualified_type_helper(t, false), LLVMGetValueName(oparam));
        insert_dict(LLVMValueRef, const Node*, p->map, oparam, nparam);
        params = append_nodes(a, params, nparam);
        if (oparam == LLVMGetLastParam(fn))
            break;
    }
    const Type* fn_type = convert_type(p, LLVMGlobalGetValueType(fn));
    assert(fn_type->tag == FnType_TAG);
    assert(fn_type->payload.fn_type.param_types.count == params.count);
    Nodes annotations = empty(a);
    switch (LLVMGetLinkage(fn)) {
        case LLVMExternalLinkage:
        case LLVMExternalWeakLinkage: {
            annotations = append_nodes(a, annotations, annotation(a, (Annotation) {.name = "Exported"}));
            break;
        }
        default:
            break;
    }
    Node* f = function(p->dst, params, LLVMGetValueName(fn), annotations, fn_type->payload.fn_type.return_types);
    FnParseCtx fn_parse_ctx = {
        .fn = f,
        .phis = new_dict(const Node*, struct List*, (HashFn) hash_node, (CmpFn) compare_node),
        .bbs = new_dict(LLVMBasicBlockRef, BBParseCtx*, (HashFn) hash_ptr, (CmpFn) compare_ptrs),
        .jumps_todo = shd_new_list(JumpTodo),
    };
    const Node* r = fn_addr_helper(a, f);
    r = prim_op_helper(a, reinterpret_op, singleton(ptr_type(a, (PtrType) { .address_space = AsGeneric, .pointed_type = unit_type(a) })), singleton(r));
    //r = prim_op_helper(a, convert_op, singleton(ptr_type(a, (PtrType) { .address_space = AsGeneric, .pointed_type = unit_type(a) })), singleton(r));
    insert_dict(LLVMValueRef, const Node*, p->map, fn, r);

    size_t bb_count = LLVMCountBasicBlocks(fn);
    if (bb_count > 0) {
        LLVMBasicBlockRef first_bb = LLVMGetEntryBasicBlock(fn);
        insert_dict(LLVMValueRef, const Node*, p->map, first_bb, f);

        //LLVMBasicBlockRef bb = LLVMGetNextBasicBlock(first_bb);
        //LARRAY(BBParseCtx, bbs, bb_count);
        //bbs[0] = (BBParseCtx) {
        BBParseCtx bb0 = {
            .nbb = f,
            .bb = first_bb,
            .instr = LLVMGetFirstInstruction(first_bb),
        };
        //BBParseCtx* bb0p = &bbs[0];
        BBParseCtx* bb0p = &bb0;
        insert_dict(LLVMBasicBlockRef, BBParseCtx*, fn_parse_ctx.bbs, first_bb, bb0p);

        write_bb_body(p, &fn_parse_ctx, &bb0);
        write_bb_tail(p, &fn_parse_ctx, &bb0);

        /*for (size_t i = 1;bb; bb = LLVMGetNextBasicBlock(bb)) {
            assert(i < bb_count);
            prepare_bb(p, &fn_parse_ctx, &bbs[i++], bb);
        }

        for (size_t i = 0; i < bb_count; i++) {
            write_bb_body(p, &fn_parse_ctx, &bbs[i]);
        }

        for (size_t i = 0; i < bb_count; i++) {
            write_bb_tail(p, &fn_parse_ctx, &bbs[i]);
        }*/
    }

    {
        size_t i = 0;
        struct List* phis_list;
        while (dict_iter(fn_parse_ctx.phis, &i, NULL, &phis_list)) {
            shd_destroy_list(phis_list);
        }
    }
    destroy_dict(fn_parse_ctx.phis);
    destroy_dict(fn_parse_ctx.bbs);
    shd_destroy_list(fn_parse_ctx.jumps_todo);

    return r;
}

const Node* convert_global(Parser* p, LLVMValueRef global) {
    const Node** found = find_value_dict(LLVMValueRef, const Node*, p->map, global);
    if (found) return *found;
    IrArena* a = get_module_arena(p->dst);

    String name = LLVMGetValueName(global);
    String intrinsic = is_llvm_intrinsic(global);
    if (intrinsic) {
        if (strcmp(intrinsic, "llvm.global.annotations") == 0) {
            return NULL;
        }
        warn_print("Skipping unknown LLVM intrinsic function: %s\n", name);
        return NULL;
    }
    debug_print("Converting global: %s\n", name);

    Node* decl = NULL;

    if (LLVMIsAGlobalVariable(global)) {
        LLVMValueRef value = LLVMGetInitializer(global);
        const Type* type = convert_type(p, LLVMGlobalGetValueType(global));
        // nb: even if we have untyped pointers, they still carry useful address space info
        const Type* ptr_t = convert_type(p, LLVMTypeOf(global));
        assert(ptr_t->tag == PtrType_TAG);
        AddressSpace as = ptr_t->payload.ptr_type.address_space;
        decl = global_var(p->dst, empty(a), type, name, as);
        if (value && as != AsUniformConstant)
            decl->payload.global_variable.init = convert_value(p, value);

        if (UNTYPED_POINTERS) {
            Node* untyped_wrapper = constant(p->dst, singleton(annotation(a, (Annotation) { .name = "Inline" })), ptr_t, format_string_interned(a, "%s_untyped", name));
            untyped_wrapper->payload.constant.value = ref_decl_helper(a, decl);
            untyped_wrapper->payload.constant.value = prim_op_helper(a, reinterpret_op, singleton(ptr_t), singleton(ref_decl_helper(a, decl)));
            decl = untyped_wrapper;
        }
    } else {
        const Type* type = convert_type(p, LLVMTypeOf(global));
        decl = constant(p->dst, empty(a), type, name);
        decl->payload.constant.value = convert_value(p, global);
    }

    assert(decl && is_declaration(decl));
    const Node* r = ref_decl_helper(a, decl);

    insert_dict(LLVMValueRef, const Node*, p->map, global, r);
    return r;
}

bool parse_llvm_into_shady(const CompilerConfig* config, size_t len, const char* data, String name, Module** dst) {
    LLVMContextRef context = LLVMContextCreate();
    LLVMModuleRef src;
    LLVMMemoryBufferRef mem = LLVMCreateMemoryBufferWithMemoryRange(data, len, "my_great_buffer", false);
    char* parsing_diagnostic = "";
    if (LLVMParseIRInContext(context, mem, &src, &parsing_diagnostic)) {
        error_print("Failed to parse LLVM IR\n");
        error_print(parsing_diagnostic);
        error_die();
    }
    info_print("LLVM IR parsed successfully\n");

    ArenaConfig aconfig = default_arena_config(&config->target);
    aconfig.check_types = false;
    aconfig.allow_fold = false;
    aconfig.optimisations.inline_single_use_bbs = false;

    IrArena* arena = new_ir_arena(&aconfig);
    Module* dirty = new_module(arena, "dirty");
    Parser p = {
        .ctx = context,
        .config = config,
        .map = new_dict(LLVMValueRef, const Node*, (HashFn) hash_opaque_ptr, (CmpFn) cmp_opaque_ptr),
        .annotations = new_dict(LLVMValueRef, ParsedAnnotation, (HashFn) hash_opaque_ptr, (CmpFn) cmp_opaque_ptr),
        .annotations_arena = shd_new_arena(),
        .src = src,
        .dst = dirty,
    };

    LLVMValueRef global_annotations = LLVMGetNamedGlobal(src, "llvm.global.annotations");
    if (global_annotations)
        process_llvm_annotations(&p, global_annotations);

    for (LLVMValueRef fn = LLVMGetFirstFunction(src); fn; fn = LLVMGetNextFunction(fn)) {
        convert_function(&p, fn);
    }

    LLVMValueRef global = LLVMGetFirstGlobal(src);
    while (global) {
        convert_global(&p, global);
        if (global == LLVMGetLastGlobal(src))
            break;
        global = LLVMGetNextGlobal(global);
    }
    log_string(DEBUGVV, "Shady module parsed from LLVM:");
    log_module(DEBUGVV, config, dirty);

    aconfig.check_types = true;
    aconfig.allow_fold = true;
    IrArena* arena2 = new_ir_arena(&aconfig);
    *dst = new_module(arena2, name);
    postprocess(&p, dirty, *dst);
    log_string(DEBUGVV, "Shady module parsed from LLVM, after cleanup:");
    log_module(DEBUGVV, config, *dst);
    verify_module(config, *dst);
    destroy_ir_arena(arena);

    destroy_dict(p.map);
    destroy_dict(p.annotations);
    shd_destroy_arena(p.annotations_arena);

    LLVMContextDispose(context);

    return true;
}
