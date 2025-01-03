#include "l2s_private.h"

#include "ir_private.h"
#include "analysis/verify.h"

#include "log.h"
#include "dict.h"
#include "list.h"

#include "llvm-c/IRReader.h"

#include <assert.h>
#include <string.h>
#include <stdlib.h>

typedef struct OpaqueRef* OpaqueRef;

static KeyHash hash_opaque_ptr(OpaqueRef* pvalue) {
    if (!pvalue)
        return 0;
    size_t ptr = *(size_t*) pvalue;
    return shd_hash(&ptr, sizeof(size_t));
}

static bool cmp_opaque_ptr(OpaqueRef* a, OpaqueRef* b) {
    if (a == b)
        return true;
    if (!a ^ !b)
        return false;
    return *a == *b;
}

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

#ifdef LLVM_VERSION_MAJOR
int vcc_get_linked_major_llvm_version() {
    return LLVM_VERSION_MAJOR;
}
#else
#error "wat"
#endif

static void write_bb_body(Parser* p, FnParseCtx* fn_ctx, BBParseCtx* bb_ctx) {
    bb_ctx->builder = shd_bld_begin(bb_ctx->nbb->arena, shd_get_abstraction_mem(bb_ctx->nbb));
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
        const Node* emitted = l2s_convert_instruction(p, fn_ctx, bb_ctx->nbb, bb_ctx->builder, instr);
        if (!emitted)
            continue;
        shd_dict_insert(LLVMValueRef, const Node*, p->map, instr, emitted);
    }
    shd_log_fmt(ERROR, "Reached end of LLVM basic block without encountering a terminator!");
    SHADY_UNREACHABLE;
}

static void write_bb_tail(Parser* p, FnParseCtx* fn_ctx, BBParseCtx* bb_ctx) {
    LLVMBasicBlockRef bb = bb_ctx->bb;
    LLVMValueRef instr = LLVMGetLastInstruction(bb);
    shd_set_abstraction_body(bb_ctx->nbb, shd_bld_finish(bb_ctx->builder, l2s_convert_instruction(p, fn_ctx, bb_ctx->nbb, bb_ctx->builder, instr)));
}

static void prepare_bb(Parser* p, FnParseCtx* fn_ctx, BBParseCtx* ctx, LLVMBasicBlockRef bb) {
    IrArena* a = shd_module_get_arena(p->dst);
    shd_debug_print("l2s: preparing BB %s %d\n", LLVMGetBasicBlockName(bb), bb);
    if (shd_log_get_level() >= DEBUG)
        LLVMDumpValue((LLVMValueRef)bb);

    struct List* phis = shd_new_list(LLVMValueRef);
    Nodes params = shd_empty(a);
    LLVMValueRef instr = LLVMGetFirstInstruction(bb);
    while (instr) {
        switch (LLVMGetInstructionOpcode(instr)) {
            case LLVMPHI: {
                const Node* nparam = param_helper(a, shd_as_qualified_type(l2s_convert_type(p, LLVMTypeOf(instr)), false), "phi");
                shd_dict_insert(LLVMValueRef, const Node*, p->map, instr, nparam);
                shd_list_append(LLVMValueRef, phis, instr);
                params = shd_nodes_append(a, params, nparam);
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
        shd_dict_insert(LLVMValueRef, const Node*, p->map, bb, nbb);
        shd_dict_insert(const Node*, struct List*, fn_ctx->phis, nbb, phis);
        *ctx = (BBParseCtx) {
            .bb = bb,
            .instr = instr,
            .nbb = nbb,
        };
    }
}

static BBParseCtx* get_bb_ctx(Parser* p, FnParseCtx* fn_ctx, LLVMBasicBlockRef bb) {
    BBParseCtx** found = shd_dict_find_value(LLVMValueRef, BBParseCtx*, fn_ctx->bbs, bb);
    if (found) return *found;

    BBParseCtx* ctx = shd_arena_alloc(p->annotations_arena, sizeof(BBParseCtx));
    prepare_bb(p, fn_ctx, ctx, bb);
    shd_dict_insert(LLVMBasicBlockRef, BBParseCtx*, fn_ctx->bbs, bb, ctx);

    return ctx;
}

const Node* l2s_convert_basic_block_header(Parser* p, FnParseCtx* fn_ctx, LLVMBasicBlockRef bb) {
    const Node** found = shd_dict_find_value(LLVMValueRef, const Node*, p->map, bb);
    if (found) return *found;

    BBParseCtx* ctx = get_bb_ctx(p, fn_ctx, bb);
    return ctx->nbb;
}

const Node* l2s_convert_basic_block_body(Parser* p, FnParseCtx* fn_ctx, LLVMBasicBlockRef bb) {
    BBParseCtx* ctx = get_bb_ctx(p, fn_ctx, bb);
    if (ctx->translated)
        return ctx->nbb;

    ctx->translated = true;
    write_bb_body(p, fn_ctx, ctx);
    write_bb_tail(p, fn_ctx, ctx);
    return ctx->nbb;
}

const Node* l2s_convert_function(Parser* p, LLVMValueRef fn) {
    if (is_llvm_intrinsic(fn)) {
        shd_warn_print("Skipping unknown LLVM intrinsic function: %s\n", LLVMGetValueName(fn));
        return NULL;
    }
    if (is_shady_intrinsic(fn)) {
        shd_warn_print("Skipping shady intrinsic function: %s\n", LLVMGetValueName(fn));
        return NULL;
    }

    const Node** found = shd_dict_find_value(LLVMValueRef, const Node*, p->map, fn);
    if (found) return *found;
    IrArena* a = shd_module_get_arena(p->dst);
    shd_debug_print("Converting function: %s\n", LLVMGetValueName(fn));

    Nodes params = shd_empty(a);
    for (LLVMValueRef oparam = LLVMGetFirstParam(fn); oparam; oparam = LLVMGetNextParam(oparam)) {
        LLVMTypeRef ot = LLVMTypeOf(oparam);
        const Type* t = l2s_convert_type(p, ot);
        const Node* nparam = param_helper(a, shd_as_qualified_type(t, false), LLVMGetValueName(oparam));
        shd_dict_insert(LLVMValueRef, const Node*, p->map, oparam, nparam);
        params = shd_nodes_append(a, params, nparam);
        if (oparam == LLVMGetLastParam(fn))
            break;
    }
    const Type* fn_type = l2s_convert_type(p, LLVMGlobalGetValueType(fn));
    assert(fn_type->tag == FnType_TAG);
    assert(fn_type->payload.fn_type.param_types.count == params.count);
    Nodes annotations = shd_empty(a);
    switch (LLVMGetLinkage(fn)) {
        case LLVMExternalLinkage:
        case LLVMExternalWeakLinkage: {
            annotations = shd_nodes_append(a, annotations, annotation(a, (Annotation) { .name = "Exported" }));
            break;
        }
        default:
            break;
    }
    Node* f = function(p->dst, params, LLVMGetValueName(fn), annotations, fn_type->payload.fn_type.return_types);
    FnParseCtx fn_parse_ctx = {
        .fn = f,
        .phis = shd_new_dict(const Node*, struct List*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
        .bbs = shd_new_dict(LLVMBasicBlockRef, BBParseCtx*, (HashFn) shd_hash_ptr, (CmpFn) shd_compare_ptrs),
        .jumps_todo = shd_new_list(JumpTodo),
    };
    const Node* r = fn_addr_helper(a, f);
    r = prim_op_helper(a, reinterpret_op, shd_singleton(ptr_type(a, (PtrType) { .address_space = AsGeneric, .pointed_type = unit_type(a) })), shd_singleton(r));
    //r = prim_op_helper(a, convert_op, singleton(ptr_type(a, (PtrType) { .address_space = AsGeneric, .pointed_type = unit_type(a) })), singleton(r));
    shd_dict_insert(LLVMValueRef, const Node*, p->map, fn, r);

    size_t bb_count = LLVMCountBasicBlocks(fn);
    if (bb_count > 0) {
        LLVMBasicBlockRef first_bb = LLVMGetEntryBasicBlock(fn);
        shd_dict_insert(LLVMValueRef, const Node*, p->map, first_bb, f);

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
        shd_dict_insert(LLVMBasicBlockRef, BBParseCtx*, fn_parse_ctx.bbs, first_bb, bb0p);

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
        while (shd_dict_iter(fn_parse_ctx.phis, &i, NULL, &phis_list)) {
            shd_destroy_list(phis_list);
        }
    }
    shd_destroy_dict(fn_parse_ctx.phis);
    shd_destroy_dict(fn_parse_ctx.bbs);
    shd_destroy_list(fn_parse_ctx.jumps_todo);

    return r;
}

const Node* l2s_convert_global(Parser* p, LLVMValueRef global) {
    const Node** found = shd_dict_find_value(LLVMValueRef, const Node*, p->map, global);
    if (found) return *found;
    IrArena* a = shd_module_get_arena(p->dst);

    String name = LLVMGetValueName(global);
    String intrinsic = is_llvm_intrinsic(global);
    if (intrinsic) {
        if (strcmp(intrinsic, "llvm.global.annotations") == 0) {
            return NULL;
        }
        shd_warn_print("Skipping unknown LLVM intrinsic function: %s\n", name);
        return NULL;
    }
    shd_debug_print("Converting global: %s\n", name);

    Node* decl = NULL;

    if (LLVMIsAGlobalVariable(global)) {
        LLVMValueRef value = LLVMGetInitializer(global);
        const Type* type = l2s_convert_type(p, LLVMGlobalGetValueType(global));
        // nb: even if we have untyped pointers, they still carry useful address space info
        const Type* ptr_t = l2s_convert_type(p, LLVMTypeOf(global));
        assert(ptr_t->tag == PtrType_TAG);
        AddressSpace as = ptr_t->payload.ptr_type.address_space;
        decl = global_var(p->dst, shd_empty(a), type, name, as);
        if (value && as != AsUniformConstant)
            decl->payload.global_variable.init = l2s_convert_value(p, value);

        if (UNTYPED_POINTERS) {
            const Node* r = prim_op_helper(a, reinterpret_op, shd_singleton(ptr_t), shd_singleton(decl));
            shd_dict_insert(LLVMValueRef, const Node*, p->map, global, r);
            return r;
        }
    } else {
        const Type* type = l2s_convert_type(p, LLVMTypeOf(global));
        decl = constant(p->dst, shd_empty(a), type, name);
        decl->payload.constant.value = l2s_convert_value(p, global);
    }

    assert(decl && is_declaration(decl));
    const Node* r = decl;

    shd_dict_insert(LLVMValueRef, const Node*, p->map, global, r);
    return r;
}

bool shd_parse_llvm(const CompilerConfig* config, size_t len, const char* data, String name, Module** dst) {
    LLVMContextRef context = LLVMContextCreate();
    LLVMModuleRef src;
    LLVMMemoryBufferRef mem = LLVMCreateMemoryBufferWithMemoryRange(data, len, "my_great_buffer", false);
    char* parsing_diagnostic = "";
    if (LLVMParseIRInContext(context, mem, &src, &parsing_diagnostic)) {
        shd_error_print("Failed to parse LLVM IR\n");
        shd_error_print(parsing_diagnostic);
        shd_error_die();
    }
    shd_info_print("LLVM IR parsed successfully\n");

    ArenaConfig aconfig = shd_default_arena_config(&config->target);
    aconfig.check_types = false;
    aconfig.allow_fold = false;
    aconfig.optimisations.inline_single_use_bbs = false;

    IrArena* arena = shd_new_ir_arena(&aconfig);
    Module* dirty = shd_new_module(arena, "dirty");
    Parser p = {
        .ctx = context,
        .config = config,
        .map = shd_new_dict(LLVMValueRef, const Node*, (HashFn) hash_opaque_ptr, (CmpFn) cmp_opaque_ptr),
        .annotations = shd_new_dict(LLVMValueRef, ParsedAnnotation, (HashFn) hash_opaque_ptr, (CmpFn) cmp_opaque_ptr),
        .annotations_arena = shd_new_arena(),
        .src = src,
        .dst = dirty,
    };

    LLVMValueRef global_annotations = LLVMGetNamedGlobal(src, "llvm.global.annotations");
    if (global_annotations)
        l2s_process_llvm_annotations(&p, global_annotations);

    for (LLVMValueRef fn = LLVMGetFirstFunction(src); fn; fn = LLVMGetNextFunction(fn)) {
        l2s_convert_function(&p, fn);
    }

    LLVMValueRef global = LLVMGetFirstGlobal(src);
    while (global) {
        l2s_convert_global(&p, global);
        if (global == LLVMGetLastGlobal(src))
            break;
        global = LLVMGetNextGlobal(global);
    }
    shd_log_fmt(DEBUGVV, "Shady module parsed from LLVM:");
    shd_log_module(DEBUGVV, config, dirty);

    aconfig.check_types = true;
    aconfig.allow_fold = true;
    IrArena* arena2 = shd_new_ir_arena(&aconfig);
    *dst = shd_new_module(arena2, name);
    l2s_postprocess(&p, dirty, *dst);
    shd_log_fmt(DEBUGVV, "Shady module parsed from LLVM, after cleanup:");
    shd_log_module(DEBUGVV, config, *dst);
    shd_verify_module(config, *dst);
    shd_destroy_ir_arena(arena);

    shd_destroy_dict(p.map);
    shd_destroy_dict(p.annotations);
    shd_destroy_arena(p.annotations_arena);

    LLVMContextDispose(context);

    return true;
}
