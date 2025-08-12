#include "l2s_private.h"

#include "shady/pass.h"

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

int shd_get_linked_major_llvm_version() {
    return LLVM_VERSION_MAJOR;
}

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
                const Node* nparam = param_helper(a, qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.bottom, l2s_convert_type(p, LLVMTypeOf(instr))));
                l2s_apply_debug_info(p, instr, nparam);
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
        Node* nbb = basic_block_helper(a, params);
        String name = LLVMGetBasicBlockName(bb);
        if (name && strlen(name) > 0)
            shd_set_debug_name(nbb, name);
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
        shd_debug_print("Skipping LLVM intrinsic function: %s\n", LLVMGetValueName(fn));
        return NULL;
    }
    if (is_shady_intrinsic(fn)) {
        shd_debug_print("Skipping shady intrinsic function: %s\n", LLVMGetValueName(fn));
        return NULL;
    }

    const Node** found = shd_dict_find_value(LLVMValueRef, const Node*, p->map, fn);
    if (found) return *found;
    IrArena* a = shd_module_get_arena(p->dst);
    shd_debug_print("Converting function: %s\n", LLVMGetValueName(fn));

    Nodes params = shd_empty(a);
    size_t param_index = 0;
    for (LLVMValueRef oparam = LLVMGetFirstParam(fn); oparam; oparam = LLVMGetNextParam(oparam)) {
        LLVMTypeRef ot = LLVMTypeOf(oparam);
        const Type* t = l2s_convert_type(p, ot);
        const Type* byval_t = l2s_get_param_byval_attr(p, fn, param_index);
        const Node* nparam = param_helper(a, qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.bottom, t));
        if (byval_t)
            shd_add_annotation(nparam, annotation_id_helper(a, "ByVal", byval_t));
        l2s_apply_debug_info(p, oparam, nparam);
        shd_dict_insert(LLVMValueRef, const Node*, p->map, oparam, nparam);
        params = shd_nodes_append(a, params, nparam);
        if (oparam == LLVMGetLastParam(fn))
            break;
        param_index++;
    }
    const Type* fn_type = l2s_convert_type(p, LLVMGlobalGetValueType(fn));
    assert(fn_type->tag == FnType_TAG);
    assert(fn_type->payload.fn_type.param_types.count == params.count);
    Node* f = function_helper(p->dst, params, fn_type->payload.fn_type.return_types);
    String name = LLVMGetValueName(fn);
    switch (LLVMGetLinkage(fn)) {
        case LLVMExternalLinkage:
        case LLVMExternalWeakLinkage:
            assert(name && "Exported LLVM functions must be named.");
            shd_module_add_export(p->dst, name, f);
            break;
        default:
            break;
    }
    if (name)
        shd_set_debug_name(f, name);
    FnParseCtx fn_parse_ctx = {
        .fn = f,
        .phis = shd_new_dict(const Node*, struct List*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
        .bbs = shd_new_dict(LLVMBasicBlockRef, BBParseCtx*, (HashFn) shd_hash_ptr, (CmpFn) shd_compare_ptrs),
        .jumps_todo = shd_new_list(JumpTodo),
    };
    const Node* r = fn_addr_helper(a, f);
    r = generic_ptr_cast_helper(a, r);
    r = bit_cast_helper(a, ptr_type(a, (PtrType) { .address_space = AsGeneric, .pointed_type = shd_uword_type(a) }), r);
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
        shd_debugvv_print("Skipping unknown LLVM intrinsic function: %s\n", name);
        return NULL;
    }
    //shd_debugvv_print("Converting global: %s\n", name);

    Node* decl = NULL;

    if (LLVMIsAGlobalVariable(global)) {
        LLVMValueRef value = LLVMGetInitializer(global);
        const Type* type = l2s_convert_type(p, LLVMGlobalGetValueType(global));
        // nb: even if we have untyped pointers, they still carry useful address space info
        const Type* ptr_t = l2s_convert_type(p, LLVMTypeOf(global));
        assert(ptr_t->tag == PtrType_TAG);
        AddressSpace as = ptr_t->payload.ptr_type.address_space;
        decl = global_variable_helper(p->dst, type, as);
        if (value && shd_is_physical_data_type(type))
            decl->payload.global_variable.init = l2s_convert_value(p, value);

        if (LLVMIsGlobalConstant(global)) {
            shd_add_annotation_named(decl, "Constant");
            shd_add_annotation_named(decl, "DoNotDemoteToReference");
        }

        switch (LLVMGetLinkage(global)) {
            case LLVMExternalLinkage:
            case LLVMExternalWeakLinkage:
                assert(name);
                shd_module_add_export(p->dst, name, decl);
                break;
            default:
                break;
        }

        if (UNTYPED_POINTERS) {
            const Node* r = bit_cast_helper(a, ptr_t, decl);
            shd_dict_insert(LLVMValueRef, const Node*, p->map, global, r);
            return r;
        }
    } else {
        const Type* type = l2s_convert_type(p, LLVMTypeOf(global));
        decl = constant_helper(p->dst, type);
        decl->payload.constant.value = l2s_convert_value(p, global);
        if (name && strlen(name) > 0)
            shd_set_debug_name(decl, name);
    }

    assert(decl && is_declaration(decl));
    const Node* r = decl;

    shd_dict_insert(LLVMValueRef, const Node*, p->map, global, r);
    return r;
}

LLVMFrontendConfig shd_get_default_llvm_frontend_config(void) {
    return (LLVMFrontendConfig) {
        .input_cf.restructure_with_heuristics = true,
    };
}

#include "shady/cli.h"

#define COMPILER_CONFIG_TOGGLE_OPTIONS(F) \
F(config->input_cf.restructure_with_heuristics, restructure-everything) \
F(config->input_cf.add_scope_annotations, add-scope-annotations) \
F(config->input_cf.has_scope_annotations, has-scope-annotations) \

void shd_parse_llvm_frontend_args(LLVMFrontendConfig* config, int* pargc, char** argv) {
    int argc = *pargc;
    for (int i = 1; i < argc; i++) {
        if (argv[i] == NULL)
            continue;

        COMPILER_CONFIG_TOGGLE_OPTIONS(PARSE_TOGGLE_OPTION)
    }

    shd_pack_remaining_args(pargc, argv);
}

RewritePass shd_pass_lower_generic_globals;
RewritePass l2s_promote_byval_params;
RewritePass shd_pass_lcssa;
RewritePass shd_pass_scope2control;
RewritePass shd_pass_remove_critical_edges;
RewritePass shd_pass_reconvergence_heuristics;

bool shd_parse_llvm(const CompilerConfig* config, const LLVMFrontendConfig* frontend_config, const TargetConfig* target_config, size_t len, const char* data, String name, Module** pmod) {
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

    ArenaConfig aconfig = shd_default_arena_config(target_config);
    aconfig.check_types = false;
    aconfig.allow_fold = false;
    aconfig.optimisations.inline_single_use_bbs = false;
    aconfig.optimisations.weaken_non_leaking_allocas = false;

    IrArena* arena = shd_new_ir_arena(&aconfig);
    Module* dirty = shd_new_module(arena, "dirty");
    Parser p = {
        .ctx = context,
        .config = frontend_config,
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
    shd_log_fmt(DEBUGVV, "Shady module parsed from LLVM:\n");
    shd_log_module(DEBUGVV, dirty);

    aconfig.check_types = true;
    aconfig.allow_fold = true;
    IrArena* arena2 = shd_new_ir_arena(&aconfig);
    *pmod = shd_new_module(arena2, name);
    l2s_postprocess(&p, dirty, *pmod);
    shd_log_fmt(DEBUGVV, "Shady module parsed from LLVM, after cleanup:\n");
    shd_log_module(DEBUGVV, *pmod);
    shd_verify_module(*pmod);
    shd_destroy_ir_arena(arena);

    RUN_PASS(shd_pass_lower_generic_globals, NULL)
    RUN_PASS(l2s_promote_byval_params, NULL);

    if (frontend_config->input_cf.has_scope_annotations) {
        // RUN_PASS(shd_pass_scope_heuristic)
        // RUN_PASS(shd_pass_lift_everything, config)
        RUN_PASS(shd_pass_lcssa, config)
        RUN_PASS(shd_pass_scope2control, config)
    } else if (frontend_config->input_cf.restructure_with_heuristics) {
        RUN_PASS(shd_pass_remove_critical_edges, config)
        RUN_PASS(shd_pass_lcssa, config)
        // RUN_PASS(shd_pass_lift_everything)
        RUN_PASS(shd_pass_reconvergence_heuristics, config)
    }

    shd_destroy_dict(p.map);
    shd_destroy_dict(p.annotations);
    shd_destroy_arena(p.annotations_arena);

    LLVMContextDispose(context);

    return true;
}
