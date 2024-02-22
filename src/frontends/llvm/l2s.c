#include "l2s_private.h"
#include "shady/ir_private.h"

#include "log.h"
#include "dict.h"
#include "list.h"
#include "util.h"

#include "llvm-c/IRReader.h"
#include "portability.h"

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

static const Node* write_bb_tail(Parser* p, Node* fn_or_bb, BodyBuilder* b, LLVMBasicBlockRef bb, LLVMValueRef first_instr) {
    LLVMValueRef instr;
    for (instr = first_instr; instr; instr = LLVMGetNextInstruction(instr)) {
        bool last = instr == LLVMGetLastInstruction(bb);
        if (last)
            assert(LLVMGetBasicBlockTerminator(bb) == instr);
        // LLVMDumpValue(instr);
        // printf("\n");
        EmittedInstr emitted = convert_instruction(p, fn_or_bb, b, instr);
        if (emitted.terminator)
            return finish_body(b, emitted.terminator);
        if (!emitted.instruction)
            continue;
        String names[] = { LLVMGetValueName(instr) };
        Nodes results = bind_instruction_explicit_result_types(b, emitted.instruction, emitted.result_types, names);
        if (emitted.result_types.count == 1) {
            const Node* result = first(results);
            insert_dict(LLVMValueRef, const Node*, p->map, instr, result);
        }
    }
    assert(false);
}

const Node* convert_basic_block(Parser* p, Node* fn, LLVMBasicBlockRef bb) {
    const Node** found = find_value_dict(LLVMValueRef, const Node*, p->map, bb);
    if (found) return *found;
    IrArena* a = get_module_arena(p->dst);

    struct List* phis = new_list(LLVMValueRef);
    Nodes params = empty(a);
    LLVMValueRef instr = LLVMGetFirstInstruction(bb);
    while (instr) {
        switch (LLVMGetInstructionOpcode(instr)) {
            case LLVMPHI: {
                const Node* nparam = var(a, convert_type(p, LLVMTypeOf(instr)), "phi");
                insert_dict(LLVMValueRef, const Node*, p->map, instr, nparam);
                append_list(LLVMValueRef, phis, instr);
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
        if (!name || strlen(name) == 0)
            name = unique_name(a, "bb");
        Node* nbb = basic_block(a, fn, params, name);
        insert_dict(LLVMValueRef, const Node*, p->map, bb, nbb);
        insert_dict(const Node*, struct List*, p->phis, nbb, phis);
        BodyBuilder* b = begin_body(a);
        nbb->payload.basic_block.body = write_bb_tail(p, nbb, b, bb, instr);
        return nbb;
    }
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
    for (LLVMValueRef oparam = LLVMGetFirstParam(fn); oparam && oparam <= LLVMGetLastParam(fn); oparam = LLVMGetNextParam(oparam)) {
        LLVMTypeRef ot = LLVMTypeOf(oparam);
        const Type* t = convert_type(p, ot);
        const Node* param = var(a, t, LLVMGetValueName(oparam));
        insert_dict(LLVMValueRef, const Node*, p->map, oparam, param);
        params = append_nodes(a, params, param);
    }
    const Type* fn_type = convert_type(p, LLVMGlobalGetValueType(fn));
    assert(fn_type->tag == FnType_TAG);
    assert(fn_type->payload.fn_type.param_types.count == params.count);
    Nodes annotations = empty(a);
    annotations = append_nodes(a, annotations, annotation(a, (Annotation) { .name = "Exported" }));
    Node* f = function(p->dst, params, LLVMGetValueName(fn), annotations, fn_type->payload.fn_type.return_types);
    const Node* r = fn_addr_helper(a, f);
    insert_dict(LLVMValueRef, const Node*, p->map, fn, r);

    if (LLVMCountBasicBlocks(fn) > 0) {
        LLVMBasicBlockRef first_bb = LLVMGetEntryBasicBlock(fn);
        BodyBuilder* b = begin_body(a);
        insert_dict(LLVMValueRef, const Node*, p->map, first_bb, f);
        f->payload.fun.body = write_bb_tail(p, f, b, first_bb, LLVMGetFirstInstruction(first_bb));
    }

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
            process_llvm_annotations(p, global);
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
    } else {
        const Type* type = convert_type(p, LLVMTypeOf(global));
        decl = constant(p->dst, empty(a), type, name);
        decl->payload.constant.instruction = convert_value(p, global);
    }

    assert(decl && is_declaration(decl));
    const Node* r = ref_decl_helper(a, decl);

    insert_dict(LLVMValueRef, const Node*, p->map, global, r);
    return r;
}

bool parse_llvm_into_shady(Module* dst, size_t len, const char* data) {
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
#if UNTYPED_POINTERS
    get_module_arena(dst)->config.untyped_ptrs = true; // tolerate untyped ptrs...
#endif

    Module* dirty = new_module(get_module_arena(dst), "dirty");
    Parser p = {
        .ctx = context,
        .map = new_dict(LLVMValueRef, const Node*, (HashFn) hash_opaque_ptr, (CmpFn) cmp_opaque_ptr),
        .annotations = new_dict(LLVMValueRef, ParsedAnnotation, (HashFn) hash_opaque_ptr, (CmpFn) cmp_opaque_ptr),
        .scopes = new_dict(const Node*, Nodes, (HashFn) hash_node, (CmpFn) compare_node),
        .phis = new_dict(const Node*, struct List*, (HashFn) hash_node, (CmpFn) compare_node),
        .annotations_arena = new_arena(),
        .src = src,
        .dst = dirty,
    };

    for (LLVMValueRef fn = LLVMGetFirstFunction(src); fn && fn <= LLVMGetNextFunction(fn); fn = LLVMGetLastFunction(src)) {
        convert_function(&p, fn);
    }

    LLVMValueRef global = LLVMGetFirstGlobal(src);
    while (global) {
        convert_global(&p, global);
        if (global == LLVMGetLastGlobal(src))
            break;
        global = LLVMGetNextGlobal(global);
    }

    postprocess(&p, dirty, dst);

    destroy_dict(p.map);
    destroy_dict(p.annotations);
    destroy_dict(p.scopes);
    {
        size_t i = 0;
        struct List* phis_list;
        while (dict_iter(p.phis, &i, NULL, &phis_list)) {
            destroy_list(phis_list);
        }
    }
    destroy_dict(p.phis);
    destroy_arena(p.annotations_arena);

    LLVMContextDispose(context);

    return true;
}
