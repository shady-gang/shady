#include "l2s_private.h"

#include "log.h"
#include "dict.h"

#include "llvm-c/IRReader.h"

#include <assert.h>
#include <string.h>

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

static const Node* write_bb_tail(Parser* p, BodyBuilder* b, LLVMBasicBlockRef bb, LLVMValueRef first_instr) {
    for (LLVMValueRef instr = first_instr; instr && instr <= LLVMGetLastInstruction(bb); instr = LLVMGetNextInstruction(instr)) {
        bool last = instr == LLVMGetLastInstruction(bb);
        if (last)
            assert(LLVMGetBasicBlockTerminator(bb) == instr);
        LLVMDumpValue(instr);
        printf("\n");
        EmittedInstr emitted = emit_instruction(p, b, instr);
        if (emitted.terminator)
            return finish_body(b, emitted.terminator);
        if (!emitted.instruction)
            continue;
        String names[] = { LLVMGetValueName(instr) };
        Nodes results = bind_instruction_extra(b, emitted.instruction, emitted.result_types.count, &emitted.result_types, names);
        if (emitted.result_types.count == 1) {
            const Node* result = first(results);
            insert_dict(LLVMValueRef, const Node*, p->map, instr, result);
        }
    }
    assert(false);
}

static const Node* emit_bb(Parser* p, Node* fn, LLVMBasicBlockRef bb) {
    const Node** found = find_value_dict(LLVMValueRef, const Node*, p->map, bb);
    if (found) return *found;
    IrArena* a = get_module_arena(p->dst);

    Nodes params = empty(a);
    LLVMValueRef instr = LLVMGetFirstInstruction(bb);
    while (instr) {
        switch (LLVMGetInstructionOpcode(instr)) {
            case LLVMPHI: {
                assert(false);
                break;
            }
            default: goto after_phis;
        }
        instr = LLVMGetNextInstruction(instr);
    }
    after_phis:
    {
        Node* nbb = basic_block(a, fn, params, LLVMGetBasicBlockName(bb));
        insert_dict(LLVMValueRef, const Node*, p->map, bb, nbb);
        BodyBuilder* b = begin_body(a);
        write_bb_tail(p, b, bb, instr);
        return nbb;
    }
}

const Node* convert_function(Parser* p, LLVMValueRef fn) {
    if (is_llvm_intrinsic(fn)) {
        warn_print("Skipping unknown LLVM intrinsic function: %s\n", LLVMGetValueName(fn));
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
    Node* f = function(p->dst, params, LLVMGetValueName(fn), empty(a), empty(a));
    insert_dict(LLVMValueRef, const Node*, p->map, fn, f);

    LLVMBasicBlockRef first_bb = LLVMGetEntryBasicBlock(fn);
    if (first_bb) {
        BodyBuilder* b = begin_body(a);
        insert_dict(LLVMValueRef, const Node*, p->map, first_bb, f);
        f->payload.fun.body = write_bb_tail(p, b, first_bb, LLVMGetFirstInstruction(first_bb));
    }

    return f;
}

const Node* convert_global(Parser* p, LLVMValueRef global) {
    const Node** found = find_value_dict(LLVMValueRef, const Node*, p->map, global);
    if (found) return *found;
    IrArena* a = get_module_arena(p->dst);

    String name = LLVMGetValueName(global);
    String intrinsic = is_llvm_intrinsic(global);
    /*if (intrinsic) {
        if (strcmp(intrinsic, "llvm.global.annotations") == 0) {
            assert(false);
        }
        warn_print("Skipping unknown LLVM intrinsic function: %s\n", name);
        return NULL;
    }*/
    debug_print("Converting global: %s\n", name);

    const Type* type = convert_type(p, LLVMTypeOf(global));
    Node* r = NULL;

    if (LLVMIsAGlobalVariable(global)) {
        LLVMValueRef value = LLVMGetInitializer(global);
        r = global_var(p->dst, empty(a), type, name, AsGeneric);
        if (value)
            r->payload.global_variable.init = convert_value(p, value);
    } else {
        r = constant(p->dst, empty(a), type, name);
        r->payload.constant.value = convert_value(p, global);
    }

    assert(r && is_declaration(r));
    const Node* ref = ref_decl_helper(a, r);
    insert_dict(LLVMValueRef, const Node*, p->map, global, ref);
    return ref;
}

bool parse_llvm_into_shady(Module* dst, size_t len, char* data) {
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

    Module* dirty = new_module(get_module_arena(dst), "dirty");
    Parser p = {
        .ctx = context,
        .map = new_dict(LLVMValueRef, const Node*, (HashFn) hash_opaque_ptr, (CmpFn) cmp_opaque_ptr),
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

    destroy_dict(p.map);

    postprocess(dirty, dst);

    LLVMContextDispose(context);

    return true;
}
