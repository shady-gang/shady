#include "l2s_private.h"

#include "log.h"
#include "dict.h"
#include "util.h"

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

static const Node* write_bb_tail(Parser* p, BodyBuilder* b, LLVMBasicBlockRef bb, LLVMValueRef first_instr) {
    LLVMValueRef instr;
    for (instr = first_instr; instr; instr = LLVMGetNextInstruction(instr)) {
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
        Nodes results = bind_instruction_explicit_result_types(b, emitted.instruction, emitted.result_types, names, false);
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
    const Type* fn_type = convert_type(p, LLVMGlobalGetValueType(fn));
    assert(fn_type->tag == FnType_TAG);
    assert(fn_type->payload.fn_type.param_types.count == params.count);
    Node* f = function(p->dst, params, LLVMGetValueName(fn), empty(a), fn_type->payload.fn_type.return_types);
    const Node* r = f;
    if (p->untyped_pointers) {
        const Type* generic_ptr_t = ptr_type(a, (PtrType) {.pointed_type = uint8_type(a), .address_space = AsGeneric});
        r = anti_quote_helper(a, prim_op_helper(a, reinterpret_op, singleton(generic_ptr_t), singleton(r)));
    }
    insert_dict(LLVMValueRef, const Node*, p->map, fn, r);

    LLVMBasicBlockRef first_bb = LLVMGetEntryBasicBlock(fn);
    if (first_bb) {
        BodyBuilder* b = begin_body(a);
        insert_dict(LLVMValueRef, const Node*, p->map, first_bb, f);
        f->payload.fun.body = write_bb_tail(p, b, first_bb, LLVMGetFirstInstruction(first_bb));
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
            const Type* t = convert_type(p, LLVMGlobalGetValueType(global));
            assert(t->tag == ArrType_TAG);
            size_t arr_size = get_int_literal_value(t->payload.arr_type.size, false);
            assert(arr_size > 0);
            const Node* value = convert_value(p, LLVMGetInitializer(global));
            assert(value->tag == Composite_TAG && value->payload.composite.contents.count == arr_size);
            for (size_t i = 0; i < arr_size; i++) {
                const Node* entry = value->payload.composite.contents.nodes[i];
                assert(entry->tag == Composite_TAG);
                const Node* annotation_payload = entry->payload.composite.contents.nodes[1];
                // eliminate dummy reinterpret cast
                if (annotation_payload->tag == AntiQuote_TAG) {
                    assert(annotation_payload->payload.anti_quote.instruction->tag == PrimOp_TAG);
                    assert(annotation_payload->payload.anti_quote.instruction->payload.prim_op.op == reinterpret_op);
                    annotation_payload = first(annotation_payload->payload.anti_quote.instruction->payload.prim_op.operands);
                }
                if (annotation_payload->tag == RefDecl_TAG) {
                    annotation_payload = annotation_payload->payload.ref_decl.decl;
                }
                if (annotation_payload->tag == GlobalVariable_TAG) {
                    annotation_payload = annotation_payload->payload.global_variable.init;
                }
                const char* ostr = get_string_literal(a, annotation_payload);
                char* str = calloc(strlen(ostr) + 1, 1);
                memcpy(str, ostr, strlen(ostr) + 1);
                if (strcmp(strtok(str, "::"), "shady") == 0) {
                    const Node* target = entry->payload.composite.contents.nodes[0];
                    if (target->tag == AntiQuote_TAG) {
                        assert(target->payload.anti_quote.instruction->tag == PrimOp_TAG);
                        assert(target->payload.anti_quote.instruction->payload.prim_op.op == reinterpret_op);
                        target = first(target->payload.anti_quote.instruction->payload.prim_op.operands);
                    }
                    if (target->tag == RefDecl_TAG) {
                        target = target->payload.ref_decl.decl;
                    }

                    char* keyword = strtok(NULL, "::");
                    if (strcmp(keyword, "entry_point") == 0) {
                        assert(target->tag == Function_TAG);
                        add_annotation(p, target, (ParsedAnnotationContents) {
                            .type = EntryPointAnnot,
                            .payload.entry_point_type = strtok(NULL, "::")
                        });
                    } else {
                        error_print("Unrecognised shady annotation '%s'\n", keyword);
                        error_die();
                    }
                } else {
                    warn_print("Ignoring annotation '%s'\n", ostr);
                }
                free(str);
                //dump_node(annotation_payload);
            }
        }
        warn_print("Skipping unknown LLVM intrinsic function: %s\n", name);
        return NULL;
    }
    debug_print("Converting global: %s\n", name);

    Node* decl = NULL;

    if (LLVMIsAGlobalVariable(global)) {
        LLVMValueRef value = LLVMGetInitializer(global);
        const Type* type = convert_type(p, LLVMTypeOf(value));
        decl = global_var(p->dst, empty(a), type, name, AsGeneric);
        if (value)
            decl->payload.global_variable.init = convert_value(p, value);
    } else {
        const Type* type = convert_type(p, LLVMTypeOf(global));
        decl = constant(p->dst, empty(a), type, name);
        decl->payload.constant.value = convert_value(p, global);
    }

    assert(decl && is_declaration(decl));
    const Node* r = ref_decl_helper(a, decl);

    if (p->untyped_pointers) {
        const Type* generic_ptr_t = ptr_type(a, (PtrType) {.pointed_type = uint8_type(a), .address_space = AsGeneric});
        r = anti_quote_helper(a, prim_op_helper(a, reinterpret_op, singleton(generic_ptr_t), singleton(r)));
    }

    insert_dict(LLVMValueRef, const Node*, p->map, global, r);
    return r;
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
        .annotations = new_dict(LLVMValueRef, ParsedAnnotationContents, (HashFn) hash_opaque_ptr, (CmpFn) cmp_opaque_ptr),
        .annotations_arena = new_arena(),
        .src = src,
        .dst = dirty,
    };

    struct { unsigned major, minor, patch; } llvm_version;
    LLVMGetVersion(&llvm_version.major, &llvm_version.minor, &llvm_version.patch);
    if (llvm_version.major >= 15)
        p.untyped_pointers = true;

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
    destroy_arena(p.annotations_arena);

    LLVMContextDispose(context);

    return true;
}
