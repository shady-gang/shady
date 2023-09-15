#include "shady_llvm.h"

#include "log.h"
#include "dict.h"

#include "llvm-c/IRReader.h"
#include "llvm-c/Core.h"
#include "llvm-c/Support.h"
#include "llvm-c/Types.h"

#include <assert.h>
#include <string.h>

typedef struct {
    LLVMContextRef ctx;
    struct Dict* map;
    LLVMModuleRef src;
    Module* dst;
} Parser;

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

static AddressSpace convert_address_space(unsigned as) {
    static bool warned = false;
    switch (as) {
        case 0: return AsGeneric;
        default:
            if (!warned)
                warn_print("Warning: unrecognised address space %d", as);
            warned = true;
            return AsGeneric;
    }
}

const Type* convert_type(Parser* p, LLVMTypeRef t) {
    const Type** found = find_value_dict(LLVMTypeRef, const Type*, p->map, t);
    if (found) return *found;
    IrArena* a = get_module_arena(p->dst);

    switch (LLVMGetTypeKind(t)) {
        case LLVMVoidTypeKind: return unit_type(a);
        case LLVMHalfTypeKind: return fp16_type(a);
        case LLVMFloatTypeKind: return fp32_type(a);
        case LLVMDoubleTypeKind: return fp64_type(a);
        case LLVMX86_FP80TypeKind:
        case LLVMFP128TypeKind:
            break;
        case LLVMLabelTypeKind:
            break;
        case LLVMIntegerTypeKind:
            switch(LLVMGetIntTypeWidth(t)) {
                case 1: return bool_type(a);
                case 8: return uint8_type(a);
                case 16: return uint16_type(a);
                case 32: return uint32_type(a);
                case 64: return uint64_type(a);
                default: error("Unsupported integer width: %d\n", LLVMGetIntTypeWidth(t)); break;
            }
        case LLVMFunctionTypeKind:
            break;
        case LLVMStructTypeKind:
            break;
        case LLVMArrayTypeKind:
            break;
        case LLVMPointerTypeKind: {
            AddressSpace as = convert_address_space(LLVMGetPointerAddressSpace(t));
            const Type* pointee = convert_type(p, LLVMGetElementType(t));
            return ptr_type(a, (PtrType) {
                .address_space = as,
                .pointed_type = pointee
            });
        }
        case LLVMVectorTypeKind:
            break;
        case LLVMMetadataTypeKind:
            assert(false && "why are we typing metadata");
            break;
        case LLVMX86_MMXTypeKind:
            break;
        case LLVMTokenTypeKind:
            break;
        case LLVMScalableVectorTypeKind:
        case LLVMBFloatTypeKind:
        case LLVMX86_AMXTypeKind:
            break;
    }

    error_print("Unsupported type: ");
    LLVMDumpType(t);
    error_die();
}

const Node* convert_function(Parser* p, LLVMValueRef fn) {
    const Node** found = find_value_dict(LLVMValueRef, const Node*, p->map, fn);
    if (found) return *found;
    IrArena* a = get_module_arena(p->dst);
    info_print("Converting: %s\n", LLVMGetValueName(fn));

    Nodes params = empty(a);
    for (LLVMValueRef oparam = LLVMGetFirstParam(fn); oparam && oparam <= LLVMGetLastParam(fn); oparam = LLVMGetNextParam(oparam)) {
        LLVMTypeRef ot = LLVMTypeOf(oparam);
        const Type* t = convert_type(p, ot);
        const Node* param = var(a, t, LLVMGetValueName(oparam));
        params = append_nodes(a, params, param);
    }
    Node* f = function(p->dst, params, LLVMGetValueName(fn), empty(a), empty(a));
    insert_dict(LLVMValueRef, const Node*, p->map, fn, f);
    return f;
}

const Node* convert_global(Parser* p, LLVMValueRef global) {
    assert(false && "TODO");
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
    size_t l;
    info_print("Module name: %s\n", LLVMGetModuleIdentifier(src, &l));

    Parser p = {
        .ctx = context,
        .map = new_dict(LLVMValueRef, const Node*, (HashFn) hash_opaque_ptr, (CmpFn) cmp_opaque_ptr),
        .src = src,
        .dst = dst,
    };

    for (LLVMValueRef fn = LLVMGetFirstFunction(src); fn && fn <= LLVMGetNextFunction(fn); fn = LLVMGetLastFunction(src)) {
        if (memcmp(LLVMGetValueName(fn), "llvm.", 5) == 0)
            continue;
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

    LLVMContextDispose(context);
    return true;
}
