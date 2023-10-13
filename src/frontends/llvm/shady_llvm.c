#include "shady_llvm.h"

#include "log.h"
#include "dict.h"
#include "portability.h"
#include "../../shady/transform/ir_gen_helpers.h"

#include "llvm-c/IRReader.h"
#include "llvm-c/Core.h"
#include "llvm-c/DebugInfo.h"
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

static const Node* convert_value(Parser* p, LLVMValueRef v);
const Node* convert_function(Parser* p, LLVMValueRef fn);

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

static String is_llvm_intrinsic(LLVMValueRef fn) {
    assert(LLVMIsAFunction(fn));
    String name = LLVMGetValueName(fn);
    if (memcmp(name, "llvm.", 5) == 0)
        return name;
    return NULL;
}

static Nodes convert_mdnode_operands(Parser* p, LLVMValueRef mdnode) {
    IrArena* a = get_module_arena(p->dst);
    assert(LLVMIsAMDNode(mdnode));

    unsigned count = LLVMGetMDNodeNumOperands(mdnode);
    LARRAY(LLVMValueRef, ops, count);
    LLVMGetMDNodeOperands(mdnode, ops);

    LARRAY(const Node*, cops, count);
    for (size_t i = 0; i < count; i++)
        cops[i] = ops[i] ? convert_value(p, ops[i]) : string_lit_helper(a, "null");
    Nodes args = nodes(a, count, cops);
    return args;
}

static const Node* convert_named_tuple_metadata(Parser* p, LLVMValueRef v, String name) {
    IrArena* a = get_module_arena(p->dst);
    Nodes args = convert_mdnode_operands(p, v);
    args = prepend_nodes(a, args, string_lit_helper(a, name));
    return tuple(a, args);
}

static const Node* convert_metadata(Parser* p, LLVMMetadataRef meta) {
    IrArena* a = get_module_arena(p->dst);
    LLVMMetadataKind kind = LLVMGetMetadataKind(meta);
    LLVMValueRef v = LLVMMetadataAsValue(p->ctx, meta);

    switch (kind) {
        case LLVMMDStringMetadataKind: {
            unsigned l;
            String name = LLVMGetMDString(v, &l);
            return string_lit_helper(a, name);
        }
        case LLVMConstantAsMetadataMetadataKind:
        case LLVMLocalAsMetadataMetadataKind: {
            Nodes ops = convert_mdnode_operands(p, v);
            assert(ops.count == 1);
            return first(ops);
        }
        case LLVMDistinctMDOperandPlaceholderMetadataKind: goto default_;
        case LLVMMDTupleMetadataKind: return tuple(a, convert_mdnode_operands(p, v));

        case LLVMDILocationMetadataKind:                 return convert_named_tuple_metadata(p, v, "DILocation");
        case LLVMDIExpressionMetadataKind:               return convert_named_tuple_metadata(p, v, "DIExpression");
        case LLVMDIGlobalVariableExpressionMetadataKind: return convert_named_tuple_metadata(p, v, "DIGlobalVariableExpression");
        case LLVMGenericDINodeMetadataKind:              return convert_named_tuple_metadata(p, v, "GenericDINode");
        case LLVMDISubrangeMetadataKind:                 return convert_named_tuple_metadata(p, v, "DISubrange");
        case LLVMDIEnumeratorMetadataKind:               return convert_named_tuple_metadata(p, v, "DIEnumerator");
        case LLVMDIBasicTypeMetadataKind:                return convert_named_tuple_metadata(p, v, "DIBasicType");
        case LLVMDIDerivedTypeMetadataKind:              return convert_named_tuple_metadata(p, v, "DIDerivedType");
        case LLVMDICompositeTypeMetadataKind:            return convert_named_tuple_metadata(p, v, "DICompositeType");
        case LLVMDISubroutineTypeMetadataKind:           return convert_named_tuple_metadata(p, v, "DISubroutineType");
        case LLVMDIFileMetadataKind:                     return convert_named_tuple_metadata(p, v, "DIFile");
        case LLVMDICompileUnitMetadataKind:              return convert_named_tuple_metadata(p, v, "DICompileUnit");
        case LLVMDISubprogramMetadataKind:               return convert_named_tuple_metadata(p, v, "DiSubprogram");
        case LLVMDILexicalBlockMetadataKind:             return convert_named_tuple_metadata(p, v, "DILexicalBlock");
        case LLVMDILexicalBlockFileMetadataKind:         return convert_named_tuple_metadata(p, v, "DILexicalBlockFile");
        case LLVMDINamespaceMetadataKind:                return convert_named_tuple_metadata(p, v, "DINamespace");
        case LLVMDIModuleMetadataKind:                   return convert_named_tuple_metadata(p, v, "DIModule");
        case LLVMDITemplateTypeParameterMetadataKind:    return convert_named_tuple_metadata(p, v, "DITemplateTypeParameter");
        case LLVMDITemplateValueParameterMetadataKind:   return convert_named_tuple_metadata(p, v, "DITemplateValueParameter");
        case LLVMDIGlobalVariableMetadataKind:           return convert_named_tuple_metadata(p, v, "DIGlobalVariable");
        case LLVMDILocalVariableMetadataKind:            return convert_named_tuple_metadata(p, v, "DILocalVariable");
        case LLVMDILabelMetadataKind:
        case LLVMDIObjCPropertyMetadataKind:
        case LLVMDIImportedEntityMetadataKind:
        case LLVMDIMacroMetadataKind:
        case LLVMDIMacroFileMetadataKind:
        case LLVMDICommonBlockMetadataKind:
        case LLVMDIStringTypeMetadataKind:
        case LLVMDIGenericSubrangeMetadataKind:
        case LLVMDIArgListMetadataKind:
        default: default_:
            error_print("Unknown metadata kind %d for ", kind);
            LLVMDumpValue(v);
            error_print(".\n");
            error_die();
    }
    /*unsigned l;
    String name = LLVMGetMDString(v, &l);
    unsigned count = LLVMGetMDNodeNumOperands(v);
    assert(count == 1);
    LLVMValueRef op;
    LLVMGetMDNodeOperands(v, &op);
    // error_print("kkk");
    LLVMDumpValue(op);
    */
}

static const Node* convert_value(Parser* p, LLVMValueRef v) {
    const Type** found = find_value_dict(LLVMTypeRef, const Type*, p->map, v);
    if (found) return *found;
    IrArena* a = get_module_arena(p->dst);

    const Node* r = NULL;

    switch (LLVMGetValueKind(v)) {
        case LLVMArgumentValueKind:
            break;
        case LLVMBasicBlockValueKind:
            break;
        case LLVMMemoryUseValueKind:
            break;
        case LLVMMemoryDefValueKind:
            break;
        case LLVMMemoryPhiValueKind:
            break;
        case LLVMFunctionValueKind:
            r = convert_function(p, v);
            break;
        case LLVMGlobalAliasValueKind:
            break;
        case LLVMGlobalIFuncValueKind:
            break;
        case LLVMGlobalVariableValueKind:
            break;
        case LLVMBlockAddressValueKind:
            break;
        case LLVMConstantExprValueKind:
            break;
        case LLVMConstantArrayValueKind:
            break;
        case LLVMConstantStructValueKind:
            break;
        case LLVMConstantVectorValueKind:
            break;
        case LLVMUndefValueValueKind:
            break;
        case LLVMConstantAggregateZeroValueKind:
            break;
        case LLVMConstantDataArrayValueKind:
            break;
        case LLVMConstantDataVectorValueKind:
            break;
        case LLVMConstantIntValueKind: {
            const Type* t = convert_type(p, LLVMTypeOf(v));
            assert(t->tag == Int_TAG);
            unsigned long long value = LLVMConstIntGetZExtValue(v);
            switch (t->payload.int_type.width) {
                case IntTy8: return uint8_literal(a, value);
                case IntTy16: return uint16_literal(a, value);
                case IntTy32: return uint32_literal(a, value);
                case IntTy64: return uint64_literal(a, value);
            }
        }
        case LLVMConstantFPValueKind:
            break;
        case LLVMConstantPointerNullValueKind:
            break;
        case LLVMConstantTokenNoneValueKind:
            break;
        case LLVMMetadataAsValueValueKind: {
            LLVMMetadataRef meta = LLVMValueAsMetadata(v);
            r = convert_metadata(p, meta);
        }
        case LLVMInlineAsmValueKind:
            break;
        case LLVMInstructionValueKind:
            break;
        case LLVMPoisonValueValueKind:
            break;
    }

    if (r) {
        insert_dict(LLVMTypeRef, const Type*, p->map, v, r);
        return r;
    }

    error_print("Failed to find value ");
    LLVMDumpValue(v);
    error_print(" in the already emitted map.");
    error_die();
}

static const Node* emit_instruction(Parser* p, BodyBuilder* b, LLVMValueRef instr) {
    IrArena* a = get_module_arena(p->dst);
    const Node* r = NULL;
    int c = 1;
    int num_ops = LLVMGetNumOperands(instr);
    LARRAY(const Node*, ops, num_ops);
    for (size_t i = 0; i < num_ops; i++) {
        LLVMValueRef op = LLVMGetOperand(instr, i);
        if (LLVMIsAFunction(op) && is_llvm_intrinsic(op))
            ops[i] = NULL;
        else
            ops[i] = convert_value(p, op);
    }
    Nodes operands = nodes(a, num_ops, ops);

    switch (LLVMGetInstructionOpcode(instr)) {
        case LLVMRet:
            return fn_ret(a, (Return) {
                .fn = NULL,
                .args = num_ops == 0 ? empty(a) : singleton(ops[0])
            });
        case LLVMBr:
            goto unimplemented;
        case LLVMSwitch:
            goto unimplemented;
        case LLVMIndirectBr:
            goto unimplemented;
        case LLVMInvoke:
            goto unimplemented;
        case LLVMUnreachable:
            return unreachable(a);
        case LLVMCallBr:
            goto unimplemented;
        case LLVMFNeg:
            goto unimplemented;
        case LLVMAdd:
            r = prim_op_helper(a, add_op, empty(a), operands);
            break;
        case LLVMFAdd:
            goto unimplemented;
        case LLVMSub:
            goto unimplemented;
        case LLVMFSub:
            goto unimplemented;
        case LLVMMul:
            goto unimplemented;
        case LLVMFMul:
            goto unimplemented;
        case LLVMUDiv:
            goto unimplemented;
        case LLVMSDiv:
            goto unimplemented;
        case LLVMFDiv:
            goto unimplemented;
        case LLVMURem:
            goto unimplemented;
        case LLVMSRem:
            goto unimplemented;
        case LLVMFRem:
            goto unimplemented;
        case LLVMShl:
            goto unimplemented;
        case LLVMLShr:
            goto unimplemented;
        case LLVMAShr:
            goto unimplemented;
        case LLVMAnd:
            goto unimplemented;
        case LLVMOr:
            goto unimplemented;
        case LLVMXor:
            goto unimplemented;
        case LLVMAlloca:
            r = prim_op_helper(a, alloca_op, singleton(convert_type(p, LLVMGetAllocatedType(instr))), empty(a));
            break;
        case LLVMLoad:
            goto unimplemented;
        case LLVMStore:
            c = 0;
            r = prim_op_helper(a, store_op, empty(a), mk_nodes(a, ops[1], ops[0]));
            break;
        case LLVMGetElementPtr:
            goto unimplemented;
        case LLVMTrunc:
            goto unimplemented;
        case LLVMZExt:
            goto unimplemented;
        case LLVMSExt:
            goto unimplemented;
        case LLVMFPToUI:
            goto unimplemented;
        case LLVMFPToSI:
            goto unimplemented;
        case LLVMUIToFP:
            goto unimplemented;
        case LLVMSIToFP:
            goto unimplemented;
        case LLVMFPTrunc:
            goto unimplemented;
        case LLVMFPExt:
            goto unimplemented;
        case LLVMPtrToInt:
            goto unimplemented;
        case LLVMIntToPtr:
            goto unimplemented;
        case LLVMBitCast:
            goto unimplemented;
        case LLVMAddrSpaceCast:
            goto unimplemented;
        case LLVMICmp:
            goto unimplemented;
        case LLVMFCmp:
            goto unimplemented;
        case LLVMPHI:
            assert(false && "We deal with phi nodes before, there shouldn't be one here");
            break;
        case LLVMCall: {
            unsigned num_args = LLVMGetNumArgOperands(instr);
            LLVMValueRef callee = LLVMGetCalledValue(instr);
            assert(num_args + 1 == num_ops);
            String intrinsic = is_llvm_intrinsic(callee);
            if (intrinsic) {
                if (strcmp(intrinsic, "llvm.dbg.declare") == 0)
                    return NULL;
            }
            r = call(a, (Call) {
                .callee = ops[num_args],
                .args = nodes(a, num_args, ops),
            });
            break;
        }
        case LLVMSelect:
            goto unimplemented;
        case LLVMUserOp1:
            goto unimplemented;
        case LLVMUserOp2:
            goto unimplemented;
        case LLVMVAArg:
            goto unimplemented;
        case LLVMExtractElement:
            goto unimplemented;
        case LLVMInsertElement:
            goto unimplemented;
        case LLVMShuffleVector:
            goto unimplemented;
        case LLVMExtractValue:
            goto unimplemented;
        case LLVMInsertValue:
            goto unimplemented;
        case LLVMFreeze:
            goto unimplemented;
        case LLVMFence:
            goto unimplemented;
        case LLVMAtomicCmpXchg:
            goto unimplemented;
        case LLVMAtomicRMW:
            goto unimplemented;
        case LLVMResume:
            goto unimplemented;
        case LLVMLandingPad:
            goto unimplemented;
        case LLVMCleanupRet:
            goto unimplemented;
        case LLVMCatchRet:
            goto unimplemented;
        case LLVMCatchPad:
            goto unimplemented;
        case LLVMCleanupPad:
            goto unimplemented;
        case LLVMCatchSwitch:
            goto unimplemented;
    }
    assert(c < 2);
    if (c == 0) {
        bind_instruction_extra(b, r, 0, NULL, NULL);
    } else if (c == 1) {
        Nodes result_types = singleton(convert_type(p, LLVMTypeOf(instr)));
        String names[] = { LLVMGetValueName(instr) };
        Nodes results = bind_instruction_extra(b, r, c, &result_types, names);
        const Node* result = first(results);
        insert_dict(LLVMValueRef, const Node*, p->map, instr, result);
    }
    return NULL;

    unimplemented:
    error_print("Shady: unimplemented LLVM instruction ");
    LLVMDumpValue(instr);
    error_print("\n");
    error_die();
}

static const Node* write_bb_tail(Parser* p, BodyBuilder* b, LLVMBasicBlockRef bb, LLVMValueRef first_instr) {
    for (LLVMValueRef instr = first_instr; instr && instr <= LLVMGetLastInstruction(bb); instr = LLVMGetNextInstruction(instr)) {
        bool last = instr == LLVMGetLastInstruction(bb);
        if (last)
            assert(LLVMGetBasicBlockTerminator(bb) == instr);
        LLVMDumpValue(instr);
        printf("\n");
        const Node* terminator = emit_instruction(p, b, instr);
        if (terminator)
            return finish_body(b, terminator);

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
    const Node** found = find_value_dict(LLVMValueRef, const Node*, p->map, fn);
    if (found) return *found;
    IrArena* a = get_module_arena(p->dst);
    info_print("Converting: %s\n", LLVMGetValueName(fn));

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

    Parser p = {
        .ctx = context,
        .map = new_dict(LLVMValueRef, const Node*, (HashFn) hash_opaque_ptr, (CmpFn) cmp_opaque_ptr),
        .src = src,
        .dst = dst,
    };

    for (LLVMValueRef fn = LLVMGetFirstFunction(src); fn && fn <= LLVMGetNextFunction(fn); fn = LLVMGetLastFunction(src)) {
        if (is_llvm_intrinsic(fn))
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
