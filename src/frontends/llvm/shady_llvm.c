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
#include "../../shady/type.h"

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

typedef struct {
    const Node* terminator;
    const Node* instruction;
    Nodes result_types;
} EmittedInstr;

static EmittedInstr emit_instruction(Parser* p, BodyBuilder* b, LLVMValueRef instr);
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
        case LLVMStructTypeKind: {
            String name = LLVMGetStructName(t);
            Node* decl = NULL;
            const Node* result = NULL;
            if (name) {
                decl = nominal_type(p->dst, empty(a), name);
                result = type_decl_ref_helper(a, decl);
                insert_dict(LLVMTypeRef, const Type*, p->map, t, result);
            }

            unsigned size = LLVMCountStructElementTypes(t);
            LARRAY(LLVMTypeRef, elements, size);
            LLVMGetStructElementTypes(t, elements);
            LARRAY(const Type*, celements, size);
            for (size_t i = 0; i < size; i++) {
                celements[i] = convert_type(p, elements[i]);
            }

            const Node* product = record_type(a, (RecordType) {
                .members = nodes(a, size, celements)
            });
            if (decl)
                decl->payload.nom_type.body = product;
            else
                result = product;
            return result;
        }
        case LLVMArrayTypeKind: {
            unsigned length = LLVMGetArrayLength(t);
            const Type* elem_t = convert_type(p, LLVMGetElementType(t));
            return arr_type(a, (ArrType) { .element_type = elem_t, .size = uint32_literal(a, length)});
        }
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
        case LLVMPPC_FP128TypeKind:
            break;
    }

    error_print("Unsupported type: ");
    LLVMDumpType(t);
    error_die();
}

static String is_llvm_intrinsic(LLVMValueRef fn) {
    assert(LLVMIsAFunction(fn) || LLVMIsConstant(fn));
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
        case LLVMDILabelMetadataKind:                    return convert_named_tuple_metadata(p, v, "DILabelMetadata");
        case LLVMDIObjCPropertyMetadataKind:             return convert_named_tuple_metadata(p, v, "DIObjCProperty");
        case LLVMDIImportedEntityMetadataKind:           return convert_named_tuple_metadata(p, v, "DIImportedEntity");
        case LLVMDIMacroMetadataKind:                    return convert_named_tuple_metadata(p, v, "DIMacroMetadata");
        case LLVMDIMacroFileMetadataKind:                return convert_named_tuple_metadata(p, v, "DIMacroFile");
        case LLVMDICommonBlockMetadataKind:              return convert_named_tuple_metadata(p, v, "DICommonBlock");
        case LLVMDIStringTypeMetadataKind:               return convert_named_tuple_metadata(p, v, "DIStringType");
        case LLVMDIGenericSubrangeMetadataKind:          return convert_named_tuple_metadata(p, v, "DIGenericSubrange");
        case LLVMDIArgListMetadataKind:                  return convert_named_tuple_metadata(p, v, "DIArgList");
        default: default_:
            error_print("Unknown metadata kind %d for ", kind);
            LLVMDumpValue(v);
            error_print(".\n");
            error_die();
    }
}

static const Node* convert_value(Parser* p, LLVMValueRef v) {
    const Type** found = find_value_dict(LLVMTypeRef, const Type*, p->map, v);
    if (found) return *found;
    IrArena* a = get_module_arena(p->dst);

    const Node* r = NULL;
    const Type* t = convert_type(p, LLVMTypeOf(v));

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
        case LLVMConstantExprValueKind: {
            BodyBuilder* bb = begin_body(a);
            EmittedInstr emitted = emit_instruction(p, bb, v);
            r = anti_quote_helper(a, emitted.instruction);
            break;
        }
        case LLVMConstantDataArrayValueKind: {
            assert(t->tag == ArrType_TAG);
            size_t arr_size = get_int_literal_value(t->payload.arr_type.size, false);
            assert(arr_size >= 0 && arr_size < INT32_MAX && "sanity check");
            LARRAY(const Node*, elements, arr_size);
            size_t idc;
            const char* raw_bytes = LLVMGetAsString(v, &idc);
            for (size_t i = 0; i < arr_size; i++) {
                const Type* et = t->payload.arr_type.element_type;
                switch (et->tag) {
                    case Int_TAG: {
                        switch (et->payload.int_type.width) {
                            case IntTy8:  elements[i] =  uint8_literal(a, ((uint8_t*) raw_bytes)[i]); break;
                            case IntTy16: elements[i] = uint16_literal(a, ((uint16_t*) raw_bytes)[i]); break;
                            case IntTy32: elements[i] = uint32_literal(a, ((uint32_t*) raw_bytes)[i]); break;
                            case IntTy64: elements[i] = uint64_literal(a, ((uint64_t*) raw_bytes)[i]); break;
                        }
                        break;
                    }
                    default: assert(false);
                }
            }
            return composite(a, t, nodes(a, arr_size, elements));
        }
        case LLVMConstantStructValueKind: {
            assert(t->tag == RecordType_TAG);
            size_t size = t->payload.record_type.members.count;
            LARRAY(const Node*, elements, size);
            for (size_t i = 0; i < size; i++) {
                LLVMValueRef value = LLVMGetOperand(v, i);
                assert(value);
                elements[i] = convert_value(p, value);
            }
            return composite(a, t, nodes(a, size, elements));
        }
        case LLVMConstantVectorValueKind:
            break;
        case LLVMUndefValueValueKind:
            break;
        case LLVMConstantAggregateZeroValueKind:
            break;
        case LLVMConstantArrayValueKind: {
            assert(t->tag == ArrType_TAG);
            if (LLVMIsConstantString(v)) {
                size_t idc;
                r = string_lit_helper(a, LLVMGetAsString(v, &idc));
                break;
            }
            size_t arr_size = get_int_literal_value(t->payload.arr_type.size, false);
            assert(arr_size >= 0 && arr_size < INT32_MAX && "sanity check");
            LARRAY(const Node*, elements, arr_size);
            for (size_t i = 0; i < arr_size; i++) {
                LLVMValueRef value = LLVMGetOperand(v, i);
                assert(value);
                elements[i] = convert_value(p, value);
            }
            return composite(a, t, nodes(a, arr_size, elements));
        }
        case LLVMConstantDataVectorValueKind:
            break;
        case LLVMConstantIntValueKind: {
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
            r = null_ptr(a, (NullPtr) { .ptr_type = t });
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
    error_print(" in the already emitted map (kind=%d)", LLVMGetValueKind(v));
    error_die();
}

static Nodes convert_operands(Parser* p, size_t num_ops, LLVMValueRef v) {
    IrArena* a = get_module_arena(p->dst);
    LARRAY(const Node*, ops, num_ops);
    for (size_t i = 0; i < num_ops; i++) {
        LLVMValueRef op = LLVMGetOperand(v, i);
        if (LLVMIsAFunction(op) && is_llvm_intrinsic(op))
            ops[i] = NULL;
        else
            ops[i] = convert_value(p, op);
    }
    Nodes operands = nodes(a, num_ops, ops);
    return operands;
}

/// instr may be an instruction or a constantexpr
static EmittedInstr emit_instruction(Parser* p, BodyBuilder* b, LLVMValueRef instr) {
    IrArena* a = get_module_arena(p->dst);
    int num_ops = LLVMGetNumOperands(instr);
    size_t num_results = 1;
    Nodes result_types = empty(a);
    const Node* r = NULL;

    LLVMOpcode opcode;
    if (LLVMIsAInstruction(instr))
        opcode = LLVMGetInstructionOpcode(instr);
    else if (LLVMIsAConstantExpr(instr))
        opcode = LLVMGetConstOpcode(instr);
    else
        assert(false);

    switch (opcode) {
        case LLVMRet: return (EmittedInstr) {
                .terminator = fn_ret(a, (Return) {
                    .fn = NULL,
                    .args = num_ops == 0 ? empty(a) : convert_operands(p, num_ops, instr)
                })
            };
        case LLVMBr:
            goto unimplemented;
        case LLVMSwitch:
            goto unimplemented;
        case LLVMIndirectBr:
            goto unimplemented;
        case LLVMInvoke:
            goto unimplemented;
        case LLVMUnreachable: return (EmittedInstr) {
            .terminator = unreachable(a)
        };
        case LLVMCallBr:
            goto unimplemented;
        case LLVMFNeg:
            goto unimplemented;
        case LLVMAdd:
            r = prim_op_helper(a, add_op, empty(a), convert_operands(p, num_ops, instr));
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
        case LLVMStore: {
            num_results = 0;
            Nodes ops = convert_operands(p, num_ops, instr);
            assert(ops.count == 2);
            r = prim_op_helper(a, store_op, empty(a), mk_nodes(a, ops.nodes[1], ops.nodes[0]));
            break;
        }
        case LLVMGetElementPtr: {
            Nodes ops = convert_operands(p, num_ops, instr);
            r = prim_op_helper(a, lea_op, empty(a), ops);
            break;
        }
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
        case LLVMIntToPtr:
        case LLVMBitCast:
        case LLVMAddrSpaceCast:{
            r = prim_op_helper(a, reinterpret_op, singleton(convert_type(p, LLVMTypeOf(instr))), convert_operands(p, num_ops, instr));
            break;
        }
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
                    return (EmittedInstr) {};
            }
            Nodes ops = convert_operands(p, num_ops, instr);
            r = call(a, (Call) {
                .callee = ops.nodes[num_args],
                .args = nodes(a, num_args, ops.nodes),
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
    if (r) {
        if (num_results == 1)
            result_types = singleton(convert_type(p, LLVMTypeOf(instr)));
        assert(result_types.count == num_results);
        return (EmittedInstr) {
            .instruction = r,
            .result_types = result_types
        };
    }

    unimplemented:
    error_print("Shady: unimplemented LLVM instruction ");
    LLVMDumpValue(instr);
    error_print(" (opcode=%d)\n", opcode);
    error_die();
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

    Parser p = {
        .ctx = context,
        .map = new_dict(LLVMValueRef, const Node*, (HashFn) hash_opaque_ptr, (CmpFn) cmp_opaque_ptr),
        .src = src,
        .dst = dst,
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

    LLVMContextDispose(context);
    return true;
}
