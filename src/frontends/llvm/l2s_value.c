#include "l2s_private.h"

#include "portability.h"
#include "log.h"
#include "dict.h"
#include "../../shady/transform/ir_gen_helpers.h"

const Node* convert_value(Parser* p, LLVMValueRef v) {
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
            return get_default_zero_value(a, convert_type(p, LLVMTypeOf(v)));
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
