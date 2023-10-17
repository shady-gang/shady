#include "l2s_private.h"

#include "portability.h"
#include "log.h"

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
EmittedInstr emit_instruction(Parser* p, BodyBuilder* b, LLVMValueRef instr) {
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
        case LLVMFAdd:
        case LLVMAdd:
            r = prim_op_helper(a, add_op, empty(a), convert_operands(p, num_ops, instr));
            break;
        case LLVMSub:
        case LLVMFSub:
            r = prim_op_helper(a, sub_op, empty(a), convert_operands(p, num_ops, instr));
            break;
        case LLVMMul:
        case LLVMFMul:
            r = prim_op_helper(a, mul_op, empty(a), convert_operands(p, num_ops, instr));
            break;
        case LLVMUDiv:
        case LLVMFDiv:
        case LLVMSDiv:
            r = prim_op_helper(a, div_op, empty(a), convert_operands(p, num_ops, instr));
            break;
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
        case LLVMAlloca: {
            const Type* dst_t = convert_type(p, LLVMTypeOf(instr));
            assert(dst_t->tag == PtrType_TAG);
            const Type* allocated_t = convert_type(p, LLVMGetAllocatedType(instr));
            r = first(bind_instruction_outputs_count(b, prim_op_helper(a, alloca_op, singleton(allocated_t), empty(a)), 1, (String[]) { "alloca_private" }, false));
            if (p->untyped_pointers) {
                const Type* untyped_private_ptr_t = ptr_type(a, (PtrType) { .pointed_type = dst_t->payload.ptr_type.pointed_type, .address_space = AsPrivatePhysical });
                r = prim_op_helper(a, reinterpret_op, singleton(untyped_private_ptr_t), singleton(r));
            }
            r = prim_op_helper(a, convert_op, singleton(dst_t), singleton(r));
            break;
        }
        case LLVMLoad: {
            Nodes ops = convert_operands(p, num_ops, instr);
            assert(ops.count == 1);
            const Node* ptr = first(ops);
            const Type* t = convert_type(p, LLVMTypeOf(instr));
            r = prim_op_helper(a, load_op, singleton(t), singleton(ptr));
            break;
        }
        case LLVMStore: {
            num_results = 0;
            Nodes ops = convert_operands(p, num_ops, instr);
            assert(ops.count == 2);
            r = prim_op_helper(a, store_op, p->untyped_pointers ? singleton(convert_type(p, LLVMTypeOf(LLVMGetOperand(instr, 0)))) : empty(a), mk_nodes(a, ops.nodes[1], ops.nodes[0]));
            break;
        }
        case LLVMGetElementPtr: {
            Nodes ops = convert_operands(p, num_ops, instr);
            r = prim_op_helper(a, lea_op, p->untyped_pointers ? singleton(convert_type(p, LLVMGetGEPSourceElementType(instr))) : empty(a), ops);
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