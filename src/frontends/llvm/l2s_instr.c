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

static const Type* change_int_t_sign(const Type* t, bool as_signed) {
    assert(t);
    assert(t->tag == Int_TAG);
    return int_type(t->arena, (Int) {
        .width = t->payload.int_type.width,
        .is_signed = as_signed
    });
}

static Nodes reinterpret_operands(BodyBuilder* b, Nodes ops, const Type* dst_t) {
    assert(ops.count > 0);
    IrArena* a = dst_t->arena;
    LARRAY(const Node*, nops, ops.count);
    for (size_t i = 0; i < ops.count; i++)
        nops[i] = first(bind_instruction_explicit_result_types(b, prim_op_helper(a, reinterpret_op, singleton(dst_t), singleton(ops.nodes[i])), singleton(dst_t), NULL, false));
    return nodes(a, ops.count, nops);
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

    const Type* t = convert_type(p, LLVMTypeOf(instr));

#define BIND_PREV_R(t) bind_instruction_explicit_result_types(b, r, singleton(t), NULL, false)

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
            r = prim_op_helper(a, neg_op, empty(a), convert_operands(p, num_ops, instr));
            break;
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
            r = prim_op_helper(a, div_op, empty(a), convert_operands(p, num_ops, instr));
            break;
        case LLVMSDiv: {
            const Type* int_t = convert_type(p, LLVMTypeOf(LLVMGetOperand(instr, 0)));
            const Type* signed_t = change_int_t_sign(int_t, true);
            r = prim_op_helper(a, div_op, empty(a), reinterpret_operands(b, convert_operands(p, num_ops, instr), signed_t));
            r = prim_op_helper(a, reinterpret_op, singleton(int_t), BIND_PREV_R(signed_t));
            break;
        } case LLVMURem:
        case LLVMFRem:
            r = prim_op_helper(a, mod_op, empty(a), convert_operands(p, num_ops, instr));
            break;
        case LLVMSRem: {
            const Type* int_t = convert_type(p, LLVMTypeOf(LLVMGetOperand(instr, 0)));
            const Type* signed_t = change_int_t_sign(int_t, true);
            r = prim_op_helper(a, mod_op, empty(a), reinterpret_operands(b, convert_operands(p, num_ops, instr), signed_t));
            r = prim_op_helper(a, reinterpret_op, singleton(int_t), BIND_PREV_R(signed_t));
            break;
        } case LLVMShl:
            r = prim_op_helper(a, lshift_op, empty(a), convert_operands(p, num_ops, instr));
            break;
        case LLVMLShr:
            r = prim_op_helper(a, rshift_logical_op, empty(a), convert_operands(p, num_ops, instr));
            break;
        case LLVMAShr:
            r = prim_op_helper(a, rshift_arithm_op, empty(a), convert_operands(p, num_ops, instr));
            break;
        case LLVMAnd:
            r = prim_op_helper(a, and_op, empty(a), convert_operands(p, num_ops, instr));
            break;
        case LLVMOr:
            r = prim_op_helper(a, or_op, empty(a), convert_operands(p, num_ops, instr));
            break;
        case LLVMXor:
            r = prim_op_helper(a, xor_op, empty(a), convert_operands(p, num_ops, instr));
            break;
        case LLVMAlloca: {
            assert(t->tag == PtrType_TAG);
            const Type* allocated_t = convert_type(p, LLVMGetAllocatedType(instr));
            const Type* allocated_ptr_t = ptr_type(a, (PtrType) { .pointed_type = allocated_t, .address_space = AsPrivatePhysical });
            r = first(bind_instruction_explicit_result_types(b, prim_op_helper(a, alloca_op, singleton(allocated_t), empty(a)), singleton(allocated_ptr_t), (String[]) { "alloca_private" }, false));
            if (UNTYPED_POINTERS) {
                const Type* untyped_private_ptr_t = ptr_type(a, (PtrType) { .pointed_type = t->payload.ptr_type.pointed_type, .address_space = AsPrivatePhysical });
                r = prim_op_helper(a, reinterpret_op, singleton(untyped_private_ptr_t), singleton(r));
            }
            r = prim_op_helper(a, convert_op, singleton(t), singleton(r));
            break;
        }
        case LLVMLoad: {
            Nodes ops = convert_operands(p, num_ops, instr);
            assert(ops.count == 1);
            const Node* ptr = first(ops);
            r = prim_op_helper(a, load_op, singleton(t), singleton(ptr));
            break;
        }
        case LLVMStore: {
            num_results = 0;
            Nodes ops = convert_operands(p, num_ops, instr);
            assert(ops.count == 2);
            r = prim_op_helper(a, store_op, UNTYPED_POINTERS ? singleton(convert_type(p, LLVMTypeOf(LLVMGetOperand(instr, 0)))) : empty(a), mk_nodes(a, ops.nodes[1], ops.nodes[0]));
            break;
        }
        case LLVMGetElementPtr: {
            Nodes ops = convert_operands(p, num_ops, instr);
            r = prim_op_helper(a, lea_op, UNTYPED_POINTERS ? singleton(convert_type(p, LLVMGetGEPSourceElementType(instr))) : empty(a), ops);
            break;
        }
        case LLVMTrunc:
            r = prim_op_helper(a, reinterpret_op, singleton(t), convert_operands(p, num_ops, instr));
            break;
        case LLVMZExt: {
            // reinterpret as unsigned, convert to change size, reinterpret back to target T
            const Type* unsigned_t = change_int_t_sign(t, false);
            r = prim_op_helper(a, convert_op, singleton(unsigned_t), reinterpret_operands(b, convert_operands(p, num_ops, instr), unsigned_t));
            r = prim_op_helper(a, reinterpret_op, singleton(t), BIND_PREV_R(unsigned_t));
            break;
        } case LLVMSExt: {
            // reinterpret as signed, convert to change size, reinterpret back to target T
            const Type* signed_t = change_int_t_sign(t, true);
            r = prim_op_helper(a, convert_op, singleton(signed_t), reinterpret_operands(b, convert_operands(p, num_ops, instr), signed_t));
            r = prim_op_helper(a, reinterpret_op, singleton(t), BIND_PREV_R(signed_t));
            break;
        } case LLVMFPToUI:
        case LLVMFPToSI:
        case LLVMUIToFP:
        case LLVMSIToFP:
            r = prim_op_helper(a, convert_op, singleton(t), convert_operands(p, num_ops, instr));
            break;
        case LLVMFPTrunc:
            goto unimplemented;
        case LLVMFPExt:
            goto unimplemented;
        case LLVMPtrToInt:
        case LLVMIntToPtr:
        case LLVMBitCast:
        case LLVMAddrSpaceCast: {
            // when constructing or deconstructing generic pointers, we need to emit a convert_op instead
            assert(num_ops == 1);
            const Node* src = first(convert_operands(p, num_ops, instr));
            Op op = reinterpret_op;
            const Type* src_t = convert_type(p, LLVMTypeOf(LLVMGetOperand(instr, 0)));
            if (src_t->tag == PtrType_TAG && t->tag == PtrType_TAG) {
                if ((t->payload.ptr_type.address_space == AsGeneric) != (src_t->payload.ptr_type.address_space == AsGeneric))
                    op = convert_op;
            } else {
                assert(opcode != LLVMAddrSpaceCast);
            }
            r = prim_op_helper(a, op, singleton(t), singleton(src));
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
            r = prim_op_helper(a, extract_dynamic_op, empty(a), convert_operands(p, num_ops, instr));
            break;
        case LLVMInsertElement:
            r = prim_op_helper(a, insert_op, empty(a), convert_operands(p, num_ops, instr));
            break;
        case LLVMShuffleVector: {
            Nodes ops = convert_operands(p, num_ops, instr);
            unsigned num_indices = LLVMGetNumMaskElements(instr);
            LARRAY(const Node*, cindices, num_indices);
            for (size_t i = 0; i < num_indices; i++)
                cindices[i] = uint32_literal(a, LLVMGetMaskValue(instr, i));
            ops = append_nodes(a, ops, tuple(a, nodes(a, num_indices, cindices)));
            assert(ops.count == 3);
            r = prim_op_helper(a, shuffle_op, empty(a), ops);
            break;
        }
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