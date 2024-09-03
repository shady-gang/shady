#include "l2s_private.h"

#include "portability.h"
#include "log.h"
#include "dict.h"
#include "list.h"

#include "../shady/type.h"
#include "../shady/transform/ir_gen_helpers.h"

#include "llvm-c/DebugInfo.h"

static Nodes convert_operands(Parser* p, size_t num_ops, LLVMValueRef v) {
    IrArena* a = get_module_arena(p->dst);
    LARRAY(const Node*, ops, num_ops);
    for (size_t i = 0; i < num_ops; i++) {
        LLVMValueRef op = LLVMGetOperand(v, i);
        if (LLVMIsAFunction(op) && (is_llvm_intrinsic(op) || is_shady_intrinsic(op)))
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
        nops[i] = first(bind_instruction_outputs_count(b, prim_op_helper(a, reinterpret_op, singleton(dst_t), singleton(ops.nodes[i])), 1));
    return nodes(a, ops.count, nops);
}

LLVMValueRef remove_ptr_bitcasts(Parser* p, LLVMValueRef v) {
    while (true) {
        if (LLVMIsAInstruction(v) || LLVMIsAConstantExpr(v)) {
            if (LLVMGetInstructionOpcode(v) == LLVMBitCast) {
                LLVMTypeRef t = LLVMTypeOf(v);
                if (LLVMGetTypeKind(t) == LLVMPointerTypeKind)
                    v = LLVMGetOperand(v, 0);
            }
        }
        break;
    }
    return v;
}

static const Node* convert_jump(Parser* p, FnParseCtx* fn_ctx, const Node* src, LLVMBasicBlockRef dst, const Node* mem) {
    IrArena* a = fn_ctx->fn->arena;
    const Node* dst_bb = convert_basic_block(p, fn_ctx, dst);
    BBPhis* phis = find_value_dict(const Node*, BBPhis, fn_ctx->phis, dst_bb);
    assert(phis);
    size_t params_count = entries_count_list(phis->list);
    LARRAY(const Node*, params, params_count);
    for (size_t i = 0; i < params_count; i++) {
        LLVMValueRef phi = read_list(LLVMValueRef, phis->list)[i];
        for (size_t j = 0; j < LLVMCountIncoming(phi); j++) {
            if (convert_basic_block(p, fn_ctx, LLVMGetIncomingBlock(phi, j)) == src) {
                params[i] = convert_value(p, LLVMGetIncomingValue(phi, j));
                goto next;
            }
        }
        assert(false && "failed to find the appropriate source");
        next: continue;
    }
    return jump_helper(a, dst_bb, nodes(a, params_count, params), mem);
}

static const Type* type_untyped_ptr(const Type* untyped_ptr_t, const Type* element_type) {
    IrArena* a = untyped_ptr_t->arena;
    assert(element_type);
    assert(untyped_ptr_t->tag == PtrType_TAG);
    assert(!untyped_ptr_t->payload.ptr_type.is_reference);
    const Type* typed_ptr_t = ptr_type(a, (PtrType) { .pointed_type = element_type, .address_space = untyped_ptr_t->payload.ptr_type.address_space });
    return typed_ptr_t;
}

/// instr may be an instruction or a constantexpr
const Node* convert_instruction(Parser* p, FnParseCtx* fn_ctx, Node* fn_or_bb, BodyBuilder* b, LLVMValueRef instr) {
    Node* fn = fn_ctx ? fn_ctx->fn : NULL;

    IrArena* a = get_module_arena(p->dst);
    int num_ops = LLVMGetNumOperands(instr);
    size_t num_results = 1;
    Nodes result_types = empty(a);
    // const Node* r = NULL;

    LLVMOpcode opcode;
    if (LLVMIsAInstruction(instr))
        opcode = LLVMGetInstructionOpcode(instr);
    else if (LLVMIsAConstantExpr(instr))
        opcode = LLVMGetConstOpcode(instr);
    else
        assert(false);

    const Type* t = convert_type(p, LLVMTypeOf(instr));

#define BIND_PREV_R(t) bind_instruction_outputs_count(b, r, 1)

    //if (LLVMIsATerminatorInst(instr)) {
    //if (LLVMIsAInstruction(instr) && !LLVMIsATerminatorInst(instr)) {
    if (LLVMIsAInstruction(instr) && p->config->input_cf.add_scope_annotations) {
        assert(fn && fn_or_bb);
        LLVMMetadataRef dbgloc = LLVMInstructionGetDebugLoc(instr);
        if (dbgloc) {
            //Nodes* found = find_value_dict(const Node*, Nodes, p->scopes, fn_or_bb);
            //if (!found) {
                Nodes str = scope_to_string(p, dbgloc);
                //insert_dict(const Node*, Nodes, p->scopes, fn_or_bb, str);
                debugv_print("Found a debug location for ");
                log_node(DEBUGV, fn_or_bb);
                debugv_print(" ");
                for (size_t i = 0; i < str.count; i++) {
                    log_node(DEBUGV, str.nodes[i]);
                    debugv_print(" -> ");
                }
                debugv_print(" (depth= %zu)\n", str.count);
                bind_instruction_single(b, ext_instr(a, (ExtInstr) {
                    .set = "shady.scope",
                    .opcode = 0,
                    .result_t = unit_type(a),
                    .mem = bb_mem(b),
                    .operands = str,
                }));
            //}
        }
    }

    switch (opcode) {
        case LLVMRet: return fn_ret(a, (Return) {
                    .args = num_ops == 0 ? empty(a) : convert_operands(p, num_ops, instr),
                    .mem = bb_mem(b),
                });
        case LLVMBr: {
            unsigned n_targets = LLVMGetNumSuccessors(instr);
            LARRAY(LLVMBasicBlockRef, targets, n_targets);
            for (size_t i = 0; i < n_targets; i++)
                targets[i] = LLVMGetSuccessor(instr, i);
            if (LLVMIsConditional(instr)) {
                assert(n_targets == 2);
                const Node* condition = convert_value(p, LLVMGetCondition(instr));
                return branch(a, (Branch) {
                        .condition = condition,
                        .true_jump = convert_jump(p, fn_ctx, fn_or_bb, targets[0], bb_mem(b)),
                        .false_jump = convert_jump(p, fn_ctx, fn_or_bb, targets[1], bb_mem(b)),
                        .mem = bb_mem(b),
                    });
            } else {
                assert(n_targets == 1);
                return convert_jump(p, fn_ctx, fn_or_bb, targets[0], bb_mem(b));
            }
        }
        case LLVMSwitch: {
            const Node* inspectee = convert_value(p, LLVMGetOperand(instr, 0));
            const Node* default_jump = convert_jump(p, fn_ctx, fn_or_bb, (LLVMBasicBlockRef) LLVMGetOperand(instr, 1), bb_mem(b));
            int n_targets = LLVMGetNumOperands(instr) / 2 - 1;
            LARRAY(const Node*, targets, n_targets);
            LARRAY(const Node*, literals, n_targets);
            for (size_t i = 0; i < n_targets; i++) {
                literals[i] = convert_value(p, LLVMGetOperand(instr, i * 2 + 2));
                targets[i] = convert_jump(p, fn_ctx, fn_or_bb, (LLVMBasicBlockRef) LLVMGetOperand(instr, i * 2 + 3), bb_mem(b));
            }
            return br_switch(a, (Switch) {
                    .switch_value = inspectee,
                    .default_jump = default_jump,
                    .case_values = nodes(a, n_targets, literals),
                    .case_jumps = nodes(a, n_targets, targets),
                    .mem = bb_mem(b),
                });
        }
        case LLVMIndirectBr:
            goto unimplemented;
        case LLVMInvoke:
            goto unimplemented;
        case LLVMUnreachable: return unreachable(a, (Unreachable) { .mem = bb_mem(b) });
        case LLVMCallBr:
            goto unimplemented;
        case LLVMFNeg:
            return prim_op_helper(a, neg_op, empty(a), convert_operands(p, num_ops, instr));
        case LLVMFAdd:
        case LLVMAdd:
            return prim_op_helper(a, add_op, empty(a), convert_operands(p, num_ops, instr));
        case LLVMSub:
        case LLVMFSub:
            return prim_op_helper(a, sub_op, empty(a), convert_operands(p, num_ops, instr));
        case LLVMMul:
        case LLVMFMul:
            return prim_op_helper(a, mul_op, empty(a), convert_operands(p, num_ops, instr));
        case LLVMUDiv:
        case LLVMFDiv:
            return prim_op_helper(a, div_op, empty(a), convert_operands(p, num_ops, instr));
        case LLVMSDiv: {
            const Type* int_t = convert_type(p, LLVMTypeOf(LLVMGetOperand(instr, 0)));
            const Type* signed_t = change_int_t_sign(int_t, true);
            return  prim_op_helper(a, reinterpret_op, singleton(int_t), singleton(prim_op_helper(a, div_op, empty(a), reinterpret_operands(b, convert_operands(p, num_ops, instr), signed_t))));
        } case LLVMURem:
        case LLVMFRem:
            return prim_op_helper(a, mod_op, empty(a), convert_operands(p, num_ops, instr));
        case LLVMSRem: {
            const Type* int_t = convert_type(p, LLVMTypeOf(LLVMGetOperand(instr, 0)));
            const Type* signed_t = change_int_t_sign(int_t, true);
            return prim_op_helper(a, reinterpret_op, singleton(int_t), singleton(prim_op_helper(a, mod_op, empty(a), reinterpret_operands(b, convert_operands(p, num_ops, instr), signed_t))));
        } case LLVMShl:
            return prim_op_helper(a, lshift_op, empty(a), convert_operands(p, num_ops, instr));
        case LLVMLShr:
            return prim_op_helper(a, rshift_logical_op, empty(a), convert_operands(p, num_ops, instr));
        case LLVMAShr:
            return prim_op_helper(a, rshift_arithm_op, empty(a), convert_operands(p, num_ops, instr));
        case LLVMAnd:
            return prim_op_helper(a, and_op, empty(a), convert_operands(p, num_ops, instr));
        case LLVMOr:
            return prim_op_helper(a, or_op, empty(a), convert_operands(p, num_ops, instr));
        case LLVMXor:
            return prim_op_helper(a, xor_op, empty(a), convert_operands(p, num_ops, instr));
        case LLVMAlloca: {
            assert(t->tag == PtrType_TAG);
            const Type* allocated_t = convert_type(p, LLVMGetAllocatedType(instr));
            const Type* allocated_ptr_t = ptr_type(a, (PtrType) { .pointed_type = allocated_t, .address_space = AsPrivate });
            const Node* r = first(bind_instruction_outputs_count(b, stack_alloc(a, (StackAlloc) { .type = allocated_t, .mem = bb_mem(b) }), 1));
            if (UNTYPED_POINTERS) {
                const Type* untyped_ptr_t = ptr_type(a, (PtrType) { .pointed_type = unit_type(a), .address_space = AsPrivate });
                r = first(bind_instruction_outputs_count(b, prim_op_helper(a, reinterpret_op, singleton(untyped_ptr_t), singleton(r)), 1));
            }
            return prim_op_helper(a, convert_op, singleton(t), singleton(r));
        }
        case LLVMLoad: {
            Nodes ops = convert_operands(p, num_ops, instr);
            assert(ops.count == 1);
            const Node* ptr = first(ops);
            if (UNTYPED_POINTERS) {
                const Type* element_t = t;
                const Type* untyped_ptr_t = convert_type(p, LLVMTypeOf(LLVMGetOperand(instr, 0)));
                const Type* typed_ptr = type_untyped_ptr(untyped_ptr_t, element_t);
                ptr = first(bind_instruction_outputs_count(b, prim_op_helper(a, reinterpret_op, singleton(typed_ptr), singleton(ptr)), 1));
            }
            return bind_instruction_single(b, load(a, (Load) { .ptr = ptr, .mem = bb_mem(b) }));
        }
        case LLVMStore: {
            num_results = 0;
            Nodes ops = convert_operands(p, num_ops, instr);
            assert(ops.count == 2);
            const Node* ptr = ops.nodes[1];
            if (UNTYPED_POINTERS) {
                const Type* element_t = convert_type(p, LLVMTypeOf(LLVMGetOperand(instr, 0)));
                const Type* untyped_ptr_t = convert_type(p, LLVMTypeOf(LLVMGetOperand(instr, 1)));
                const Type* typed_ptr = type_untyped_ptr(untyped_ptr_t, element_t);
                ptr = first(bind_instruction_outputs_count(b, prim_op_helper(a, reinterpret_op, singleton(typed_ptr), singleton(ptr)), 1));
            }
            return bind_instruction_single(b, store(a, (Store) { .ptr = ptr, .value = ops.nodes[0], .mem = bb_mem(b) }));
        }
        case LLVMGetElementPtr: {
            Nodes ops = convert_operands(p, num_ops, instr);
            const Node* ptr = first(ops);
            if (UNTYPED_POINTERS) {
                const Type* element_t = convert_type(p, LLVMGetGEPSourceElementType(instr));
                const Type* untyped_ptr_t = convert_type(p, LLVMTypeOf(LLVMGetOperand(instr, 0)));
                const Type* typed_ptr = type_untyped_ptr(untyped_ptr_t, element_t);
                ptr = first(bind_instruction_outputs_count(b, prim_op_helper(a, reinterpret_op, singleton(typed_ptr), singleton(ptr)), 1));
            }
            ops = change_node_at_index(a, ops, 0, ptr);
            const Node* r = lea_helper(a, ops.nodes[0], ops.nodes[1], nodes(a, ops.count - 2, &ops.nodes[2]));
            if (UNTYPED_POINTERS) {
                const Type* element_t = convert_type(p, LLVMGetGEPSourceElementType(instr));
                const Type* untyped_ptr_t = convert_type(p, LLVMTypeOf(LLVMGetOperand(instr, 0)));
                bool idk;
                //element_t = qualified_type_helper(element_t, false);
                enter_composite_indices(&element_t, &idk, nodes(a, ops.count - 2, &ops.nodes[2]), true);
                const Type* typed_ptr = type_untyped_ptr(untyped_ptr_t, element_t);
                r = prim_op_helper(a, reinterpret_op, singleton(untyped_ptr_t), BIND_PREV_R(typed_ptr));
            }
            return r;
        }
        case LLVMTrunc:
        case LLVMZExt: {
            const Type* src_t = convert_type(p, LLVMTypeOf(LLVMGetOperand(instr, 0)));
            Nodes ops = convert_operands(p, num_ops, instr);
            const Node* r;
            if (src_t->tag == Bool_TAG) {
                assert(t->tag == Int_TAG);
                const Node* zero = int_literal(a, (IntLiteral) { .value = 0, .width = t->payload.int_type.width, .is_signed = t->payload.int_type.is_signed });
                const Node* one  = int_literal(a, (IntLiteral) { .value = 1, .width = t->payload.int_type.width, .is_signed = t->payload.int_type.is_signed });
                r = prim_op_helper(a, select_op, empty(a), mk_nodes(a, first(ops), one, zero));
            } else if (t->tag == Bool_TAG) {
                assert(src_t->tag == Int_TAG);
                const Node* one  = int_literal(a, (IntLiteral) { .value = 1, .width = src_t->payload.int_type.width, .is_signed = false });
                r = prim_op_helper(a, and_op, empty(a), mk_nodes(a, first(ops), one));
                r = prim_op_helper(a, eq_op, empty(a), mk_nodes(a, first(BIND_PREV_R(int_type(a, (Int) { .width = src_t->payload.int_type.width, .is_signed = false }))), one));
            } else {
                // reinterpret as unsigned, convert to change size, reinterpret back to target T
                const Type* unsigned_src_t = change_int_t_sign(src_t, false);
                const Type* unsigned_dst_t = change_int_t_sign(t, false);
                r = prim_op_helper(a, convert_op, singleton(unsigned_dst_t), reinterpret_operands(b, ops, unsigned_src_t));
                r = prim_op_helper(a, reinterpret_op, singleton(t), BIND_PREV_R(unsigned_dst_t));
            }
            return r;
        } case LLVMSExt: {
            const Node* r;
            // reinterpret as signed, convert to change size, reinterpret back to target T
            const Type* src_t = convert_type(p, LLVMTypeOf(LLVMGetOperand(instr, 0)));
            Nodes ops = convert_operands(p, num_ops, instr);
            if (src_t->tag == Bool_TAG) {
                assert(t->tag == Int_TAG);
                const Node* zero = int_literal(a, (IntLiteral) { .value = 0, .width = t->payload.int_type.width, .is_signed = t->payload.int_type.is_signed });
                uint64_t i = UINT64_MAX >> (64 - int_size_in_bytes(t->payload.int_type.width) * 8);
                const Node* ones = int_literal(a, (IntLiteral) { .value = i, .width = t->payload.int_type.width, .is_signed = t->payload.int_type.is_signed });
                r = prim_op_helper(a, select_op, empty(a), mk_nodes(a, first(ops), ones, zero));
            } else {
                const Type* signed_src_t = change_int_t_sign(src_t, true);
                const Type* signed_dst_t = change_int_t_sign(t, true);
                r = prim_op_helper(a, convert_op, singleton(signed_dst_t), reinterpret_operands(b, ops, signed_src_t));
                r = prim_op_helper(a, reinterpret_op, singleton(t), singleton(r));
            }
            return r;
        } case LLVMFPToUI:
        case LLVMFPToSI:
        case LLVMUIToFP:
        case LLVMSIToFP:
            return prim_op_helper(a, convert_op, singleton(t), convert_operands(p, num_ops, instr));
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
                if ((t->payload.ptr_type.address_space == AsGeneric)) {
                    switch (src_t->payload.ptr_type.address_space) {
                        case AsGeneric: // generic-to-generic isn't a conversion.
                            break;
                        default: {
                            op = convert_op;
                            break;
                        }
                    }
                }
            } else {
                assert(opcode != LLVMAddrSpaceCast);
            }
            return prim_op_helper(a, op, singleton(t), singleton(src));
        }
        case LLVMICmp: {
            Op op;
            bool cast_to_signed = false;
            switch(LLVMGetICmpPredicate(instr)) {
                case LLVMIntEQ:
                    op = eq_op;
                    break;
                case LLVMIntNE:
                    op = neq_op;
                    break;
                case LLVMIntUGT:
                    op = gt_op;
                    break;
                case LLVMIntUGE:
                    op = gte_op;
                    break;
                case LLVMIntULT:
                    op = lt_op;
                    break;
                case LLVMIntULE:
                    op = lte_op;
                    break;
                case LLVMIntSGT:
                    op = gt_op;
                    cast_to_signed = true;
                    break;
                case LLVMIntSGE:
                    op = gte_op;
                    cast_to_signed = true;
                    break;
                case LLVMIntSLT:
                    op = lt_op;
                    cast_to_signed = true;
                    break;
                case LLVMIntSLE:
                    op = lte_op;
                    cast_to_signed = true;
                    break;
            }
            Nodes ops = convert_operands(p, num_ops, instr);
            if (cast_to_signed) {
                const Type* unsigned_t = convert_type(p, LLVMTypeOf(LLVMGetOperand(instr, 0)));
                assert(unsigned_t->tag == Int_TAG);
                const Type* signed_t = change_int_t_sign(unsigned_t, true);
                ops = reinterpret_operands(b, ops, signed_t);
            }
            return prim_op_helper(a, op, empty(a), ops);
        }
        case LLVMFCmp: {
            Op op;
            bool cast_to_signed = false;
            switch(LLVMGetFCmpPredicate(instr)) {
                case LLVMRealUEQ:
                case LLVMRealOEQ:
                    op = eq_op;
                    break;
                case LLVMRealUNE:
                case LLVMRealONE:
                    op = neq_op;
                    break;
                case LLVMRealUGT:
                case LLVMRealOGT:
                    op = gt_op;
                    break;
                case LLVMRealUGE:
                case LLVMRealOGE:
                    op = gte_op;
                    break;
                case LLVMRealULT:
                case LLVMRealOLT:
                    op = lt_op;
                    break;
                case LLVMRealULE:
                case LLVMRealOLE:
                    op = lte_op;
                    break;
                default: goto unimplemented;
            }
            Nodes ops = convert_operands(p, num_ops, instr);
            return prim_op_helper(a, op, empty(a), ops);
            break;
        }
        case LLVMPHI:
            assert(false && "We deal with phi nodes before, there shouldn't be one here");
            break;
        case LLVMCall: {
            const Node* r = NULL;
            unsigned num_args = LLVMGetNumArgOperands(instr);
            LLVMValueRef callee = LLVMGetCalledValue(instr);
            callee = remove_ptr_bitcasts(p, callee);
            assert(num_args + 1 == num_ops);
            String intrinsic = NULL;
            if (LLVMIsAFunction(callee) || LLVMIsAConstant(callee)) {
                intrinsic = is_llvm_intrinsic(callee);
                if (!intrinsic)
                    intrinsic = is_shady_intrinsic(callee);
            }
            if (intrinsic) {
                assert(LLVMIsAFunction(callee));
                if (strcmp(intrinsic, "llvm.dbg.declare") == 0) {
                    const Node* target = convert_value(p, LLVMGetOperand(instr, 0));
                    const Node* meta = convert_value(p, LLVMGetOperand(instr, 1));
                    assert(meta->tag == RefDecl_TAG);
                    meta = meta->payload.ref_decl.decl;
                    assert(meta->tag == GlobalVariable_TAG);
                    meta = meta->payload.global_variable.init;
                    assert(meta && meta->tag == Composite_TAG);
                    const Node* name_node = meta->payload.composite.contents.nodes[2];
                    String name = get_string_literal(target->arena, name_node);
                    assert(name);
                    set_value_name((Node*) target, name);
                    return NULL;
                }
                if (strcmp(intrinsic, "llvm.dbg.label") == 0) {
                    // TODO
                    return NULL;
                }
                if (strcmp(intrinsic, "llvm.dbg.value") == 0) {
                    // TODO
                    return NULL;
                }
                if (string_starts_with(intrinsic, "llvm.lifetime")) {
                    // don't care
                    return NULL;
                }
                if (string_starts_with(intrinsic, "llvm.experimental.noalias.scope.decl")) {
                    // don't care
                    return NULL;
                }
                if (string_starts_with(intrinsic, "llvm.memcpy")) {
                    Nodes ops = convert_operands(p, num_ops, instr);
                    return bind_instruction_single(b, copy_bytes(a, (CopyBytes) { .dst = ops.nodes[0], .src = ops.nodes[1], .count = ops.nodes[2], .mem = bb_mem(b) }));
                } else if (string_starts_with(intrinsic, "llvm.memset")) {
                    Nodes ops = convert_operands(p, num_ops, instr);
                    return bind_instruction_single(b, fill_bytes(a, (FillBytes) { .dst = ops.nodes[0], .src = ops.nodes[1], .count = ops.nodes[2], .mem = bb_mem(b) }));
                } else if (string_starts_with(intrinsic, "llvm.fmuladd")) {
                    Nodes ops = convert_operands(p, num_ops, instr);
                    return prim_op_helper(a, fma_op, empty(a), nodes(a, 3, ops.nodes));
                } else if (string_starts_with(intrinsic, "llvm.fabs")) {
                    Nodes ops = convert_operands(p, num_ops, instr);
                    return prim_op_helper(a, abs_op, empty(a), nodes(a, 1, ops.nodes));
                } else if (string_starts_with(intrinsic, "llvm.floor")) {
                    Nodes ops = convert_operands(p, num_ops, instr);
                    return prim_op_helper(a, floor_op, empty(a), nodes(a, 1, ops.nodes));
                }

                typedef struct {
                    bool is_byval;
                } DecodedParamAttr;

                size_t params_count = 0;
                for (LLVMValueRef oparam = LLVMGetFirstParam(callee); oparam && oparam <= LLVMGetLastParam(callee); oparam = LLVMGetNextParam(oparam)) {
                    params_count++;
                }
                LARRAY(DecodedParamAttr, decoded, params_count);
                memset(decoded, 0, sizeof(DecodedParamAttr) * params_count);
                size_t param_index = 0;
                for (LLVMValueRef oparam = LLVMGetFirstParam(callee); oparam && oparam <= LLVMGetLastParam(callee); oparam = LLVMGetNextParam(oparam)) {
                    size_t num_attrs = LLVMGetAttributeCountAtIndex(callee, param_index + 1);
                    LARRAY(LLVMAttributeRef, attrs, num_attrs);
                    LLVMGetAttributesAtIndex(callee, param_index + 1, attrs);
                    bool is_byval = false;
                    for (size_t i = 0; i < num_attrs; i++) {
                        LLVMAttributeRef attr = attrs[i];
                        size_t k = LLVMGetEnumAttributeKind(attr);
                        size_t e = LLVMGetEnumAttributeKindForName("byval", 5);
                        uint64_t value = LLVMGetEnumAttributeValue(attr);
                        // printf("p = %zu, i = %zu, k = %zu, e = %zu\n", param_index, i, k, e);
                        if (k == e)
                            decoded[param_index].is_byval = true;
                    }
                    param_index++;
                }

                String ostr = intrinsic;
                char* str = calloc(strlen(ostr) + 1, 1);
                memcpy(str, ostr, strlen(ostr) + 1);

                if (strcmp(strtok(str, "::"), "shady") == 0) {
                    char* keyword = strtok(NULL, "::");
                    if (strcmp(keyword, "prim_op") == 0) {
                        char* opname = strtok(NULL, "::");
                        Op op;
                        size_t i;
                        for (i = 0; i < PRIMOPS_COUNT; i++) {
                            if (strcmp(get_primop_name(i), opname) == 0) {
                                op = (Op) i;
                                break;
                            }
                        }
                        assert(i != PRIMOPS_COUNT);
                        Nodes ops = convert_operands(p, num_args, instr);
                        LARRAY(const Node*, processed_ops, ops.count);
                        for (i = 0; i < num_args; i++) {
                            if (decoded[i].is_byval)
                                processed_ops[i] = first(bind_instruction_outputs_count(b, load(a, (Load) { .ptr = ops.nodes[i], .mem = bb_mem(b) }), 1));
                            else
                                processed_ops[i] = ops.nodes[i];
                        }
                        r = prim_op_helper(a, op, empty(a), nodes(a, num_args, processed_ops));
                        free(str);
                        goto finish;
                    } else if (strcmp(keyword, "instruction") == 0) {
                        char* instructionname = strtok(NULL, "::");
                        Nodes ops = convert_operands(p, num_args, instr);
                        if (strcmp(instructionname, "DebugPrintf") == 0) {
                            if (ops.count == 0)
                                error("DebugPrintf called without arguments");
                            size_t whocares;
                            gen_debug_printf(b, LLVMGetAsString(LLVMGetInitializer(LLVMGetOperand(instr, 0)), &whocares), nodes(a, ops.count - 1, &ops.nodes[1]));
                            return tuple_helper(a, empty(a));
                        }

                        error_print("Unrecognised shady instruction '%s'\n", instructionname);
                        error_die();
                    } else {
                        error_print("Unrecognised shady intrinsic '%s'\n", keyword);
                        error_die();
                    }
                }

                error_print("Unhandled intrinsic '%s'\n", intrinsic);
                error_die();
            }
            finish:

            if (!r) {
                Nodes ops = convert_operands(p, num_ops, instr);
                r = bind_instruction_single(b, call(a, (Call) {
                    .mem = bb_mem(b),
                    .callee = ops.nodes[num_args],
                    .args = nodes(a, num_args, ops.nodes),
                }));
            }
            return r;
        }
        case LLVMSelect:
            return prim_op_helper(a, select_op, empty(a), convert_operands(p, num_ops, instr));
        case LLVMUserOp1:
            goto unimplemented;
        case LLVMUserOp2:
            goto unimplemented;
        case LLVMVAArg:
            goto unimplemented;
        case LLVMExtractElement:
            return prim_op_helper(a, extract_dynamic_op, empty(a), convert_operands(p, num_ops, instr));
        case LLVMInsertElement:
            return prim_op_helper(a, insert_op, empty(a), convert_operands(p, num_ops, instr));
        case LLVMShuffleVector: {
            Nodes ops = convert_operands(p, num_ops, instr);
            unsigned num_indices = LLVMGetNumMaskElements(instr);
            LARRAY(const Node*, cindices, num_indices);
            for (size_t i = 0; i < num_indices; i++)
                cindices[i] = uint32_literal(a, LLVMGetMaskValue(instr, i));
            ops = concat_nodes(a, ops, nodes(a, num_indices, cindices));
            return prim_op_helper(a, shuffle_op, empty(a), ops);
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
    /*shortcut:
    if (r) {
        if (num_results == 1)
            result_types = singleton(convert_type(p, LLVMTypeOf(instr)));
        assert(result_types.count == num_results);
        return (EmittedInstr) {
            .instruction = r,
            .result_types = result_types,
        };
    }*/

    unimplemented:
    error_print("Shady: unimplemented LLVM instruction ");
    LLVMDumpValue(instr);
    error_print(" (opcode=%d)\n", opcode);
    error_die();
}
