#include "passes.h"

#include "../rewrite.h"
#include "../type.h"
#include "../ir_private.h"
#include "../transform/ir_gen_helpers.h"

#include "log.h"
#include "portability.h"

#include <assert.h>

typedef struct {
    Rewriter rewriter;
} Context;

static const Node* lower_ptr_arithm(Context* ctx, BodyBuilder* bb, const Type* pointer_type, const Node* base, const Node* offset, size_t n_indices, const Node** indices) {
    IrArena* a = ctx->rewriter.dst_arena;
    const Type* emulated_ptr_t = int_type(a, (Int) { .width = a->config.memory.ptr_size, .is_signed = false });
    assert(pointer_type->tag == PtrType_TAG);

    const Node* ptr = base;

    const IntLiteral* offset_value = resolve_to_literal(offset);
    bool offset_is_zero = offset_value && offset_value->value.i64 == 0;
    if (!offset_is_zero) {
        const Type* arr_type = pointer_type->payload.ptr_type.pointed_type;
        assert(arr_type->tag == ArrType_TAG);
        const Type* element_type = arr_type->payload.arr_type.element_type;

        const Node* element_t_size = gen_primop_e(bb, size_of_op, singleton(element_type), empty(a));

        const Node* new_offset = convert_int_extend_according_to_src_t(bb, emulated_ptr_t, offset);
        const Node* physical_offset = gen_primop_ce(bb, mul_op, 2, (const Node* []) { new_offset, element_t_size});

        ptr = gen_primop_ce(bb, add_op, 2, (const Node* []) { ptr, physical_offset});
    }

    for (size_t i = 0; i < n_indices; i++) {
        assert(pointer_type->tag == PtrType_TAG);
        const Type* pointed_type = pointer_type->payload.ptr_type.pointed_type;
        switch (pointed_type->tag) {
            case ArrType_TAG: {
                const Type* element_type = pointed_type->payload.arr_type.element_type;

                const Node* element_t_size = gen_primop_e(bb, size_of_op, singleton(element_type), empty(a));

                const Node* new_index = convert_int_extend_according_to_src_t(bb, emulated_ptr_t, indices[i]);
                const Node* physical_offset = gen_primop_ce(bb, mul_op, 2, (const Node* []) {new_index, element_t_size});

                ptr = gen_primop_ce(bb, add_op, 2, (const Node* []) { ptr, physical_offset });

                pointer_type = ptr_type(a, (PtrType) {
                    .pointed_type = element_type,
                    .address_space = pointer_type->payload.ptr_type.address_space
                });
                break;
            }
            case TypeDeclRef_TAG: {
                const Node* nom_decl = pointed_type->payload.type_decl_ref.decl;
                assert(nom_decl && nom_decl->tag == NominalType_TAG);
                pointed_type = nom_decl->payload.nom_type.body;
                SHADY_FALLTHROUGH
            }
            case RecordType_TAG: {
                Nodes member_types = pointed_type->payload.record_type.members;

                const IntLiteral* selector_value = resolve_to_literal(indices[i]);
                assert(selector_value && "selector value must be known for LEA into a record");
                size_t n = selector_value->value.u64;
                assert(n < member_types.count);

                const Node* offset_of = gen_primop_e(bb, offset_of_op, singleton(pointed_type), singleton(uint64_literal(a, n)));
                ptr = gen_primop_ce(bb, add_op, 2, (const Node* []) { ptr, offset_of });

                pointer_type = ptr_type(a, (PtrType) {
                    .pointed_type = member_types.nodes[n],
                    .address_space = pointer_type->payload.ptr_type.address_space
                });
                break;
            }
            default: error("cannot index into this")
        }
    }

    return ptr;
}

static const Node* process(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

    const Type* emulated_ptr_t = int_type(ctx->rewriter.dst_arena, (Int) { .width = ctx->rewriter.dst_arena->config.memory.ptr_size, .is_signed = false });

    switch (old->tag) {
        case PrimOp_TAG: {
            switch (old->payload.prim_op.op) {
                case lea_op: {
                    Nodes old_ops = old->payload.prim_op.operands;
                    const Node* old_base = first(old_ops);
                    const Type* old_base_ptr_t = old_base->type;
                    deconstruct_qualified_type(&old_base_ptr_t);
                    assert(old_base_ptr_t->tag == PtrType_TAG);
                    const Node* old_result_t = old->type;
                    deconstruct_qualified_type(&old_result_t);
                    // Leave logical ptrs alone
                    if (!is_physical_as(old_base_ptr_t->payload.ptr_type.address_space))
                        break;
                    BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
                    Nodes new_ops = rewrite_nodes(&ctx->rewriter, old_ops);
                    const Node* cast_base = gen_reinterpret_cast(bb, emulated_ptr_t, first(new_ops));
                    const Type* new_base_t = rewrite_node(&ctx->rewriter, old_base_ptr_t);
                    const Node* result = lower_ptr_arithm(ctx, bb, new_base_t, cast_base, new_ops.nodes[1], new_ops.count - 2, &new_ops.nodes[2]);
                    const Type* new_ptr_t = rewrite_node(&ctx->rewriter, old_result_t);
                    const Node* cast_result = gen_reinterpret_cast(bb, new_ptr_t, result);
                    return yield_values_and_wrap_in_block(bb, singleton(cast_result));
                }
                default: break;
            }
            break;
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, old);
}

void lower_lea(CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process)
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
