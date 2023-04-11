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

static AddressSpace generic_ptr_tags[4] = { AsGlobalPhysical, AsSharedPhysical, AsSubgroupPhysical, AsPrivatePhysical };

static size_t generic_ptr_tag_bitwidth = 2;

static AddressSpace get_addr_space_from_tag(size_t tag) {
    size_t max_tag = sizeof(generic_ptr_tags) / sizeof(generic_ptr_tags[0]);
    assert(tag < max_tag);
    return generic_ptr_tags[tag];
}

static uint64_t get_tag_for_addr_space(AddressSpace as) {
    size_t max_tag = sizeof(generic_ptr_tags) / sizeof(generic_ptr_tags[0]);
    for (size_t i = 0; i < max_tag; i++) {
        if (generic_ptr_tags[i] == as)
            return (uint64_t) i;
    }
    error("this address space can't be converted to generic");
}

static const Node* size_t_literal(Context* ctx, uint64_t value) {
    IrArena* a = ctx->rewriter.dst_arena;
    return int_literal(a, (IntLiteral) { .width = a->config.memory.ptr_size, .is_signed = false, .value.u64 = value });
}

static const Node* recover_full_pointer(Context* ctx, BodyBuilder* bb, uint64_t tag, const Node* nptr, const Type* element_type) {
    IrArena* a = ctx->rewriter.dst_arena;
    size_t max_tag = sizeof(generic_ptr_tags) / sizeof(generic_ptr_tags[0]);
    const Node* generic_ptr_type = int_type(a, (Int) {.width = a->config.memory.ptr_size, .is_signed = false});

    //          first_non_tag_bit = nptr >> (64 - 2 - 1)
    const Node* first_non_tag_bit = gen_primop_e(bb, rshift_logical_op, empty(a), mk_nodes(a, nptr, size_t_literal(ctx, get_type_bitwidth(generic_ptr_type) - generic_ptr_tag_bitwidth - 1)));
    //          first_non_tag_bit &= 1
    first_non_tag_bit = gen_primop_e(bb, and_op, empty(a), mk_nodes(a, first_non_tag_bit, size_t_literal(ctx, 1)));
    //          needs_sign_extension = first_non_tag_bit == 1
    const Node* needs_sign_extension = gen_primop_e(bb, eq_op, empty(a), mk_nodes(a, nptr, size_t_literal(ctx, 1)));
    //          sign_extension_patch = needs_sign_extension ? ((1 << 2) - 1) << (64 - 2) : 0
    const Node* sign_extension_patch = gen_primop_e(bb, select_op, empty(a), mk_nodes(a, needs_sign_extension, size_t_literal(ctx, ((size_t) ((1 << max_tag) - 1)) << (get_type_bitwidth(generic_ptr_type) - generic_ptr_tag_bitwidth)), size_t_literal(ctx, 0)));
    //          patched_ptr = nptr | sign_extension_patch
    const Node* patched_ptr = gen_primop_e(bb, or_op, empty(a), mk_nodes(a, nptr, sign_extension_patch));
    const Type* dst_ptr_t = ptr_type(a, (PtrType) { .pointed_type = element_type, .address_space = get_addr_space_from_tag(tag) });
    const Node* reinterpreted_ptr = gen_reinterpret_cast(bb, dst_ptr_t, patched_ptr);
    return reinterpreted_ptr;
}

static const Node* process(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

    IrArena* a = ctx->rewriter.dst_arena;
    Module* m = ctx->rewriter.dst_module;
    size_t max_tag = sizeof(generic_ptr_tags) / sizeof(generic_ptr_tags[0]);
    const Node* generic_ptr_type = int_type(a, (Int) {.width = a->config.memory.ptr_size, .is_signed = false});

    switch (old->tag) {
        case PtrType_TAG: {
            if (old->payload.ptr_type.address_space == AsGeneric) {
                return generic_ptr_type;
            }
            break;
        }
        case PrimOp_TAG: {
            switch (old->payload.prim_op.op) {
                case convert_op: {
                    const Node* old_src = first(old->payload.prim_op.operands);
                    const Type* old_src_t = old_src->type;
                    deconstruct_qualified_type(&old_src_t);
                    const Type* old_dst_t = first(old->payload.prim_op.type_arguments);
                    if (old_dst_t->tag == PtrType_TAG && old_dst_t->payload.ptr_type.address_space == AsGeneric) {
                        // cast _into_ generic
                        AddressSpace src_as = old_src_t->payload.ptr_type.address_space;
                        size_t tag = get_tag_for_addr_space(src_as);
                        BodyBuilder* bb = begin_body(a);
                        String x = format_string(a, "Generated generic ptr convert src %d tag %d", src_as, tag);
                        gen_comment(bb, x);
                        const Node* src_ptr = rewrite_node(&ctx->rewriter, old_src);
                        const Node* generic_ptr = gen_reinterpret_cast(bb, generic_ptr_type, src_ptr);
                        const Node* ptr_mask = size_t_literal(ctx, (UINT64_MAX >> (uint64_t) (generic_ptr_tag_bitwidth)));
                        //          generic_ptr = generic_ptr & 0x001111 ... 111
                                    generic_ptr = gen_primop_e(bb, and_op, empty(a), mk_nodes(a, generic_ptr, ptr_mask));
                        const Node* shifted_tag = size_t_literal(ctx, (tag << (uint64_t) (get_type_bitwidth(generic_ptr_type) -generic_ptr_tag_bitwidth)));
                        //          generic_ptr = generic_ptr | 01000000 ... 000
                                    generic_ptr = gen_primop_e(bb, or_op, empty(a), mk_nodes(a, generic_ptr, shifted_tag));
                        return yield_values_and_wrap_in_block(bb, singleton(generic_ptr));
                    } else if (old_src_t->tag == PtrType_TAG && old_src_t->payload.ptr_type.address_space == AsGeneric) {
                        // cast _from_ generic
                        error("TODO");
                    }
                    break;
                }
                case load_op: {
                    const Node* result_t = rewrite_node(&ctx->rewriter, old->type);
                    deconstruct_qualified_type(&result_t);

                    const Node* old_ptr = first(old->payload.prim_op.operands);
                    const Type* old_ptr_t = old_ptr->type;
                    deconstruct_qualified_type(&old_ptr_t);
                    if (old_ptr_t->payload.ptr_type.address_space == AsGeneric) {
                        const Node* nptr = rewrite_node(&ctx->rewriter, old_ptr);
                        LARRAY(const Node*, literals, max_tag);
                        LARRAY(const Node*, cases, max_tag);
                        for (size_t tag = 0; tag < max_tag; tag++) {
                            literals[tag] = size_t_literal(ctx, tag);
                            BodyBuilder* case_bb = begin_body(a);
                            const Node* reinterpreted_ptr = recover_full_pointer(ctx, case_bb, tag, nptr, rewrite_node(&ctx->rewriter, old_ptr_t->payload.ptr_type.pointed_type));
                            const Node* loaded_value = gen_load(case_bb, reinterpreted_ptr);
                            cases[tag] = lambda(a, empty(a), finish_body(case_bb, yield(a, (Yield) {
                                .args = singleton(loaded_value),
                            })));
                        }

                        BodyBuilder* bb = begin_body(a);
                        gen_comment(bb, "Generated generic ptr store");
                        //          extracted_tag = nptr >> (64 - 2), for example
                        const Node* extracted_tag = gen_primop_e(bb, rshift_logical_op, empty(a), mk_nodes(a, nptr, size_t_literal(ctx, get_type_bitwidth(generic_ptr_type) - generic_ptr_tag_bitwidth)));

                        const Node* loaded_value = first(bind_instruction(bb, match_instr(a, (Match) {
                            .inspect = extracted_tag,
                            .yield_types = singleton(result_t),
                            .literals = nodes(a, max_tag, literals),
                            .cases = nodes(a, max_tag, cases),
                            .default_case = lambda(a, empty(a), unreachable(a)),
                        })));
                        return yield_values_and_wrap_in_block(bb, singleton(loaded_value));
                    }
                    break;
                }
                case store_op: {
                    const Node* old_ptr = first(old->payload.prim_op.operands);
                    const Type* old_ptr_t = old_ptr->type;
                    deconstruct_qualified_type(&old_ptr_t);
                    if (old_ptr_t->payload.ptr_type.address_space == AsGeneric) {
                        const Node* nptr = rewrite_node(&ctx->rewriter, old_ptr);
                        LARRAY(const Node*, literals, max_tag);
                        LARRAY(const Node*, cases, max_tag);
                        for (size_t tag = 0; tag < max_tag; tag++) {
                            literals[tag] = size_t_literal(ctx, tag);
                            BodyBuilder* case_bb = begin_body(a);
                            const Node* reinterpreted_ptr = recover_full_pointer(ctx, case_bb, tag, nptr, rewrite_node(&ctx->rewriter, old_ptr_t->payload.ptr_type.pointed_type));
                            gen_store(case_bb, reinterpreted_ptr, rewrite_node(&ctx->rewriter, old->payload.prim_op.operands.nodes[1]));
                            cases[tag] = lambda(a, empty(a), finish_body(case_bb, yield(a, (Yield) {
                                    .args = empty(a),
                            })));
                        }

                        BodyBuilder* bb = begin_body(a);
                        gen_comment(bb, "Generated generic ptr store");
                        //          extracted_tag = nptr >> (64 - 2), for example
                        const Node* extracted_tag = gen_primop_e(bb, rshift_logical_op, empty(a), mk_nodes(a, nptr, size_t_literal(ctx, get_type_bitwidth(generic_ptr_type) - generic_ptr_tag_bitwidth)));

                        bind_instruction(bb, match_instr(a, (Match) {
                                .inspect = extracted_tag,
                                .yield_types = empty(a),
                                .literals = nodes(a, max_tag, literals),
                                .cases = nodes(a, max_tag, cases),
                                .default_case = lambda(a, empty(a), unreachable(a)),
                        }));
                        return yield_values_and_wrap_in_block(bb, empty(a));
                    }
                    break;
                }
            }
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, old);
}

void lower_generic_ptrs(CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process)
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
