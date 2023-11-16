#include "passes.h"

#include "log.h"
#include "portability.h"
#include "util.h"
#include "dict.h"

#include "../rewrite.h"
#include "../type.h"
#include "../ir_private.h"
#include "../transform/ir_gen_helpers.h"

#include <assert.h>

typedef struct {
    Rewriter rewriter;
    const Node* generic_ptr_type;
    struct Dict* fns;
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
    return int_literal(a, (IntLiteral) { .width = a->config.memory.ptr_size, .is_signed = false, .value = value });
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
    const Node* needs_sign_extension = gen_primop_e(bb, eq_op, empty(a), mk_nodes(a, first_non_tag_bit, size_t_literal(ctx, 1)));
    //          sign_extension_patch = needs_sign_extension ? ((1 << 2) - 1) << (64 - 2) : 0
    const Node* sign_extension_patch = gen_primop_e(bb, select_op, empty(a), mk_nodes(a, needs_sign_extension, size_t_literal(ctx, ((size_t) ((1 << max_tag) - 1)) << (get_type_bitwidth(generic_ptr_type) - generic_ptr_tag_bitwidth)), size_t_literal(ctx, 0)));
    //          patched_ptr = nptr & 0b00111 ... 111
    const Node* patched_ptr = gen_primop_e(bb, and_op, empty(a), mk_nodes(a, nptr, size_t_literal(ctx, SIZE_MAX >> generic_ptr_tag_bitwidth)));
    //          patched_ptr = patched_ptr | sign_extension_patch
                patched_ptr = gen_primop_e(bb, or_op, empty(a), mk_nodes(a, patched_ptr, sign_extension_patch));
    const Type* dst_ptr_t = ptr_type(a, (PtrType) { .pointed_type = element_type, .address_space = get_addr_space_from_tag(tag) });
    const Node* reinterpreted_ptr = gen_reinterpret_cast(bb, dst_ptr_t, patched_ptr);
    return reinterpreted_ptr;
}

static bool allowed(IrArena* a, AddressSpace as) {
    if (as == AsSharedPhysical && !a->config.allow_shared_memory)
        return false;
    if (as == AsSubgroupPhysical && !a->config.allow_subgroup_memory)
        return false;
    return true;
}

typedef enum { LoadFn, StoreFn } WhichFn;
static const Node* get_or_make_access_fn(Context* ctx, WhichFn which, bool uniform_ptr, const Type* t) {
    IrArena* a = ctx->rewriter.dst_arena;
    String name;
    switch (which) {
        case LoadFn: name = format_string_interned(a, "generated_generic_load_%s", name_type_safe(a, t)); break;
        case StoreFn: name = format_string_interned(a, "generated_generic_store_%s", name_type_safe(a, t)); break;
    }

    const Node** found = find_value_dict(String, const Node*, ctx->fns, name);
    if (found)
        return *found;

    const Node* ptr_param = var(a, qualified_type_helper(ctx->generic_ptr_type, false), "ptr");
    const Node* value_param;
    Nodes params = singleton(ptr_param);
    Nodes return_ts = empty(a);
    switch (which) {
        case LoadFn:
            return_ts = singleton(qualified_type_helper(t, false));
            break;
        case StoreFn:
            value_param = var(a, qualified_type_helper(t, false), "value");
            params = append_nodes(a, params, value_param);
            break;
    }
    Node* new_fn = function(ctx->rewriter.dst_module, params, name, singleton(annotation(a, (Annotation) { .name = "Generated" })), return_ts);
    insert_dict(String, const Node*, ctx->fns, name, new_fn);

    size_t max_tag = sizeof(generic_ptr_tags) / sizeof(generic_ptr_tags[0]);
    switch (which) {
        case LoadFn: {
            LARRAY(const Node*, literals, max_tag);
            LARRAY(const Node*, cases, max_tag);
            for (size_t tag = 0; tag < max_tag; tag++) {
                literals[tag] = size_t_literal(ctx, tag);
                if (!allowed(a, generic_ptr_tags[tag])) {
                    cases[tag] = case_(a, empty(a), unreachable(a));
                    continue;
                }
                BodyBuilder* case_bb = begin_body(a);
                const Node* reinterpreted_ptr = recover_full_pointer(ctx, case_bb, tag, ptr_param, t);
                const Node* loaded_value = gen_load(case_bb, reinterpreted_ptr);
                cases[tag] = case_(a, empty(a), finish_body(case_bb, yield(a, (Yield) {
                        .args = singleton(loaded_value),
                })));
            }

            BodyBuilder* bb = begin_body(a);
            gen_comment(bb, "Generated generic ptr store");
            //          extracted_tag = nptr >> (64 - 2), for example
            const Node* extracted_tag = gen_primop_e(bb, rshift_logical_op, empty(a), mk_nodes(a, ptr_param, size_t_literal(ctx, get_type_bitwidth(ctx->generic_ptr_type) - generic_ptr_tag_bitwidth)));

            const Node* loaded_value = first(bind_instruction(bb, match_instr(a, (Match) {
                    .inspect = extracted_tag,
                    .yield_types = singleton(t),
                    .literals = nodes(a, max_tag, literals),
                    .cases = nodes(a, max_tag, cases),
                    .default_case = case_(a, empty(a), unreachable(a)),
            })));
            new_fn->payload.fun.body = finish_body(bb, fn_ret(a, (Return) { .args = singleton(loaded_value), .fn = new_fn }));
            break;
        }
        case StoreFn: {
            LARRAY(const Node*, literals, max_tag);
            LARRAY(const Node*, cases, max_tag);
            for (size_t tag = 0; tag < max_tag; tag++) {
                literals[tag] = size_t_literal(ctx, tag);
                if (!allowed(a, generic_ptr_tags[tag])) {
                    cases[tag] = case_(a, empty(a), unreachable(a));
                    continue;
                }
                BodyBuilder* case_bb = begin_body(a);
                const Node* reinterpreted_ptr = recover_full_pointer(ctx, case_bb, tag, ptr_param, t);
                gen_store(case_bb, reinterpreted_ptr, value_param);
                cases[tag] = case_(a, empty(a), finish_body(case_bb, yield(a, (Yield) {
                        .args = empty(a),
                })));
            }

            BodyBuilder* bb = begin_body(a);
            gen_comment(bb, "Generated generic ptr store");
            //          extracted_tag = nptr >> (64 - 2), for example
            const Node* extracted_tag = gen_primop_e(bb, rshift_logical_op, empty(a), mk_nodes(a, ptr_param, size_t_literal(ctx, get_type_bitwidth(ctx->generic_ptr_type) - generic_ptr_tag_bitwidth)));

            bind_instruction(bb, match_instr(a, (Match) {
                    .inspect = extracted_tag,
                    .yield_types = empty(a),
                    .literals = nodes(a, max_tag, literals),
                    .cases = nodes(a, max_tag, cases),
                    .default_case = case_(a, empty(a), unreachable(a)),
            }));
            new_fn->payload.fun.body = finish_body(bb, fn_ret(a, (Return) { .args = empty(a), .fn = new_fn }));
            break;
        }
    }
    return new_fn;
}

static const Node* process(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

    IrArena* a = ctx->rewriter.dst_arena;
    Module* m = ctx->rewriter.dst_module;
    size_t max_tag = sizeof(generic_ptr_tags) / sizeof(generic_ptr_tags[0]);

    switch (old->tag) {
        case PtrType_TAG: {
            if (old->payload.ptr_type.address_space == AsGeneric) {
                return ctx->generic_ptr_type;
            }
            break;
        }
        case NullPtr_TAG: {
            if (old->payload.null_ptr.ptr_type->payload.ptr_type.address_space == AsGeneric)
                return size_t_literal(ctx, 0);
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
                        String x = format_string_arena(a->arena, "Generated generic ptr convert src %d tag %d", src_as, tag);
                        gen_comment(bb, x);
                        const Node* src_ptr = rewrite_node(&ctx->rewriter, old_src);
                        const Node* generic_ptr = gen_reinterpret_cast(bb, ctx->generic_ptr_type, src_ptr);
                        const Node* ptr_mask = size_t_literal(ctx, (UINT64_MAX >> (uint64_t) (generic_ptr_tag_bitwidth)));
                        //          generic_ptr = generic_ptr & 0x001111 ... 111
                                    generic_ptr = gen_primop_e(bb, and_op, empty(a), mk_nodes(a, generic_ptr, ptr_mask));
                        const Node* shifted_tag = size_t_literal(ctx, (tag << (uint64_t) (get_type_bitwidth(ctx->generic_ptr_type) - generic_ptr_tag_bitwidth)));
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
                    const Type* old_ptr_t = first(old->payload.prim_op.operands)->type;
                    deconstruct_qualified_type(&old_ptr_t);
                    if (old_ptr_t->payload.ptr_type.address_space == AsGeneric) {
                        return call(a, (Call) {
                            .callee = fn_addr_helper(a, get_or_make_access_fn(ctx, LoadFn, false, rewrite_node(&ctx->rewriter, old_ptr_t->payload.ptr_type.pointed_type))),
                            .args = singleton(rewrite_node(&ctx->rewriter, first(old->payload.prim_op.operands))),
                        });
                    }
                    break;
                }
                case store_op: {
                    const Type* old_ptr_t = first(old->payload.prim_op.operands)->type;
                    deconstruct_qualified_type(&old_ptr_t);
                    if (old_ptr_t->payload.ptr_type.address_space == AsGeneric) {
                        return call(a, (Call) {
                            .callee = fn_addr_helper(a, get_or_make_access_fn(ctx, StoreFn, false, rewrite_node(&ctx->rewriter, old_ptr_t->payload.ptr_type.pointed_type))),
                            .args = rewrite_nodes(&ctx->rewriter, old->payload.prim_op.operands),
                        });
                    }
                    break;
                }
                default: break;
            }
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, old);
}

KeyHash hash_string(const char** string);
bool compare_string(const char** a, const char** b);

Module* lower_generic_ptrs(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process),
        .fns = new_dict(String, const Node*, (HashFn) hash_string, (CmpFn) compare_string),
        .generic_ptr_type = int_type(a, (Int) {.width = a->config.memory.ptr_size, .is_signed = false}),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    destroy_dict(ctx.fns);
    return dst;
}
