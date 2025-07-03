#include "shady/pass.h"
#include "shady/ir/cast.h"
#include "shady/ir/memory_layout.h"

#include "ir_private.h"

#include "log.h"
#include "portability.h"
#include "util.h"
#include "dict.h"

#include <assert.h>

typedef struct {
    Rewriter rewriter;
    const Node* generic_ptr_type;
    struct Dict* fns;
    const CompilerConfig* config;
} Context;

static AddressSpace generic_ptr_tags[8] = {
    [0x0] = AsGlobal,
    [0x1] = AsShared,
    [0x2] = AsSubgroup,
    [0x3] = AsPrivate,
    [0x7] = AsGlobal
};

static size_t generic_ptr_tag_bitwidth = 3;

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
    shd_error("address space '%s' can't be converted to generic", shd_get_address_space_name(as));
}

static const Node* recover_full_pointer(Context* ctx, BodyBuilder* bb, uint64_t tag, const Node* nptr, const Type* element_type) {
    IrArena* a = ctx->rewriter.dst_arena;
    size_t max_tag = sizeof(generic_ptr_tags) / sizeof(generic_ptr_tags[0]);
    const Node* generic_ptr_type = int_type(a, (Int) {.width = a->config.target.memory.ptr_size, .is_signed = false});

    //          first_non_tag_bit = nptr >> (64 - 2 - 1)
    const Node* first_non_tag_bit = prim_op_helper(a, rshift_logical_op, mk_nodes(a, nptr, size_t_literal(a, shd_get_type_bitwidth(generic_ptr_type) - generic_ptr_tag_bitwidth - 1)));
    //          first_non_tag_bit &= 1
    first_non_tag_bit = prim_op_helper(a, and_op, mk_nodes(a, first_non_tag_bit, size_t_literal(a, 1)));
    //          needs_sign_extension = first_non_tag_bit == 1
    const Node* needs_sign_extension = prim_op_helper(a, eq_op, mk_nodes(a, first_non_tag_bit, size_t_literal(a, 1)));
    //          sign_extension_patch = needs_sign_extension ? ((1 << 2) - 1) << (64 - 2) : 0
    const Node* sign_extension_patch = prim_op_helper(a, select_op, mk_nodes(a, needs_sign_extension, size_t_literal(a, ((size_t) ((1 << max_tag) - 1)) << (shd_get_type_bitwidth(generic_ptr_type) - generic_ptr_tag_bitwidth)), size_t_literal(a, 0)));
    //          patched_ptr = nptr & 0b00111 ... 111
    const Node* patched_ptr = prim_op_helper(a, and_op, mk_nodes(a, nptr, size_t_literal(a, SIZE_MAX >> generic_ptr_tag_bitwidth)));
    //          patched_ptr = patched_ptr | sign_extension_patch
                patched_ptr = prim_op_helper(a, or_op, mk_nodes(a, patched_ptr, sign_extension_patch));
    const Type* dst_ptr_t = ptr_type(a, (PtrType) { .pointed_type = element_type, .address_space = get_addr_space_from_tag(tag) });
    const Node* reinterpreted_ptr = shd_bld_bitcast(bb, dst_ptr_t, patched_ptr);
    return reinterpreted_ptr;
}

static bool allowed(Context* ctx, AddressSpace as) {
    // some tags aren't in use
    if (as == AsGeneric)
        return false;
    // if an address space is logical-only, or isn't allowed at all in the module, we can skip emitting a case for it.
    if (!ctx->rewriter.dst_arena->config.target.memory.address_spaces[as].physical || !ctx->rewriter.dst_arena->config.target.memory.address_spaces[as].allowed)
        return false;
    return true;
}

typedef enum { LoadFn, StoreFn } WhichFn;
static const Node* get_or_make_access_fn(Context* ctx, WhichFn which, ShdScope ptr_scope, const Type* t) {
    IrArena* a = ctx->rewriter.dst_arena;
    String name;
    switch (which) {
        case LoadFn: name = shd_fmt_string_irarena(a, "generated_load_Generic_%s_%s", shd_get_type_name(a, t), shd_get_scope_name(ptr_scope)); break;
        case StoreFn: name = shd_fmt_string_irarena(a, "generated_store_Generic_%s_%s", shd_get_type_name(a, t), shd_get_scope_name(ptr_scope)); break;
    }

    const Node** found = shd_dict_find_value(String, const Node*, ctx->fns, name);
    if (found)
        return *found;

    const Node* ptr_param = param_helper(a, qualified_type_helper(a, ptr_scope, ctx->generic_ptr_type));
    shd_set_debug_name(ptr_param, "ptr");
    const Node* value_param;
    Nodes params = shd_singleton(ptr_param);
    Nodes return_ts = shd_empty(a);
    switch (which) {
        case LoadFn:
            return_ts = shd_singleton(qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.bottom, t));
            break;
        case StoreFn:
            value_param = param_helper(a, qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.bottom, t));
            shd_set_debug_name(value_param, "value");
            params = shd_nodes_append(a, params, value_param);
            break;
    }
    Node* new_fn = function_helper(ctx->rewriter.dst_module, params, return_ts);
    shd_add_annotation_named(new_fn, "Generated");
    shd_add_annotation_named(new_fn, "Leaf");
    shd_set_debug_name(new_fn, name);
    shd_dict_insert(String, const Node*, ctx->fns, name, new_fn);

    size_t max_tag = sizeof(generic_ptr_tags) / sizeof(generic_ptr_tags[0]);
    switch (which) {
        case LoadFn: {
            BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(new_fn));
            shd_bld_comment(bb, "Generated generic ptr store");
            begin_control_t r = shd_bld_begin_control(bb, shd_singleton(t));
            const Node* final_loaded_value = shd_first(r.results);

            LARRAY(const Node*, literals, max_tag);
            LARRAY(const Node*, jumps, max_tag);
            for (size_t tag = 0; tag < max_tag; tag++) {
                literals[tag] = size_t_literal(a, tag);
                if (!allowed(ctx, generic_ptr_tags[tag])) {
                    Node* tag_case = basic_block_helper(a, shd_empty(a));
                    shd_set_abstraction_body(tag_case, unreachable(a, (Unreachable) { .mem = shd_get_abstraction_mem(tag_case) }));
                    jumps[tag] = jump_helper(a, shd_get_abstraction_mem(r.case_), tag_case, shd_empty(a));
                    continue;
                }
                Node* tag_case = basic_block_helper(a, shd_empty(a));
                BodyBuilder* case_bb = shd_bld_begin(a, shd_get_abstraction_mem(tag_case));
                const Node* reinterpreted_ptr = recover_full_pointer(ctx, case_bb, tag, ptr_param, t);
                const Node* loaded_value = shd_bld_load(case_bb, reinterpreted_ptr);
                shd_set_abstraction_body(tag_case, shd_bld_join(case_bb, r.jp, shd_singleton(loaded_value)));
                jumps[tag] = jump_helper(a, shd_get_abstraction_mem(r.case_), tag_case, shd_empty(a));
            }
            //          extracted_tag = nptr >> (64 - 2), for example
            const Node* extracted_tag = prim_op_helper(a, rshift_logical_op, mk_nodes(a, ptr_param, size_t_literal(a, shd_get_type_bitwidth(ctx->generic_ptr_type) - generic_ptr_tag_bitwidth)));

            Node* default_case = basic_block_helper(a, shd_empty(a));
            shd_set_abstraction_body(default_case, unreachable(a, (Unreachable) { .mem = shd_get_abstraction_mem(default_case) }));
            shd_set_abstraction_body(r.case_, br_switch(a, (Switch) {
                .mem = shd_get_abstraction_mem(r.case_),
                .switch_value = extracted_tag,
                .case_values = shd_nodes(a, max_tag, literals),
                .case_jumps = shd_nodes(a, max_tag, jumps),
                .default_jump = jump_helper(a, shd_get_abstraction_mem(r.case_), default_case, shd_empty(a))
            }));
            shd_set_abstraction_body(new_fn, shd_bld_finish(bb, fn_ret(a, (Return) { .args = shd_singleton(final_loaded_value), .mem = shd_bld_mem(bb) })));
            break;
        }
        case StoreFn: {
            BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(new_fn));
            shd_bld_comment(bb, "Generated generic ptr store");
            begin_control_t r = shd_bld_begin_control(bb, shd_empty(a));

            LARRAY(const Node*, literals, max_tag);
            LARRAY(const Node*, jumps, max_tag);
            for (size_t tag = 0; tag < max_tag; tag++) {
                literals[tag] = size_t_literal(a, tag);
                if (!allowed(ctx, generic_ptr_tags[tag])) {
                    Node* tag_case = basic_block_helper(a, shd_empty(a));
                    shd_set_abstraction_body(tag_case, unreachable(a, (Unreachable) { .mem = shd_get_abstraction_mem(tag_case) }));
                    jumps[tag] = jump_helper(a, shd_get_abstraction_mem(r.case_), tag_case, shd_empty(a));
                    continue;
                }
                Node* tag_case = basic_block_helper(a, shd_empty(a));
                BodyBuilder* case_bb = shd_bld_begin(a, shd_get_abstraction_mem(tag_case));
                const Node* reinterpreted_ptr = recover_full_pointer(ctx, case_bb, tag, ptr_param, t);
                shd_bld_store(case_bb, reinterpreted_ptr, value_param);
                shd_set_abstraction_body(tag_case, shd_bld_join(case_bb, r.jp, shd_empty(a)));
                jumps[tag] = jump_helper(a, shd_get_abstraction_mem(r.case_), tag_case, shd_empty(a));
            }
            //          extracted_tag = nptr >> (64 - 2), for example
            const Node* extracted_tag = prim_op_helper(a, rshift_logical_op, mk_nodes(a, ptr_param, size_t_literal(a, shd_get_type_bitwidth(ctx->generic_ptr_type) - generic_ptr_tag_bitwidth)));

            Node* default_case = basic_block_helper(a, shd_empty(a));
            shd_set_abstraction_body(default_case, unreachable(a, (Unreachable) { .mem = shd_get_abstraction_mem(default_case) }));
            shd_set_abstraction_body(r.case_, br_switch(a, (Switch) {
                .mem = shd_get_abstraction_mem(r.case_),
                .switch_value = extracted_tag,
                .case_values = shd_nodes(a, max_tag, literals),
                .case_jumps = shd_nodes(a, max_tag, jumps),
                .default_jump = jump_helper(a, shd_get_abstraction_mem(r.case_), default_case, shd_empty(a))
            }));
            shd_set_abstraction_body(new_fn, shd_bld_finish(bb, fn_ret(a, (Return) { .args = shd_empty(a), .mem = shd_bld_mem(bb) })));
            break;
        }
    }
    return new_fn;
}

static const Node* process(Context* ctx, const Node* old) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    Module* m = ctx->rewriter.dst_module;
    size_t max_tag = sizeof(generic_ptr_tags) / sizeof(generic_ptr_tags[0]);

    switch (old->tag) {
        case PtrType_TAG: {
            if (old->payload.ptr_type.address_space == AsGeneric) {
                if (old->payload.ptr_type.pointed_type->tag == FnType_TAG) {
                    PtrType npayload = old->payload.ptr_type;
                    npayload.pointed_type = shd_rewrite_node(r, npayload.pointed_type);
                    npayload.address_space = AsCode;
                    return ptr_type(a, npayload);
                }
                return ctx->generic_ptr_type;
            }
            break;
        }
        case NullPtr_TAG: {
            if (old->payload.null_ptr.ptr_type->payload.ptr_type.address_space == AsGeneric)
                return size_t_literal(a, 0);
            break;
        }
        case Load_TAG: {
            Load payload = old->payload.load;
            const Type* old_ptr_t = payload.ptr->type;
            ShdScope ptr_scope = shd_deconstruct_qualified_type(&old_ptr_t);
            if (old_ptr_t->payload.ptr_type.address_space == AsGeneric) {
                return call(a, (Call) {
                    .callee = get_or_make_access_fn(ctx, LoadFn, ptr_scope, shd_rewrite_node(r, old_ptr_t->payload.ptr_type.pointed_type)),
                    .args = shd_singleton(shd_rewrite_node(&ctx->rewriter, payload.ptr)),
                    .mem = shd_rewrite_node(r, payload.mem)
                });
            }
            break;
        }
        case Store_TAG: {
            Store payload = old->payload.store;
            const Type* old_ptr_t = payload.ptr->type;
            ShdScope ptr_scope = shd_deconstruct_qualified_type(&old_ptr_t);
            if (old_ptr_t->payload.ptr_type.address_space == AsGeneric) {
                return call(a, (Call) {
                    .callee = get_or_make_access_fn(ctx, StoreFn, ptr_scope, shd_rewrite_node(r, old_ptr_t->payload.ptr_type.pointed_type)),
                    .args = mk_nodes(a, shd_rewrite_node(r, payload.ptr), shd_rewrite_node(r, payload.value)),
                    .mem = shd_rewrite_node(r, payload.mem),
                });
            }
            break;
        }
        case GenericPtrCast_TAG: {
            GenericPtrCast payload = old->payload.generic_ptr_cast;
            const Node* old_src = payload.src;
            const Type* old_src_t = old_src->type;
            shd_deconstruct_qualified_type(&old_src_t);

            if (old_src_t->payload.ptr_type.address_space == AsCode)
                return bit_cast_helper(a, size_t_type(a), shd_rewrite_node(r, old_src));

            if (old_src_t->payload.ptr_type.pointed_type->tag == FnType_TAG) {
                PtrType npayload = old_src_t->payload.ptr_type;
                npayload.pointed_type = shd_rewrite_node(r, npayload.pointed_type);
                npayload.address_space = AsCode;
                return bit_cast_helper(a, ptr_type(a, npayload), shd_rewrite_node(r, old_src));
            }

            // cast _into_ generic
            AddressSpace src_as = old_src_t->payload.ptr_type.address_space;
            size_t tag = get_tag_for_addr_space(src_as);
            BodyBuilder* bb = shd_bld_begin_pure(a);
            // TODO: find another way to annotate this ?
            // String x = format_string_arena(a->arena, "Generated generic ptr convert src %d tag %d", src_as, tag);
            // gen_comment(bb, x);
            const Node* src_ptr = shd_rewrite_node(&ctx->rewriter, old_src);
            const Node* generic_ptr = shd_bld_bitcast(bb, ctx->generic_ptr_type, src_ptr);
            const Node* ptr_mask = size_t_literal(a, (UINT64_MAX >> (uint64_t) (generic_ptr_tag_bitwidth)));
            //          generic_ptr = generic_ptr & 0x001111 ... 111
            generic_ptr = prim_op_helper(a, and_op, mk_nodes(a, generic_ptr, ptr_mask));
            const Node* shifted_tag = size_t_literal(a, (tag << (uint64_t) (shd_get_type_bitwidth(ctx->generic_ptr_type) - generic_ptr_tag_bitwidth)));
            //          generic_ptr = generic_ptr | 01000000 ... 000
            generic_ptr = prim_op_helper(a, or_op, mk_nodes(a, generic_ptr, shifted_tag));
            return shd_bld_to_instr_yield_values(bb, shd_singleton(generic_ptr));
            break;
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, old);
}

KeyHash shd_hash_string(const char** string);
bool shd_compare_string(const char** a, const char** b);

Module* shd_pass_lower_generic_ptrs(const CompilerConfig* config, SHADY_UNUSED const void* unused, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .fns = shd_new_dict(String, const Node*, (HashFn) shd_hash_string, (CmpFn) shd_compare_string),
        .generic_ptr_type = int_type(a, (Int) {.width = a->config.target.memory.ptr_size, .is_signed = false}),
        .config = config,
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    shd_destroy_dict(ctx.fns);
    return dst;
}
