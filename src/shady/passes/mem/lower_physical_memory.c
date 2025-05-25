#include "shady/pass.h"
#include "shady/ir/cast.h"
#include "shady/ir/memory_layout.h"

#include "ir_private.h"

#include "log.h"
#include "portability.h"
#include "util.h"
#include "list.h"
#include "dict.h"

#include <string.h>
#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    const CompilerConfig* config;

    Nodes collected[NumAddressSpaces];

    struct Dict* fns;

    const Node* fake_private_memory;
    const Node* fake_subgroup_memory;
    const Node* fake_shared_memory;
} Context;

static void store_init_data(Context* ctx, AddressSpace as, Nodes collected, BodyBuilder* bb);

static bool is_as_emulated(Context* ctx, AddressSpace as) {
    return !shd_get_arena_config(ctx->rewriter.dst_arena)->target.memory.address_spaces[as].physical;
}

/// The emulated memory arrays are not realistically going to be bigger than 4GiB, therefore all the address computations
/// should be done on 32-bit for improved performance. Emulated ptr size for ABI purposes is still dictated by the target
static IntSizes get_shortptr_type_size(Context* ctx, AddressSpace as) {
    return IntTy32;
}

static const Type* get_shortptr_type(Context* ctx, AddressSpace as) {
    return int_type_helper(ctx->rewriter.dst_arena, get_shortptr_type_size(ctx, as), false);
}

static const Node* shortptr_literal(Context* ctx, AddressSpace as, uint64_t value) {
    return int_literal_helper(ctx->rewriter.dst_arena, get_shortptr_type_size(ctx, as), false, value);
}

static const Node** get_emulated_as_word_array(Context* ctx, AddressSpace as) {
    switch (as) {
        case AsPrivate:  return &ctx->fake_private_memory;
        case AsSubgroup: return &ctx->fake_subgroup_memory;
        case AsShared:   return &ctx->fake_shared_memory;
        default: shd_error("Emulation of this AS is not supported");
    }
}

static IntSizes bytes_to_int_size(int bytes) {
    switch (bytes) {
        case 1: return IntTy8;
        case 2: return IntTy16;
        case 4: return IntTy32;
        case 8: return IntTy64;
        default: shd_error("TODO");
    }
}

static const Node* add(const Node* a, const Node* b) {
    IrArena* arena = a->arena;
    return prim_op_helper(arena, add_op, mk_nodes(arena, a, b));
}

static const Node* mul(const Node* a, const Node* b) {
    IrArena* arena = a->arena;
    return prim_op_helper(arena, mul_op, mk_nodes(arena, a, b));
}

static const Node* swizzle_offset(Context* ctx, BodyBuilder* bb, const Node* offset) {
    // can't really swizzle in RT mode
    if (shd_is_rt_execution_model(shd_get_arena_config(ctx->rewriter.src_arena)->target.execution_model))
        return offset;
    IrArena* a = ctx->rewriter.dst_arena;
    const Node* subgroup_size = shd_bld_builtin_load(ctx->rewriter.dst_module, bb, BuiltinSubgroupSize);
    subgroup_size = shd_convert_int_zero_extend(a, shd_get_unqualified_type(offset->type), subgroup_size);
    const Node* subgroup_local_id = shd_bld_builtin_load(ctx->rewriter.dst_module, bb, BuiltinSubgroupLocalInvocationId);
    subgroup_local_id = shd_convert_int_zero_extend(a, shd_get_unqualified_type(offset->type), subgroup_local_id);
    return add(mul(offset, subgroup_size), subgroup_local_id);
}

static const Node* gen_load_base(Context* ctx, BodyBuilder* bb, AddressSpace as, const Node* offset) {
    assert(shd_get_unqualified_type(offset->type) == get_shortptr_type(ctx, as));
    IrArena* a = ctx->rewriter.dst_arena;
    // swizzle the address here !
    if (ctx->config->lower.use_scratch_for_private)
        offset = swizzle_offset(ctx, bb, offset);
    const Node* arr = *get_emulated_as_word_array(ctx, as);
    const Node* value = shd_bld_load(bb, ptr_composite_element_helper(a, arr, offset));
    IntSizes width = a->config.target.memory.word_size;
    if (ctx->config->printf_trace.memory_accesses) {
        String template = shd_fmt_string_irarena(a, "loaded %s at %s:0x%s\n", width == IntTy64 ? "%lu" : "%u", shd_get_address_space_name(as), "%lx");
        const Node* widened = value;
        if (width < IntTy32)
            widened = shd_bld_conversion(bb, shd_uint32_type(a), value);
        shd_bld_debug_printf(bb, template, mk_nodes(a, widened, offset));
    }
    return value;
}

static void gen_store_base(Context* ctx, BodyBuilder* bb, AddressSpace as, const Node* offset, const Node* value) {
    assert(shd_get_unqualified_type(offset->type) == get_shortptr_type(ctx, as));
    IrArena* a = ctx->rewriter.dst_arena;
    // swizzle the address here !
    if (ctx->config->lower.use_scratch_for_private)
        offset = swizzle_offset(ctx, bb, offset);
    IntSizes width = a->config.target.memory.word_size;
    if (ctx->config->printf_trace.memory_accesses) {
        String template = shd_fmt_string_irarena(a, "storing %s at %s:0x%s\n", width == IntTy64 ? "%lu" : "%u", shd_get_address_space_name(as), "%lx");
        const Node* widened = value;
        if (width < IntTy32)
            widened = shd_bld_conversion(bb, shd_uint32_type(a), value);
        shd_bld_debug_printf(bb, template, mk_nodes(a, widened, offset));
    }
    const Node* arr = *get_emulated_as_word_array(ctx, as);
    shd_bld_store(bb, ptr_composite_element_helper(a, arr, offset), value);
}

static const Node* gen_load_for_type(Context* ctx, BodyBuilder* bb, const Type* element_type, AddressSpace as, const Node* address) {
    assert(shd_get_unqualified_type(address->type) == get_shortptr_type(ctx, as));
    IrArena* a = ctx->rewriter.dst_arena;
    const CompilerConfig* config = ctx->config;
    const Type* word_t = int_type(a, (Int) { .width = a->config.target.memory.word_size, .is_signed = false });
    switch (element_type->tag) {
        case Int_TAG: {
            assert(element_type->tag == Int_TAG);
            const Node* acc = int_literal(a, (IntLiteral) { .width = element_type->payload.int_type.width, .is_signed = false, .value = 0 });
            size_t length_in_bytes = int_size_in_bytes(element_type->payload.int_type.width);
            size_t word_size_in_bytes = int_size_in_bytes(a->config.target.memory.word_size);
            const Node* offset = shd_bytes_to_words(bb, address);
            const Node* shift = int_literal(a, (IntLiteral) { .width = element_type->payload.int_type.width, .is_signed = false, .value = 0 });
            const Node* word_bitwidth = int_literal(a, (IntLiteral) { .width = element_type->payload.int_type.width, .is_signed = false, .value = word_size_in_bytes * 8 });
            for (size_t byte = 0; byte < length_in_bytes; byte += word_size_in_bytes) {
                const Node* word = gen_load_base(ctx, bb, as, offset);
                word = shd_bld_conversion(bb, int_type(a, (Int) { .width = element_type->payload.int_type.width, .is_signed = false }), word); // widen/truncate the word we just loaded
                word = prim_op_helper(a, lshift_op, mk_nodes(a, word, shift)); // shift it
                acc = prim_op_helper(a, or_op, mk_nodes(a, acc, word));

                offset = prim_op_helper(a, add_op, mk_nodes(a, offset, shortptr_literal(ctx, as, 1)));
                shift = prim_op_helper(a, add_op, mk_nodes(a, shift, word_bitwidth));
            }
            acc = shd_bld_bitcast(bb, int_type(a, (Int) { .width = element_type->payload.int_type.width, .is_signed = element_type->payload.int_type.is_signed }), acc);
            return acc;
        }
        case Bool_TAG: {
            const Node* value = gen_load_for_type(ctx, bb, word_t, as, address);
            return prim_op_helper(a, neq_op, mk_nodes(a, value, int_literal(a, (IntLiteral) { .value = 0, .width = a->config.target.memory.word_size })));
        }
        case PtrType_TAG: {
            TypeMemLayout layout = shd_get_mem_layout(a, element_type);
            assert(layout.size_in_bytes <= int_size_in_bytes(IntSizeMax));
            const Type* ptr_int_t = int_type(a, (Int) { .width = bytes_to_int_size(layout.size_in_bytes), .is_signed = false });
            const Node* unsigned_int = gen_load_for_type(ctx, bb, ptr_int_t, as, address);
            return shd_bld_bitcast(bb, element_type, unsigned_int);
        }
        case Float_TAG: {
            const Type* unsigned_int_t = int_type(a, (Int) {.width = shd_float_to_int_width(element_type->payload.float_type.width), .is_signed = false });
            const Node* unsigned_int = gen_load_for_type(ctx, bb, unsigned_int_t, as, address);
            return shd_bld_bitcast(bb, element_type, unsigned_int);
        }
        case NominalType_TAG:
        case RecordType_TAG: {
            const Type* compound_type = element_type;
            compound_type = shd_get_maybe_nominal_type_body(compound_type);

            Nodes member_types = compound_type->payload.record_type.members;
            LARRAY(const Node*, loaded, member_types.count);
            for (size_t i = 0; i < member_types.count; i++) {
                const Node* field_offset = offset_of_helper(a, element_type, size_t_literal(a, i));
                field_offset = shd_convert_int_zero_extend(a, get_shortptr_type(ctx, as), field_offset);
                const Node* adjusted_offset = prim_op_helper(a, add_op, mk_nodes(a, address, field_offset));
                loaded[i] = gen_load_for_type(ctx, bb, member_types.nodes[i], as, adjusted_offset);
            }
            return composite_helper(a, element_type, shd_nodes(a, member_types.count, loaded));
        }
        case ArrType_TAG:
        case VectorType_TAG: {
            const Node* size = shd_get_fill_type_size(element_type);
            if (size->tag != IntLiteral_TAG) {
                shd_error_print("Size of type ");
                shd_log_node(ERROR, element_type);
                shd_error_print(" is not known a compile-time!\n");
            }
            size_t components_count = shd_get_int_literal_value(*shd_resolve_to_int_literal(size), 0);
            const Type* component_type = shd_get_fill_type_element_type(element_type);
            LARRAY(const Node*, components, components_count);
            const Node* offset = address;
            for (size_t i = 0; i < components_count; i++) {
                components[i] = gen_load_for_type(ctx, bb, component_type, as, offset);
                const Node* component_type_width = size_of_helper(a, component_type);
                component_type_width = shd_convert_int_zero_extend(a, get_shortptr_type(ctx, as), component_type_width);
                offset = add(offset, component_type_width);
            }
            return composite_helper(a, element_type, shd_nodes(a, components_count, components));
        }
        default: shd_error("TODO");
    }
}

static void gen_store_for_type(Context* ctx, BodyBuilder* bb, const Type* element_type, AddressSpace as, const Node* address, const Node* value) {
    assert(shd_get_unqualified_type(address->type) == get_shortptr_type(ctx, as));
    IrArena* a = ctx->rewriter.dst_arena;
    const CompilerConfig* config = ctx->config;
    const Type* word_t = int_type(a, (Int) { .width = a->config.target.memory.word_size, .is_signed = false });
    switch (element_type->tag) {
        case Int_TAG: {
            assert(element_type->tag == Int_TAG);
            // First bitcast to unsigned so we always get zero-extension and not sign-extension afterwards
            const Type* element_t_unsigned = int_type(a, (Int) { .width = element_type->payload.int_type.width, .is_signed = false});
            value = shd_convert_int_extend_according_to_src_t(a, element_t_unsigned, value);

            // const Node* acc = int_literal(a, (IntLiteral) { .width = element_type->payload.int_type.width, .is_signed = false, .value = 0 });
            size_t length_in_bytes = int_size_in_bytes(element_type->payload.int_type.width);
            size_t word_size_in_bytes = int_size_in_bytes(a->config.target.memory.word_size);
            const Node* offset = shd_bytes_to_words(bb, address);
            const Node* shift = int_literal(a, (IntLiteral) { .width = element_type->payload.int_type.width, .is_signed = false, .value = 0 });
            const Node* word_bitwidth = int_literal(a, (IntLiteral) { .width = element_type->payload.int_type.width, .is_signed = false, .value = word_size_in_bytes * 8 });
            for (size_t byte = 0; byte < length_in_bytes; byte += word_size_in_bytes) {
                bool is_last_word = byte + word_size_in_bytes >= length_in_bytes;
                /*bool needs_patch = is_last_word && word_size_in_bytes < length_in_bytes;
                const Node* original_word = NULL;
                if (needs_patch) {
                    original_word = gen_load_base(ctx, bb, arr, base_offset));
                    shd_error_print("TODO");
                    shd_error_die();
                    // word = gen_conversion(bb, int_type(a, (Int) { .width = element_type->payload.int_type.width, .is_signed = false }), word); // widen/truncate the word we just loaded
                }*/
                const Node* word = value;
                word = (prim_op_helper(a, rshift_logical_op, mk_nodes(a, word, shift))); // shift it
                word = shd_bld_conversion(bb, int_type(a, (Int) { .width = a->config.target.memory.word_size, .is_signed = false }), word); // widen/truncate the word we want to store
                gen_store_base(ctx, bb, as, offset, word);

                offset = prim_op_helper(a, add_op, mk_nodes(a, offset, shortptr_literal(ctx, as, 1)));
                shift = prim_op_helper(a, add_op, mk_nodes(a, shift, word_bitwidth));
            }
            return;
        }
        case Bool_TAG: {
            const Node* zero_b = int_literal(a, (IntLiteral) { .value = 0, .width = a->config.target.memory.word_size });
            const Node* one_b =  int_literal(a, (IntLiteral) { .value = 1, .width = a->config.target.memory.word_size });
            const Node* int_value = prim_op_helper(a, select_op, mk_nodes(a, value, one_b, zero_b));
            gen_store_for_type(ctx, bb, word_t, as, address, int_value);
            return;
        }
        case PtrType_TAG: {
            TypeMemLayout layout = shd_get_mem_layout(a, element_type);
            assert(layout.size_in_bytes <= int_size_in_bytes(IntSizeMax));
            const Type* ptr_int_t = int_type(a, (Int) { .width = bytes_to_int_size(layout.size_in_bytes), .is_signed = false });
            const Node* unsigned_value = bit_cast_helper(a, ptr_int_t, value);
            return gen_store_for_type(ctx, bb, ptr_int_t, as, address, unsigned_value);
        }
        case Float_TAG: {
            const Type* unsigned_int_t = int_type(a, (Int) {.width = shd_float_to_int_width(element_type->payload.float_type.width), .is_signed = false });
            const Node* unsigned_value = bit_cast_helper(a, unsigned_int_t, value);
            return gen_store_for_type(ctx, bb, unsigned_int_t, as, address, unsigned_value);
        }
        case RecordType_TAG: {
            Nodes member_types = element_type->payload.record_type.members;
            for (size_t i = 0; i < member_types.count; i++) {
                const Node* extracted_value = shd_extract_literal(a, value, i);
                const Node* field_offset = offset_of_helper(a, element_type, size_t_literal(a, i));
                field_offset = shd_convert_int_zero_extend(a, get_shortptr_type(ctx, as), field_offset);
                const Node* adjusted_offset = prim_op_helper(a, add_op, mk_nodes(a, address, field_offset));
                gen_store_for_type(ctx, bb, member_types.nodes[i], as, adjusted_offset, extracted_value);
            }
            return;
        }
        case NominalType_TAG: {
            gen_store_for_type(ctx, bb, element_type->payload.nom_type.body, as, address, value);
            return;
        }
        case ArrType_TAG:
        case VectorType_TAG: {
            const Node* size = shd_get_fill_type_size(element_type);
            if (size->tag != IntLiteral_TAG) {
                shd_error_print("Size of type ");
                shd_log_node(ERROR, element_type);
                shd_error_print(" is not known a compile-time!\n");
            }
            size_t components_count = shd_get_int_literal_value(*shd_resolve_to_int_literal(size), 0);
            const Type* component_type = shd_get_fill_type_element_type(element_type);
            const Node* offset = address;
            for (size_t i = 0; i < components_count; i++) {
                gen_store_for_type(ctx, bb, component_type, as, offset, shd_extract_helper(a, value, shd_singleton(shd_int32_literal(a, i))));
                const Node* component_type_width = size_of_helper(a, component_type);
                component_type_width = shd_convert_int_zero_extend(a, get_shortptr_type(ctx, as), component_type_width);
                offset = add(offset, component_type_width);
            }
            return;
        }
        default: shd_error("TODO");
    }
}

static const Node* gen_serdes_fn(Context* ctx, const Type* element_type, ShdScope address_scope, bool ser, AddressSpace as) {
    assert(is_as_emulated(ctx, as));
    IrArena* a = ctx->rewriter.dst_arena;

    String fn_name = shd_fmt_string_irarena(a, "generated_%s_%s_%s_%s", ser ? "Store" : "Load", shd_get_scope_name(address_scope), shd_get_type_name(a, element_type), shd_get_address_space_name(as));

    const Node** found = shd_dict_find_value(String, const Node*, ctx->fns, fn_name);
    if (found)
        return *found;

    const Type* emulated_ptr_type = int_type(a, (Int) { .width = a->config.target.memory.ptr_size, .is_signed = false });
    const Node* address_param = param_helper(a, qualified_type(a, (QualifiedType) { .scope = address_scope, .type = emulated_ptr_type }));
    shd_set_debug_name(address_param, "ptr");

    const Type* input_value_t = qualified_type(a, (QualifiedType) { .scope = shd_get_arena_config(a)->target.scopes.bottom, .type = element_type });
    const Node* value_param = NULL;
    if (ser) {
        value_param = param_helper(a, input_value_t);
        shd_set_debug_name(value_param, "value");
    }
    Nodes params = ser ? mk_nodes(a, address_param, value_param) : shd_singleton(address_param);

    const Type* return_value_t = qualified_type(a, (QualifiedType) { .scope = shd_combine_scopes(address_scope, shd_get_addr_space_scope(as)), .type = element_type });
    Nodes return_ts = ser ? shd_empty(a) : shd_singleton(return_value_t);

    Node* fun = function_helper(ctx->rewriter.dst_module, params, return_ts);
    shd_set_debug_name(fun, fn_name);
    shd_add_annotation_named(fun, "Generated");
    shd_add_annotation_named(fun, "Leaf");

    shd_dict_insert(String, Node*, ctx->fns, fn_name, fun);

    BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(fun));
    // convert the pointer to the internal size here
    const Type* shortptr_t = get_shortptr_type(ctx, as);
    address_param = shd_convert_int_zero_extend(a, shortptr_t, address_param);
    if (ser) {
        gen_store_for_type(ctx, bb, element_type, as, address_param, value_param);
        shd_set_abstraction_body(fun, shd_bld_return(bb, shd_empty(a)));
    } else {
        const Node* loaded_value = gen_load_for_type(ctx, bb, element_type, as, address_param);
        assert(loaded_value);
        shd_set_abstraction_body(fun, shd_bld_return(bb, shd_singleton(loaded_value)));
    }
    return fun;
}

static const Node* process_node(Context* ctx, const Node* old) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (old->tag) {
        case Load_TAG: {
            Load payload = old->payload.load;
            const Type* ptr_type = payload.ptr->type;
            ShdScope ptr_scope = shd_deconstruct_qualified_type(&ptr_type);
            assert(ptr_type->tag == PtrType_TAG);
            if (ptr_type->payload.ptr_type.is_reference || !is_as_emulated(ctx, ptr_type->payload.ptr_type.address_space))
                break;
            BodyBuilder* bb = shd_bld_begin_pseudo_instr(a, shd_rewrite_node(r, payload.mem));
            const Type* element_type = shd_rewrite_node(&ctx->rewriter, ptr_type->payload.ptr_type.pointed_type);
            const Node* pointer_as_offset = shd_rewrite_node(&ctx->rewriter, payload.ptr);
            const Node* fn = gen_serdes_fn(ctx, element_type, ptr_scope, false, ptr_type->payload.ptr_type.address_space);
            Nodes results = shd_bld_call(bb, fn, shd_singleton(pointer_as_offset));
            return shd_bld_to_instr_yield_values(bb, results);
        }
        case Store_TAG: {
            Store payload = old->payload.store;
            const Type* ptr_type = payload.ptr->type;
            ShdScope ptr_scope = shd_deconstruct_qualified_type(&ptr_type);
            assert(ptr_type->tag == PtrType_TAG);
            if (ptr_type->payload.ptr_type.is_reference || !is_as_emulated(ctx, ptr_type->payload.ptr_type.address_space))
                break;
            BodyBuilder* bb = shd_bld_begin_pseudo_instr(a, shd_rewrite_node(r, payload.mem));

            const Type* element_type = shd_rewrite_node(&ctx->rewriter, ptr_type->payload.ptr_type.pointed_type);
            const Node* pointer_as_offset = shd_rewrite_node(&ctx->rewriter, payload.ptr);
            const Node* fn = gen_serdes_fn(ctx, element_type, ptr_scope, true, ptr_type->payload.ptr_type.address_space);

            const Node* value = shd_rewrite_node(&ctx->rewriter, payload.value);
            shd_bld_call(bb, fn, mk_nodes(a, pointer_as_offset, value));
            return shd_bld_to_instr_yield_values(bb, shd_empty(a));
        }
        case PtrType_TAG: {
            if (!old->payload.ptr_type.is_reference && is_as_emulated(ctx, old->payload.ptr_type.address_space))
                return int_type(a, (Int) { .width = a->config.target.memory.ptr_size, .is_signed = false });
            break;
        }
        case NullPtr_TAG: {
            if (is_as_emulated(ctx, old->payload.null_ptr.ptr_type->payload.ptr_type.address_space))
                return size_t_literal(a, 0);
            break;
        }
        case GlobalVariable_TAG: {
            GlobalVariable payload = old->payload.global_variable;
            // Global variables into emulated address spaces become integer constants (to index into arrays used for emulation of said address space)
            if (!payload.is_ref && is_as_emulated(ctx, payload.address_space)) {
                assert(false);
            }
            break;
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, old);
}

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

static Nodes collect_globals(Context* ctx, AddressSpace as) {
    IrArena* a = ctx->rewriter.dst_arena;
    Nodes oglobals = shd_module_collect_reachable_globals(ctx->rewriter.src_module);
    LARRAY(const Type*, collected, oglobals.count);
    size_t members_count = 0;

    for (size_t i = 0; i < oglobals.count; i++) {
        const Node* oglobal = oglobals.nodes[i];
        GlobalVariable payload = oglobal->payload.global_variable;
        if (payload.is_ref || payload.address_space != as)
            continue;
        collected[members_count] = oglobal;
        members_count++;
    }

    return shd_nodes(a, members_count, collected);
}

/// Collects all global variables in a specific AS, and creates a record type for them.
static const Node* make_record_type(Context* ctx, AddressSpace as, Nodes collected) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    Module* m = r->dst_module;

    String as_name = shd_get_address_space_name(as);
    Node* global_struct_t = nominal_type_helper(m);
    shd_set_debug_name(global_struct_t, shd_format_string_arena(a->arena, "globals_physical_%s_t", as_name));
    shd_add_annotation_named(global_struct_t, "Generated");

    LARRAY(String, member_names, collected.count);
    LARRAY(const Type*, member_tys, collected.count);

    for (size_t i = 0; i < collected.count; i++) {
        const Node* decl = collected.nodes[i];
        const Type* type = decl->payload.global_variable.type;

        member_tys[i] = shd_rewrite_node(r, type);
        member_names[i] = shd_get_node_name_safe(decl);

        // Turn the old global variable into a pointer (which are also now integers)
        const Type* emulated_ptr_type = int_type(a, (Int) { .width = a->config.target.memory.ptr_size, .is_signed = false });
        Node* new_address = constant_helper(m, emulated_ptr_type);
        shd_set_debug_name(new_address, shd_get_node_name_safe(decl));
        shd_rewrite_annotations(r, decl, new_address);

        // we need to compute the actual pointer by getting the offset and dividing it
        // after lower_memory_layout, optimisations will eliminate this and resolve to a value
        BodyBuilder* bb = shd_bld_begin_pure(a);
        const Node* offset = offset_of_helper(a, global_struct_t, size_t_literal(a, i));
        new_address->payload.constant.value = shd_bld_to_instr_pure_with_values(bb, shd_singleton(offset));

        shd_register_processed(r, decl, new_address);
    }

    const Type* record_t = record_type(a, (RecordType) {
        .members = shd_nodes(a, collected.count, member_tys),
        .names = shd_strings(a, collected.count, member_names)
    });

    //return record_t;
    global_struct_t->payload.nom_type.body = record_t;
    return global_struct_t;
}

static void store_init_data(Context* ctx, AddressSpace as, Nodes collected, BodyBuilder* bb) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    IrArena* oa = ctx->rewriter.src_arena;
    for (size_t i = 0; i < collected.count; i++) {
        const Node* old_decl = collected.nodes[i];
        assert(old_decl->tag == GlobalVariable_TAG);
        const Node* old_init = old_decl->payload.global_variable.init;
        if (old_init) {
            const Node* value = shd_rewrite_node(r, old_init);
            const Node* fn = gen_serdes_fn(ctx, shd_get_unqualified_type(value->type), false, true, old_decl->payload.global_variable.address_space);
            shd_bld_call(bb, fn, mk_nodes(a, shd_rewrite_node(r, old_decl), value));
        }
    }
}

static void construct_emulated_memory_array(Context* ctx, AddressSpace as) {
    IrArena* a = ctx->rewriter.dst_arena;
    Module* m = ctx->rewriter.dst_module;
    String as_name = shd_get_address_space_name(as);

    const Type* word_type = int_type(a, (Int) { .width = a->config.target.memory.word_size, .is_signed = false });
    const Type* ptr_size_type = int_type(a, (Int) { .width = a->config.target.memory.ptr_size, .is_signed = false });

    ctx->collected[as] = collect_globals(ctx, as);
    if (ctx->collected[as].count == 0) {
        const Type* words_array_type = arr_type(a, (ArrType) {
            .element_type = word_type,
            .size = NULL
        });
        *get_emulated_as_word_array(ctx, as) = undef(a, (Undef) { .type = ptr_type(a, (PtrType) { .address_space = as, .pointed_type = words_array_type, .is_reference = true }) });
        return;
    }

    const Node* global_struct_t = make_record_type(ctx, as, ctx->collected[as]);

    // compute the size
    BodyBuilder* bb = shd_bld_begin_pure(a);
    const Node* size_of = size_of_helper(a, global_struct_t);
    const Node* size_in_words = shd_bytes_to_words(bb, size_of);

    Node* memory_size_constant = constant_helper(m, ptr_size_type);
    shd_set_debug_name(memory_size_constant, shd_fmt_string_irarena(a, "memory_%s_size", as_name));
    shd_add_annotation_named(memory_size_constant, "Generated");
    memory_size_constant->payload.constant.value = shd_bld_to_instr_pure_with_values(bb, shd_singleton(size_in_words));

    const Type* words_array_type = arr_type(a, (ArrType) {
        .element_type = word_type,
        .size = memory_size_constant
    });

    AddressSpace ass = as;
    if (ctx->config->lower.use_scratch_for_private && as == AsPrivate) {
        ass = AsGlobal;
    }
    Node* words_array = shd_global_var(m, (GlobalVariable) {
        .address_space = ass,
        .type = words_array_type,
        .is_ref = !ctx->config->lower.use_scratch_for_private
    });
    String name = shd_format_string_arena(a->arena, "memory_%s", as_name);
    shd_set_debug_name(words_array, name);
    shd_add_annotation_named(words_array, "Generated");

    if (ctx->config->lower.use_scratch_for_private && as == AsPrivate) {
        shd_add_annotation_named(words_array, "AllocateInScratchMemory");
        shd_add_annotation_named(words_array, "DoNotDemoteToReference");
        //shd_module_add_export(m, name, words_array);
    }

    *get_emulated_as_word_array(ctx, as) = words_array;
}

KeyHash shd_hash_string(const char** string);
bool shd_compare_string(const char** a, const char** b);

Module* shd_pass_lower_physical_memory(const CompilerConfig* config, const TargetConfig* target, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    for (int i = 0; i < NumAddressSpaces; i++) {
        aconfig.target.memory.address_spaces[i].physical = target->memory.address_spaces[i].physical;
    }

    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process_node),
        .config = config,
    };

    if (is_as_emulated(&ctx, AsPrivate))
        construct_emulated_memory_array(&ctx, AsPrivate);
    if (is_as_emulated(&ctx, AsSubgroup) && dst->arena->config.target.memory.address_spaces[AsSubgroup].allowed)
        construct_emulated_memory_array(&ctx, AsSubgroup);
    if (is_as_emulated(&ctx, AsSubgroup) && dst->arena->config.target.memory.address_spaces[AsShared].allowed)
        construct_emulated_memory_array(&ctx, AsShared);

    ctx.fns = shd_new_dict(String, Node*, (HashFn) shd_hash_string, (CmpFn) shd_compare_string);

    Rewriter* r = &ctx.rewriter;
    Node* ninit;
    const Node* oinit = shd_module_get_init_fn(src);
    BodyBuilder* ninit_bld = shd_bld_begin_fn_rewrite(r, oinit, &ninit);
    shd_rewrite_module(&ctx.rewriter);
    for (AddressSpace as = 0; as < NumAddressSpaces; as++) {
        if (is_as_emulated(&ctx, as))
            store_init_data(&ctx, as, ctx.collected[as], ninit_bld);
    }
    shd_bld_finish_fn_rewrite(r, oinit, ninit, ninit_bld);

    shd_destroy_rewriter(&ctx.rewriter);

    shd_destroy_dict(ctx.fns);

    return dst;
}
