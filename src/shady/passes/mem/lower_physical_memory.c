#include "shady/pass.h"
#include "shady/ir/cast.h"
#include "shady/ir/memory_layout.h"
#include "shady/ir/composite.h"

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
static ShdIntSize get_shortptr_type_size(Context* ctx, AddressSpace as) {
    return ShdIntSize32;
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

static ShdIntSize bytes_to_int_size(int bytes) {
    switch (bytes) {
        case 1: return ShdIntSize8;
        case 2: return ShdIntSize16;
        case 4: return ShdIntSize32;
        case 8: return ShdIntSize64;
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

static const Node* load_word(Context* ctx, BodyBuilder* bb, AddressSpace as, const Node* offset) {
    assert(shd_get_unqualified_type(offset->type) == get_shortptr_type(ctx, as));
    IrArena* a = ctx->rewriter.dst_arena;
    const Node* arr = *get_emulated_as_word_array(ctx, as);
    const Node* value = shd_bld_load(bb, ptr_composite_element_helper(a, arr, offset));
    ShdIntSize width = a->config.target.memory.word_size;
    if (ctx->config->printf_trace.memory_accesses) {
        String template = shd_fmt_string_irarena(a, "loaded %s at %s:0x%s\n", width == ShdIntSize64 ? "%lu" : "%u", shd_get_address_space_name(as), "%lx");
        const Node* widened = value;
        if (width < ShdIntSize32)
            widened = shd_bld_conversion(bb, shd_uint32_type(a), value);
        shd_bld_debug_printf(bb, template, mk_nodes(a, widened, offset));
    }
    return value;
}

static void store_word(Context* ctx, BodyBuilder* bb, AddressSpace as, const Node* offset, const Node* value) {
    assert(shd_get_unqualified_type(offset->type) == get_shortptr_type(ctx, as));
    IrArena* a = ctx->rewriter.dst_arena;
    ShdIntSize width = a->config.target.memory.word_size;
    if (ctx->config->printf_trace.memory_accesses) {
        String template = shd_fmt_string_irarena(a, "storing %s at %s:0x%s\n", width == ShdIntSize64 ? "%lu" : "%u", shd_get_address_space_name(as), "%lx");
        const Node* widened = value;
        if (width < ShdIntSize32)
            widened = shd_bld_conversion(bb, shd_uint32_type(a), value);
        shd_bld_debug_printf(bb, template, mk_nodes(a, widened, offset));
    }
    const Node* arr = *get_emulated_as_word_array(ctx, as);
    shd_bld_store(bb, ptr_composite_element_helper(a, arr, offset), value);
}

static const Node* swizzle_offset(Context* ctx, BodyBuilder* bb, const Node* offset) {
    // can't really swizzle in RT mode
    if (shd_is_rt_execution_model(shd_get_arena_config(ctx->rewriter.src_arena)->target.execution_model))
        return offset;
    IrArena* a = ctx->rewriter.dst_arena;
    const Node* subgroup_size = shd_bld_builtin_load(ctx->rewriter.dst_module, bb, ShdBuiltinSubgroupSize);
    subgroup_size = shd_convert_int_zero_extend(a, shd_get_unqualified_type(offset->type), subgroup_size);
    const Node* subgroup_local_id = shd_bld_builtin_load(ctx->rewriter.dst_module, bb, ShdBuiltinSubgroupLocalInvocationId);
    subgroup_local_id = shd_convert_int_zero_extend(a, shd_get_unqualified_type(offset->type), subgroup_local_id);
    return add(mul(offset, subgroup_size), subgroup_local_id);
}

/// Implements accesses for arbitrary uint sizes
static Node* uint_access(Context* ctx, BodyBuilder* bb, AddressSpace as, ShdIntSize width, /*const Node* old_access,*/ AccessTag tag, Nodes new_operands) {
    IrArena* a = ctx->rewriter.dst_arena;
    const Node* address = shd_first(new_operands);
    //AccessTag tag = is_access(old_access);

    const Node* output = NULL;
    const Node* input = NULL;

    switch (tag) {
        case Access_Load_TAG: {
            output = int_literal(a, (IntLiteral) { .width = width, .is_signed = false, .value = 0 });
            break;
        }
        case Access_Store_TAG: {
            input = new_operands.nodes[1];
            const Type* input_t = shd_get_unqualified_type(input->type);
            assert(input_t->tag == Int_TAG && !input_t->payload.int_type.is_signed);
            break;
        }
        default: assert(false);
    }

    size_t length_in_bytes = int_size_in_bytes(width);
    size_t word_size_in_bytes = int_size_in_bytes(a->config.target.memory.word_size);
    const Node* offset = shd_bytes_to_words(bb, address);
    const Node* shift = int_literal(a, (IntLiteral) { .width = width, .is_signed = false, .value = 0 });
    const Node* word_bitwidth = int_literal(a, (IntLiteral) { .width = width, .is_signed = false, .value = word_size_in_bytes * 8 });
    for (size_t byte = 0; byte < length_in_bytes; byte += word_size_in_bytes) {
        const Node* final_offset = offset;
        // swizzle the address here !
        if (as == AsPrivate && ctx->config->lower.use_scratch_for_private)
            final_offset = swizzle_offset(ctx, bb, final_offset);
        switch (tag) {
            case Access_Load_TAG: {
                const Node* word = load_word(ctx, bb, as, final_offset);
                word = shd_bld_conversion(bb, int_type(a, (Int) { .width = width, .is_signed = false }), word); // widen/truncate the word we just loaded
                word = prim_op_helper(a, lshift_op, mk_nodes(a, word, shift)); // shift it
                output = prim_op_helper(a, or_op, mk_nodes(a, output, word));
                break;
            }
            case Access_Store_TAG: {
                const Node* word = input;
                word = (prim_op_helper(a, rshift_logical_op, mk_nodes(a, word, shift))); // shift it
                word = shd_bld_conversion(bb, int_type(a, (Int) { .width = a->config.target.memory.word_size, .is_signed = false }), word); // widen/truncate the word we want to store
                store_word(ctx, bb, as, final_offset, word);
                break;
            }
            default: assert(false);
        }

        offset = prim_op_helper(a, add_op, mk_nodes(a, offset, shortptr_literal(ctx, as, 1)));
        shift = prim_op_helper(a, add_op, mk_nodes(a, shift, word_bitwidth));
    }

    return output;
}

static const Node* gen_load_for_type(Context* ctx, BodyBuilder* bb, const Type* element_type, AddressSpace as, const Node* address) {
    assert(shd_get_unqualified_type(address->type) == get_shortptr_type(ctx, as));
    IrArena* a = ctx->rewriter.dst_arena;

    const Type* word_t = int_type(a, (Int) { .width = a->config.target.memory.word_size, .is_signed = false });
    switch (element_type->tag) {
        case Int_TAG: {
            Int payload = element_type->payload.int_type;
            const Node* acc = uint_access(ctx, bb, as, payload.width, Access_Load_TAG, mk_nodes(a, address));
            acc = shd_bld_bitcast(bb, int_type(a, (Int) { .width = element_type->payload.int_type.width, .is_signed = element_type->payload.int_type.is_signed }), acc);
            return acc;
        }
        case Bool_TAG: {
            const Node* value = gen_load_for_type(ctx, bb, word_t, as, address);
            return prim_op_helper(a, neq_op, mk_nodes(a, value, int_literal(a, (IntLiteral) { .value = 0, .width = a->config.target.memory.word_size })));
        }
        case PtrType_TAG: {
            TypeMemLayout layout = shd_get_mem_layout(a, element_type);
            assert(layout.size_in_bytes <= int_size_in_bytes(ShdIntSizeMax));
            const Type* ptr_int_t = int_type(a, (Int) { .width = bytes_to_int_size(layout.size_in_bytes), .is_signed = false });
            const Node* unsigned_int = gen_load_for_type(ctx, bb, ptr_int_t, as, address);
            return shd_bld_bitcast(bb, element_type, unsigned_int);
        }
        case Float_TAG: {
            const Type* unsigned_int_t = int_type(a, (Int) {.width = shd_float_to_int_width(element_type->payload.float_type.width), .is_signed = false });
            const Node* unsigned_int = gen_load_for_type(ctx, bb, unsigned_int_t, as, address);
            return shd_bld_bitcast(bb, element_type, unsigned_int);
        }
        case StructType_TAG: {
            const Type* compound_type = element_type;
            Nodes member_types = compound_type->payload.struct_type.members;
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
    const Type* word_t = int_type(a, (Int) { .width = a->config.target.memory.word_size, .is_signed = false });
    switch (element_type->tag) {
        case Int_TAG: {
            Int payload = element_type->payload.int_type;
            const Type* element_t_unsigned = int_type(a, (Int) { .width = element_type->payload.int_type.width, .is_signed = false});
            value = shd_convert_int_extend_according_to_src_t(a, element_t_unsigned, value);
            uint_access(ctx, bb, as, payload.width, Access_Store_TAG, mk_nodes(a, address, value));
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
            assert(layout.size_in_bytes <= int_size_in_bytes(ShdIntSizeMax));
            const Type* ptr_int_t = int_type(a, (Int) { .width = bytes_to_int_size(layout.size_in_bytes), .is_signed = false });
            const Node* unsigned_value = bit_cast_helper(a, ptr_int_t, value);
            return gen_store_for_type(ctx, bb, ptr_int_t, as, address, unsigned_value);
        }
        case Float_TAG: {
            const Type* unsigned_int_t = int_type(a, (Int) {.width = shd_float_to_int_width(element_type->payload.float_type.width), .is_signed = false });
            const Node* unsigned_value = bit_cast_helper(a, unsigned_int_t, value);
            return gen_store_for_type(ctx, bb, unsigned_int_t, as, address, unsigned_value);
        }
        case StructType_TAG: {
            Nodes member_types = element_type->payload.struct_type.members;
            for (size_t i = 0; i < member_types.count; i++) {
                const Node* extracted_value = shd_extract_literal(a, value, i);
                const Node* field_offset = offset_of_helper(a, element_type, size_t_literal(a, i));
                field_offset = shd_convert_int_zero_extend(a, get_shortptr_type(ctx, as), field_offset);
                const Node* adjusted_offset = prim_op_helper(a, add_op, mk_nodes(a, address, field_offset));
                gen_store_for_type(ctx, bb, member_types.nodes[i], as, adjusted_offset, extracted_value);
            }
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

static const Node* get_emulating_function(Context* ctx, const Node* old_op) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    AccessTag tag = is_access(old_op);
    assert(tag != NotAnAccess);
    const Node* old_ptr = get_access_ptr(old_op);
    const Node* old_ptr_type = old_ptr->type;
    ShdScope address_scope = shd_deconstruct_qualified_type(&old_ptr_type);
    assert(old_ptr_type->tag == PtrType_TAG);
    AddressSpace as = old_ptr_type->payload.ptr_type.address_space;
    const Node* element_type = shd_rewrite_node(r, old_ptr_type->payload.ptr_type.pointed_type);

    assert(is_as_emulated(ctx, as));

    String fn_name = shd_fmt_string_irarena(a, "emulated_%s_%s_%s_%s", shd_get_node_tag_string(old_op->tag), shd_get_scope_name(address_scope), shd_get_type_name(a, element_type), shd_get_address_space_name(as));

    const Node** found = shd_dict_find_value(String, const Node*, ctx->fns, fn_name);
    if (found)
        return *found;

    const Type* emulated_ptr_type = int_type(a, (Int) { .width = a->config.target.memory.ptr_size, .is_signed = false });
    const Node* address_param = param_helper(a, qualified_type(a, (QualifiedType) { .scope = address_scope, .type = emulated_ptr_type }));
    shd_set_debug_name(address_param, "ptr");

    const Type* input_value_t = qualified_type(a, (QualifiedType) { .scope = shd_get_arena_config(a)->target.scopes.bottom, .type = element_type });

    Nodes params;
    Nodes return_types;
    switch (tag) {
        case Access_Load_TAG: {
            params = shd_singleton(address_param);
            shd_set_debug_name(params.nodes[0], "value");
            //const Type* return_value_t = qualified_type(a, (QualifiedType) { .scope = shd_combine_scopes(address_scope, shd_get_addr_space_scope(as)), .type = element_type });
            return_types = shd_singleton(shd_rewrite_node(r, old_op->type));
            break;
        }
        case Access_Store_TAG: {
            params = mk_nodes(a, address_param, param_helper(a, input_value_t));
            return_types = shd_empty(a);
            break;
        }
        default: shd_error("Unimplemented access!");
    }

    Node* fun = function_helper(ctx->rewriter.dst_module, params, return_types);
    shd_set_debug_name(fun, fn_name);
    shd_add_annotation_named(fun, "Generated");
    shd_add_annotation_named(fun, "Leaf");
    shd_dict_insert(String, Node*, ctx->fns, fn_name, fun);

    BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(fun));
    // convert the pointer to the internal size here
    const Type* shortptr_t = get_shortptr_type(ctx, as);
    address_param = shd_convert_int_zero_extend(a, shortptr_t, address_param);

    switch (tag) {
        case Access_Load_TAG: {
            const Node* loaded_value = gen_load_for_type(ctx, bb, element_type, as, address_param);
            assert(loaded_value);
            shd_set_abstraction_body(fun, shd_bld_return(bb, shd_singleton(loaded_value)));
            break;
        }
        case Access_Store_TAG: {
            gen_store_for_type(ctx, bb, element_type, as, address_param, params.nodes[1]);
            shd_set_abstraction_body(fun, shd_bld_return(bb, shd_empty(a)));
            break;
        }
        default: shd_error("Unimplemented access!");
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
            shd_deconstruct_qualified_type(&ptr_type);
            assert(ptr_type->tag == PtrType_TAG);
            if (ptr_type->payload.ptr_type.is_reference || !is_as_emulated(ctx, ptr_type->payload.ptr_type.address_space))
                break;
            BodyBuilder* bb = shd_bld_begin_pseudo_instr(a, shd_rewrite_node(r, payload.mem));
            const Node* pointer_as_offset = shd_rewrite_node(&ctx->rewriter, payload.ptr);
            const Node* fn = get_emulating_function(ctx, old);
            Nodes results = shd_bld_call(bb, fn, shd_singleton(pointer_as_offset));
            return shd_bld_to_instr_yield_values(bb, results);
        }
        case Store_TAG: {
            Store payload = old->payload.store;
            const Type* ptr_type = payload.ptr->type;
            shd_deconstruct_qualified_type(&ptr_type);
            assert(ptr_type->tag == PtrType_TAG);
            if (ptr_type->payload.ptr_type.is_reference || !is_as_emulated(ctx, ptr_type->payload.ptr_type.address_space))
                break;
            
            BodyBuilder* bb = shd_bld_begin_pseudo_instr(a, shd_rewrite_node(r, payload.mem));
            const Node* pointer_as_offset = shd_rewrite_node(&ctx->rewriter, payload.ptr);
            const Node* fn = get_emulating_function(ctx, old);
            const Node* value = shd_rewrite_node(&ctx->rewriter, payload.value);
            shd_bld_call(bb, fn, mk_nodes(a, pointer_as_offset, value));
            return shd_bld_to_instr_yield_values(bb, shd_empty(a));
        }
        case AtomicAccess_TAG: {
            AtomicAccess old_payload = old->payload.atomic_access;
            const Type* ptr_type = old_payload.ptr->type;
            shd_deconstruct_qualified_type(&ptr_type);
            assert(ptr_type->tag == PtrType_TAG);
            if (ptr_type->payload.ptr_type.is_reference || !is_as_emulated(ctx, ptr_type->payload.ptr_type.address_space))
                break;

            AddressSpace as = ptr_type->payload.ptr_type.address_space;
            const Node* element_type = ptr_type->payload.ptr_type.pointed_type;
            assert(shd_get_type_bitwidth(element_type) == int_size_in_bytes(a->config.target.memory.word_size) * 8);

            BodyBuilder* bb = shd_bld_begin_pseudo_instr(a, shd_rewrite_node(r, old_payload.mem));

            const Node* ptr = shd_rewrite_node(r, old_payload.ptr);
            const Node* offset = shd_bytes_to_words(bb, ptr);

            const Node* arr = *get_emulated_as_word_array(ctx, as);

            AtomicAccess rewritten = {
                .mem = shd_bld_mem(bb),
                .op = old_payload.op,
                .result_t = shd_rewrite_node(r, old_payload.result_t),
                .ptr = ptr_composite_element_helper(a, arr, offset),
                .scope = shd_rewrite_node(r, old_payload.scope),
                .semantics = shd_rewrite_node(r, old_payload.semantics),
                //.ops = shd_rewrite_nodes(r, old_payload.ops),
            };

            const Type* word_type = int_type(a, (Int) { .width = a->config.target.memory.word_size, .is_signed = false });
            LARRAY(const Node*, nops, old_payload.ops.count);
            for (size_t i = 0; i < old_payload.ops.count; i++) {
                nops[i] = shd_rewrite_node(r, old_payload.ops.nodes[i]);
                nops[i] = bit_cast_helper(a, word_type, nops[i]);
            }
            rewritten.ops = shd_nodes(a, old_payload.ops.count, nops);

            const Node* result = shd_bld_add_instruction(bb, atomic_access(a, rewritten));

            if (old_payload.result_t) {
                result = bit_cast_helper(a, shd_get_unqualified_type(rewritten.result_t), result);
                return shd_bld_to_instr_yield_values(bb, shd_singleton(result));
            }
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
    Node* global_struct_t = struct_type_helper(a, 0);
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

    shd_struct_type_set_members_named(global_struct_t, shd_nodes(a, collected.count, member_tys), shd_strings(a, collected.count, member_names));
    return global_struct_t;
}

// TODO: this should be a dedicated pass
static void store_init_data(Context* ctx, AddressSpace as, Nodes collected, BodyBuilder* bb) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    IrArena* oa = ctx->rewriter.src_arena;
    for (size_t i = 0; i < collected.count; i++) {
        const Node* old_decl = collected.nodes[i];
        assert(old_decl->tag == GlobalVariable_TAG);
        const Node* old_init = old_decl->payload.global_variable.init;
        if (old_init) {
            // obtain the appropriate emulating function for the store
            const Node* old_dummy_store = store_helper(oa, NULL, old_decl, old_init);
            const Node* fn = get_emulating_function(ctx, old_dummy_store);
            // and then just call it!
            shd_bld_call(bb, fn, mk_nodes(a, shd_rewrite_node(r, old_decl), shd_rewrite_node(r, old_init)));
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
