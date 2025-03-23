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

// TODO: make this configuration-dependant
static bool is_as_emulated(SHADY_UNUSED Context* ctx, AddressSpace as) {
    switch (as) {
        case AsPrivate:  return true; // TODO have a config option to do this with swizzled global memory
        case AsSubgroup: return true;
        case AsShared:   return true;
        case AsGlobal:  return false; // TODO have a config option to do this with SSBOs
        default: return false;
    }
}

static const Node** get_emulated_as_word_array(Context* ctx, AddressSpace as) {
    switch (as) {
        case AsPrivate:  return &ctx->fake_private_memory;
        case AsSubgroup: return &ctx->fake_subgroup_memory;
        case AsShared:   return &ctx->fake_shared_memory;
        default: shd_error("Emulation of this AS is not supported");
    }
}

static const Node* gen_deserialisation(Context* ctx, BodyBuilder* bb, const Type* element_type, const Node* arr, const Node* address) {
    IrArena* a = ctx->rewriter.dst_arena;
    const CompilerConfig* config = ctx->config;
    const Node* zero = size_t_literal(a, 0);
    switch (element_type->tag) {
        case Bool_TAG: {
            const Node* logical_ptr = lea_helper(a, arr, zero, shd_singleton(address));
            const Node* value = shd_bld_load(bb, logical_ptr);
            return prim_op_helper(a, neq_op, mk_nodes(a, value, int_literal(a, (IntLiteral) { .value = 0, .width = a->config.target.memory.word_size })));
        }
        case PtrType_TAG: switch (element_type->payload.ptr_type.address_space) {
            case AsGlobal: {
                // TODO: add a per-as size configuration
                const Type* ptr_int_t = int_type(a, (Int) {.width = a->config.target.memory.ptr_size, .is_signed = false });
                const Node* unsigned_int = gen_deserialisation(ctx, bb, ptr_int_t, arr, address);
                return shd_bld_bitcast(bb, element_type, unsigned_int);
            }
            default: shd_error("TODO")
        }
        case Int_TAG: ser_int: {
            assert(element_type->tag == Int_TAG);
            const Node* acc = int_literal(a, (IntLiteral) { .width = element_type->payload.int_type.width, .is_signed = false, .value = 0 });
            size_t length_in_bytes = int_size_in_bytes(element_type->payload.int_type.width);
            size_t word_size_in_bytes = int_size_in_bytes(a->config.target.memory.word_size);
            const Node* offset = shd_bytes_to_words(bb, address);
            const Node* shift = int_literal(a, (IntLiteral) { .width = element_type->payload.int_type.width, .is_signed = false, .value = 0 });
            const Node* word_bitwidth = int_literal(a, (IntLiteral) { .width = element_type->payload.int_type.width, .is_signed = false, .value = word_size_in_bytes * 8 });
            for (size_t byte = 0; byte < length_in_bytes; byte += word_size_in_bytes) {
                const Node* word = shd_bld_load(bb, lea_helper(a, arr, zero, shd_singleton(offset)));
                            word = shd_bld_conversion(bb, int_type(a, (Int) { .width = element_type->payload.int_type.width, .is_signed = false }), word); // widen/truncate the word we just loaded
                word = prim_op_helper(a, lshift_op, mk_nodes(a, word, shift)); // shift it
                acc = prim_op_helper(a, or_op, mk_nodes(a, acc, word));

                offset = prim_op_helper(a, add_op, mk_nodes(a, offset, size_t_literal(a, 1)));
                shift = prim_op_helper(a, add_op, mk_nodes(a, shift, word_bitwidth));
            }
            if (config->printf_trace.memory_accesses) {
                AddressSpace as = shd_get_unqualified_type(arr->type)->payload.ptr_type.address_space;
                String template = shd_fmt_string_irarena(a, "loaded %s at %s:0x%s\n", element_type->payload.int_type.width == IntTy64 ? "%lu" : "%u", shd_get_address_space_name(as), "%lx");
                const Node* widened = acc;
                if (element_type->payload.int_type.width < IntTy32)
                    widened = shd_bld_conversion(bb, shd_uint32_type(a), acc);
                shd_bld_debug_printf(bb, template, mk_nodes(a, widened, address));
            }
            acc = shd_bld_bitcast(bb, int_type(a, (Int) { .width = element_type->payload.int_type.width, .is_signed = element_type->payload.int_type.is_signed }), acc);\
            return acc;
        }
        case Float_TAG: {
            const Type* unsigned_int_t = int_type(a, (Int) {.width = shd_float_to_int_width(element_type->payload.float_type.width), .is_signed = false });
            const Node* unsigned_int = gen_deserialisation(ctx, bb, unsigned_int_t, arr, address);
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
                const Node* adjusted_offset = prim_op_helper(a, add_op, mk_nodes(a, address, field_offset));
                loaded[i] = gen_deserialisation(ctx, bb, member_types.nodes[i], arr, adjusted_offset);
            }
            return composite_helper(a, element_type, shd_nodes(a, member_types.count, loaded));
        }
        case ArrType_TAG:
        case PackType_TAG: {
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
                components[i] = gen_deserialisation(ctx, bb, component_type, arr, offset);
                offset = prim_op_helper(a, add_op, mk_nodes(a, offset, size_of_helper(a, component_type)));
            }
            return composite_helper(a, element_type, shd_nodes(a, components_count, components));
        }
        default: shd_error("TODO");
    }
}

static void gen_serialisation(Context* ctx, BodyBuilder* bb, const Type* element_type, const Node* arr, const Node* address, const Node* value) {
    IrArena* a = ctx->rewriter.dst_arena;
    const CompilerConfig* config = ctx->config;
    const Node* zero = size_t_literal(a, 0);
    switch (element_type->tag) {
        case Bool_TAG: {
            const Node* logical_ptr = lea_helper(a, arr, zero, shd_singleton(address));
            const Node* zero_b = int_literal(a, (IntLiteral) { .value = 1, .width = a->config.target.memory.word_size });
            const Node* one_b =  int_literal(a, (IntLiteral) { .value = 0, .width = a->config.target.memory.word_size });
            const Node* int_value = prim_op_helper(a, select_op, mk_nodes(a, value, one_b, zero_b));
            shd_bld_store(bb, logical_ptr, int_value);
            return;
        }
        case PtrType_TAG: switch (element_type->payload.ptr_type.address_space) {
            case AsGlobal: {
                const Type* ptr_int_t = int_type(a, (Int) {.width = a->config.target.memory.ptr_size, .is_signed = false });
                const Node* unsigned_value = bit_cast_helper(a, ptr_int_t, value);
                return gen_serialisation(ctx, bb, ptr_int_t, arr, address, unsigned_value);
            }
            default: shd_error("TODO")
        }
        case Int_TAG: des_int: {
            assert(element_type->tag == Int_TAG);
            // First bitcast to unsigned so we always get zero-extension and not sign-extension afterwards
            const Type* element_t_unsigned = int_type(a, (Int) { .width = element_type->payload.int_type.width, .is_signed = false});
            value = shd_bld_convert_int_extend_according_to_src_t(bb, element_t_unsigned, value);

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
                    original_word = gen_load(bb, gen_lea(bb, arr, zero, singleton(base_offset)));
                    shd_error_print("TODO");
                    shd_error_die();
                    // word = gen_conversion(bb, int_type(a, (Int) { .width = element_type->payload.int_type.width, .is_signed = false }), word); // widen/truncate the word we just loaded
                }*/
                const Node* word = value;
                word = (prim_op_helper(a, rshift_logical_op, mk_nodes(a, word, shift))); // shift it
                word = shd_bld_conversion(bb, int_type(a, (Int) { .width = a->config.target.memory.word_size, .is_signed = false }), word); // widen/truncate the word we want to store
                shd_bld_store(bb, lea_helper(a, arr, zero, shd_singleton(offset)), word);

                offset = (prim_op_helper(a, add_op, mk_nodes(a, offset, size_t_literal(a, 1))));
                shift = (prim_op_helper(a, add_op, mk_nodes(a, shift, word_bitwidth)));
            }
            if (config->printf_trace.memory_accesses) {
                AddressSpace as = shd_get_unqualified_type(arr->type)->payload.ptr_type.address_space;
                String template = shd_fmt_string_irarena(a, "stored %s at %s:0x%s\n", element_type->payload.int_type.width == IntTy64 ? "%lu" : "%u", shd_get_address_space_name(as), "%lx");
                const Node* widened = value;
                if (element_type->payload.int_type.width < IntTy32)
                    widened = shd_bld_conversion(bb, shd_uint32_type(a), value);
                shd_bld_debug_printf(bb, template, mk_nodes(a, widened, address));
            }
            return;
        }
        case Float_TAG: {
            const Type* unsigned_int_t = int_type(a, (Int) {.width = shd_float_to_int_width(element_type->payload.float_type.width), .is_signed = false });
            const Node* unsigned_value = bit_cast_helper(a, unsigned_int_t, value);
            return gen_serialisation(ctx, bb, unsigned_int_t, arr, address, unsigned_value);
        }
        case RecordType_TAG: {
            Nodes member_types = element_type->payload.record_type.members;
            for (size_t i = 0; i < member_types.count; i++) {
                const Node* extracted_value = prim_op_helper(a, extract_op, mk_nodes(a, value, shd_int32_literal(a, i)));
                const Node* field_offset = offset_of_helper(a, element_type, size_t_literal(a, i));
                const Node* adjusted_offset = prim_op_helper(a, add_op, mk_nodes(a, address, field_offset));
                gen_serialisation(ctx, bb, member_types.nodes[i], arr, adjusted_offset, extracted_value);
            }
            return;
        }
        case NominalType_TAG: {
            gen_serialisation(ctx, bb, element_type->payload.nom_type.body, arr, address, value);
            return;
        }
        case ArrType_TAG:
        case PackType_TAG: {
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
                gen_serialisation(ctx, bb, component_type, arr, offset, shd_extract_helper(a, value, shd_singleton(shd_int32_literal(a, i))));
                offset = prim_op_helper(a, add_op, mk_nodes(a, offset, size_of_helper(a, component_type)));
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
    const Node* base = *get_emulated_as_word_array(ctx, as);
    if (ser) {
        gen_serialisation(ctx, bb, element_type, base, address_param, value_param);
        shd_set_abstraction_body(fun, shd_bld_return(bb, shd_empty(a)));
    } else {
        const Node* loaded_value = gen_deserialisation(ctx, bb, element_type, base, address_param);
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
        case StackAlloc_TAG: shd_error("This needs to be lowered (see setup_stack_frames.c)")
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
        *get_emulated_as_word_array(ctx, as) = undef(a, (Undef) { .type = ptr_type(a, (PtrType) { .address_space = as, .pointed_type = words_array_type }) });
        return;
    }

    const Node* global_struct_t = make_record_type(ctx, as, ctx->collected[as]);

    // compute the size
    BodyBuilder* bb = shd_bld_begin_pure(a);
    const Node* size_of = size_of_helper(a, global_struct_t);
    const Node* size_in_words = shd_bytes_to_words(bb, size_of);

    Node* constant_decl = constant_helper(m, ptr_size_type);
    shd_set_debug_name(constant_decl, shd_fmt_string_irarena(a, "memory_%s_size", as_name));
    shd_add_annotation_named(constant_decl, "Generated");
    constant_decl->payload.constant.value = shd_bld_to_instr_pure_with_values(bb, shd_singleton(size_in_words));

    const Type* words_array_type = arr_type(a, (ArrType) {
        .element_type = word_type,
        .size = constant_decl
    });

    Node* words_array = global_variable_helper(m, words_array_type, as);
    shd_set_debug_name(words_array, shd_format_string_arena(a->arena, "memory_%s", as_name));
    shd_add_annotation_named(words_array, "Generated");

    *get_emulated_as_word_array(ctx, as) = words_array;
}

KeyHash shd_hash_string(const char** string);
bool shd_compare_string(const char** a, const char** b);

Module* shd_pass_lower_physical_ptrs(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    aconfig.target.address_spaces[AsPrivate].physical = false;
    aconfig.target.address_spaces[AsShared].physical = false;
    aconfig.target.address_spaces[AsSubgroup].physical = false;

    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process_node),
        .config = config,
    };

    construct_emulated_memory_array(&ctx, AsPrivate);
    if (dst->arena->config.target.address_spaces[AsSubgroup].allowed)
        construct_emulated_memory_array(&ctx, AsSubgroup);
    if (dst->arena->config.target.address_spaces[AsShared].allowed)
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
