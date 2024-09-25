#include "shady/pass.h"

#include "../ir_private.h"
#include "../type.h"

#include "../transform/ir_gen_helpers.h"
#include "../transform/memory_layout.h"

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

    struct Dict*   serialisation_uniform[NumAddressSpaces];
    struct Dict* deserialisation_uniform[NumAddressSpaces];

    struct Dict*   serialisation_varying[NumAddressSpaces];
    struct Dict* deserialisation_varying[NumAddressSpaces];

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
            const Node* logical_ptr = gen_lea(bb, arr, zero, singleton(address));
            const Node* value = gen_load(bb, logical_ptr);
            return gen_primop_ce(bb, neq_op, 2, (const Node*[]) {value, int_literal(a, (IntLiteral) { .value = 0, .width = a->config.memory.word_size })});
        }
        case PtrType_TAG: switch (element_type->payload.ptr_type.address_space) {
            case AsGlobal: {
                // TODO: add a per-as size configuration
                const Type* ptr_int_t = int_type(a, (Int) {.width = a->config.memory.ptr_size, .is_signed = false });
                const Node* unsigned_int = gen_deserialisation(ctx, bb, ptr_int_t, arr, address);
                return gen_reinterpret_cast(bb, element_type, unsigned_int);
            }
            default: shd_error("TODO")
        }
        case Int_TAG: ser_int: {
            assert(element_type->tag == Int_TAG);
            const Node* acc = int_literal(a, (IntLiteral) { .width = element_type->payload.int_type.width, .is_signed = false, .value = 0 });
            size_t length_in_bytes = int_size_in_bytes(element_type->payload.int_type.width);
            size_t word_size_in_bytes = int_size_in_bytes(a->config.memory.word_size);
            const Node* offset = bytes_to_words(bb, address);
            const Node* shift = int_literal(a, (IntLiteral) { .width = element_type->payload.int_type.width, .is_signed = false, .value = 0 });
            const Node* word_bitwidth = int_literal(a, (IntLiteral) { .width = element_type->payload.int_type.width, .is_signed = false, .value = word_size_in_bytes * 8 });
            for (size_t byte = 0; byte < length_in_bytes; byte += word_size_in_bytes) {
                const Node* word = gen_load(bb, gen_lea(bb, arr, zero, singleton(offset)));
                            word = gen_conversion(bb, int_type(a, (Int) { .width = element_type->payload.int_type.width, .is_signed = false }), word); // widen/truncate the word we just loaded
                            word = first(gen_primop(bb, lshift_op, empty(a), mk_nodes(a, word, shift))); // shift it
                acc = gen_primop_e(bb, or_op, empty(a), mk_nodes(a, acc, word));

                offset = first(gen_primop(bb, add_op, empty(a), mk_nodes(a, offset, size_t_literal(a, 1))));
                shift = first(gen_primop(bb, add_op, empty(a), mk_nodes(a, shift, word_bitwidth)));
            }
            if (config->printf_trace.memory_accesses) {
                AddressSpace as = get_unqualified_type(arr->type)->payload.ptr_type.address_space;
                String template = format_string_interned(a, "loaded %s at %s:0x%s\n", element_type->payload.int_type.width == IntTy64 ? "%lu" : "%u", get_address_space_name(as), "%lx");
                const Node* widened = acc;
                if (element_type->payload.int_type.width < IntTy32)
                    widened = gen_conversion(bb, uint32_type(a), acc);
                gen_debug_printf(bb, template, mk_nodes(a, widened, address));
            }
            acc = gen_reinterpret_cast(bb, int_type(a, (Int) { .width = element_type->payload.int_type.width, .is_signed = element_type->payload.int_type.is_signed }), acc);\
            return acc;
        }
        case Float_TAG: {
            const Type* unsigned_int_t = int_type(a, (Int) {.width = float_to_int_width(element_type->payload.float_type.width), .is_signed = false });
            const Node* unsigned_int = gen_deserialisation(ctx, bb, unsigned_int_t, arr, address);
            return gen_reinterpret_cast(bb, element_type, unsigned_int);
        }
        case TypeDeclRef_TAG:
        case RecordType_TAG: {
            const Type* compound_type = element_type;
            compound_type = get_maybe_nominal_type_body(compound_type);

            Nodes member_types = compound_type->payload.record_type.members;
            LARRAY(const Node*, loaded, member_types.count);
            for (size_t i = 0; i < member_types.count; i++) {
                const Node* field_offset = gen_primop_e(bb, offset_of_op, singleton(element_type), singleton(size_t_literal(a, i)));
                const Node* adjusted_offset = gen_primop_e(bb, add_op, empty(a), mk_nodes(a, address, field_offset));
                loaded[i] = gen_deserialisation(ctx, bb, member_types.nodes[i], arr, adjusted_offset);
            }
            return composite_helper(a, element_type, nodes(a, member_types.count, loaded));
        }
        case ArrType_TAG:
        case PackType_TAG: {
            const Node* size = get_fill_type_size(element_type);
            if (size->tag != IntLiteral_TAG) {
                shd_error_print("Size of type ");
                shd_log_node(ERROR, element_type);
                shd_error_print(" is not known a compile-time!\n");
            }
            size_t components_count = get_int_literal_value(*resolve_to_int_literal(size), 0);
            const Type* component_type = get_fill_type_element_type(element_type);
            LARRAY(const Node*, components, components_count);
            const Node* offset = address;
            for (size_t i = 0; i < components_count; i++) {
                components[i] = gen_deserialisation(ctx, bb, component_type, arr, offset);
                offset = gen_primop_e(bb, add_op, empty(a), mk_nodes(a, offset, gen_primop_e(bb, size_of_op, singleton(component_type), empty(a))));
            }
            return composite_helper(a, element_type, nodes(a, components_count, components));
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
            const Node* logical_ptr = gen_lea(bb, arr, zero, singleton(address));
            const Node* zero_b = int_literal(a, (IntLiteral) { .value = 1, .width = a->config.memory.word_size });
            const Node* one_b =  int_literal(a, (IntLiteral) { .value = 0, .width = a->config.memory.word_size });
            const Node* int_value = gen_primop_ce(bb, select_op, 3, (const Node*[]) { value, one_b, zero_b });
            gen_store(bb, logical_ptr, int_value);
            return;
        }
        case PtrType_TAG: switch (element_type->payload.ptr_type.address_space) {
            case AsGlobal: {
                const Type* ptr_int_t = int_type(a, (Int) {.width = a->config.memory.ptr_size, .is_signed = false });
                const Node* unsigned_value = gen_primop_e(bb, reinterpret_op, singleton(ptr_int_t), singleton(value));
                return gen_serialisation(ctx, bb, ptr_int_t, arr, address, unsigned_value);
            }
            default: shd_error("TODO")
        }
        case Int_TAG: des_int: {
            assert(element_type->tag == Int_TAG);
            // First bitcast to unsigned so we always get zero-extension and not sign-extension afterwards
            const Type* element_t_unsigned = int_type(a, (Int) { .width = element_type->payload.int_type.width, .is_signed = false});
            value = convert_int_extend_according_to_src_t(bb, element_t_unsigned, value);

            // const Node* acc = int_literal(a, (IntLiteral) { .width = element_type->payload.int_type.width, .is_signed = false, .value = 0 });
            size_t length_in_bytes = int_size_in_bytes(element_type->payload.int_type.width);
            size_t word_size_in_bytes = int_size_in_bytes(a->config.memory.word_size);
            const Node* offset = bytes_to_words(bb, address);
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
                word = first(gen_primop(bb, rshift_logical_op, empty(a), mk_nodes(a, word, shift))); // shift it
                word = gen_conversion(bb, int_type(a, (Int) { .width = a->config.memory.word_size, .is_signed = false }), word); // widen/truncate the word we want to store
                gen_store(bb, gen_lea(bb, arr, zero, singleton(offset)), word);

                offset = first(gen_primop(bb, add_op, empty(a), mk_nodes(a, offset, size_t_literal(a, 1))));
                shift = first(gen_primop(bb, add_op, empty(a), mk_nodes(a, shift, word_bitwidth)));
            }
            if (config->printf_trace.memory_accesses) {
                AddressSpace as = get_unqualified_type(arr->type)->payload.ptr_type.address_space;
                String template = format_string_interned(a, "stored %s at %s:0x%s\n", element_type->payload.int_type.width == IntTy64 ? "%lu" : "%u", get_address_space_name(as), "%lx");
                const Node* widened = value;
                if (element_type->payload.int_type.width < IntTy32)
                    widened = gen_conversion(bb, uint32_type(a), value);
                gen_debug_printf(bb, template, mk_nodes(a, widened, address));
            }
            return;
        }
        case Float_TAG: {
            const Type* unsigned_int_t = int_type(a, (Int) {.width = float_to_int_width(element_type->payload.float_type.width), .is_signed = false });
            const Node* unsigned_value = gen_primop_e(bb, reinterpret_op, singleton(unsigned_int_t), singleton(value));
            return gen_serialisation(ctx, bb, unsigned_int_t, arr, address, unsigned_value);
        }
        case RecordType_TAG: {
            Nodes member_types = element_type->payload.record_type.members;
            for (size_t i = 0; i < member_types.count; i++) {
                const Node* extracted_value = prim_op(a, (PrimOp) { .op = extract_op, .operands = mk_nodes(a, value, int32_literal(a, i)), .type_arguments = empty(a) });
                const Node* field_offset = gen_primop_e(bb, offset_of_op, singleton(element_type), singleton(size_t_literal(a, i)));
                const Node* adjusted_offset = gen_primop_e(bb, add_op, empty(a), mk_nodes(a, address, field_offset));
                gen_serialisation(ctx, bb, member_types.nodes[i], arr, adjusted_offset, extracted_value);
            }
            return;
        }
        case TypeDeclRef_TAG: {
            const Node* nom = element_type->payload.type_decl_ref.decl;
            assert(nom && nom->tag == NominalType_TAG);
            gen_serialisation(ctx, bb, nom->payload.nom_type.body, arr, address, value);
            return;
        }
        case ArrType_TAG:
        case PackType_TAG: {
            const Node* size = get_fill_type_size(element_type);
            if (size->tag != IntLiteral_TAG) {
                shd_error_print("Size of type ");
                shd_log_node(ERROR, element_type);
                shd_error_print(" is not known a compile-time!\n");
            }
            size_t components_count = get_int_literal_value(*resolve_to_int_literal(size), 0);
            const Type* component_type = get_fill_type_element_type(element_type);
            const Node* offset = address;
            for (size_t i = 0; i < components_count; i++) {
                gen_serialisation(ctx, bb, component_type, arr, offset, gen_extract(bb, value, singleton(int32_literal(a, i))));
                offset = gen_primop_e(bb, add_op, empty(a), mk_nodes(a, offset, gen_primop_e(bb, size_of_op, singleton(component_type), empty(a))));
            }
            return;
        }
        default: shd_error("TODO");
    }
}

static const Node* gen_serdes_fn(Context* ctx, const Type* element_type, bool uniform_address, bool ser, AddressSpace as) {
    assert(is_as_emulated(ctx, as));
    struct Dict* cache;

    if (uniform_address)
        cache = ser ? ctx->serialisation_uniform[as] : ctx->deserialisation_uniform[as];
    else
        cache = ser ? ctx->serialisation_varying[as] : ctx->deserialisation_varying[as];

    const Node** found = shd_dict_find_value(const Node*, const Node*, cache, element_type);
    if (found)
        return *found;

    IrArena* a = ctx->rewriter.dst_arena;

    const Type* emulated_ptr_type = int_type(a, (Int) { .width = a->config.memory.ptr_size, .is_signed = false });
    const Node* address_param = param(a, qualified_type(a, (QualifiedType) { .is_uniform = !a->config.is_simt || uniform_address, .type = emulated_ptr_type }), "ptr");

    const Type* input_value_t = qualified_type(a, (QualifiedType) { .is_uniform = !a->config.is_simt || (uniform_address && is_addr_space_uniform(a, as) && false), .type = element_type });
    const Node* value_param = ser ? param(a, input_value_t, "value") : NULL;
    Nodes params = ser ? mk_nodes(a, address_param, value_param) : singleton(address_param);

    const Type* return_value_t = qualified_type(a, (QualifiedType) { .is_uniform = !a->config.is_simt || (uniform_address && is_addr_space_uniform(a, as)), .type = element_type });
    Nodes return_ts = ser ? empty(a) : singleton(return_value_t);

    String name = shd_format_string_arena(a->arena, "generated_%s_%s_%s_%s", ser ? "store" : "load", get_address_space_name(as), uniform_address ? "uniform" : "varying", name_type_safe(a, element_type));
    Node* fun = function(ctx->rewriter.dst_module, params, name, mk_nodes(a, annotation(a, (Annotation) { .name = "Generated" }), annotation(a, (Annotation) { .name = "Leaf" })), return_ts);
    shd_dict_insert(const Node*, Node*, cache, element_type, fun);

    BodyBuilder* bb = begin_body_with_mem(a, get_abstraction_mem(fun));
    const Node* base = *get_emulated_as_word_array(ctx, as);
    if (ser) {
        gen_serialisation(ctx, bb, element_type, base, address_param, value_param);
        set_abstraction_body(fun, finish_body_with_return(bb, empty(a)));
    } else {
        const Node* loaded_value = gen_deserialisation(ctx, bb, element_type, base, address_param);
        assert(loaded_value);
        set_abstraction_body(fun, finish_body_with_return(bb, singleton(loaded_value)));
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
            bool uniform_ptr = deconstruct_qualified_type(&ptr_type);
            assert(ptr_type->tag == PtrType_TAG);
            if (ptr_type->payload.ptr_type.is_reference || !is_as_emulated(ctx, ptr_type->payload.ptr_type.address_space))
                break;
            BodyBuilder* bb = begin_block_with_side_effects(a, rewrite_node(r, payload.mem));
            const Type* element_type = rewrite_node(&ctx->rewriter, ptr_type->payload.ptr_type.pointed_type);
            const Node* pointer_as_offset = rewrite_node(&ctx->rewriter, payload.ptr);
            const Node* fn = gen_serdes_fn(ctx, element_type, uniform_ptr, false, ptr_type->payload.ptr_type.address_space);
            Nodes results = gen_call(bb, fn_addr_helper(a, fn), singleton(pointer_as_offset));
            return yield_values_and_wrap_in_block(bb, results);
        }
        case Store_TAG: {
            Store payload = old->payload.store;
            const Type* ptr_type = payload.ptr->type;
            bool uniform_ptr = deconstruct_qualified_type(&ptr_type);
            assert(ptr_type->tag == PtrType_TAG);
            if (ptr_type->payload.ptr_type.is_reference || !is_as_emulated(ctx, ptr_type->payload.ptr_type.address_space))
                break;
            BodyBuilder* bb = begin_block_with_side_effects(a, rewrite_node(r, payload.mem));

            const Type* element_type = rewrite_node(&ctx->rewriter, ptr_type->payload.ptr_type.pointed_type);
            const Node* pointer_as_offset = rewrite_node(&ctx->rewriter, payload.ptr);
            const Node* fn = gen_serdes_fn(ctx, element_type, uniform_ptr, true, ptr_type->payload.ptr_type.address_space);

            const Node* value = rewrite_node(&ctx->rewriter, payload.value);
            gen_call(bb, fn_addr_helper(a, fn), mk_nodes(a, pointer_as_offset, value));
            return yield_values_and_wrap_in_block(bb, empty(a));
        }
        case StackAlloc_TAG: shd_error("This needs to be lowered (see setup_stack_frames.c)")
        case PtrType_TAG: {
            if (!old->payload.ptr_type.is_reference && is_as_emulated(ctx, old->payload.ptr_type.address_space))
                return int_type(a, (Int) { .width = a->config.memory.ptr_size, .is_signed = false });
            break;
        }
        case NullPtr_TAG: {
            if (is_as_emulated(ctx, old->payload.null_ptr.ptr_type->payload.ptr_type.address_space))
                return size_t_literal(a, 0);
            break;
        }
        case GlobalVariable_TAG: {
            const GlobalVariable* old_gvar = &old->payload.global_variable;
            // Global variables into emulated address spaces become integer constants (to index into arrays used for emulation of said address space)
            if (!lookup_annotation(old, "Logical") && is_as_emulated(ctx, old_gvar->address_space)) {
                assert(false);
            }
            break;
        }
        case Function_TAG: {
            if (strcmp(get_abstraction_name(old), "generated_init") == 0) {
                Node* new = recreate_decl_header_identity(&ctx->rewriter, old);
                BodyBuilder *bb = begin_body_with_mem(a, get_abstraction_mem(new));
                for (AddressSpace as = 0; as < NumAddressSpaces; as++) {
                    if (is_as_emulated(ctx, as))
                        store_init_data(ctx, as, ctx->collected[as], bb);
                }
                register_processed(&ctx->rewriter, get_abstraction_mem(old), bb_mem(bb));
                set_abstraction_body(new, finish_body(bb, rewrite_node(&ctx->rewriter, old->payload.fun.body)));
                return new;
            }
            break;
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, old);
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

static Nodes collect_globals(Context* ctx, AddressSpace as) {
    IrArena* a = ctx->rewriter.dst_arena;
    Nodes old_decls = get_module_declarations(ctx->rewriter.src_module);
    LARRAY(const Type*, collected, old_decls.count);
    size_t members_count = 0;

    for (size_t i = 0; i < old_decls.count; i++) {
        const Node* decl = old_decls.nodes[i];
        if (decl->tag != GlobalVariable_TAG) continue;
        if (decl->payload.global_variable.address_space != as) continue;
        if (lookup_annotation(decl, "Logical")) continue;
        collected[members_count] = decl;
        members_count++;
    }

    return nodes(a, members_count, collected);
}

/// Collects all global variables in a specific AS, and creates a record type for them.
static const Node* make_record_type(Context* ctx, AddressSpace as, Nodes collected) {
    IrArena* a = ctx->rewriter.dst_arena;
    Module* m = ctx->rewriter.dst_module;

    String as_name = get_address_space_name(as);
    Node* global_struct_t = nominal_type(m, singleton(annotation(a, (Annotation) { .name = "Generated" })), shd_format_string_arena(a->arena, "globals_physical_%s_t", as_name));

    LARRAY(String, member_names, collected.count);
    LARRAY(const Type*, member_tys, collected.count);

    for (size_t i = 0; i < collected.count; i++) {
        const Node* decl = collected.nodes[i];
        const Type* type = decl->payload.global_variable.type;

        member_tys[i] = rewrite_node(&ctx->rewriter, type);
        member_names[i] = decl->payload.global_variable.name;

        // Turn the old global variable into a pointer (which are also now integers)
        const Type* emulated_ptr_type = int_type(a, (Int) { .width = a->config.memory.ptr_size, .is_signed = false });
        Nodes annotations = rewrite_nodes(&ctx->rewriter, decl->payload.global_variable.annotations);
        Node* new_address = constant(ctx->rewriter.dst_module, annotations, emulated_ptr_type, decl->payload.global_variable.name);

        // we need to compute the actual pointer by getting the offset and dividing it
        // after lower_memory_layout, optimisations will eliminate this and resolve to a value
        BodyBuilder* bb = begin_block_pure(a);
        const Node* offset = gen_primop_e(bb, offset_of_op, singleton(type_decl_ref(a, (TypeDeclRef) { .decl = global_struct_t })), singleton(size_t_literal(a,  i)));
        new_address->payload.constant.value = yield_values_and_wrap_in_compound_instruction(bb, singleton(offset));

        register_processed(&ctx->rewriter, decl, new_address);
    }

    const Type* record_t = record_type(a, (RecordType) {
        .members = nodes(a, collected.count, member_tys),
        .names = strings(a, collected.count, member_names)
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
            const Node* value = rewrite_node(r, old_init);
            const Node* fn = gen_serdes_fn(ctx, get_unqualified_type(value->type), false, true, old_decl->payload.global_variable.address_space);
            gen_call(bb, fn_addr_helper(a, fn), mk_nodes(a, rewrite_node(r, ref_decl_helper(oa, old_decl)), value));
        }
    }
}

static void construct_emulated_memory_array(Context* ctx, AddressSpace as) {
    IrArena* a = ctx->rewriter.dst_arena;
    Module* m = ctx->rewriter.dst_module;
    String as_name = get_address_space_name(as);

    const Type* word_type = int_type(a, (Int) { .width = a->config.memory.word_size, .is_signed = false });
    const Type* ptr_size_type = int_type(a, (Int) { .width = a->config.memory.ptr_size, .is_signed = false });

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

    Nodes annotations = singleton(annotation(a, (Annotation) { .name = "Generated" }));

    // compute the size
    BodyBuilder* bb = begin_block_pure(a);
    const Node* size_of = gen_primop_e(bb, size_of_op, singleton(type_decl_ref(a, (TypeDeclRef) { .decl = global_struct_t })), empty(a));
    const Node* size_in_words = bytes_to_words(bb, size_of);

    Node* constant_decl = constant(m, annotations, ptr_size_type, format_string_interned(a, "memory_%s_size", as_name));
    constant_decl->payload.constant.value = yield_values_and_wrap_in_compound_instruction(bb, singleton(size_in_words));

    const Type* words_array_type = arr_type(a, (ArrType) {
        .element_type = word_type,
        .size = ref_decl_helper(a, constant_decl)
    });

    Node* words_array = global_var(m, append_nodes(a, annotations, annotation(a, (Annotation) { .name = "Logical"})), words_array_type, shd_format_string_arena(a->arena, "memory_%s", as_name), as);

    *get_emulated_as_word_array(ctx, as) = ref_decl_helper(a, words_array);
}

Module* lower_physical_ptrs(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    aconfig.address_spaces[AsPrivate].physical = false;
    aconfig.address_spaces[AsShared].physical = false;
    aconfig.address_spaces[AsSubgroup].physical = false;

    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));

    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process_node),
        .config = config,
    };

    construct_emulated_memory_array(&ctx, AsPrivate);
    if (dst->arena->config.address_spaces[AsSubgroup].allowed)
        construct_emulated_memory_array(&ctx, AsSubgroup);
    if (dst->arena->config.address_spaces[AsShared].allowed)
        construct_emulated_memory_array(&ctx, AsShared);

    for (size_t i = 0; i < NumAddressSpaces; i++) {
        if (is_as_emulated(&ctx, i)) {
            ctx.serialisation_varying[i] = shd_new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
            ctx.deserialisation_varying[i] = shd_new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
            ctx.serialisation_uniform[i] = shd_new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
            ctx.deserialisation_uniform[i] = shd_new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
        }
    }

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);

    for (size_t i = 0; i < NumAddressSpaces; i++) {
        if (is_as_emulated(&ctx, i)) {
            shd_destroy_dict(ctx.serialisation_varying[i]);
            shd_destroy_dict(ctx.deserialisation_varying[i]);
            shd_destroy_dict(ctx.serialisation_uniform[i]);
            shd_destroy_dict(ctx.deserialisation_uniform[i]);
        }
    }

    return dst;
}
