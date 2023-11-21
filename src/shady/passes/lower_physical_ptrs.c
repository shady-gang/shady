#include "passes.h"

#include "../transform/ir_gen_helpers.h"
#include "../transform/memory_layout.h"

#include "../ir_private.h"
#include "../rewrite.h"
#include "../type.h"

#include "log.h"
#include "portability.h"
#include "util.h"

#include "list.h"
#include "dict.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    const CompilerConfig* config;

    struct Dict*   serialisation_uniform[NumAddressSpaces];
    struct Dict* deserialisation_uniform[NumAddressSpaces];

    struct Dict*   serialisation_varying[NumAddressSpaces];
    struct Dict* deserialisation_varying[NumAddressSpaces];

    const Node* fake_private_memory;
    const Node* fake_subgroup_memory;
    const Node* fake_shared_memory;
} Context;

// TODO: make this configuration-dependant
static bool is_as_emulated(SHADY_UNUSED Context* ctx, AddressSpace as) {
    switch (as) {
        case AsPrivatePhysical:  return true; // TODO have a config option to do this with swizzled global memory
        case AsSubgroupPhysical: return true;
        case AsSharedPhysical:   return true;
        case AsGlobalPhysical:  return false; // TODO have a config option to do this with SSBOs
        default: return false;
    }
}

static const Node** get_emulated_as_word_array(Context* ctx, AddressSpace as) {
    switch (as) {
        case AsPrivatePhysical:  return &ctx->fake_private_memory;
        case AsSubgroupPhysical: return &ctx->fake_subgroup_memory;
        case AsSharedPhysical:   return &ctx->fake_shared_memory;
        default: error("Emulation of this AS is not supported");
    }
}

static const Node* gen_deserialisation(Context* ctx, BodyBuilder* bb, const Type* element_type, const Node* arr, const Node* base_offset) {
    IrArena* a = ctx->rewriter.dst_arena;
    const CompilerConfig* config = ctx->config;
    const Node* zero = size_t_literal(a, 0);
    switch (element_type->tag) {
        case Bool_TAG: {
            const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset });
            const Node* value = gen_load(bb, logical_ptr);
            return gen_primop_ce(bb, neq_op, 2, (const Node*[]) {value, int_literal(a, (IntLiteral) { .value = 0, .width = a->config.memory.word_size })});
        }
        case PtrType_TAG: switch (element_type->payload.ptr_type.address_space) {
            case AsGlobalPhysical: {
                const Type* ptr_int_t = int_type(a, (Int) {.width = a->config.memory.ptr_size, .is_signed = false });
                const Node* unsigned_int = gen_deserialisation(ctx, bb, ptr_int_t, arr, base_offset);
                return gen_reinterpret_cast(bb, element_type, unsigned_int);
            }
            default: error("TODO")
        }
        case Int_TAG: ser_int: {
            assert(element_type->tag == Int_TAG);
            const Node* acc = int_literal(a, (IntLiteral) { .width = element_type->payload.int_type.width, .is_signed = false, .value = 0 });
            size_t length_in_bytes = int_size_in_bytes(element_type->payload.int_type.width);
            size_t word_size_in_bytes = int_size_in_bytes(a->config.memory.word_size);
            const Node* offset = base_offset;
            const Node* shift = int_literal(a, (IntLiteral) { .width = element_type->payload.int_type.width, .is_signed = false, .value = 0 });
            const Node* word_bitwidth = int_literal(a, (IntLiteral) { .width = element_type->payload.int_type.width, .is_signed = false, .value = word_size_in_bytes * 8 });
            for (size_t byte = 0; byte < length_in_bytes; byte += word_size_in_bytes) {
                const Node* word = gen_load(bb, gen_primop_ce(bb, lea_op, 3, (const Node* []) {arr, zero, offset}));
                            word = gen_conversion(bb, int_type(a, (Int) { .width = element_type->payload.int_type.width, .is_signed = false }), word); // widen/truncate the word we just loaded
                            word = first(gen_primop(bb, lshift_op, empty(a), mk_nodes(a, word, shift))); // shift it
                acc = gen_primop_e(bb, or_op, empty(a), mk_nodes(a, acc, word));

                offset = first(gen_primop(bb, add_op, empty(a), mk_nodes(a, offset, size_t_literal(a, 1))));
                shift = first(gen_primop(bb, add_op, empty(a), mk_nodes(a, shift, word_bitwidth)));
            }
            if (config->printf_trace.memory_accesses) {
                AddressSpace as = get_unqualified_type(arr->type)->payload.ptr_type.address_space;
                String template = format_string_interned(a, "loaded %s at %s:%s\n", element_type->payload.int_type.width == IntTy64 ? "%lu" : "%u", get_address_space_name(as), "%lx");
                const Node* widened = acc;
                if (element_type->payload.int_type.width < IntTy32)
                    widened = gen_conversion(bb, uint32_type(a), acc);
                bind_instruction(bb, prim_op(a, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(a, string_lit(a, (StringLiteral) { .string = template }), widened, base_offset) }));
            }
            acc = gen_reinterpret_cast(bb, int_type(a, (Int) { .width = element_type->payload.int_type.width, .is_signed = element_type->payload.int_type.is_signed }), acc);\
            return acc;
        }
        case Float_TAG: {
            const Type* unsigned_int_t = int_type(a, (Int) {.width = float_to_int_width(element_type->payload.float_type.width), .is_signed = false });
            const Node* unsigned_int = gen_deserialisation(ctx, bb, unsigned_int_t, arr, base_offset);
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
                            field_offset = bytes_to_words(bb, field_offset);
                const Node* adjusted_offset = gen_primop_e(bb, add_op, empty(a), mk_nodes(a, base_offset, field_offset));
                loaded[i] = gen_deserialisation(ctx, bb, member_types.nodes[i], arr, adjusted_offset);
            }
            return composite_helper(a, element_type, nodes(a, member_types.count, loaded));
        }
        case ArrType_TAG:
        case PackType_TAG: {
            const Node* size = get_fill_type_size(element_type);
            if (size->tag != IntLiteral_TAG) {
                error_print("Size of type ");
                log_node(ERROR, element_type);
                error_print(" is not known a compile-time!\n");
            }
            size_t components_count = get_int_literal_value(size, 0);
            const Type* component_type = get_fill_type_element_type(element_type);
            LARRAY(const Node*, components, components_count);
            const Node* offset = base_offset;
            for (size_t i = 0; i < components_count; i++) {
                components[i] = gen_deserialisation(ctx, bb, component_type, arr, offset);
                offset = gen_primop_e(bb, add_op, empty(a), mk_nodes(a, offset, gen_primop_e(bb, size_of_op, singleton(component_type), empty(a))));
            }
            return composite_helper(a, element_type, nodes(a, components_count, components));
        }
        default: error("TODO");
    }
}

static void gen_serialisation(Context* ctx, BodyBuilder* bb, const Type* element_type, const Node* arr, const Node* base_offset, const Node* value) {
    IrArena* a = ctx->rewriter.dst_arena;
    const CompilerConfig* config = ctx->config;
    const Node* zero = size_t_literal(a, 0);
    switch (element_type->tag) {
        case Bool_TAG: {
            const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset });
            const Node* zero_b = int_literal(a, (IntLiteral) { .value = 1, .width = a->config.memory.word_size });
            const Node* one_b =  int_literal(a, (IntLiteral) { .value = 0, .width = a->config.memory.word_size });
            const Node* int_value = gen_primop_ce(bb, select_op, 3, (const Node*[]) { value, one_b, zero_b });
            gen_store(bb, logical_ptr, int_value);
            return;
        }
        case PtrType_TAG: switch (element_type->payload.ptr_type.address_space) {
            case AsGlobalPhysical: {
                const Type* ptr_int_t = int_type(a, (Int) {.width = a->config.memory.ptr_size, .is_signed = false });
                const Node* unsigned_value = gen_primop_e(bb, reinterpret_op, singleton(ptr_int_t), singleton(value));
                return gen_serialisation(ctx, bb, ptr_int_t, arr, base_offset, unsigned_value);
            }
            default: error("TODO")
        }
        case Int_TAG: des_int: {
            assert(element_type->tag == Int_TAG);
            // First bitcast to unsigned so we always get zero-extension and not sign-extension afterwards
            const Type* element_t_unsigned = int_type(a, (Int) { .width = element_type->payload.int_type.width, .is_signed = false});
            value = convert_int_extend_according_to_src_t(bb, element_t_unsigned, value);

            // const Node* acc = int_literal(a, (IntLiteral) { .width = element_type->payload.int_type.width, .is_signed = false, .value = 0 });
            size_t length_in_bytes = int_size_in_bytes(element_type->payload.int_type.width);
            size_t word_size_in_bytes = int_size_in_bytes(a->config.memory.word_size);
            const Node* offset = base_offset;
            const Node* shift = int_literal(a, (IntLiteral) { .width = element_type->payload.int_type.width, .is_signed = false, .value = 0 });
            const Node* word_bitwidth = int_literal(a, (IntLiteral) { .width = element_type->payload.int_type.width, .is_signed = false, .value = word_size_in_bytes * 8 });
            for (size_t byte = 0; byte < length_in_bytes; byte += word_size_in_bytes) {
                bool is_last_word = byte + word_size_in_bytes >= length_in_bytes;
                /*bool needs_patch = is_last_word && word_size_in_bytes < length_in_bytes;
                const Node* original_word = NULL;
                if (needs_patch) {
                    original_word = gen_load(bb, gen_primop_ce(bb, lea_op, 3, (const Node* []) {arr, zero, offset}));
                    error_print("TODO");
                    error_die();
                    // word = gen_conversion(bb, int_type(a, (Int) { .width = element_type->payload.int_type.width, .is_signed = false }), word); // widen/truncate the word we just loaded
                }*/
                const Node* word = value;
                word = first(gen_primop(bb, rshift_logical_op, empty(a), mk_nodes(a, word, shift))); // shift it
                word = gen_conversion(bb, int_type(a, (Int) { .width = a->config.memory.word_size, .is_signed = false }), word); // widen/truncate the word we want to store
                gen_store(bb, gen_primop_ce(bb, lea_op, 3, (const Node* []) {arr, zero, offset}), word);

                offset = first(gen_primop(bb, add_op, empty(a), mk_nodes(a, offset, size_t_literal(a, 1))));
                shift = first(gen_primop(bb, add_op, empty(a), mk_nodes(a, shift, word_bitwidth)));
            }
            if (config->printf_trace.memory_accesses) {
                AddressSpace as = get_unqualified_type(arr->type)->payload.ptr_type.address_space;
                String template = format_string_interned(a, "stored %s at %s:%s\n", element_type->payload.int_type.width == IntTy64 ? "%lu" : "%u", get_address_space_name(as), "%lx");
                const Node* widened = value;
                if (element_type->payload.int_type.width < IntTy32)
                    widened = gen_conversion(bb, uint32_type(a), value);
                bind_instruction(bb, prim_op(a, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(a, string_lit(a, (StringLiteral) { .string = template }), widened, base_offset) }));
            }
            return;
        }
        case Float_TAG: {
            const Type* unsigned_int_t = int_type(a, (Int) {.width = float_to_int_width(element_type->payload.float_type.width), .is_signed = false });
            const Node* unsigned_value = gen_primop_e(bb, reinterpret_op, singleton(unsigned_int_t), singleton(value));
            return gen_serialisation(ctx, bb, unsigned_int_t, arr, base_offset, unsigned_value);
        }
        case RecordType_TAG: {
            Nodes member_types = element_type->payload.record_type.members;
            for (size_t i = 0; i < member_types.count; i++) {
                const Node* extracted_value = first(bind_instruction(bb, prim_op(a, (PrimOp) { .op = extract_op, .operands = mk_nodes(a, value, int32_literal(a, i)), .type_arguments = empty(a) })));
                const Node* field_offset = gen_primop_e(bb, offset_of_op, singleton(element_type), singleton(size_t_literal(a, i)));
                            field_offset = bytes_to_words(bb, field_offset);
                const Node* adjusted_offset = gen_primop_e(bb, add_op, empty(a), mk_nodes(a, base_offset, field_offset));
                gen_serialisation(ctx, bb, member_types.nodes[i], arr, adjusted_offset, extracted_value);
            }
            return;
        }
        case TypeDeclRef_TAG: {
            const Node* nom = element_type->payload.type_decl_ref.decl;
            assert(nom && nom->tag == NominalType_TAG);
            gen_serialisation(ctx, bb, nom->payload.nom_type.body, arr, base_offset, value);
            return;
        }
        case ArrType_TAG:
        case PackType_TAG: {
            const Node* size = get_fill_type_size(element_type);
            if (size->tag != IntLiteral_TAG) {
                error_print("Size of type ");
                log_node(ERROR, element_type);
                error_print(" is not known a compile-time!\n");
            }
            size_t components_count = get_int_literal_value(size, 0);
            const Type* component_type = get_fill_type_element_type(element_type);
            const Node* offset = base_offset;
            for (size_t i = 0; i < components_count; i++) {
                gen_serialisation(ctx, bb, component_type, arr, offset, gen_extract(bb, value, singleton(int32_literal(a, i))));
                offset = gen_primop_e(bb, add_op, empty(a), mk_nodes(a, offset, gen_primop_e(bb, size_of_op, singleton(component_type), empty(a))));
            }
            return;
        }
        default: error("TODO");
    }
}

static const Node* gen_serdes_fn(Context* ctx, const Type* element_type, bool uniform_address, bool ser, AddressSpace as) {
    assert(is_as_emulated(ctx, as));
    struct Dict* cache;

    if (uniform_address)
        cache = ser ? ctx->serialisation_uniform[as] : ctx->deserialisation_uniform[as];
    else
        cache = ser ? ctx->serialisation_varying[as] : ctx->deserialisation_varying[as];

    const Node** found = find_value_dict(const Node*, const Node*, cache, element_type);
    if (found)
        return *found;

    IrArena* a = ctx->rewriter.dst_arena;

    const Type* emulated_ptr_type = int_type(a, (Int) { .width = a->config.memory.ptr_size, .is_signed = false });
    const Node* address_param = var(a, qualified_type(a, (QualifiedType) { .is_uniform = !a->config.is_simt || uniform_address, .type = emulated_ptr_type }), "ptr");

    const Type* input_value_t = qualified_type(a, (QualifiedType) { .is_uniform = !a->config.is_simt || (uniform_address && is_addr_space_uniform(a, as) && false), .type = element_type });
    const Node* value_param = ser ? var(a, input_value_t, "value") : NULL;
    Nodes params = ser ? mk_nodes(a, address_param, value_param) : singleton(address_param);

    const Type* return_value_t = qualified_type(a, (QualifiedType) { .is_uniform = !a->config.is_simt || (uniform_address && is_addr_space_uniform(a, as)), .type = element_type });
    Nodes return_ts = ser ? empty(a) : singleton(return_value_t);

    String name = format_string_arena(a->arena, "generated_%s_%s_%s_%s", ser ? "store" : "load", get_address_space_name(as), uniform_address ? "uniform" : "varying", name_type_safe(a, element_type));
    Node* fun = function(ctx->rewriter.dst_module, params, name, singleton(annotation(a, (Annotation) { .name = "Generated" })), return_ts);
    insert_dict(const Node*, Node*, cache, element_type, fun);

    BodyBuilder* bb = begin_body(a);
    const Node* address = bytes_to_words(bb, address_param);
    const Node* base = ref_decl_helper(a, *get_emulated_as_word_array(ctx, as));
    if (ser) {
        gen_serialisation(ctx, bb, element_type, base, address, value_param);
        fun->payload.fun.body = finish_body(bb, fn_ret(a, (Return) { .fn = fun, .args = empty(a) }));
    } else {
        const Node* loaded_value = gen_deserialisation(ctx, bb, element_type, base, address);
        assert(loaded_value);
        fun->payload.fun.body = finish_body(bb, fn_ret(a, (Return) { .fn = fun, .args = singleton(loaded_value) }));
    }
    return fun;
}

static const Node* process_let(Context* ctx, const Node* node) {
    assert(node->tag == Let_TAG);
    IrArena* a = ctx->rewriter.dst_arena;

    const Node* tail = rewrite_node(&ctx->rewriter, node->payload.let.tail);
    const Node* old_instruction = node->payload.let.instruction;

    if (old_instruction->tag == PrimOp_TAG) {
        const PrimOp* oprim_op = &old_instruction->payload.prim_op;
        switch (oprim_op->op) {
            case alloca_subgroup_op:
            case alloca_op: error("This needs to be lowered (see setup_stack_frames.c)")
            // lowering for either kind of memory accesses is similar
            case load_op:
            case store_op: {
                const Node* old_ptr = oprim_op->operands.nodes[0];
                const Type* ptr_type = old_ptr->type;
                bool uniform_ptr = deconstruct_qualified_type(&ptr_type);
                assert(ptr_type->tag == PtrType_TAG);
                if (!is_as_emulated(ctx, ptr_type->payload.ptr_type.address_space))
                    break;
                BodyBuilder* bb = begin_body(a);

                const Type* element_type = rewrite_node(&ctx->rewriter, ptr_type->payload.ptr_type.pointed_type);
                const Node* pointer_as_offset = rewrite_node(&ctx->rewriter, old_ptr);
                const Node* fn = gen_serdes_fn(ctx, element_type, uniform_ptr, oprim_op->op == store_op, ptr_type->payload.ptr_type.address_space);

                if (oprim_op->op == load_op) {
                    return finish_body(bb, let(a, call(a, (Call) {.callee = fn_addr_helper(a, fn), .args = singleton(pointer_as_offset)}), tail));
                } else {
                    const Node* value = rewrite_node(&ctx->rewriter, oprim_op->operands.nodes[1]);
                    return finish_body(bb, let(a, call(a, (Call) { .callee = fn_addr_helper(a, fn), .args = mk_nodes(a, pointer_as_offset, value) }), tail));
                }
                SHADY_UNREACHABLE;
            }
            default: break;
        }
    }

    return let(a, rewrite_node(&ctx->rewriter, old_instruction), tail);
}

static const Node* process_node(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

    IrArena* a = ctx->rewriter.dst_arena;

    switch (old->tag) {
        case Let_TAG: return process_let(ctx, old);
        case PtrType_TAG: {
            if (is_as_emulated(ctx, old->payload.ptr_type.address_space))
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
            if (is_as_emulated(ctx, old_gvar->address_space)) {
                assert(false);
            }
            break;
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, old);
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

/// Collects all global variables in a specific AS, and creates a record type for them.
static void collect_globals_into_record_type(Context* ctx, Node* global_struct_t, AddressSpace as) {
    IrArena* a = ctx->rewriter.dst_arena;
    Module* m = ctx->rewriter.dst_module;
    Nodes old_decls = get_module_declarations(ctx->rewriter.src_module);

    LARRAY(String, member_names, old_decls.count);
    LARRAY(const Type*, member_tys, old_decls.count);
    size_t members_count = 0;

    for (size_t i = 0; i < old_decls.count; i++) {
        const Node* decl = old_decls.nodes[i];
        if (decl->tag != GlobalVariable_TAG) continue;
        if (decl->payload.global_variable.address_space != as) continue;
        const Type* type = decl->payload.global_variable.type;

        member_tys[members_count] = rewrite_node(&ctx->rewriter, type);
        member_names[members_count] = decl->payload.global_variable.name;

        // Turn the old global variable into a pointer (which are also now integers)
        const Type* emulated_ptr_type = int_type(a, (Int) { .width = a->config.memory.ptr_size, .is_signed = false });
        Nodes annotations = rewrite_nodes(&ctx->rewriter, decl->payload.global_variable.annotations);
        Node* cnst = constant(ctx->rewriter.dst_module, annotations, emulated_ptr_type, decl->payload.global_variable.name);

        // we need to compute the actual pointer by getting the offset and dividing it
        // after lower_memory_layout, optimisations will eliminate this and resolve to a value
        BodyBuilder* bb = begin_body(a);
        const Node* offset = gen_primop_e(bb, offset_of_op, singleton(type_decl_ref(a, (TypeDeclRef) { .decl = global_struct_t })), singleton(size_t_literal(a,  members_count)));
        // const Node* offset_in_words = bytes_to_words(bb, offset);
        cnst->payload.constant.instruction = yield_values_and_wrap_in_block(bb, singleton(offset));

        register_processed(&ctx->rewriter, decl, cnst);

        members_count++;
    }

    // add some dummy thing so we don't end up with a zero-sized thing, which SPIR-V hates
    if (members_count == 0) {
        member_tys[0] = int32_type(a);
        member_names[0] = "dummy";
        members_count++;
    }

    const Type* record_t = record_type(a, (RecordType) {
        .members = nodes(a, members_count, member_tys),
        .names = strings(a, members_count, member_names)
    });

    //return record_t;
    global_struct_t->payload.nom_type.body = record_t;
}

static void construct_emulated_memory_array(Context* ctx, AddressSpace as, AddressSpace logical_as) {
    IrArena* a = ctx->rewriter.dst_arena;
    Module* m = ctx->rewriter.dst_module;
    String as_name = get_address_space_name(as);
    Nodes annotations = singleton(annotation(a, (Annotation) { .name = "Generated" }));

    Node* global_struct_t = nominal_type(m, annotations, format_string_arena(a->arena, "globals_physical_%s_t", as_name));
    //global_struct_t->payload.nom_type.body = collect_globals_into_record_type(ctx, as);
    collect_globals_into_record_type(ctx, global_struct_t, as);

    // compute the size
    BodyBuilder* bb = begin_body(a);
    const Node* size_of = gen_primop_e(bb, size_of_op, singleton(type_decl_ref(a, (TypeDeclRef) { .decl = global_struct_t })), empty(a));
    const Node* size_in_words = bytes_to_words(bb, size_of);

    const Type* word_type = int_type(a, (Int) { .width = a->config.memory.word_size, .is_signed = false });
    const Type* ptr_size_type = int_type(a, (Int) { .width = a->config.memory.ptr_size, .is_signed = false });

    Node* constant_decl = constant(m, annotations, ptr_size_type, format_string_interned(a, "globals_physical_%s_size", as_name));
    constant_decl->payload.constant.instruction = yield_values_and_wrap_in_block(bb, singleton(size_in_words));

    const Type* words_array_type = arr_type(a, (ArrType) {
        .element_type = word_type,
        .size = ref_decl_helper(a, constant_decl)
    });

    Node* words_array = global_var(m, annotations, words_array_type, format_string_arena(a->arena, "addressable_word_memory_%s", as_name), logical_as);
    *get_emulated_as_word_array(ctx, as) = words_array;
}

Module* lower_physical_ptrs(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));

    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process_node),
        .config = config,
    };

    construct_emulated_memory_array(&ctx, AsPrivatePhysical, AsPrivateLogical);
    if (dst->arena->config.allow_subgroup_memory)
        construct_emulated_memory_array(&ctx, AsSubgroupPhysical, AsSubgroupLogical);
    if (dst->arena->config.allow_shared_memory)
        construct_emulated_memory_array(&ctx, AsSharedPhysical, AsSharedLogical);

    for (size_t i = 0; i < NumAddressSpaces; i++) {
        if (is_as_emulated(&ctx, i)) {
            ctx.serialisation_varying[i] = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
            ctx.deserialisation_varying[i] = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
            ctx.serialisation_uniform[i] = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
            ctx.deserialisation_uniform[i] = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
        }
    }

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);

    for (size_t i = 0; i < NumAddressSpaces; i++) {
        if (is_as_emulated(&ctx, i)) {
            destroy_dict(ctx.serialisation_varying[i]);
            destroy_dict(ctx.deserialisation_varying[i]);
            destroy_dict(ctx.serialisation_uniform[i]);
            destroy_dict(ctx.deserialisation_uniform[i]);
        }
    }

    return dst;
}
