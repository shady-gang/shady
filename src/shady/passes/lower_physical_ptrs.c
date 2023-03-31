#include "passes.h"

#include "../transform/ir_gen_helpers.h"

#include "../ir_private.h"
#include "../rewrite.h"
#include "../type.h"
#include "log.h"
#include "portability.h"

#include "list.h"
#include "dict.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    CompilerConfig* config;

    struct Dict*   serialisation_uniform[NumAddressSpaces];
    struct Dict* deserialisation_uniform[NumAddressSpaces];

    struct Dict*   serialisation_varying[NumAddressSpaces];
    struct Dict* deserialisation_varying[NumAddressSpaces];

    const Node* fake_private_memory;
    const Node* fake_subgroup_memory;
    const Node* fake_shared_memory;
} Context;

static IntSizes float_to_int_width(FloatSizes width) {
    switch (width) {
        case FloatTy16: return IntTy16;
        case FloatTy32: return IntTy32;
        case FloatTy64: return IntTy64;
    }
}

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

static const Node* size_t_literal(Context* ctx, uint64_t value) {
    IrArena* a = ctx->rewriter.dst_arena;
    return int_literal(a, (IntLiteral) { .width = a->config.memory.ptr_size, .is_signed = false, .value.u64 = value });
}

static const Node* bytes_to_words(Context* ctx, BodyBuilder* bb, const Node* bytes) {
    IrArena* a = bb->arena;
    const Type* word_type = int_type(a, (Int) { .width = a->config.memory.word_size, .is_signed = false });
    size_t word_width = get_type_bitwidth(word_type);
    const Node* bytes_per_word = size_t_literal(ctx, word_width / 8);
    return gen_primop_e(bb, div_op, empty(a), mk_nodes(a, bytes, bytes_per_word));
}

static uint64_t bytes_to_words_static(Context* ctx, uint64_t bytes) {
    IrArena* a = ctx->rewriter.dst_arena;
    const Type* word_type = int_type(a, (Int) { .width = a->config.memory.word_size, .is_signed = false });
    uint64_t word_width = get_type_bitwidth(word_type);
    return bytes / ( word_width / 8 );
}

static const Node* gen_deserialisation(Context* ctx, BodyBuilder* bb, const Type* element_type, const Node* arr, const Node* base_offset) {
    const CompilerConfig* config = ctx->config;
    const Node* zero = size_t_literal(ctx, 0);
    switch (element_type->tag) {
        case Bool_TAG: {
            const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset });
            const Node* value = gen_load(bb, logical_ptr);
            return gen_primop_ce(bb, neq_op, 2, (const Node*[]) {value, zero});
        }
        case PtrType_TAG: switch (element_type->payload.ptr_type.address_space) {
            default: error("TODO")
        }
        case Int_TAG: ser_int: {
            if (element_type->payload.int_type.width != IntTy64) {
                // One load suffices
                const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset });
                const Node* value = gen_load(bb, logical_ptr);
                // cast into the appropriate width and throw other bits away
                // note: folding gets rid of identity casts
                value = gen_primop_e(bb, reinterpret_op, singleton(element_type), singleton(value));
                if (config->printf_trace.memory_accesses)
                    bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(bb->arena, string_lit(bb->arena, (StringLiteral) { .string = "loaded: %u at %lu as=%d" }),
                        value, base_offset, int32_literal(bb->arena, get_unqualified_type(arr->type)->payload.ptr_type.address_space)) }));
                return value;
            } else {
                // We need to decompose this into two loads, then we use the merge routine
                const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset });
                const Node* lo = gen_load(bb, logical_ptr);
                if (config->printf_trace.memory_accesses)
                    bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(bb->arena, string_lit(bb->arena, (StringLiteral) { .string = "loaded i64 lo: %u at %lu as=%d" }),
                       lo, base_offset, int32_literal(bb->arena, get_unqualified_type(arr->type)->payload.ptr_type.address_space)) }));
                const Node* hi_destination_offset = gen_primop_ce(bb, add_op, 2, (const Node* []) { base_offset, size_t_literal(ctx, bytes_to_words_static(ctx, 4)) });
                logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, hi_destination_offset });
                const Node* hi = gen_load(bb, logical_ptr);
                if (config->printf_trace.memory_accesses)
                    bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(bb->arena, string_lit(bb->arena, (StringLiteral) { .string = "loaded i64 hi: %u at %lu as=%d" }),
                       hi, hi_destination_offset, int32_literal(bb->arena, get_unqualified_type(arr->type)->payload.ptr_type.address_space)) }));

                const Node* merged = gen_merge_halves(bb, lo, hi);
                return gen_reinterpret_cast(bb, element_type, merged);
            }
        }
        case Float_TAG: {
            const Type* unsigned_int_t = int_type(bb->arena, (Int) {.width = float_to_int_width(element_type->payload.float_type.width), .is_signed = false });
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
                const Node* field_offset = gen_primop_e(bb, offset_of_op, singleton(element_type), singleton(size_t_literal(ctx, i)));
                            field_offset = bytes_to_words(ctx, bb, field_offset);
                const Node* adjusted_offset = gen_primop_e(bb, add_op, empty(bb->arena), mk_nodes(bb->arena, base_offset, field_offset));
                loaded[i] = gen_deserialisation(ctx, bb, member_types.nodes[i], arr, adjusted_offset);
            }
            return composite(bb->arena, element_type, nodes(bb->arena, member_types.count, loaded));
        }
        default: error("TODO");
    }
}

static void gen_serialisation(Context* ctx, BodyBuilder* bb, const Type* element_type, const Node* arr, const Node* base_offset, const Node* value) {
    const CompilerConfig* config = ctx->config;
    const Node* zero = size_t_literal(ctx, 0);
    switch (element_type->tag) {
        case Bool_TAG: {
            const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset });
            const Node* one = size_t_literal(ctx, 1);
            const Node* int_value = gen_primop_ce(bb, select_op, 3, (const Node*[]) { value, one, zero });
            gen_store(bb, logical_ptr, int_value);
            return;
        }
        case PtrType_TAG: switch (element_type->payload.ptr_type.address_space) {
            case AsProgramCode: goto des_int;
            default: error("TODO")
        }
        case Int_TAG: des_int: {
            // First bitcast to unsigned so we always get zero-extension and not sign-extension afterwards
            const Type* element_t_unsigned = int_type(bb->arena, (Int) { .width = element_type->payload.int_type.width, .is_signed = false});
            const Node* unsigned_value = gen_primop_e(bb, reinterpret_op, singleton(element_t_unsigned), singleton(value));
            if (element_type->payload.int_type.width != IntTy64) {
                value = unsigned_value;
                value = gen_primop_e(bb, convert_op, singleton(uint32_type(bb->arena)), singleton(value));
                const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset});
                if (config->printf_trace.memory_accesses)
                    bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(bb->arena, string_lit(bb->arena, (StringLiteral) { .string = "stored: %u at %lu as=%d" }),
                        value, base_offset, int32_literal(bb->arena, get_unqualified_type(arr->type)->payload.ptr_type.address_space)) }));
                gen_store(bb, logical_ptr, value);
            } else {
                const Node* lo = unsigned_value;
                            lo = gen_primop_e(bb, convert_op, singleton(uint32_type(bb->arena)), singleton(lo));
                const Node* hi = unsigned_value;
                            hi = gen_primop_e(bb, rshift_logical_op, empty(bb->arena), mk_nodes(bb->arena, hi, uint64_literal(bb->arena, 32)));
                            hi = gen_primop_e(bb, convert_op, singleton(uint32_type(bb->arena)), singleton(hi));
                // TODO: make this dependant on the emulation array type
                const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset});
                gen_store(bb, logical_ptr, lo);
                if (config->printf_trace.memory_accesses)
                    bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(bb->arena, string_lit(bb->arena, (StringLiteral) { .string = "stored i64 lo: %u at %lu as=%d" }),
                        lo, base_offset, int32_literal(bb->arena, get_unqualified_type(arr->type)->payload.ptr_type.address_space)) }));
                const Node* hi_destination_offset = gen_primop_ce(bb, add_op, 2, (const Node* []) { base_offset, size_t_literal(ctx, bytes_to_words_static(ctx, 4)) });
                logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, hi_destination_offset});
                gen_store(bb, logical_ptr, hi);
                if (config->printf_trace.memory_accesses)
                    bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(bb->arena, string_lit(bb->arena, (StringLiteral) { .string = "stored i64 hi: %u at %lu as=%d" }),
                        hi, hi_destination_offset, int32_literal(bb->arena, get_unqualified_type(arr->type)->payload.ptr_type.address_space)) }));
            }
            return;
        }
        case Float_TAG: {
            const Type* unsigned_int_t = int_type(bb->arena, (Int) {.width = float_to_int_width(element_type->payload.float_type.width), .is_signed = false });
            const Node* unsigned_value = gen_primop_e(bb, reinterpret_op, singleton(unsigned_int_t), singleton(value));
            return gen_serialisation(ctx, bb, unsigned_int_t, arr, base_offset, unsigned_value);
        }
        case RecordType_TAG: {
            Nodes member_types = element_type->payload.record_type.members;
            for (size_t i = 0; i < member_types.count; i++) {
                const Node* extracted_value = first(bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = extract_op, .operands = mk_nodes(bb->arena, value, int32_literal(bb->arena, i)), .type_arguments = empty(bb->arena) })));
                const Node* field_offset = gen_primop_e(bb, offset_of_op, singleton(element_type), singleton(size_t_literal(ctx, i)));
                            field_offset = bytes_to_words(ctx, bb, field_offset);
                const Node* adjusted_offset = gen_primop_e(bb, add_op, empty(bb->arena), mk_nodes(bb->arena, base_offset, field_offset));
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
        default: error("TODO");
    }
}

static const Node* gen_serdes_fn(Context* ctx, const Type* element_type, bool uniform, bool ser, AddressSpace as) {
    assert(is_as_emulated(ctx, as));
    struct Dict* cache;

    if (uniform)
        cache = ser ? ctx->serialisation_uniform[as] : ctx->deserialisation_uniform[as];
    else
        cache = ser ? ctx->serialisation_varying[as] : ctx->deserialisation_varying[as];

    const Node** found = find_value_dict(const Node*, const Node*, cache, element_type);
    if (found)
        return *found;

    IrArena* arena = ctx->rewriter.dst_arena;

    const Type* emulated_ptr_type = int_type(arena, (Int) { .width = arena->config.memory.ptr_size, .is_signed = false });
    const Node* address_param = var(arena, qualified_type(arena, (QualifiedType) { .is_uniform = !arena->config.is_simt || uniform, .type = emulated_ptr_type }), "ptr");

    const Type* input_value_t = qualified_type(arena, (QualifiedType) { .is_uniform = !arena->config.is_simt || uniform, .type = element_type });
    const Node* value_param = ser ? var(arena, input_value_t, "value") : NULL;
    Nodes params = ser ? mk_nodes(arena, address_param, value_param) : singleton(address_param);

    const Type* return_value_t = qualified_type(arena, (QualifiedType) { .is_uniform = !arena->config.is_simt || (uniform && is_addr_space_uniform(arena, as)), .type = element_type });
    Nodes return_ts = ser ? empty(arena) : singleton(return_value_t);

    String name = format_string(arena, "generated_%s_as%d_%s_%s", ser ? "store" : "load", as, uniform ? "uniform" : "varying", name_type_safe(arena, element_type));
    Node* fun = function(ctx->rewriter.dst_module, params, name, singleton(annotation(arena, (Annotation) { .name = "Generated" })), return_ts);
    insert_dict(const Node*, Node*, cache, element_type, fun);

    BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
    const Node* address = bytes_to_words(ctx, bb, address_param);
    const Node* base = ref_decl(arena, (RefDecl) { .decl = *get_emulated_as_word_array(ctx, as) });
    if (ser) {
        gen_serialisation(ctx, bb, element_type, base, address, value_param);
        fun->payload.fun.body = finish_body(bb, fn_ret(arena, (Return) { .fn = fun, .args = empty(arena) }));
    } else {
        const Node* loaded_value = gen_deserialisation(ctx, bb, element_type, base, address);
        assert(loaded_value);
        fun->payload.fun.body = finish_body(bb, fn_ret(arena, (Return) { .fn = fun, .args = singleton(loaded_value) }));
    }
    return fun;
}

static const Node* process_let(Context* ctx, const Node* node) {
    assert(node->tag == Let_TAG);
    IrArena* arena = ctx->rewriter.dst_arena;

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
                BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);

                const Type* element_type = rewrite_node(&ctx->rewriter, ptr_type->payload.ptr_type.pointed_type);
                const Node* pointer_as_offset = rewrite_node(&ctx->rewriter, old_ptr);
                const Node* fn = gen_serdes_fn(ctx, element_type, uniform_ptr, oprim_op->op == store_op, ptr_type->payload.ptr_type.address_space);

                if (oprim_op->op == load_op) {
                    const Node* result = first(bind_instruction(bb, leaf_call(arena, (LeafCall) { .callee = fn, .args = singleton(pointer_as_offset) })));
                    return finish_body(bb, let(arena, quote_single(arena, result), tail));
                } else {
                    const Node* value = rewrite_node(&ctx->rewriter, oprim_op->operands.nodes[1]);

                    bind_instruction(bb, leaf_call(arena, (LeafCall) { .callee = fn, .args = mk_nodes(arena, pointer_as_offset, value) }));
                    return finish_body(bb, let(arena, unit(arena), tail));
                }
                SHADY_UNREACHABLE;
            }
            default: break;
        }
    }

    return let(arena, rewrite_node(&ctx->rewriter, old_instruction), tail);
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

            return recreate_node_identity(&ctx->rewriter, old);
        }
        case GlobalVariable_TAG: {
            const GlobalVariable* old_gvar = &old->payload.global_variable;
            // Global variables into emulated address spaces become integer constants (to index into arrays used for emulation of said address space)
            if (is_as_emulated(ctx, old_gvar->address_space)) {
                assert(false);
            }
            return recreate_node_identity(&ctx->rewriter, old);
        }
        default: return recreate_node_identity(&ctx->rewriter, old);
    }
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
        BodyBuilder* bb = begin_body(m);
        const Node* offset = gen_primop_e(bb, offset_of_op, singleton(type_decl_ref(a, (TypeDeclRef) { .decl = global_struct_t })), singleton(size_t_literal(ctx,  members_count)));
        // const Node* offset_in_words = bytes_to_words(ctx, bb, offset);
        cnst->payload.constant.value = anti_quote(a, (AntiQuote) {
            .instruction = yield_values_and_wrap_in_block(bb, singleton(offset))
        });

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
    String as_name = format_string(a, "as_%d", as);
    Nodes annotations = singleton(annotation(a, (Annotation) { .name = "Generated" }));

    Node* global_struct_t = nominal_type(m, annotations, format_string(a, "globals_physical_%s_t", as_name));
    //global_struct_t->payload.nom_type.body = collect_globals_into_record_type(ctx, as);
    collect_globals_into_record_type(ctx, global_struct_t, as);

    // compute the size
    BodyBuilder* bb = begin_body(m);
    const Node* size_of = gen_primop_e(bb, size_of_op, singleton(type_decl_ref(a, (TypeDeclRef) { .decl = global_struct_t })), empty(a));
    const Node* size_in_words = bytes_to_words(ctx, bb, size_of);

    const Type* word_type = int_type(a, (Int) { .width = a->config.memory.word_size, .is_signed = false });
    const Type* words_array_type = arr_type(a, (ArrType) {
        .element_type = word_type,
        .size = anti_quote(a, (AntiQuote) {
            .instruction = yield_values_and_wrap_in_block(bb, singleton(size_in_words))
        }),
    });

    Node* words_array = global_var(m, annotations, words_array_type, format_string(a, "addressable_word_memory_%s", as_name), logical_as);
    *get_emulated_as_word_array(ctx, as) = words_array;
}

void lower_physical_ptrs(CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process_node),
        .config = config,
    };

    construct_emulated_memory_array(&ctx, AsPrivatePhysical, AsPrivateLogical);
    construct_emulated_memory_array(&ctx, AsSubgroupPhysical, AsSubgroupLogical);
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
}
