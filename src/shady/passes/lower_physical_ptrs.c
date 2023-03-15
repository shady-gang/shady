#include "passes.h"

#include "../transform/memory_layout.h"
#include "../transform/ir_gen_helpers.h"

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

    IntSizes emulated_ptr_width;
    IntSizes emulated_memory_word_size;

    struct Dict*   serialisation_uniform[NumAddressSpaces];
    struct Dict* deserialisation_uniform[NumAddressSpaces];

    struct Dict*   serialisation_varying[NumAddressSpaces];
    struct Dict* deserialisation_varying[NumAddressSpaces];

    /// Bytes used up by static allocations
    uint32_t preallocated_private_memory;
    uint32_t preallocated_subgroup_memory;

    const Node* thread_private_memory;
    const Node* subgroup_shared_memory;

    bool tpm_is_block_buffer;
    bool ssm_is_block_buffer;
} Context;

static IntSizes float_to_int_width(FloatSizes width) {
    switch (width) {
        case FloatTy16: return IntTy16;
        case FloatTy32: return IntTy32;
        case FloatTy64: return IntTy64;
    }
}

static const Node* gen_deserialisation(const CompilerConfig* config, BodyBuilder* bb, const Type* element_type, const Node* arr, const Node* base_offset) {
    assert(get_unqualified_type(base_offset->type) == uint32_type(bb->arena));
    const Node* zero = uint32_literal(bb->arena, 0);
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
                    bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(bb->arena, string_lit(bb->arena, (StringLiteral) { .string = "loaded: %u at %d as=%d" }),
                        value, base_offset, int32_literal(bb->arena, get_unqualified_type(arr->type)->payload.ptr_type.address_space)) }));
                return value;
            } else {
                // We need to decompose this into two loads, then we use the merge routine
                const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset });
                const Node* lo = gen_load(bb, logical_ptr);
                if (config->printf_trace.memory_accesses)
                    bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(bb->arena, string_lit(bb->arena, (StringLiteral) { .string = "loaded i64 lo: %u at %d as=%d" }),
                       lo, base_offset, int32_literal(bb->arena, get_unqualified_type(arr->type)->payload.ptr_type.address_space)) }));
                const Node* hi_destination_offset = gen_primop_ce(bb, add_op, 2, (const Node* []) { base_offset, uint32_literal(bb->arena, 1) });
                logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, hi_destination_offset });
                const Node* hi = gen_load(bb, logical_ptr);
                if (config->printf_trace.memory_accesses)
                    bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(bb->arena, string_lit(bb->arena, (StringLiteral) { .string = "loaded i64 hi: %u at %d as=%d" }),
                       hi, hi_destination_offset, int32_literal(bb->arena, get_unqualified_type(arr->type)->payload.ptr_type.address_space)) }));

                const Node* merged = gen_merge_halves(bb, lo, hi);
                return gen_reinterpret_cast(bb, element_type, merged);
            }
        }
        case Float_TAG: {
            const Type* unsigned_int_t = int_type(bb->arena, (Int) {.width = float_to_int_width(element_type->payload.float_type.width), .is_signed = false });
            const Node* unsigned_int = gen_deserialisation(config, bb, unsigned_int_t, arr, base_offset);
            return gen_reinterpret_cast(bb, element_type, unsigned_int);
        }
        case TypeDeclRef_TAG:
        case RecordType_TAG: {
            const Type* compound_type = element_type;
            compound_type = get_maybe_nominal_type_body(compound_type);

            Nodes member_types = compound_type->payload.record_type.members;
            LARRAY(FieldLayout, fields, member_types.count);
            get_record_layout(config, bb->arena, compound_type, fields);
            LARRAY(const Node*, loaded, member_types.count);
            for (size_t i = 0; i < member_types.count; i++) {
                const Node* field_offset = gen_primop_e(bb, add_op, empty(bb->arena), mk_nodes(bb->arena, base_offset, uint32_literal(bb->arena, bytes_to_i32_cells(fields[i].offset_in_bytes))));
                loaded[i] = gen_deserialisation(config, bb, member_types.nodes[i], arr, field_offset);
            }
            return composite(bb->arena, element_type, nodes(bb->arena, member_types.count, loaded));
        }
        default: error("TODO");
    }
}

static void gen_serialisation(const CompilerConfig* config, BodyBuilder* bb, const Type* element_type, const Node* arr, const Node* base_offset, const Node* value) {
    assert(get_unqualified_type(base_offset->type) == uint32_type(bb->arena));
    const Node* zero = uint32_literal(bb->arena, 0);
    switch (element_type->tag) {
        case Bool_TAG: {
            const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset });
            const Node* one = uint32_literal(bb->arena, 1);
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
                    bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(bb->arena, string_lit(bb->arena, (StringLiteral) { .string = "stored: %u at %d as=%d" }),
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
                    bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(bb->arena, string_lit(bb->arena, (StringLiteral) { .string = "stored i64 lo: %u at %d as=%d" }),
                        lo, base_offset, int32_literal(bb->arena, get_unqualified_type(arr->type)->payload.ptr_type.address_space)) }));
                const Node* hi_destination_offset = gen_primop_ce(bb, add_op, 2, (const Node* []) { base_offset, uint32_literal(bb->arena, 1) });
                logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, hi_destination_offset});
                gen_store(bb, logical_ptr, hi);
                if (config->printf_trace.memory_accesses)
                    bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(bb->arena, string_lit(bb->arena, (StringLiteral) { .string = "stored i64 hi: %u at %d as=%d" }),
                        hi, hi_destination_offset, int32_literal(bb->arena, get_unqualified_type(arr->type)->payload.ptr_type.address_space)) }));
            }
            return;
        }
        case Float_TAG: {
            const Type* unsigned_int_t = int_type(bb->arena, (Int) {.width = float_to_int_width(element_type->payload.float_type.width), .is_signed = false });
            const Node* unsigned_value = gen_primop_e(bb, reinterpret_op, singleton(unsigned_int_t), singleton(value));
            return gen_serialisation(config, bb, unsigned_int_t, arr, base_offset, unsigned_value);
        }
        case RecordType_TAG: {
            Nodes member_types = element_type->payload.record_type.members;
            LARRAY(FieldLayout, fields, member_types.count);
            get_record_layout(config, bb->arena, element_type, fields);
            for (size_t i = 0; i < member_types.count; i++) {
                const Node* extracted_value = first(bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = extract_op, .operands = mk_nodes(bb->arena, value, int32_literal(bb->arena, i)), .type_arguments = empty(bb->arena) })));
                const Node* field_offset = gen_primop_e(bb, add_op, empty(bb->arena), mk_nodes(bb->arena, base_offset, uint32_literal(bb->arena, bytes_to_i32_cells(fields[i].offset_in_bytes))));
                gen_serialisation(config, bb, member_types.nodes[i], arr, field_offset, extracted_value);
            }
            return;
        }
        case TypeDeclRef_TAG: {
            const Node* nom = element_type->payload.type_decl_ref.decl;
            assert(nom && nom->tag == NominalType_TAG);
            gen_serialisation(config, bb, nom->payload.nom_type.body, arr, base_offset, value);
            return;
        }
        default: error("TODO");
    }
}

static const Node* gen_serdes_fn(Context* ctx, const Type* element_type, bool uniform, bool ser, AddressSpace as) {
    struct Dict* cache;

    if (uniform)
        cache = ser ? ctx->serialisation_uniform[as] : ctx->deserialisation_uniform[as];
    else
        cache = ser ? ctx->serialisation_varying[as] : ctx->deserialisation_varying[as];

    const Node** found = find_value_dict(const Node*, const Node*, cache, element_type);
    if (found)
        return *found;

    IrArena* arena = ctx->rewriter.dst_arena;

    const Node* addr_param = var(arena, qualified_type(arena, (QualifiedType) { .is_uniform = !arena->config.is_simt || uniform, .type = uint32_type(arena) }), "ptr");

    const Type* input_value_t = qualified_type(arena, (QualifiedType) { .is_uniform = !arena->config.is_simt || uniform, .type = element_type });
    const Node* value_param = ser ? var(arena, input_value_t, "value") : NULL;
    Nodes params = ser ? mk_nodes(arena, addr_param, value_param) : singleton(addr_param);

    const Type* return_value_t = qualified_type(arena, (QualifiedType) { .is_uniform = !arena->config.is_simt || (uniform && is_addr_space_uniform(arena, as)), .type = element_type });
    Nodes return_ts = ser ? empty(arena) : singleton(return_value_t);

    String name = format_string(arena, "generated_%s_as%d_%s_%s", ser ? "store" : "load", as, uniform ? "uniform" : "varying", name_type_safe(arena, element_type));
    Node* fun = function(ctx->rewriter.dst_module, params, name, singleton(annotation(arena, (Annotation) { .name = "Generated" })), return_ts);
    insert_dict(const Node*, Node*, cache, element_type, fun);

    BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);

    const Node* base = NULL;
    bool is_backed_by_block_buffer;
    switch (as) {
        case AsPrivatePhysical:
            base = ctx->thread_private_memory;
            is_backed_by_block_buffer = ctx->tpm_is_block_buffer;
            break;
        case AsSubgroupPhysical:
            base = ctx->subgroup_shared_memory;
            is_backed_by_block_buffer = ctx->ssm_is_block_buffer;
            break;
        default: error("Emulation of this AS is not supported");
    }

    // some address spaces need to put the data in a special 'Block'-based record, so we need an extra lea to match
    if (is_backed_by_block_buffer)
        base = gen_lea(bb, base, NULL, nodes(arena, 1, (const Node* []) { uint32_literal(arena, 0) }));

    if (ser) {
        gen_serialisation(ctx->config, bb, element_type, base, addr_param, value_param);
        fun->payload.fun.body = finish_body(bb, fn_ret(arena, (Return) { .fn = fun, .args = empty(arena) }));
    } else {
        const Node* loaded_value = gen_deserialisation(ctx->config, bb, element_type, base, addr_param);
        assert(loaded_value);
        fun->payload.fun.body = finish_body(bb, fn_ret(arena, (Return) { .fn = fun, .args = singleton(loaded_value) }));
    }
    return fun;
}

// TODO: make this configuration-dependant
static bool is_as_emulated(SHADY_UNUSED Context* ctx, AddressSpace as) {
    switch (as) {
        case AsSubgroupPhysical: return true;
        case AsPrivatePhysical:  return true;
        case AsSharedPhysical: return false; // error("TODO");
        case AsGlobalPhysical: return false; // TODO config
        default: return false;
    }
}

static const Node* convert_offset(BodyBuilder* bb, const Type* dst_type, const Node* src) {
    const Type* src_type = get_unqualified_type(src->type);
    assert(src_type->tag == Int_TAG);
    assert(dst_type->tag == Int_TAG);

    // first convert to final bitsize then bitcast
    const Type* extended_src_t = int_type(bb->arena, (Int) { .width = dst_type->payload.int_type.width, .is_signed = src_type->payload.int_type.is_signed });
    const Node* val = src;
                val = gen_primop_e(bb, convert_op, singleton(extended_src_t), singleton(val));
                val = gen_primop_e(bb, reinterpret_op, singleton(dst_type), singleton(val));
    return val;
}

// TODO consume layouts from memory_layout.h
static const Node* lower_lea(Context* ctx, BodyBuilder* instructions, const PrimOp* lea) {
    IrArena* dst_arena = ctx->rewriter.dst_arena;
    const Node* old_pointer = lea->operands.nodes[0];
    const Node* faked_pointer = rewrite_node(&ctx->rewriter, old_pointer);
    const Type* pointer_type = get_unqualified_type(old_pointer->type);
    const Type* emulated_ptr_t = int_type(dst_arena, (Int) { .width = ctx->emulated_ptr_width, .is_signed = false });
    assert(pointer_type->tag == PtrType_TAG);

    const Node* old_offset = lea->operands.nodes[1];
    const IntLiteral* offset_value = resolve_to_literal(old_offset);
    bool offset_is_zero = offset_value && offset_value->value.i64 == 0;
    if (!offset_is_zero) {
        const Type* arr_type = pointer_type->payload.ptr_type.pointed_type;
        assert(arr_type->tag == ArrType_TAG);
        const Type* element_type = arr_type->payload.arr_type.element_type;
        TypeMemLayout element_t_layout = get_mem_layout(ctx->config, ctx->rewriter.dst_arena, element_type);

        const Node* elem_size_val = uint32_literal(dst_arena, bytes_to_i32_cells(element_t_layout.size_in_bytes));
        const Node* new_offset = convert_offset(instructions, emulated_ptr_t, rewrite_node(&ctx->rewriter, old_offset));
        const Node* physical_offset = gen_primop_ce(instructions, mul_op, 2, (const Node* []) { new_offset, elem_size_val});

        faked_pointer = gen_primop_ce(instructions, add_op, 2, (const Node* []) { faked_pointer, physical_offset});
    }

    for (size_t i = 2; i < lea->operands.count; i++) {
        assert(pointer_type->tag == PtrType_TAG);
        const Type* pointed_type = pointer_type->payload.ptr_type.pointed_type;
        switch (pointed_type->tag) {
            case ArrType_TAG: {
                const Type* element_type = pointed_type->payload.arr_type.element_type;

                TypeMemLayout element_t_layout = get_mem_layout(ctx->config, ctx->rewriter.dst_arena, element_type);

                const Node* elem_size_val = uint32_literal(dst_arena, bytes_to_i32_cells(element_t_layout.size_in_bytes));
                const Node* new_index = convert_offset(instructions, emulated_ptr_t, rewrite_node(&ctx->rewriter, lea->operands.nodes[i]));
                const Node* physical_offset = gen_primop_ce(instructions, mul_op, 2, (const Node* []) {new_index, elem_size_val});

                faked_pointer = gen_primop_ce(instructions, add_op, 2, (const Node* []) {faked_pointer, physical_offset});

                pointer_type = ptr_type(dst_arena, (PtrType) {
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

                const IntLiteral* selector_value = resolve_to_literal(rewrite_node(&ctx->rewriter, lea->operands.nodes[i]));
                assert(selector_value && "selector value must be known for LEA into a record");
                size_t n = selector_value->value.u64;
                assert(n < member_types.count);

                size_t field_offset = get_record_field_offset_in_bytes(ctx->config, dst_arena, pointed_type, n);
                faked_pointer = gen_primop_ce(instructions, add_op, 2, (const Node* []) { faked_pointer, uint32_literal(dst_arena, bytes_to_i32_cells(field_offset))});

                pointer_type = ptr_type(dst_arena, (PtrType) {
                    .pointed_type = member_types.nodes[n],
                    .address_space = pointer_type->payload.ptr_type.address_space
                });
                break;
            }
            default: error("cannot index into this")
        }
    }

    return faked_pointer;
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
            case lea_op: {
                BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
                const Type* ptr_type = oprim_op->operands.nodes[0]->type;
                ptr_type = get_unqualified_type(ptr_type);
                assert(ptr_type->tag == PtrType_TAG);
                if (!is_as_emulated(ctx, ptr_type->payload.ptr_type.address_space))
                    break;
                const Node* new = lower_lea(ctx, bb, oprim_op);
                return finish_body(bb, let(arena, quote_single(arena, new), tail));
            }
            case reinterpret_op: {
                const Type* dest_type = first(oprim_op->type_arguments);
                assert(is_data_type(dest_type));
                if (dest_type->tag != PtrType_TAG || !is_as_emulated(ctx, dest_type->payload.ptr_type.address_space))
                    break;
                BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
                // emulated physical pointers do not care about pointers, they're just ints :frog:
                const Node* imported = rewrite_node(&ctx->rewriter, first(oprim_op->operands));
                return finish_body(bb, let(arena, quote_single(arena, imported), tail));
            }
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

    IrArena* arena = ctx->rewriter.dst_arena;

    switch (old->tag) {
        case Let_TAG: return process_let(ctx, old);
        case PtrType_TAG: {
            if (is_as_emulated(ctx, old->payload.ptr_type.address_space))
                return int_type(arena, (Int) { .width = ctx->emulated_ptr_width, .is_signed = false });

            return recreate_node_identity(&ctx->rewriter, old);
        }
        case GlobalVariable_TAG: {
            const GlobalVariable* old_gvar = &old->payload.global_variable;
            // Global variables into emulated address spaces become integer constants (to index into arrays used for emulation of said address space)
            if (old_gvar->address_space == AsSubgroupPhysical || old_gvar->address_space == AsPrivatePhysical) {
                Nodes annotations = rewrite_nodes(&ctx->rewriter, old_gvar->annotations); // We keep the old annotations
                annotations = append_nodes(arena, annotations, annotation(arena, (Annotation) { .name = "Generated" }));

                const char* emulated_heap_name = old_gvar->address_space == AsPrivatePhysical ? "private" : "subgroup";

                const Type* emulated_ptr_type = int_type(arena, (Int) { .width = ctx->emulated_ptr_width, .is_signed = false });
                Node* cnst = constant(ctx->rewriter.dst_module, annotations, emulated_ptr_type, format_string(arena, "%s_offset_%s_arr", old_gvar->name, emulated_heap_name));

                uint32_t* preallocated = old_gvar->address_space == AsSubgroupPhysical ? &ctx->preallocated_subgroup_memory : &ctx->preallocated_private_memory;

                const Type* contents_type = rewrite_node(&ctx->rewriter, old_gvar->type);
                assert(is_data_type(contents_type));
                uint32_t required_space = bytes_to_i32_cells(get_mem_layout(ctx->config, arena, contents_type).size_in_bytes);

                cnst->payload.constant.value = uint32_literal(arena, *preallocated);
                *preallocated += required_space;

                register_processed(&ctx->rewriter, old, cnst);
                return cnst;
            }
            return recreate_node_identity(&ctx->rewriter, old);
        }
        default: return recreate_node_identity(&ctx->rewriter, old);
    }
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

void lower_physical_ptrs(CompilerConfig* config, Module* src, Module* dst) {
    IrArena* dst_arena = get_module_arena(dst);

    uint32_t per_thread_private_memory = 0, per_subgroup_memory = 0;
    Nodes old_decls = get_module_declarations(src);
    for (size_t i = 0; i < old_decls.count; i++) {
        const Node* decl = old_decls.nodes[i];
        if (decl->tag != GlobalVariable_TAG) continue;
        const Type* type = decl->payload.global_variable.type;
        TypeMemLayout layout = get_mem_layout(config, dst_arena, type);
        switch (decl->payload.global_variable.address_space) {
            case AsPrivatePhysical:
                per_thread_private_memory += bytes_to_i32_cells(layout.size_in_bytes);
                break;
            case AsSubgroupPhysical:
                per_subgroup_memory += bytes_to_i32_cells(layout.size_in_bytes);
                break;
            default: continue;
        }
    }

    // TODO make everything else use this and then make it configurable...
    IntSizes emulated_physical_pointer_width = IntTy32;
    IntSizes emulated_memory_word_size = IntTy32;

    const Type* emulated_memory_base_type = int_type(dst_arena, (Int) { .width = emulated_memory_word_size, .is_signed = false });
    const Type* private_memory_arr_type = arr_type(dst_arena, (ArrType) {
        .element_type = emulated_memory_base_type,
        .size = uint32_literal(dst_arena, per_thread_private_memory),
    });
    const Type* subgroup_memory_arr_type = arr_type(dst_arena, (ArrType) {
        .element_type = emulated_memory_base_type,
        .size = uint32_literal(dst_arena, per_subgroup_memory),
    });

    Nodes annotations = singleton(annotation(dst_arena, (Annotation) { .name = "Generated" }));

    // divide memory up between subgroups in a workgroup
    // TODO decide between shared/global memory for this purpose
    SHADY_UNUSED const Type* wrapped_type = record_type(dst_arena, (RecordType) {
        .members = nodes(dst_arena, 1, (const Node* []) {private_memory_arr_type }),
        .special = DecorateBlock,
        .names = strings(dst_arena, 0, NULL)
    });

    Node* thread_private_memory = global_var(dst, annotations, private_memory_arr_type, "emulated_private_memory", AsPrivateLogical);
    Node* subgroup_shared_memory = global_var(dst, annotations, subgroup_memory_arr_type, "emulated_subgroup_memory", AsSubgroupLogical);

    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process_node),

        .config = config,

        .preallocated_private_memory = 0,
        .preallocated_subgroup_memory = 0,

        .tpm_is_block_buffer = false,
        .ssm_is_block_buffer = false,

        .emulated_memory_word_size = emulated_memory_word_size,
        .emulated_ptr_width = emulated_physical_pointer_width,

        .thread_private_memory = ref_decl(dst_arena, (RefDecl) { .decl = thread_private_memory }),
        .subgroup_shared_memory = ref_decl(dst_arena, (RefDecl) { .decl = subgroup_shared_memory }),
    };

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
