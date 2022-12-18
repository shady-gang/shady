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
#include <string.h>

typedef struct Context_ {
    Rewriter rewriter;
    CompilerConfig* config;

    struct Dict* ser[NumAddressSpaces];
    struct Dict* des[NumAddressSpaces];

    /// Bytes used up by static allocations
    uint32_t preallocated_private_memory;
    uint32_t preallocated_subgroup_memory;

    const Node* thread_private_memory;
    const Node* subgroup_shared_memory;

    bool tpm_is_block_buffer;
    bool ssm_is_block_buffer;
} Context;

static const Node* gen_deserialisation(const CompilerConfig* config, BodyBuilder* bb, const Type* element_type, const Node* arr, const Node* base_offset) {
    const Node* zero = int32_literal(bb->arena, 0);
    switch (element_type->tag) {
        case Bool_TAG: {
            const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset });
            const Node* value = gen_load(bb, logical_ptr);
            return gen_primop_ce(bb, neq_op, 2, (const Node*[]) {value, zero});
        }
        case PtrType_TAG: switch (element_type->payload.ptr_type.address_space) {
            case AsProgramCode: goto ser_int;
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
                    bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(bb->arena, string_lit(bb->arena, (StringLiteral) { .string = "loaded: %d at %d as=%d" }),
                        value, base_offset, int32_literal(bb->arena, extract_operand_type(arr->type)->payload.ptr_type.address_space)) }));
                return value;
            } else {
                // We need to decompose this into two loads, then we use the merge routine
                const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset });
                const Node* lo = gen_load(bb, logical_ptr);
                if (config->printf_trace.memory_accesses)
                    bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(bb->arena, string_lit(bb->arena, (StringLiteral) { .string = "loaded i64 lo: %d at %d as=%d" }),
                       lo, base_offset, int32_literal(bb->arena, extract_operand_type(arr->type)->payload.ptr_type.address_space)) }));
                const Node* hi_destination_offset = gen_primop_ce(bb, add_op, 2, (const Node* []) { base_offset, int32_literal(bb->arena, 1) });
                logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, hi_destination_offset });
                const Node* hi = gen_load(bb, logical_ptr);
                if (config->printf_trace.memory_accesses)
                    bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(bb->arena, string_lit(bb->arena, (StringLiteral) { .string = "loaded i64 hi: %d at %d as=%d" }),
                       hi, hi_destination_offset, int32_literal(bb->arena, extract_operand_type(arr->type)->payload.ptr_type.address_space)) }));

                return gen_merge_i32s_i64(bb, lo, hi);
            }
        }
        case RecordType_TAG: {
            Nodes member_types = element_type->payload.record_type.members;
            LARRAY(FieldLayout, fields, member_types.count);
            get_record_layout(config, bb->arena, element_type, fields);
            LARRAY(const Node*, loaded, member_types.count);
            for (size_t i = 0; i < member_types.count; i++) {
                const Node* field_offset = gen_primop_e(bb, add_op, empty(bb->arena), mk_nodes(bb->arena, base_offset, int32_literal(bb->arena, bytes_to_i32_cells(fields[i].offset_in_bytes))));
                loaded[i] = gen_deserialisation(config, bb, member_types.nodes[i], arr, field_offset);
                TypeMemLayout member_layout = get_mem_layout(config, bb->arena, member_types.nodes[i]);
            }
            return tuple(bb->arena, nodes(bb->arena, member_types.count, loaded));
        }
        case TypeDeclRef_TAG: {
            const Node* nom = element_type->payload.type_decl_ref.decl;
            assert(nom && nom->tag == NominalType_TAG);
            const Node* body = gen_deserialisation(config, bb, nom->payload.nom_type.body, arr, base_offset);
            return first(bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = make_op, .type_arguments = singleton(element_type), .operands = singleton(body) })));
        }
        default: error("TODO");
    }
}

static void gen_serialisation(const CompilerConfig* config, BodyBuilder* bb, const Type* element_type, const Node* arr, const Node* base_offset, const Node* value) {
    const Node* zero = int32_literal(bb->arena, 0);
    switch (element_type->tag) {
        case Bool_TAG: {
            const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset });
            const Node* one = int32_literal(bb->arena, 1);
            const Node* int_value = gen_primop_ce(bb, select_op, 3, (const Node*[]) { value, one, zero });
            gen_store(bb, logical_ptr, int_value);
            return;
        }
        case PtrType_TAG: switch (element_type->payload.ptr_type.address_space) {
            case AsProgramCode: goto des_int;
            default: error("TODO")
        }
        case Int_TAG: des_int: {
            // Same story as for deser
            if (element_type->payload.int_type.width != IntTy64) {
                value = gen_primop_e(bb, reinterpret_op, singleton(int32_type(bb->arena)), singleton(value));
                const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset});
                if (config->printf_trace.memory_accesses)
                    bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(bb->arena, string_lit(bb->arena, (StringLiteral) { .string = "stored: %d at %d as=%d" }),
                        value, base_offset, int32_literal(bb->arena, extract_operand_type(arr->type)->payload.ptr_type.address_space)) }));
                gen_store(bb, logical_ptr, value);
            } else {
                const Node* lo = gen_primop_e(bb, reinterpret_op, singleton(int32_type(bb->arena)), singleton(value));
                const Node* hi = gen_primop_ce(bb, rshift_logical_op, 2, (const Node* []){ value, int64_literal(bb->arena, 32) });
                hi = gen_primop_e(bb, reinterpret_op, singleton(int32_type(bb->arena)), singleton(hi));
                // TODO: make this dependant on the emulation array type
                const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset});
                gen_store(bb, logical_ptr, lo);
                if (config->printf_trace.memory_accesses)
                    bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(bb->arena, string_lit(bb->arena, (StringLiteral) { .string = "stored i64 lo: %d at %d as=%d" }),
                        lo, base_offset, int32_literal(bb->arena, extract_operand_type(arr->type)->payload.ptr_type.address_space)) }));
                const Node* hi_destination_offset = gen_primop_ce(bb, add_op, 2, (const Node* []) { base_offset, int32_literal(bb->arena, 1) });
                logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, hi_destination_offset});
                gen_store(bb, logical_ptr, hi);
                if (config->printf_trace.memory_accesses)
                    bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(bb->arena, string_lit(bb->arena, (StringLiteral) { .string = "stored i64 hi: %d at %d as=%d" }),
                        hi, hi_destination_offset, int32_literal(bb->arena, extract_operand_type(arr->type)->payload.ptr_type.address_space)) }));
            }
            return;
        }
        case RecordType_TAG: {
            Nodes member_types = element_type->payload.record_type.members;
            LARRAY(FieldLayout, fields, member_types.count);
            get_record_layout(config, bb->arena, element_type, fields);
            for (size_t i = 0; i < member_types.count; i++) {
                const Node* extracted_value = first(bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = extract_op, .operands = mk_nodes(bb->arena, value, int32_literal(bb->arena, i)), .type_arguments = empty(bb->arena) })));
                const Node* field_offset = gen_primop_e(bb, add_op, empty(bb->arena), mk_nodes(bb->arena, base_offset, int32_literal(bb->arena, bytes_to_i32_cells(fields[i].offset_in_bytes))));
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

static const Node* gen_serdes_fn(Context* ctx, const Type* element_type, bool ser, AddressSpace as) {
    struct Dict* cache = ser ? ctx->ser[as] : ctx->des[as];

    const Node** found = find_value_dict(const Node*, const Node*, cache, element_type);
    if (found)
        return *found;

    IrArena* arena = ctx->rewriter.dst_arena;

    const Node* addr_param = var(arena, qualified_type(arena, (QualifiedType) { .is_uniform = false, .type = int32_type(arena) }), "ptr");

    const Type* input_value_t = qualified_type(arena, (QualifiedType) { .is_uniform = false, .type = element_type });
    const Node* value_param = ser ? var(arena, input_value_t, "value") : NULL;
    Nodes params = ser ? mk_nodes(arena, addr_param, value_param) : singleton(addr_param);

    const Type* return_value_t = qualified_type(arena, (QualifiedType) { .is_uniform = is_addr_space_uniform(as), .type = element_type });
    Nodes return_ts = ser ? empty(arena) : singleton(return_value_t);

    String name = format_string(arena, "generated_%s_as%d_%s", ser ? "store" : "load", as, name_type_safe(arena, element_type));
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
        base = gen_lea(bb, base, NULL, nodes(arena, 1, (const Node* []) { int32_literal(arena, 0) }));

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

// TODO consume layouts from memory_layout.h
static const Node* lower_lea(Context* ctx, BodyBuilder* instructions, const PrimOp* lea) {
    IrArena* dst_arena = ctx->rewriter.dst_arena;
    const Node* old_pointer = lea->operands.nodes[0];
    const Node* faked_pointer = rewrite_node(&ctx->rewriter, old_pointer);
    const Type* pointer_type = extract_operand_type(old_pointer->type);
    assert(pointer_type->tag == PtrType_TAG);

    const Node* old_offset = lea->operands.nodes[1];
    const IntLiteral* offset_value = resolve_to_literal(old_offset);
    bool offset_is_zero = offset_value && offset_value->value.i64 == 0;
    if (!offset_is_zero) {
        const Type* arr_type = pointer_type->payload.ptr_type.pointed_type;
        assert(arr_type->tag == ArrType_TAG);
        const Type* element_type = arr_type->payload.arr_type.element_type;
        TypeMemLayout element_t_layout = get_mem_layout(ctx->config, ctx->rewriter.dst_arena, element_type);

        const Node* elem_size_val = int32_literal(dst_arena, bytes_to_i32_cells(element_t_layout.size_in_bytes));
        const Node* computed_offset = gen_primop_ce(instructions, mul_op, 2, (const Node* []) { rewrite_node(&ctx->rewriter, old_offset), elem_size_val});

        faked_pointer = gen_primop_ce(instructions, add_op, 2, (const Node* []) { faked_pointer, computed_offset});
    }

    for (size_t i = 2; i < lea->operands.count; i++) {
        assert(pointer_type->tag == PtrType_TAG);
        const Type* pointed_type = pointer_type->payload.ptr_type.pointed_type;
        switch (pointed_type->tag) {
            case ArrType_TAG: {
                const Type* element_type = pointed_type->payload.arr_type.element_type;

                TypeMemLayout element_t_layout = get_mem_layout(ctx->config, ctx->rewriter.dst_arena, element_type);

                const Node* elem_size_val = int32_literal(dst_arena, bytes_to_i32_cells(element_t_layout.size_in_bytes));
                const Node* computed_offset = gen_primop_ce(instructions, mul_op, 2, (const Node* []) { rewrite_node(&ctx->rewriter, lea->operands.nodes[i]), elem_size_val});

                faked_pointer = gen_primop_ce(instructions, add_op, 2, (const Node* []) { faked_pointer, computed_offset});

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
                faked_pointer = gen_primop_ce(instructions, add_op, 2, (const Node* []) { faked_pointer, int32_literal(dst_arena, field_offset)});

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
            case get_stack_base_uniform_op:
            case get_stack_base_op: {
                BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
                const Node* stack_pointer = access_decl(&ctx->rewriter, ctx->rewriter.src_module, oprim_op->op == get_stack_base_op ? "stack_ptr" : "uniform_stack_ptr");
                const Node* stack_size = gen_load(bb, stack_pointer);
                if (ctx->config->printf_trace.stack_size)
                    bind_instruction(bb, prim_op(arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(arena, string_lit(arena, (StringLiteral) { .string = "trace: stack_size=%d uniform=%d" }), stack_size, int32_literal(arena, oprim_op->op != get_stack_base_op )) }));
                return finish_body(bb, let(arena, quote_single(arena, stack_size), tail));
            }
            case lea_op: {
                BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
                const Type* ptr_type = oprim_op->operands.nodes[0]->type;
                ptr_type = extract_operand_type(ptr_type);
                assert(ptr_type->tag == PtrType_TAG);
                if (!is_as_emulated(ctx, ptr_type->payload.ptr_type.address_space))
                    break;
                const Node* new = lower_lea(ctx, bb, oprim_op);
                return finish_body(bb, let(arena, quote_single(arena, new), tail));
            }
            case reinterpret_op: {
                const Type* source_type = first(oprim_op->type_arguments);
                assert(!contains_qualified_type(source_type));
                if (source_type->tag != PtrType_TAG || !is_as_emulated(ctx, source_type->payload.ptr_type.address_space))
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
                ptr_type = extract_operand_type(ptr_type);
                assert(ptr_type->tag == PtrType_TAG);
                if (!is_as_emulated(ctx, ptr_type->payload.ptr_type.address_space))
                    break;
                BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);

                const Type* element_type = rewrite_node(&ctx->rewriter, ptr_type->payload.ptr_type.pointed_type);
                const Node* pointer_as_offset = rewrite_node(&ctx->rewriter, old_ptr);
                const Node* fn = gen_serdes_fn(ctx, element_type, oprim_op->op == store_op, ptr_type->payload.ptr_type.address_space);

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

    if (old->tag == Function_TAG && strcmp(get_abstraction_name(old), "generated_init") == 0) {
        Node* new = recreate_decl_header_identity(&ctx->rewriter, old);
        BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);

        // Make sure to zero-init the stack pointers
        const Node* uniform_stack_pointer = access_decl(&ctx->rewriter, ctx->rewriter.src_module, "uniform_stack_ptr");
        gen_store(bb, uniform_stack_pointer, int32_literal(arena, 0));
        const Node* stack_pointer = access_decl(&ctx->rewriter, ctx->rewriter.src_module, "stack_ptr");
        gen_store(bb, stack_pointer, int32_literal(arena, 0));
        new->payload.fun.body = finish_body(bb, rewrite_node(&ctx->rewriter, old->payload.fun.body));
        return new;
    }

    switch (old->tag) {
        case Let_TAG: return process_let(ctx, old);
        case PtrType_TAG: {
            if (is_as_emulated(ctx, old->payload.ptr_type.address_space))
                return int32_type(arena);

            return recreate_node_identity(&ctx->rewriter, old);
        }
        case GlobalVariable_TAG: {
            const GlobalVariable* old_gvar = &old->payload.global_variable;
            // Global variables into emulated address spaces become integer constants (to index into arrays used for emulation of said address space)
            if (old_gvar->address_space == AsSubgroupPhysical || old_gvar->address_space == AsPrivatePhysical) {
                Nodes annotations = rewrite_nodes(&ctx->rewriter, old_gvar->annotations); // We keep the old annotations

                const char* emulated_heap_name = old_gvar->address_space == AsSubgroupPhysical ? "private" : "subgroup";

                Node* cnst = constant(ctx->rewriter.dst_module, annotations, int32_type(arena), format_string(arena, "%s_offset_%s_arr", old_gvar->name, emulated_heap_name));

                uint32_t* preallocated = old_gvar->address_space == AsSubgroupPhysical ? &ctx->preallocated_subgroup_memory : &ctx->preallocated_private_memory;

                const Type* contents_type = rewrite_node(&ctx->rewriter, old_gvar->type);
                assert(!contains_qualified_type(contents_type));
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

    const Type* stack_base_element = int32_type(dst_arena);
    const Type* stack_arr_type = arr_type(dst_arena, (ArrType) {
        .element_type = stack_base_element,
        .size = uint32_literal(dst_arena, config->per_thread_stack_size * 2),
    });
    const Type* uniform_stack_arr_type = arr_type(dst_arena, (ArrType) {
        .element_type = stack_base_element,
        .size = uint32_literal(dst_arena, config->per_subgroup_stack_size * 2),
    });

    // TODO add a @Synthetic annotation to tag those
    Nodes annotations = nodes(dst_arena, 0, NULL);

    // divide memory up between subgroups in a workgroup
    // TODO decide between shared/global memory for this purpose
    SHADY_UNUSED const Type* wrapped_type = record_type(dst_arena, (RecordType) {
        .members = nodes(dst_arena, 1, (const Node* []) { stack_arr_type }),
        .special = DecorateBlock,
        .names = strings(dst_arena, 0, NULL)
    });

    Node* thread_private_memory = global_var(dst, annotations, stack_arr_type, "physical_private_buffer", AsPrivateLogical);
    Node* subgroup_shared_memory = global_var(dst, annotations, uniform_stack_arr_type, "physical_subgroup_buffer", AsSubgroupLogical);

    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process_node),

        .config = config,

        .preallocated_private_memory = 0,
        .preallocated_subgroup_memory = 0,

        .tpm_is_block_buffer = false,
        .ssm_is_block_buffer = false,

        .thread_private_memory = ref_decl(dst_arena, (RefDecl) { .decl = thread_private_memory }),
        .subgroup_shared_memory = ref_decl(dst_arena, (RefDecl) { .decl = subgroup_shared_memory }),
    };

    for (size_t i = 0; i < NumAddressSpaces; i++) {
        if (is_as_emulated(&ctx, i)) {
            ctx.ser[i] = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
            ctx.des[i] = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
        }
    }

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);

    for (size_t i = 0; i < NumAddressSpaces; i++) {
        if (is_as_emulated(&ctx, i)) {
            destroy_dict(ctx.ser[i]);
            destroy_dict(ctx.des[i]);
        }
    }
}
