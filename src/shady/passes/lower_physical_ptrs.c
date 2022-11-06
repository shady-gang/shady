#include "passes.h"

#include "../transform/memory_layout.h"
#include "../transform/ir_gen_helpers.h"

#include "../rewrite.h"
#include "../type.h"
#include "log.h"
#include "portability.h"

#include "list.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    CompilerConfig* config;

    /// Bytes used up by static allocations
    uint32_t preallocated_private_memory;
    uint32_t preallocated_subgroup_memory;

    const Node* thread_private_memory;
    const Node* subgroup_shared_memory;

    bool tpm_is_block_buffer;
    bool ssm_is_block_buffer;
} Context;

// TODO: make this configuration-dependant
static bool is_as_emulated(SHADY_UNUSED Context* ctx, AddressSpace as) {
    switch (as) {
        case AsSubgroupPhysical: return true;
        case AsPrivatePhysical:  return true;
        case AsSharedPhysical: error("TODO");
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
    if (old_offset) {
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
            case RecordType_TAG: {
                error("TODO");
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
            case alloca_op: error("This has to be the slot variant")
            case alloca_slot_op: {
                BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
                const Type* element_type = rewrite_node(&ctx->rewriter, first(oprim_op->type_arguments));
                TypeMemLayout layout = get_mem_layout(ctx->config, arena, element_type);

                const Node* stack_pointer = access_decl(&ctx->rewriter, ctx->rewriter.src_module, "stack_ptr");
                const Node* stack_size = gen_load(bb, stack_pointer);
                stack_size = gen_primop_ce(bb, add_op, 2, (const Node* []) { stack_size, int32_literal(arena, bytes_to_i32_cells(layout.size_in_bytes)) });
                gen_store(bb, stack_pointer, stack_size);
                return finish_body(bb, let(arena, quote(arena, stack_size), tail));
            }
            case lea_op: {
                BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
                const Type* ptr_type = oprim_op->operands.nodes[0]->type;
                ptr_type = extract_operand_type(ptr_type);
                assert(ptr_type->tag == PtrType_TAG);
                if (!is_as_emulated(ctx, ptr_type->payload.ptr_type.address_space))
                    break;
                const Node* new = lower_lea(ctx, bb, oprim_op);
                return finish_body(bb, let(arena, quote(arena, new), tail));
            }
            case reinterpret_op: {
                const Type* source_type = first(oprim_op->type_arguments);
                assert(!contains_qualified_type(source_type));
                if (source_type->tag != PtrType_TAG || !is_as_emulated(ctx, source_type->payload.ptr_type.address_space))
                    break;
                BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
                // emulated physical pointers do not care about pointers, they're just ints :frog:
                const Node* imported = rewrite_node(&ctx->rewriter, first(oprim_op->operands));
                return finish_body(bb, let(arena, quote(arena, imported), tail));
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

                const Node* base = NULL;
                bool is_backed_by_block_buffer;
                switch (ptr_type->payload.ptr_type.address_space) {
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

                const Node* pointer_as_offset = rewrite_node(&ctx->rewriter, old_ptr);
                if (oprim_op->op == load_op) {
                    const Node* result = gen_deserialisation(ctx->config, bb, element_type, base, pointer_as_offset);
                    return finish_body(bb, let(arena, quote(arena, result), tail));
                } else {
                    const Node* value = rewrite_node(&ctx->rewriter, oprim_op->operands.nodes[1]);
                    gen_serialisation(ctx->config, bb, element_type, base, pointer_as_offset, value);
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

    switch (old->tag) {
        case Let_TAG: return process_let(ctx, old);
        case PtrType_TAG: {
            if (is_as_emulated(ctx, old->payload.ptr_type.address_space))
                return int32_type(ctx->rewriter.dst_arena);

            return recreate_node_identity(&ctx->rewriter, old);
        }
        case GlobalVariable_TAG: {
            const GlobalVariable* old_gvar = &old->payload.global_variable;
            // Global variables into emulated address spaces become integer constants (to index into arrays used for emulation of said address space)
            if (old_gvar->address_space == AsSubgroupPhysical || old_gvar->address_space == AsPrivatePhysical) {
                Nodes annotations = rewrite_nodes(&ctx->rewriter, old_gvar->annotations); // We keep the old annotations

                const char* emulated_heap_name = old_gvar->address_space == AsSubgroupPhysical ? "private" : "subgroup";

                Node* cnst = constant(ctx->rewriter.dst_module, annotations, int32_type(ctx->rewriter.dst_arena), format_string(ctx->rewriter.dst_arena, "%s_offset_%s_arr", old_gvar->name, emulated_heap_name));

                uint32_t* preallocated = old_gvar->address_space == AsSubgroupPhysical ? &ctx->preallocated_subgroup_memory : &ctx->preallocated_private_memory;
                const Type* contents_type = rewrite_node(&ctx->rewriter, old_gvar->type);
                assert(!contains_qualified_type(contents_type));
                uint32_t required_space = bytes_to_i32_cells(get_mem_layout(ctx->config, ctx->rewriter.dst_arena, contents_type).size_in_bytes);

                cnst->payload.constant.value = uint32_literal(ctx->rewriter.dst_arena, *preallocated);
                *preallocated += required_space;

                register_processed(&ctx->rewriter, old, cnst);
                return cnst;
            }
            return recreate_node_identity(&ctx->rewriter, old);
        }
        default: return recreate_node_identity(&ctx->rewriter, old);
    }
}

static void update_base_stack_ptrs(Context* ctx) {
    Node* per_thread_stack_ptr = (Node*) find_or_process_decl(&ctx->rewriter, ctx->rewriter.src_module, "stack_ptr");
    assert(per_thread_stack_ptr && per_thread_stack_ptr->tag == GlobalVariable_TAG);
    per_thread_stack_ptr->payload.global_variable.init = uint32_literal(ctx->rewriter.dst_arena, ctx->preallocated_private_memory);
    Node* subgroup_stack_ptr = (Node*) find_or_process_decl(&ctx->rewriter, ctx->rewriter.src_module, "uniform_stack_ptr");
    assert(subgroup_stack_ptr && subgroup_stack_ptr->tag == GlobalVariable_TAG);
    subgroup_stack_ptr->payload.global_variable.init = uint32_literal(ctx->rewriter.dst_arena, ctx->preallocated_subgroup_memory);
}

void lower_physical_ptrs(CompilerConfig* config, Module* src, Module* dst) {
    IrArena* dst_arena = get_module_arena(dst);

    const Type* stack_base_element = int32_type(dst_arena);
    const Type* stack_arr_type = arr_type(dst_arena, (ArrType) {
        .element_type = stack_base_element,
        .size = uint32_literal(dst_arena, config->per_thread_stack_size),
    });
    const Type* uniform_stack_arr_type = arr_type(dst_arena, (ArrType) {
        .element_type = stack_base_element,
        .size = uint32_literal(dst_arena, config->per_subgroup_stack_size),
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
    Node* subgroup_shared_memory = global_var(dst, annotations, uniform_stack_arr_type, "physical_subgroup_buffer", AsSharedLogical);

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

    rewrite_module(&ctx.rewriter);
    update_base_stack_ptrs(&ctx);
    destroy_rewriter(&ctx.rewriter);
}
